from typing import Optional
from types import SimpleNamespace
from functools import partial

import numpy as np
from gym.spaces import Space

import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import optax

from safe_adaptation_agents.agents.on_policy import ppo_lagrangian, vpg
from safe_adaptation_agents.agents.on_policy.safe_vpg import Evaluation
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState
from safe_adaptation_agents import utils
from safe_adaptation_agents import episodic_trajectory_buffer as etb
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData


class MamlPpoLagrangian(ppo_lagrangian.PpoLagrangian):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(MamlPpoLagrangian,
          self).__init__(observation_space, action_space, config, logger, actor,
                         critic, safety_critic)
    num_steps = self.config.time_limit // self.config.action_repeat
    self.buffer = etb.EpisodicTrajectoryBuffer(
        self.config.num_trajectories + self.config.num_query_trajectories,
        num_steps, observation_space.shape, action_space.shape,
        self.config.task_batch_size)
    self.task_id = -1
    lrs = tuple(
        map(utils.inv_softplus,
            (self.config.lagrangian_inner_lr, self.config.policy_inner_lr)))
    self.inner_lrs = utils.Learner(
        lrs, next(self.rng_seq), self.config.inner_lr_opt,
        utils.get_mixed_precision_policy(config.precision))
    # The adapted/fine-tuned policy parameters for the last seen task.
    self.pi_posterior = None
    # Save a posterior of each task for the outer update steps. See ProMP,
    # Rothfuss et al (2018) @ https://arxiv.org/abs/1810.06784
    self.pi_posteriors = [None] * self.config.task_batch_size

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if train and self.buffer.full:
      assert adapt, (
          'Should train at the first step of adaptation (after filling up the '
          'buffer with adaptation and query data)')
      self.train(self.buffer.dump())
      self.logger.log_metrics(self.training_step)
    # Use the prior parameters on only adaptation phase, otherwise use prior.
    policy_params = self.actor.params if adapt else self.pi_posterior
    action = self.policy(observation, policy_params, next(self.rng_seq), train,
                         adapt)
    return action

  @partial(jax.jit, static_argnums=(0, 4, 5))
  def policy(self, observation: jnp.ndarray, params: hk.Params,
             key: jnp.ndarray, train: bool, adapt: bool) -> jnp.ndarray:
    policy = self.actor.apply(params, observation)
    # Take the mode only on query episodes in which we evaluate the agent.
    if not adapt and not train:
      action = policy.mode()
    else:
      action = policy.sample(seed=key)
    return action

  def observe_task_id(self, task_id: Optional[str] = None):
    self.task_id = (self.task_id + 1) % self.config.task_batch_size
    self.buffer.set_task(self.task_id)

  def train(self, trajectory_data: TrajectoryData):
    sum_trajectories = (
        self.config.num_trajectories + self.config.num_query_trajectories)
    assert sum_trajectories == self.buffer.observation.shape[1]
    split = lambda x: np.split(
        x,
        (self.config.num_trajectories, sum_trajectories),
        axis=1,
    )
    support, query, _ = zip(*map(split, trajectory_data))
    support = TrajectoryData(*support)
    query = TrajectoryData(*query)
    (
        self.lagrangian.state,
        self.actor.state,
        self.inner_lrs.state,
        info,
    ) = self.update_priors(self.lagrangian.state, self.actor.state,
                           self.inner_lrs.state, support, query,
                           utils.pytrees_stack(self.pi_posteriors))
    info['agent/lagrangian_lr'] = jnn.softplus(self.inner_lrs.params[0])
    info['agent/policy_lr'] = jnn.softplus(self.inner_lrs.params[1])
    for k, v in info.items():
      self.logger[k] = v.mean()

  @partial(jax.jit, static_argnums=0)
  def update_priors(self, lagrangian_state: LearningState,
                    actor_state: LearningState, inner_lr_state: LearningState,
                    support: TrajectoryData, query: TrajectoryData,
                    old_pi_posteriors: hk.Params) -> [LearningState, dict]:
    if self.config.safe:
      (lagrangian_loss, lagrangians), grads = jax.value_and_grad(
          self.lagrangian_meta_loss, (0, 1),
          has_aux=True)(lagrangian_state.params, inner_lr_state.params, support,
                        query)
      lagrangian_grads, lr_grads = grads
      new_lagrangian_state = self.lagrangian.grad_step(lagrangian_grads,
                                                       lagrangian_state)
      inner_lr_state = self.inner_lrs.grad_step(lr_grads, inner_lr_state)
    else:
      lagrangians = jnp.zeros((support.o.shape[0],))
      new_lagrangian_state = lagrangian_state
    support_eval, query_eval = self.evaluate_support_and_query(
        self.critic.state, self.safety_critic.state, support, query)
    old_pi_support = self.actor.apply(actor_state.params, support.o[:, :, :-1])
    old_pi_support_logprob = old_pi_support.log_prob(support.a)
    old_posterior_pi_logprobs = (
        lambda params, o, a: self.actor.apply(params, o).log_prob(a))
    old_pi_query_logprob = jax.vmap(old_posterior_pi_logprobs)(
        old_pi_posteriors, query.o[:, :, :-1], query.a)

    def cond(val):
      iter_, *_, info = val
      kl = info['agent/actor/delta_kl']
      # Returns Truthy if iter is smaller than pi_iters and kl smaller than
      # kl threshold to continue iterating. Enforces a hard constraint on the
      # KL divergences between the new and old priors (see
      # https://spinningup.openai.com/en/latest/algorithms/ppo.html).
      return jax.lax.bitwise_not(
          jax.lax.bitwise_or(
              kl > self.config.kl_margin * self.config.target_kl,
              iter_ == self.config.pi_iters,
          ))

    def body(val):
      (iter_, actor_state, lr_state, _) = val
      (loss, kl_d), grads = jax.value_and_grad(
          self.policy_meta_loss, (0, 1),
          has_aux=True)(actor_state.params, lr_state.params, lagrangians,
                        support, query, support_eval, query_eval,
                        old_pi_support_logprob, old_pi_query_logprob,
                        old_pi_posteriors)
      pi_grads, lr_grads = grads
      new_actor_state = self.actor.grad_step(pi_grads, actor_state)
      new_lr_state = self.inner_lrs.grad_step(lr_grads, inner_lr_state)
      new_pi = self.actor.apply(new_actor_state.params, support.o[:, :, :-1])
      report = {
          'agent/actor/loss': loss,
          'agent/actor/entropy': new_pi.entropy().mean(),
          'agent/actor/delta_kl': kl_d,
          'agent/actor/pi_grads': optax.global_norm(pi_grads)
      }
      out = (iter_ + 1, new_actor_state, new_lr_state, report)
      return out

    init_state = (0, actor_state, inner_lr_state, {
        'agent/actor/loss': 0.,
        'agent/actor/entropy': 0.,
        'agent/actor/delta_kl': 0.,
        'agent/actor/pi_grads': 0.
    })
    (
        iters,
        new_actor_state,
        new_lr_state,
        info,
    ) = jax.lax.while_loop(cond, body, init_state)  # noqa
    new_lagrangian = self.lagrangian.apply(new_lagrangian_state.params)
    info['agent/lagrangian_prior'] = jnn.softplus(new_lagrangian)
    info['agent/actor/update_iters'] = iters
    return new_lagrangian_state, new_actor_state, new_lr_state, info

  @partial(jax.jit, static_argnums=0)
  def evaluate_support_and_query(
      self, critic_state: LearningState, safety_critic_state: LearningState,
      support: TrajectoryData,
      query: TrajectoryData) -> [Evaluation, Evaluation]:
    # Fit a critic for each task, evaluate the policy's performance with it.
    batched_adapt_critic_and_evaluate = jax.vmap(
        partial(
            self.adapt_critics_and_evaluate,
            critic_state,
            safety_critic_state,
        ))
    support_eval = batched_adapt_critic_and_evaluate(support.o, support.r,
                                                     support.c)
    query_eval = batched_adapt_critic_and_evaluate(query.o, query.r, query.c)
    return support_eval, query_eval

  def lagrangian_meta_loss(self, prior: hk.Params, learning_rates: hk.Params,
                           support: TrajectoryData, query: TrajectoryData):
    learning_rate, _ = learning_rates
    support_constraints = support.c.sum(2).mean(1)
    query_constraints = query.c.sum(2).mean(1)

    def task_loss(support_constraint, query_constraint):
      lagrangian_posterior = self.adapt_lagrangian(prior, learning_rate,
                                                   support_constraint)
      loss = self.lagrangian_loss(lagrangian_posterior, query_constraint)
      lagrangian_posterior = self.lagrangian.apply(lagrangian_posterior)
      lagrangian_posterior = jnn.softplus(lagrangian_posterior)
      return loss, lagrangian_posterior

    task_loss = jax.vmap(task_loss)
    losses, lagrangians = task_loss(support_constraints, query_constraints)
    return losses.mean(), lagrangians

  def policy_meta_loss(self, prior: hk.Params, learning_rates: hk.Params,
                       lagrangians: jnp.ndarray, support: TrajectoryData,
                       query: TrajectoryData, support_eval: Evaluation,
                       query_eval: Evaluation,
                       old_pi_support_logprob: jnp.ndarray,
                       old_pi_query_logprob: jnp.ndarray,
                       old_pi_posteriors: hk.Params):
    _, learning_rate = learning_rates

    def task_loss(support, query, support_eval, query_eval,
                  old_pi_support_logprob, lagrangian, old_pi_query_logprob,
                  old_pi_posterior):
      pi_posterior = self.adapt_policy(prior, learning_rate, lagrangian,
                                       support.o[:, :-1], support.a,
                                       support_eval.advantage,
                                       support_eval.cost_advantage,
                                       old_pi_support_logprob)
      loss = self.policy_loss(pi_posterior, query.o[:, :-1], query.a,
                              query_eval.advantage, query_eval.cost_advantage,
                              lagrangian, old_pi_query_logprob)
      # Compute the KL-divs of the new posterior. We could compute old_pi
      # outside but computing it here makes the code clearer.
      pi = self.actor.apply(pi_posterior, query.o[:, :-1])
      old_pi = self.actor.apply(old_pi_posterior, query.o[:, :-1])
      kl = pi.kl_divergence(old_pi)
      return loss, kl.mean()

    task_loss = jax.vmap(task_loss)
    losses, kls = task_loss(support, query, support_eval, query_eval,
                            old_pi_support_logprob, lagrangians,
                            old_pi_query_logprob, old_pi_posteriors)
    return losses.mean(), kls.mean()

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    # Evaluate the policy given the recent data.
    eval_ = self.adapt_critics_and_evaluate(self.critic.state,
                                            self.safety_critic.state,
                                            observation, reward, cost)
    lagrangian_lr, pi_lr = self.inner_lrs.params
    old_pi = self.actor.apply(self.actor.params, observation[:, :-1])
    old_pi_logprob = old_pi.log_prob(action)
    constraint = cost.sum(1).mean()
    # Adapt the lagrangian and policy to get a MAP of the policy's poterior.
    pi_posterior = self.task_adaptation(self.lagrangian.params,
                                        self.actor.params, lagrangian_lr, pi_lr,
                                        observation[:, :-1], action,
                                        eval_.advantage, eval_.cost_advantage,
                                        constraint, old_pi_logprob)
    # Keep the posterior's MAP to the query data set.
    self.pi_posterior = pi_posterior
    # Keep this posterior for later use as part of training.
    if train:
      self.pi_posteriors[self.task_id] = pi_posterior

  def adapt_lagrangian(self, prior: hk.Params, learning_rate: jnp.ndarray,
                       costraint: jnp.ndarray):
    learning_rate = jnn.softplus(learning_rate)
    new_lagrangian = prior
    for _ in range(self.config.inner_steps):
      grads = jax.grad(self.lagrangian_loss)(prior, costraint)
      new_lagrangian = utils.gradient_descent(grads, new_lagrangian,
                                              learning_rate)
    return new_lagrangian

  def adapt_policy(self, prior: hk.Params, learning_rate: jnp.ndarray,
                   lagrangian: jnp.ndarray, observation: jnp.ndarray,
                   action: jnp.ndarray, advantage: jnp.ndarray,
                   cost_advantage: jnp.ndarray, old_pi_logprob: jnp.ndarray):
    learning_rate = jnn.softplus(learning_rate)
    new_pi = prior
    for _ in range(self.config.inner_steps):
      grads = jax.grad(self.policy_loss)(
          new_pi,
          observation,
          action,
          advantage,
          cost_advantage,
          lagrangian,
          old_pi_logprob,
          clip=False)
      new_pi = utils.gradient_descent(grads, new_pi, learning_rate)
    return new_pi

  @partial(jax.jit, static_argnums=0)
  def task_adaptation(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, lagrangian_lr: jnp.ndarray,
                      pi_lr: jnp.ndarray, observation: jnp.ndarray,
                      action: jnp.ndarray, advantage: jnp.ndarray,
                      cost_advantage: jnp.ndarray, constraint: jnp.ndarray,
                      old_pi_logprob: np.ndarray) -> [hk.Params]:
    if self.config.safe:
      lagrangian_posterior = self.adapt_lagrangian(lagrangian_prior,
                                                   lagrangian_lr, constraint)
      lagrangian = jnn.softplus(self.lagrangian.apply(lagrangian_posterior))
    else:
      lagrangian = 0.
    policy_posterior = self.adapt_policy(policy_prior, pi_lr, lagrangian,
                                         observation, action, advantage,
                                         cost_advantage, old_pi_logprob)
    return policy_posterior

  @partial(jax.jit, static_argnums=0)
  def adapt_critics_and_evaluate(self, critic_state: LearningState,
                                 safety_critic_state: LearningState,
                                 observation: np.ndarray, reward: np.ndarray,
                                 cost: np.ndarray) -> Evaluation:
    # Fit the critics to the returns of the given task and return the
    # evaluation.
    reward_return = vpg.discounted_cumsum(reward, self.config.discount)
    cost_return = vpg.discounted_cumsum(cost, self.config.cost_discount)
    critic_state, _ = self.update_critic(critic_state, observation[:, :-1],
                                         reward_return)
    if self.safe:
      safety_critic_states, _ = self.update_safety_critic(
          safety_critic_state, observation[:, :-1], cost_return)
    else:
      safety_critic_states = critic_state
    return self.evaluate_with_safety(critic_state.params,
                                     safety_critic_states.params, observation,
                                     reward, cost)
