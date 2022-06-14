from typing import Tuple, Optional
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
      (iter_, lagrangian_state, actor_state, lr_state, _) = val
      (loss, kl_d), grads = jax.value_and_grad(
          self.meta_loss, (0, 1, 2),
          has_aux=True)(lagrangian_state.params, actor_state.params,
                        lr_state.params, support, query, support_eval,
                        query_eval, old_pi_support_logprob,
                        old_pi_query_logprob, old_pi_posteriors)
      lagrangian_grads, pi_grads, lr_grads = grads
      new_actor_state = self.actor.grad_step(pi_grads, actor_state)
      new_lagrangian_state = self.lagrangian.grad_step(lagrangian_grads,
                                                       lagrangian_state)
      new_lr_state = self.inner_lrs.grad_step(lr_grads, inner_lr_state)
      # TODO (yarden): maybe should be actor state!
      new_pi = self.actor.apply(new_actor_state.params, support.o[:, :, :-1])
      report = {
          'agent/actor/loss': loss,
          'agent/actor/entropy': new_pi.entropy().mean(),
          'agent/actor/delta_kl': kl_d,
          'agent/actor/lagrangian_grads': optax.global_norm(lagrangian_grads),
          'agent/actor/pi_grads': optax.global_norm(pi_grads),
          'agent/actor/lr_grads': optax.global_norm(lr_grads)
      }
      out = (iter_ + 1, new_lagrangian_state, new_actor_state, new_lr_state,
             report)
      return out

    init_state = (0, lagrangian_state, actor_state, inner_lr_state, {
        'agent/actor/loss': 0.,
        'agent/actor/entropy': 0.,
        'agent/actor/delta_kl': 0.,
        'agent/actor/lagrangian_grads': 0.,
        'agent/actor/pi_grads': 0.,
        'agent/actor/lr_grads': 0.
    })
    (
        iters,
        new_lagrangian_state,
        new_actor_state,
        new_lr_state,
        info,
    ) = jax.lax.while_loop(cond, body, init_state)  # noqa
    info['agent/actor/update_iters'] = iters
    new_lagrangian = self.lagrangian.apply(new_lagrangian_state.params)
    info['agent/lagrangian'] = new_lagrangian
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

  def meta_loss(self, lagrangian_prior: hk.Params, policy_prior: hk.Params,
                inner_lrs: Tuple[float, float], support: TrajectoryData,
                query: TrajectoryData, support_eval: Evaluation,
                query_eval: Evaluation, old_pi_support_logprob: jnp.ndarray,
                old_pi_query_logprob: jnp.ndarray,
                old_pi_posteriors: hk.Params):
    lagrangian_lr, pi_lr = inner_lrs

    def task_loss(support, query, support_eval, query_eval,
                  old_pi_support_logprob, old_pi_query_logprob,
                  old_pi_posterior):
      constraint = support.c.sum(1).mean()
      lagrangian_posterior, pi_posterior = self.task_adaptation(
          lagrangian_prior, policy_prior, lagrangian_lr, pi_lr,
          support.o[:, :-1], support.a, support_eval.advantage,
          support_eval.cost_advantage, constraint, old_pi_support_logprob)
      if self.safe:
        lagrangian = jnn.softplus(self.lagrangian.apply(lagrangian_posterior))
      else:
        lagrangian = jnp.zeros_like(constraint)
      loss = self.policy_loss(pi_posterior, query.o[:, :-1], query.a,
                              query_eval.advantage, query_eval.cost_advantage,
                              lagrangian, old_pi_query_logprob)
      # Compute the KL-divs of the new posterior. We could compute old_pi
      # outside but computing it here makes the code clearer.
      pi = self.actor.apply(pi_posterior, query.o[:, :-1])
      old_pi = self.actor.apply(old_pi_posterior, query.o[:, :-1])
      # TODO (yarden): this could be actually not the reverse kl-divergence
      kl = old_pi.kl_divergence(pi)
      return loss, kl.mean()

    task_loss = jax.vmap(task_loss)
    losses, kls = task_loss(support, query, support_eval, query_eval,
                            old_pi_support_logprob, old_pi_query_logprob,
                            old_pi_posteriors)
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
    # Adapt the policy to get a MAP of the policy's poterior.
    _, pi_posterior = self.task_adaptation(self.lagrangian.params,
                                           self.actor.params, lagrangian_lr,
                                           pi_lr, observation[:, :-1], action,
                                           eval_.advantage,
                                           eval_.cost_advantage, constraint,
                                           old_pi_logprob)
    # Keep the posterior's MAP to the query data set.
    self.pi_posterior = pi_posterior
    # Keep this posterior for later use as part of training.
    if train:
      self.pi_posteriors[self.task_id] = pi_posterior

  @partial(jax.jit, static_argnums=0)
  def task_adaptation(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, lagrangian_lr: float,
                      pi_lr: float, observation: jnp.ndarray,
                      action: jnp.ndarray, advantage: jnp.ndarray,
                      cost_advantage: jnp.ndarray, constraint: jnp.ndarray,
                      old_pi_logprob: np.ndarray) -> [hk.Params, hk.Params]:
    lagrangian_lr, pi_lr = map(jnn.softplus, (lagrangian_lr, pi_lr))
    new_lagrangian, new_pi = lagrangian_prior, policy_prior
    lagrangian_loss = lambda p: (-self.lagrangian.apply(p) *
                                 (constraint - self.config.cost_limit))[0]
    for _ in range(self.config.inner_steps):
      if self.safe:
        # Fine tune the lagrangian for the given task.
        lagrangian_grads = jax.grad(lagrangian_loss)(new_lagrangian)
        new_lagrangian = utils.gradient_descent(lagrangian_grads,
                                                new_lagrangian, lagrangian_lr)
        lagrangian = jnn.softplus(self.lagrangian.apply(new_lagrangian))
      else:
        lagrangian = 0.
      # Use the newly-tuned lagrangian and task evaluation to fine-tune the
      # policy.
      policy_grads = jax.grad(self.policy_loss)(
          new_pi,
          observation,
          action,
          advantage,
          cost_advantage,
          lagrangian,
          old_pi_logprob,
          clip=False)
      new_pi = utils.gradient_descent(policy_grads, new_pi, pi_lr)
    return new_lagrangian, new_pi

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
