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

from safe_adaptation_agents.agents.on_policy import cpo, vpg
from safe_adaptation_agents.agents.on_policy.safe_vpg import Evaluation
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState
from safe_adaptation_agents import utils
from safe_adaptation_agents import episodic_trajectory_buffer as etb
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData


class MamlCpo(cpo.Cpo):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(MamlCpo, self).__init__(observation_space, action_space, config,
                                  logger, actor, critic, safety_critic)
    num_steps = self.config.time_limit // self.config.action_repeat
    self.buffer = etb.EpisodicTrajectoryBuffer(
        self.config.num_trajectories + self.config.num_query_trajectories,
        num_steps, observation_space.shape, action_space.shape,
        self.config.task_batch_size)
    self.task_id = -1
    self.inner_lr = utils.Learner(
        utils.inv_softplus(self.config.policy_inner_lr), next(self.rng_seq),
        self.config.inner_lr_opt,
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
        self.actor.state,
        self.inner_lr.state,
        info,
    ) = self.update_priors(self.actor.state, self.inner_lr.state, support,
                           query, utils.pytrees_stack(self.pi_posteriors))
    info['agent/policy_lr'] = jnn.softplus(self.inner_lr.params)
    for k, v in info.items():
      self.logger[k] = v.mean()

  @partial(jax.jit, static_argnums=0)
  def update_priors(self, actor_state: LearningState,
                    inner_lr_state: LearningState, support: TrajectoryData,
                    query: TrajectoryData,
                    old_pi_posteriors: hk.Params) -> [LearningState, dict]:
    support_eval, query_eval = self.evaluate_support_and_query(
        self.critic.state, self.safety_critic.state, support, query)
    pass

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

  def meta_loss(self, policy_prior: hk.Params, inner_lr: jnp.ndarray,
                support: TrajectoryData, query: TrajectoryData,
                support_eval: Evaluation, query_eval: Evaluation,
                old_pi_query_logprob: jnp.ndarray,
                old_pi_posteriors: hk.Params):

    def task_loss(support, query, support_eval, query_eval,
                  old_pi_query_logprob, old_pi_posterior):
      pi_posterior = self.task_adaptation(policy_prior, inner_lr,
                                          support.o[:, :-1], support.a,
                                          support_eval.advantage)
      loss, surrogate_cost = self.policy_loss(pi_posterior, query.o[:, :-1],
                                              query.a, query_eval.advantage,
                                              query_eval.cost_advantage,
                                              old_pi_query_logprob)
      # Compute the KL-divs of the new posterior. We could compute old_pi
      # outside but computing it here makes the code clearer.
      pi = self.actor.apply(pi_posterior, query.o[:, :-1])
      old_pi = self.actor.apply(old_pi_posterior, query.o[:, :-1])
      kl = pi.kl_divergence(old_pi)
      return loss, surrogate_cost, kl.mean()

    task_loss = jax.vmap(task_loss)
    losses, surrogate_costs, kls = task_loss(support, query, support_eval,
                                             query_eval, old_pi_query_logprob,
                                             old_pi_posteriors)
    return losses.mean(), surrogate_costs.mean(), kls.mean()

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    # Evaluate the policy given the recent data.
    eval_ = self.adapt_critics_and_evaluate(self.critic.state,
                                            self.safety_critic.state,
                                            observation, reward, cost)
    # Adapt the policy to get a MAP of the policy's poterior.
    _, pi_posterior = self.task_adaptation(self.actor.params,
                                           self.inner_lr.params,
                                           observation[:, :-1], action,
                                           eval_.advantage)
    # Keep the posterior's MAP to the query data set.
    self.pi_posterior = pi_posterior
    # Keep this posterior for later use as part of training.
    if train:
      self.pi_posteriors[self.task_id] = pi_posterior

  @partial(jax.jit, static_argnums=0)
  def task_adaptation(self, policy_prior: hk.Params, pi_lr: jnp.ndarray,
                      observation: jnp.ndarray, action: jnp.ndarray,
                      advantage: jnp.ndarray) -> [hk.Params, hk.Params]:
    pi_lr = jnn.softplus(pi_lr)
    new_pi = policy_prior
    loss = partial(vpg.VanillaPolicyGrandients.policy_loss, self)
    for _ in range(self.config.inner_steps):
      # Shamelessly ignore safety and use only the objective to
      # adapt the policy.
      policy_grads = jax.grad(loss)(new_pi, observation, action, advantage)
      new_pi = utils.gradient_descent(policy_grads, new_pi, pi_lr)
    return new_pi

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
