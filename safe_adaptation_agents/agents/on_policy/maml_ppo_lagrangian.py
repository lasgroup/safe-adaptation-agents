from typing import Tuple, Optional
from types import SimpleNamespace
from enum import Enum
import functools

import numpy as np
from gym.spaces import Space

import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk

from safe_adaptation_agents.agents import Transition
from safe_adaptation_agents.agents.on_policy import ppo_lagrangian, vpg
from safe_adaptation_agents.agents.on_policy import trajectory_buffer
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState
from safe_adaptation_agents import utils

# Vmap over the task axis.
discounted_cumsum = jax.vmap(vpg.discounted_cumsum, in_axes=[0, None])


class MamlPpoLagrangian(ppo_lagrangian.PpoLagrangian):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(MamlPpoLagrangian,
          self).__init__(observation_space, action_space, config, logger, actor,
                         critic, safety_critic)
    self.buffer = trajectory_buffer.TrajectoryBuffer(
        self.config.num_trajectories + self.config.num_query_trajectories,
        self.config.time_limit, observation_space.shape, action_space.shape,
        self.config.task_batch_size)
    self.adaptation_buffer = trajectory_buffer.TrajectoryBuffer(
        self.config.num_trajectories, self.config.time_limit,
        observation_space.shape, action_space.shape,
        self.config.task_batch_size)
    self.task_id = 0
    self.lagrangian_inner_lr = self.config.init_lagrangian_inner_lr
    self.policy_inner_lr = self.config.init_policy_inner_lr
    # Map task id to it's corresponding fine-tuned/adapted parameters.
    self.pi_posterior = [None] * self.config.task_batch_size

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if train and self.buffer.full:
      assert adapt, ('Should train (and dump the trajectory buffer) only after '
                     'finished querying.')
      self.train(*self.buffer.dump())
      self.task_id = 0
    if self.adaptation_buffer.full:
      assert not adapt, 'Should not adapt while collecting adaptation data.'
      self.adapt(*self.adaptation_buffer.dump())
    policy_params = (
        self.task_posterior_params if not adapt else self.actor.params)
    action = self.policy(observation, policy_params, next(self.rng_seq), train)
    return action

  def observe(self, transition: Transition, train: bool, adapt: bool):
    super(MamlPpoLagrangian, self).observe(transition, train, adapt)
    if adapt:
      self.adaptation_buffer.add(transition)

  def observe_task_id(self, task_id: Optional[str] = None):
    self.buffer.set_task(self.task_id)
    self.adaptation_buffer.set_task(self.task_id)
    self.task_id += 1

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, running_cost: np.ndarray):
    # Find parameters that fit well the critics for the new tasks.
    return_, cost_return = self.returns(reward, cost)
    critic_states, _ = self.update_critic(self.critic.state,
                                          observation[:, :, :-1], return_)
    if self.safe:
      safety_critic_states, _ = self.update_safety_critic(
          self.safety_critic.state, observation[:, :, :-1], cost_return)
    else:
      safety_critic_states = critic_states
    # Evaluate with the task specific value functions.
    (advantage, return_, cost_advantage,
     cost_return) = self.evaluate_with_safety(critic_states.params,
                                              safety_critic_states.params,
                                              observation, reward, cost)
    _, pi_posteriors = self.adaptation_step(
        self.lagrangian.params, self.actor.params, self.lagrangian_inner_lr,
        self.policy_inner_lr, observation[:, :, :-1], action, advantage,
        cost_advantage, running_cost)
    for i in range(self.config.task_batch_size):
      self.pi_posterior[i] = jax.tree_util.tree_map(lambda node: node[i],
                                                    pi_posteriors)

  @functools.partial(jax.jit, static_argnums=0)
  def adaptation_step(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, lagrangian_lr: float,
                      policy_lr: float, observation: np.ndarray,
                      action: np.ndarray, advantage: np.ndarray,
                      cost_advantage: np.ndarray,
                      running_cost: np.ndarray) -> [hk.Params, hk.Params]:
    per_task_adaptation = jax.vmap(
        functools.partial(self.task_adaptation, lagrangian_prior, policy_prior,
                          lagrangian_lr, policy_lr))
    return per_task_adaptation(observation, action, advantage, cost_advantage,
                               running_cost)

  def task_adaptation(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, lagrangian_lr: float,
                      policy_lr: float, observation: jnp.ndarray,
                      action: jnp.ndarray, advantage: jnp.ndarray,
                      cost_advantage: jnp.ndarray,
                      running_cost: jnp.ndarray) -> [hk.Params, hk.Params]:
    """
    Finds policy's and lagrangian MAP paramters for a single task.
    """
    old_pi = self.actor.apply(policy_prior, observation)
    old_pi_logprob = old_pi.log_prob(action)

    def inner_step(prior: Tuple[hk.Params, hk.Params], _):
      lagrangian_prior, policy_prior = prior
      if self.safe:
        lagrange_grads = jax.grad(lambda prior: self.lagrangian.apply(prior) *
                                  (running_cost - self.config.cost_limit))(
                                      lagrangian_prior)
        lagrangian_posterior = utils.gradient_descent(lagrange_grads,
                                                      lagrangian_prior,
                                                      lagrangian_lr)
        lagrangian = jnn.softplus(self.lagrangian.apply(lagrangian_posterior))
      else:
        lagrangian = 0.
        lagrangian_posterior = lagrangian_prior
      policy_grads = jax.grad(self.policy_loss)(policy_prior, observation,
                                                action, advantage,
                                                cost_advantage, lagrangian,
                                                old_pi_logprob)
      policy_posterior = utils.gradient_descent(policy_grads, policy_prior,
                                                policy_lr)
      return (lagrangian_posterior, policy_posterior), None

    (lagrangian_posteriors,
     policy_posteriors), _ = jax.lax.scan(inner_step,
                                          (lagrangian_prior, policy_prior),
                                          jnp.arange(self.config.inner_steps))
    return lagrangian_posteriors, policy_posteriors

  @functools.partial(jax.jit, static_argnums=0)
  def evaluate_with_safety(
      self, critic_params: hk.Params, safety_critic_params: hk.Params,
      observation: jnp.ndarray, reward: jnp.ndarray, cost: jnp.ndarray
  ) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    task_batched = jax.vmap(super(MamlPpoLagrangian, self).evaluate_with_safety)
    return task_batched(critic_params, safety_critic_params, observation,
                        reward, cost)

  @functools.partial(jax.jit, static_argnums=0)
  def update_critic(self, state: LearningState, observation: jnp.ndarray,
                    return_: jnp.ndarray) -> [LearningState, dict]:
    partial = functools.partial(
        super(MamlPpoLagrangian, self).update_critic, state)
    # vmap over the task axis.
    return jax.vmap(partial)(observation, return_)

  @functools.partial(jax.jit, static_argnums=0)
  def update_safety_critic(self, state: LearningState, observation: jnp.ndarray,
                           cost_return: jnp.ndarray) -> [LearningState, dict]:
    partial = functools.partial(
        super(MamlPpoLagrangian, self).update_safety_critic, state)
    # vmap over the task axis.
    return jax.vmap(partial)(observation, cost_return)

  @functools.partial(jax.jit, static_argnums=0)
  def returns(self, rewards: jnp.ndarray, costs: jnp.ndarray):
    reward_return = discounted_cumsum(rewards, self.config.discount)
    cost_return = discounted_cumsum(costs, self.config.cost_discount)
    return reward_return, cost_return

  @property
  def task_posterior_params(self):
    if None in self.pi_posterior:
      return self.actor.params
    return self.pi_posterior[self.task_id]
