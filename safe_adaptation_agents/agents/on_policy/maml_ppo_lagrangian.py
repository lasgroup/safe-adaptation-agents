from typing import Optional
from types import SimpleNamespace
from enum import Enum
import functools

import numpy as np
from gym.spaces import Space

import optax
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


class Mode(Enum):
  ADAPT = 0
  QUERY = 1


class MamlPpoLagrangian(ppo_lagrangian.PpoLagrangian):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(MamlPpoLagrangian,
          self).__init__(observation_space, action_space, config, logger, actor,
                         critic, safety_critic)
    self.buffer = trajectory_buffer.TrajectoryBuffer(
        self.config.num_trajectories + self.config.query_num_trajectories,
        self.config.time_limit, observation_space.shape, action_space.shape,
        self.config.num_train_tasks)
    self.meta_test_buffer = trajectory_buffer.TrajectoryBuffer(
        self.config.num_trajectories, self.config.time_limit,
        observation_space.shape, action_space.shape, self.config.num_test_tasks)
    self.task_id = -1
    self.mode = Mode.ADAPT
    # A dictionary mapping task_id to policy parameters
    self.pi_posterior = dict()

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if self.time_to_meta_train(adapt):
      self.meta_train()
      self.mode = Mode.ADAPT
      self.task_id = 0
    elif self.time_to_adapt(adapt):
      self.adapt(**self.adaptation_buffer.dump())
      self.mode = Mode.QUERY
    action = self.policy(observation, self.task_posterior_params,
                         next(self.rng_seq), train)
    return action

  def observe(self, transition: Transition, train: bool, adapt: bool):
    super(MamlPpoLagrangian, self).observe(transition, train, adapt)
    if self.mode == Mode.ADAPT:
      self.adaptation_buffer.add(transition)

  def observe_task_id(self, task_id: Optional[str] = None):
    self.task_id += 1
    if self.mode == Mode.ADAPT:
      self.buffer.set_task(self.task_id)
    elif self.mode == Mode.QUERY:
      self.adaptation_buffer.set_task(self.task_id)
    else:
      raise AssertionError('Mode is wrong.')

  def meta_train(self):
    # pi_posterior is a dict that maps a task id to policy parameters.
    self.pi_posterior = '456'

  def train(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, running_cost: np.ndarray):
    (advantage, return_, cost_advantage,
     cost_return) = self.evaluate_with_safety(self.critic.params,
                                              self.safety_critic.params,
                                              observation, reward, cost)
    if self.safe:
      self.lagrangian.state, lagrangian_report = self.lagrangian_update_step(
          self.lagrangian.state, running_cost)
      lagrangian = jnn.softplus(self.lagrangian.apply(self.lagrangian.params))
    else:
      lagrangian = 0.
      lagrangian_report = {}
    self.actor.state, actor_report = self.actor_update_step(
        self.actor.state,
        observation[:, :-1],
        action=action,
        advantage=advantage,
        lagrangian=lagrangian,
        cost_advantage=cost_advantage)
    self.critic.state, critic_report = self.critic_update_step(
        self.critic.state, observation[:, :-1], return_)
    if self.safe:
      self.safety_critic.state, safety_report = self.safe_critic_update_step(
          self.safety_critic.state, observation[:, :, :-1], cost_return)
      critic_report.update({**safety_report, **lagrangian_report})
    for k, v in {**actor_report, **critic_report}.items():
      self.logger[k] = v.mean()

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, running_cost: np.ndarray):
    # Find parameters that fit well the critics for the new tasks.
    return_, cost_return = self.returns(reward, cost)
    critic_states, _ = self.critic_update_step(self.critic.state,
                                               observation[:, :, :-1], return_)
    safety_critic_states, _ = self.safe_critic_update_step(
        self.safety_critic.state, observation[:, :, :-1], cost_return)
    # Evaluate with the task specific value functions.
    (advantage, return_, cost_advantage,
     cost_return) = self.evaluate_with_safety(critic_states.params,
                                              safety_critic_states.params,
                                              observation, reward, cost)
    lagrangian_posterior, policy_posterior = self.adaptation_step(
        self.lagrangian.params, self.actor.params, observation, action,
        advantage, cost_advantage, return_, cost_return)
    # pi_posterior is a dict that maps a task id to policy parameters.
    self.pi_posterior = '123'

  @functools.partial(jax.jit, static_argnums=0)
  def adaptation_step(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, observation: jnp.ndarray,
                      action: jnp.ndarray, advantage: jnp.ndarray,
                      cost_advantage: jnp.ndarray, cost_return: jnp.ndarray):
    if self.safe:
      # TODO (yarden): this line is experimental!!!
      running_cost = cost_return.sum(1).mean()
      lagrange_grads = jax.grad(lambda prior: self.lagrangian.apply(prior) *
                                (running_cost - self.config.cost_limit))(
                                    lagrangian_prior)
      lagrangian_posterior = utils.gradient_decent(
          lagrange_grads, lagrangian_prior, self.config.lagrange_inner_lr)
      lagrangian = jnn.softplus(self.lagrangian.apply(lagrangian_posterior))
    else:
      lagrangian = 0.
    old_pi = self.actor.apply(policy_prior, observation)
    old_pi_logprob = old_pi.log_prob(action)
    policy_loss = jax.vmap(self.policy_loss, [None, 0, 0, 0, 0])
    policy_grads = jax.grad(policy_loss)(policy_prior, observation, action,
                                         advantage, cost_advantage, lagrangian,
                                         old_pi_logprob)
    policy_posterior = utils.gradient_decent(policy_grads, policy_prior,
                                             self.config.policy_inner_lr)
    return policy_posterior

  @functools.partial(jax.jit, static_argnums=0)
  def evaluate_with_safety(
      self, critic_params: hk.Params, safety_critic_params: hk.Params,
      observation: jnp.ndarray, reward: jnp.ndarray, cost: jnp.ndarray
  ) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jax.vmap(super(MamlPpoLagrangian,
                          self).evaluate_with_safety)(critic_params,
                                                      safety_critic_params,
                                                      observation, reward, cost)

  @functools.partial(jax.jit, static_argnums=0)
  def critic_update_step(self, state: LearningState, observation: jnp.ndarray,
                         return_: jnp.ndarray) -> [LearningState, dict]:
    partial = functools.partial(
        super(MamlPpoLagrangian, self).critic_update_step, state=state)
    # vmap over the task axis.
    return jax.vmap(partial)(observation, return_)

  @functools.partial(jax.jit, static_argnums=0)
  def safe_critic_update_step(
      self, state: LearningState, observation: jnp.ndarray,
      cost_return: jnp.ndarray) -> [LearningState, dict]:
    partial = functools.partial(
        super(MamlPpoLagrangian, self).safe_critic_update_step, state=state)
    # vmap over the task axis.
    return jax.vmap(partial)(observation, cost_return)

  @functools.partial(jax.jit, static_argnums=0)
  def returns(self, rewards: jnp.ndarray, costs: jnp.ndarray):
    reward_return = discounted_cumsum(rewards, self.config.discount)
    cost_return = discounted_cumsum(costs, self.config.cost_discount)
    return reward_return, cost_return

  def time_to_adapt(self, adapt: bool):
    assert self.meta_test_buffer.full
    # Collected support data, adapt with it before querying the query set.
    return not adapt and self.mode == Mode.ADAPT

  def time_to_meta_train(self, adapt: bool):
    assert self.buffer.full
    # Collect query data and switch to adaptation mode train if needed.
    return adapt and self.mode == Mode.QUERY

  @property
  def task_posterior_params(self):
    if self.pi_posterior is None:
      return self.actor.params
    return self.pi_posterior[self.task_id]

    def partition_cond():
      self.task_id
      self.pi_posterior
      pass

    return hk.data_structures.partition(partition_cond)
