import functools
from types import SimpleNamespace
from typing import Optional

import numpy as np
from gym.spaces import Space

import jax
import jax.numpy as jnp
import haiku as hk

from safe_adaptation_agents.agents import Agent, Transition
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.agents.on_policy.trajectory_buffer import (
    TrajectoryBuffer)
from safe_adaptation_agents import utils
from safe_adaptation_agents.utils import LearningState

discounted_cumsum = jax.vmap(utils.discounted_cumsum, in_axes=[0, None])


class VanillaPolicyGrandients(Agent):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed):
    super().__init__(config, logger)
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.training_step = 0
    self.buffer = TrajectoryBuffer(self.config.num_trajectories,
                                   self.config.time_limit,
                                   observation_space.shape, action_space.shape)
    self.buffer.set_task(0)
    self.actor = utils.Learner(
        actor, next(self.rng_seq), config.actor_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
    self.critic = utils.Learner(
        critic, next(self.rng_seq), config.critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if self.time_to_update and train:
      self.train(*self.buffer.dump())
      self.logger.log_metrics(self.training_step)
    action = self.policy(observation, self.actor.params, next(self.rng_seq),
                         train)
    return np.clip(action, -1.0, 1.0)

  def observe(self, transition: Transition):
    self.buffer.add(transition)
    self.training_step += sum(transition.steps)

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  @functools.partial(jax.jit, static_argnums=(0, 4))
  def policy(self,
             observation: jnp.ndarray,
             params: hk.Params,
             key: jnp.ndarray,
             training=True) -> jnp.ndarray:
    policy = self.actor.apply(params, observation)
    action = policy.sample(seed=key) if training else policy.mode(seed=key)
    return action

  def train(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, _: np.ndarray):
    return_ = discounted_cumsum(reward, self.config.discount)
    advantage = self._advantage(self.critic.params, observation, reward)
    self.actor.learning_state, actor_report = self.actor_update_step(
        self.actor.learning_state, observation, action, advantage,
        next(self.rng_seq))
    for k, v in actor_report.items():
      self.logger[k] = v
    for _ in range(self.config.value_update_steps):
      (self.critic.learning_state, report) = self.critic_update_step(
          self.actor.learning_state, self.critic.learning_state, observation,
          action, advantage, return_)
      for k, v in report.items():
        self.logger[k] = v

  @functools.partial(jax.jit, static_argnums=0)
  def critic_update_step(self, critic_state: LearningState,
                         return_: jnp.ndarray) -> [LearningState, dict]:
    value_loss, value_grads = jax.value_and_grad(self.critic_loss)(
        critic_state.params, observation, return_)
    new_critic_state = self.critic.grad_step(value_grads, critic_state)
    return new_critic_state, {'agent/critic/loss': value_loss}

  @functools.partial(jax.jit, static_argnums=0)
  def actor_update_step(self, actor_state: LearningState,
                        observation: jnp.ndarray, actions: jnp.ndarray,
                        advantage: jnp.ndarray,
                        seed: jnp.ndarray) -> [LearningState, dict]:
    policy_loss, policy_grads = jax.value_and_grad(self.policy_loss)(
        actor_state.params, observation, actions, advantage, seed)
    new_actor_state = self.actor.grad_step(policy_grads, actor_state)
    return new_actor_state, {'agent/actor/loss': policy_loss}

  def policy_loss(self, actor_params: hk.Params, observation: jnp.ndarray,
                  actions: jnp.ndarray, advantage: jnp.ndarray,
                  seed: jnp.ndarray):
    pi = self.actor.apply(actor_params, observation[:, :-1])
    objective = (
        pi.log_prob(actions) * advantage +
        self.config.entropy_regularization * pi.entropy(seed=seed))
    return -objective.mean()

  def critic_loss(self, critic_params: hk.Params, observation: jnp.ndarray,
                  return_: jnp.ndarray):
    return -self.critic.apply(critic_params,
                              observation[:, :-1]).log_prob(return_).mean()

  @functools.partial(jax.jit, static_argnums=0)
  def _advantage(self, critic_params: hk.Params, observation: jnp.ndarray,
                 reward: jnp.ndarray):
    bootstrap = self.critic.apply(critic_params, observation).mode()
    diff = reward + (
        self.config.discount * bootstrap[..., 1:] - bootstrap[..., :-1])
    advantage = discounted_cumsum(diff,
                                  self.config.lambda_ * self.config.discount)
    mean, stddev = advantage.mean(), advantage.std()
    return (advantage - mean) / (stddev + 1e-8)

  @property
  def time_to_update(self):
    return self.buffer.full
