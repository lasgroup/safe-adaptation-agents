import functools
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
from gym.spaces import Space

import jax
import jax.numpy as jnp
import haiku as hk

from safe_adaptation_agents.agents import Agent, Transition
from safe_adaptation_agents.logger import TrainingLogger
from safe_adaptation_agents.agents.on_policy.trajectory_buffer import (
    TrajectoryBuffer)
from safe_adaptation_agents import utils
from safe_adaptation_agents.utils import LearningState

discounted_cumsum = jax.vmap(utils.discounted_cumsum, in_axes=[0, None])


class VanillaPolicyGrandients(Agent):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed):
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.config = config
    self.logger = logger
    self.training_steps = 0
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
    action = self.policy(observation, self.actor.params, next(self.rng_seq),
                         train)
    return np.clip(action, -1.0, 1.0)

  def observe(self, transition: Transition):
    self.buffer.add(transition)
    self.training_steps += self.config.action_repeat

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  @functools.partial(jax.jit, static_argnums=0)
  def policy(self,
             observation: np.ndarray,
             params: hk.Params,
             key: jnp.ndarray,
             training=True) -> jnp.ndarray:
    policy = self.actor.apply(params, observation)
    action = policy.sample(seed=key) if training else policy.mode(seed=key)
    return action

  def train(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, _: np.ndarray, terminal: np.ndarray):
    return_ = discounted_cumsum(reward, self.config.discount)
    advantage = self._advantage(self.critic.params, observation, reward,
                                terminal)
    for _ in range(self.config.update_steps):
      self.actor.learning_state, self.critic.learning_state = self._update_step(
          self.actor.learning_state, self.critic.learning_state, observation,
          action, advantage, return_)

  @functools.partial(jax.jit, static_argnums=0)
  def _update_step(
      self, actor_state: LearningState, critic_state: LearningState,
      observation: jnp.ndarray, actions: jnp.ndarray, advantage: jnp.ndarray,
      return_: jnp.ndarray) -> [utils.LearningState, LearningState]:
    policy_grads = jax.grad(self.policy_loss)(actor_state.params, observation,
                                              actions, advantage)
    new_actor_state = self.actor.grad_step(policy_grads, actor_state)
    value_grads = jax.grad(self.critic_loss)(observation, return_)
    new_critic_state = self.critic.grad_step(value_grads, critic_state)
    return new_actor_state, new_critic_state

  def policy_loss(self, actor_params: hk.Params, observation: jnp.ndarray,
                  actions: jnp.ndarray, advantage: jnp.ndarray):
    pi = self.actor.apply(actor_params, observation)
    objective = (
        pi.log_prob(actions) * advantage +
        self.config.entropy_regularization * pi.entropy())
    return -objective.mean()

  def critic_loss(self, critic_params: hk.Params, observation: jnp.ndarray,
                  return_: jnp.ndarray):
    return -self.critic.apply(critic_params,
                              observation).log_prob(return_).mean()

  @functools.partial(jax.jit, static_argnums=0)
  def _advantage(self, critic_params: hk.Params, observation: jnp.ndarray,
                 reward: jnp.ndarray, _: jnp.ndarray, terminal: jnp.ndarray):
    bootstrap = self.critic.apply(critic_params, observation) * (1. - terminal)
    diff = reward[:-1] + self.config.discount * bootstrap[1:] - bootstrap[:-1]
    return discounted_cumsum(diff, self.config.lambda_ * self.config.discount)

  @property
  def time_to_update(self):
    return self.training_steps % self.config.update_every == 0
