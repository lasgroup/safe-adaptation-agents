import functools
from types import SimpleNamespace
from typing import Optional

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


class PolicyGrandiens(Agent):

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
    self.actor = utils.Learner(
        actor, next(self.rng_seq), config.actor_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
    self.critic = utils.Learner(
        critic, next(self.rng_seq), config.critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
    self.safety_critic = utils.Learner(
        critic, next(self.rng_seq), config.safety_critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
    self.current_task_id = -1
    self.adapting = True

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    self.adapting = self.toggle_mode(adapt)
    if self.time_to_update and train:
      self.train(self.buffer.dump())
    action = self.policy(observation, self.actor.params,
                         next(self.rng_seq), train)
    return np.clip(action, -1.0, 1.0)

  def observe(self, transition: Transition):
    self.buffer.add(transition)

  def observe_task_id(self, task_id: Optional[str] = None):
    self.current_task_id += 1
    self.buffer.set_task(self.current_task_id)

  @functools.partial(jax.jit, static_argnums=0)
  def policy(self,
             observation: np.ndarray,
             params: hk.Params,
             key: jnp.ndarray,
             training=True) -> jnp.ndarray:
    policy = self.actor.apply(params, observation)
    action = policy.sample(seed=key) if training else policy.mode(
        seed=key)
    return action

  @property
  def time_to_update(self):
    return self.training_steps % self.config.update_every == 0

  def toggle_mode(self, adapt: bool):
    # Toggle mode.
    if (self.adapting and not adapt) or (not self.adapting and adapt):
      adaptation_mode = not self.adapting
      self.current_task_id = -1
    else:
      adaptation_mode = self.adapting
    return adaptation_mode
