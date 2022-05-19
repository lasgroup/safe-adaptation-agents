import functools
from types import SimpleNamespace
from typing import Optional

import numpy as np
import optax
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
    self.rs = np.random.RandomState(config.seed)
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
    return action

  def observe(self, transition: Transition, train: bool, adapt: bool):
    if train:
      self.buffer.add(transition)
      self.training_step += sum(transition.steps)

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  @functools.partial(jax.jit, static_argnums=(0, 4))
  def policy(self,
             observation: jnp.ndarray,
             params: hk.Params,
             key: jnp.ndarray,
             training: bool = True) -> jnp.ndarray:
    policy = self.actor.apply(params, observation)
    action = policy.sample(seed=key) if training else policy.mode()
    return action

  def train(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, _, __):
    advantage, return_ = self.evaluate(self.critic.params, observation, reward)
    self.actor.state, actor_report = self.actor_update_step(
        self.actor.state, observation[:, :-1], action, advantage)
    (self.critic.state,
     critic_report) = self.critic_update_step(self.critic.state,
                                              observation[:, :-1], return_)
    for k, v in {**actor_report, **critic_report}.items():
      self.logger[k] = v.mean()

  @functools.partial(jax.jit, static_argnums=0)
  def critic_update_step(self, state: LearningState, observation: jnp.ndarray,
                         return_: jnp.ndarray) -> [LearningState, dict]:

    def update(critic_state: LearningState):
      loss, grads = jax.value_and_grad(self.critic_loss)(critic_state.params,
                                                         observation, return_)
      new_critic_state = self.critic.grad_step(grads, critic_state)
      return new_critic_state, {
          'agent/critic/loss': loss,
          'agent/critic/grad': optax.global_norm(grads)
      }

    return jax.lax.scan(lambda state, _: update(state), state,
                        jnp.arange(self.config.vf_iters))

  @functools.partial(jax.jit, static_argnums=0)
  def actor_update_step(self, state: LearningState, *args,
                        **kwargs) -> [LearningState, dict]:
    observation, *_ = args
    loss, grads = jax.value_and_grad(self.policy_loss)(state.params,
                                                       *args, **kwargs)
    new_actor_state = self.actor.grad_step(grads, state)
    entropy = self.actor.apply(state.params,
                               observation).entropy().mean()
    return new_actor_state, {
        'agent/actor/loss': loss,
        'agent/actor/grad': optax.global_norm(grads),
        'agent/actor/entropy': entropy
    }

  def policy_loss(self, params: hk.Params, *args, **kwargs):
    observation, actions, advantage = args
    pi = self.actor.apply(params, observation)
    objective = (
        pi.log_prob(actions) * advantage +
        self.config.entropy_regularization * pi.entropy())
    return -objective.mean()

  def critic_loss(self, params: hk.Params, observation: jnp.ndarray,
                  return_: jnp.ndarray):
    return -self.critic.apply(params, observation).log_prob(return_).mean()

  @functools.partial(jax.jit, static_argnums=0)
  def evaluate(self, critic_params: hk.Params, observation: jnp.ndarray,
               reward: jnp.ndarray):
    value = self.critic.apply(critic_params, observation).mode()
    diff = reward + (self.config.discount * value[..., 1:] - value[..., :-1])
    advantage = discounted_cumsum(diff,
                                  self.config.lambda_ * self.config.discount)
    mean, stddev = advantage.mean(), advantage.std()
    return_ = discounted_cumsum(reward, self.config.discount)
    return (advantage - mean) / (stddev + 1e-8), return_

  @property
  def time_to_update(self):
    return self.buffer.full
