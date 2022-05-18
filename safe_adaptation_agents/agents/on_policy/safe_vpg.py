import abc
from abc import ABC
from typing import Optional
from types import SimpleNamespace
import functools

import numpy as np
from gym.spaces import Space

import optax
import jax
import jax.numpy as jnp
import haiku as hk

from safe_adaptation_agents.agents.on_policy import vpg
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents import utils
from safe_adaptation_agents.utils import LearningState


class SafeVanillaPolicyGradients(vpg.VanillaPolicyGrandients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(SafeVanillaPolicyGradients,
          self).__init__(observation_space, action_space, config, logger, actor,
                         critic)
    self.safety_critic = utils.Learner(
        safety_critic, next(self.rng_seq), config.critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())

  @abc.abstractmethod
  def train(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, running_cost: np.ndarray):
    """
    Implements a training loop for safe policy-grandients algorithm.
    """

  @functools.partial(jax.jit, static_argnums=0)
  def safe_critic_update_step(
      self, critic_state: LearningState, observation: jnp.ndarray,
      cost_return: jnp.ndarray) -> [LearningState, dict]:

    def update(critic_state: LearningState):
      loss, grads = jax.value_and_grad(self.safety_critic_loss)(
          critic_state.params, observation, cost_return)
      new_critic_state = self.safety_critic.grad_step(grads, critic_state)
      return new_critic_state, {
          'agent/safety_critic/loss': loss,
          'agent/safety_critic/grad': optax.global_norm(grads)
      }

    return jax.lax.scan(lambda state, _: update(state), critic_state,
                        jnp.arange(self.config.vf_iters))

  @abc.abstractmethod
  def policy_loss(self, params: hk.Params, *args, **kwargs) -> float:
    """
    Implements a loss for the policy.
    """

  def safety_critic_loss(self, params: hk.Params, observation: jnp.ndarray,
                         return_: jnp.ndarray) -> float:
    return -self.safety_critic.apply(params,
                                     observation).log_prob(return_).mean()

  @functools.partial(jax.jit, static_argnums=0)
  def evaluate_with_safety(
      self, critic_params: hk.Params, safety_critic_params: hk.Params,
      observation: jnp.ndarray, reward: jnp.ndarray, cost: jnp.ndarray
  ) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    advantage, return_ = self.evaluate(critic_params, observation, reward)
    if not self.safe:
      return advantage, return_, jnp.zeros_like(advantage), jnp.zeros_like(
          return_)
    cost_value = self.safety_critic.apply(safety_critic_params,
                                          observation).mode()
    cost_return = vpg.discounted_cumsum(cost, self.config.cost_discount)
    diff = cost + (
        self.config.cost_discount * cost_value[..., 1:] - cost_value[..., :-1])
    cost_advantage = vpg.discounted_cumsum(
        diff, self.config.lambda_ * self.config.cost_discount)
    # Centering advantage, but not normalize, as in
    # https://github.com/openai/safety-starter-agents/blob
    # /4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/buffer.py#L71
    cost_advantage -= cost_advantage.mean()
    return advantage, return_, cost_advantage, cost_return

  @property
  def safe(self):
    return self.config.safe
