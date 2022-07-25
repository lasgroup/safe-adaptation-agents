import abc
import functools
from types import SimpleNamespace
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gym.spaces import Space

from safe_adaptation_agents import utils
from safe_adaptation_agents.agents.on_policy import vpg
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState


class Evaluation(NamedTuple):
  advantage: np.ndarray
  return_: np.ndarray
  cost_advantage: np.ndarray
  cost_return: np.ndarray


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
  def train(self, trajectory_data: TrajectoryData):
    """
    Implements a training loop for safe policy-grandients algorithm.
    """

  @functools.partial(jax.jit, static_argnums=0)
  def update_safety_critic(self, state: LearningState, observation: jnp.ndarray,
                           cost_return: jnp.ndarray) -> [LearningState, dict]:

    def update(critic_state: LearningState):
      loss, grads = jax.value_and_grad(self.safety_critic_loss)(
          critic_state.params, observation, cost_return)
      new_critic_state = self.safety_critic.grad_step(grads, critic_state)
      return new_critic_state, {
          'agent/safety_critic/loss': loss,
          'agent/safety_critic/grad': optax.global_norm(grads)
      }

    return jax.lax.scan(lambda state, _: update(state), state,
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
  def evaluate_with_safety(self, critic_params: hk.Params,
                           safety_critic_params: hk.Params,
                           observation: jnp.ndarray, reward: jnp.ndarray,
                           cost: jnp.ndarray) -> Evaluation:
    advantage, return_ = self.evaluate(critic_params, observation, reward)
    if not self.safe:
      return Evaluation(advantage, return_, jnp.zeros_like(advantage),
                        jnp.zeros_like(return_))
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
    return Evaluation(advantage, return_, cost_advantage, cost_return)

  @property
  def safe(self):
    return self.config.safe
