from typing import Optional, NamedTuple
from types import SimpleNamespace

import numpy as np
from gym.spaces import Space

import jax.numpy as jnp
import haiku as hk

from safe_adaptation_agents import models
from safe_adaptation_agents import nets
from safe_adaptation_agents.agents import Transition
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.agents.on_policy import safe_vpg


def _initial_hidden_state(batch_size: int, hidden_size: int):
  return jnp.zeros((batch_size, hidden_size))


class State(NamedTuple):
  hidden: jnp.ndarray
  prev_action: jnp.ndarray
  prev_reward: jnp.ndarray
  prev_cost: jnp.ndarray
  done: jnp.ndarray

  @property
  def vec(self):
    return jnp.concatenate(
        [self.prev_action, self.prev_reward, self.prev_cost, self.done],
        -1), self.hidden


class GruPolicy(hk.Module):

  def __init__(self,
               hidden_size: int,
               actor_config: dict,
               initialization: str = 'glorot'):
    super(GruPolicy, self).__init__()
    self._cell = hk.GRU(
        hidden_size,
        w_i_init=nets.initializer(initialization),
        w_h_init=hk.initializers.Orthogonal())
    self._head = models.Actor(**actor_config)

  def __call__(self, observation: jnp.ndarray, state: State):
    embeddings, hidden = state.vec
    ins = jnp.concatenate([observation, embeddings], -1)
    outs, hidden = self._cell(ins, hidden)
    return self._head(outs), hidden


class Rl2Cpo(safe_vpg):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(Rl2Cpo, self).__init__(observation_space, action_space, config,
                                 logger, actor, critic, safety_critic)
    parallel_envs = self.config.parallel_envs
    hidden_state = _initial_hidden_state(parallel_envs, self.config.hidden_size)
    self.state = State(hidden_state,
                       jnp.zeros((parallel_envs,) + action_space.shape),
                       jnp.zeros((parallel_envs,)), jnp.zeros(parallel_envs,),
                       jnp.zeros((parallel_envs,)))
    self.task_id = -1

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
    action, hidden = self.policy(observation, policy_params, next(self.rng_seq),
                                 train, adapt)
    # Update only hidden here, update the rest of the attributes
    # in 'self.observe(...)'
    self.state = State(hidden, self.state.prev_action, self.state.prev_reward,
                       self.state.prev_cost, self.state.done)
    return action

  def policy(self, observation: jnp.ndarray, params: hk.Params, state: State,
             key: jnp.ndarray, train: bool,
             adapt: bool) -> [jnp.ndarray, jnp.ndarray]:
    policy, hidden = self.actor.apply(params, observation, state)
    # Take the mode only on query episodes in which we evaluate the agent.
    if not adapt and not train:
      action = policy.mode()
    else:
      action = policy.sample(seed=key)
    return action, hidden

  def observe(self, transition: Transition, adapt: bool):
    self.buffer.add(transition)
    self.training_step += sum(transition.steps)
    # Keep prev_hidden after computing a forward pass in `self.policy(...)`
    hidden = self.state.hidden
    self.state = State(hidden, transition.action, transition.reward,
                       transition.cost, transition.done)

  def observe_task_id(self, task_id: Optional[str] = None):
    self.task_id = (self.task_id + 1) % self.config.task_batch_size
    self.buffer.set_task(self.task_id)
    self.state = State(*map(jnp.zeros_like, self.state))
