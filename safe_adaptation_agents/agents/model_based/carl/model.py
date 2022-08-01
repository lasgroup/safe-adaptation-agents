from typing import (
    Sequence,
    Optional,
    Callable,
    Union,
    Dict,
    Any,
    NamedTuple,
    Tuple,
)

import haiku as hk
import jax.lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random
from tensorflow_probability.substrates import jax as tfp

from safe_adaptation_agents import models, nets, utils

tfd = tfp.distributions
tfb = tfp.bijectors


class WorldState(NamedTuple):
  next_state: tfd.Distribution
  reward: tfd.Distribution
  cost: tfd.Distribution


Action = jnp.ndarray
Observation = jnp.ndarray
Model = Callable[[Observation, Action], WorldState]


class WorldModel(hk.Module):

  def __init__(self, output_shape: Sequence[int], hidden_size: int,
               state_decoder_config: Dict, reward_decoder_config: Dict,
               cost_decoder_config: Dict):
    super(WorldModel, self).__init__()
    self.output_shape = tuple(map(lambda x: 2 * x, output_shape))
    self.hidden_size = hidden_size
    self.reward_config = reward_decoder_config
    self.cost_config = cost_decoder_config
    self.state_decoder_config = state_decoder_config

  def __call__(self, observation: Observation,
               action: Action) -> [tfd.Distribution, ...]:
    x = jnp.concatenate([observation, action], -1)
    x = jnn.relu(hk.Linear(self.hidden_size)(x))
    reward = models.DenseDecoder((1,), **self.reward_config, dist='normal')(x)
    cost = models.DenseDecoder((1,), **self.cost_config, dist='bernoulli')(x)
    outs = nets.mlp(
        x,
        output_sizes=tuple(self.state_decoder_config['layers'] +
                           list(self.output_shape)),
        initializer=nets.initializer(
            self.state_decoder_config.get('initializer', 'glorot')),
        activation=self.state_decoder_config.get('activation', jnn.relu))
    mu, stddev = jnp.split(outs, 2, -1)
    mu += jax.lax.stop_gradient(observation)
    init_stddev = utils.inv_softplus(self.state_decoder_config['init_stddev'])
    stddev = jnn.softplus(stddev + init_stddev)
    min_stddev, max_stddev = map(
        utils.inv_softplus,
        (self.state_decoder_config['min_stddev'],
         self.state_decoder_config['max_stddev']),
    )
    stddev = jnp.clip(
        stddev * self.state_decoder_config['stddev_scale'],
        min_stddev,
        max_stddev,
    )
    next_observation = tfd.MultivariateNormalDiag(mu, stddev)
    predictions = WorldState(next_observation, reward, cost)
    return predictions


def sample_trajectories(
    model: Model,
    init_state: Observation,
    actions: Action,
    key: jax.random.PRNGKey,
) -> Tuple[tfd.Distribution, tfd.Distribution, tfd.Distribution]:
  assert init_state.shape[0] == actions.shape[0]

  def step(carry, ins):
    seed = carry[0]
    obs = carry[1]
    acs = ins
    next_obs, reward, cost = model(obs, acs)
    seed, model_seed = jax.random.split(seed)
    carry = seed, next_obs.sample(seed=model_seed)
    outs = next_obs, reward, cost
    return carry, outs
  # `jax.lax.scan` scans over the first dimension, transpose the inputs.
  ins = actions.swapaxes(0, 1)
  carry = (key, init_state)
  _, outs = jax.lax.scan(step, carry, ins)
  # Transpose back such that batch_dim is the leading dimension.
  outs = jax.tree_map(lambda x: x.swapaxes(0, 1), outs)
  return tuple(outs)  # noqa
