from typing import Tuple, Sequence

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from safe_adaptation_agents import models
from safe_adaptation_agents import nets
from safe_adaptation_agents.agents.la_mbda import rssm

tfd = tfp.distributions


def create_model(config, observation_space):

  def model():
    _model = WorldModel(observation_space, config)

    def filter_state(prev_state, prev_action, observation):
      return _model(prev_state, prev_action, observation)

    def generate_sequence(initial_state, policy, policy_params, actions=None):
      return _model.generate_sequence(initial_state, policy, policy_params,
                                      actions)

    def observe_sequence(observations, actions):
      return _model.observe_sequence(observations, actions)

    def decode(feature):
      return _model.decode(feature)

    def init(observations, actions):
      return _model.observe_sequence(observations, actions)

    return init, (filter_state, generate_sequence, observe_sequence, decode)

  return hk.multi_transform(model)


class WorldModel(hk.Module):

  def __init__(self, observation_space, config):
    super(WorldModel, self).__init__()
    self.rssm = rssm.RSSM(config)
    self.encoder = Encoder(config.encoder['depth'],
                           tuple(config.encoder['kernels']),
                           config.initialization)
    self.decoder = Decoder(config.decoder['depth'],
                           tuple(config.decoder['kernels']),
                           observation_space.shape, config.initialization)
    self.reward = models.DenseDecoder(
        tuple(config.reward['output_sizes']) + (1,), 'normal',
        config.initialization, 'reward')
    self.cost = models.DenseDecoder(
        tuple(config.const['output_sizes']) + (1,), 'bernoulli',
        config.initialization, 'cost')

  def __call__(
      self, prev_state: rssm.State, prev_action: jnp.ndarray,
      observation: jnp.ndarray
  ) -> Tuple[Tuple[tfd.MultivariateNormalDiag, tfd.MultivariateNormalDiag],
             rssm.State]:
    observation = jnp.squeeze(self.encoder(observation[None, None]))
    return self.rssm(prev_state, prev_action, observation)

  def generate_sequence(
      self,
      initial_features: jnp.ndarray,
      actor: hk.Transformed,
      actor_params: hk.Params,
      actions=None) -> Tuple[jnp.ndarray, tfd.Normal, tfd.Bernoulli]:
    features = self.rssm.generate_sequence(initial_features, actor,
                                           actor_params, actions)
    reward = self.reward(features)
    cost = self.cost(features)
    return features, reward, cost

  def observe_sequence(
      self, observations: jnp.ndarray, actions: jnp.ndarray
  ) -> Tuple[Tuple[tfd.MultivariateNormalDiag, tfd.MultivariateNormalDiag],
             jnp.ndarray, tfd.Normal, tfd.Normal, tfd.Bernoulli]:
    observations = self.encoder(observations)
    (prior,
     posterior), features = self.rssm.observe_sequence(observations, actions)
    reward = self.reward(features)
    cost = self.cost(features)
    decoded = self.decode(features)
    return (prior, posterior), features, decoded, reward, cost

  def decode(self, featuers: jnp.ndarray) -> tfd.Normal:
    return self.decoder(featuers)


class Encoder(hk.Module):

  def __init__(self, depth: int, kernels: Sequence[int], initialization: str):
    super(Encoder, self).__init__()
    self._depth = depth
    self._kernels = kernels
    self._initialization = initialization

  def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:

    def cnn(x):
      kwargs = {
          'stride': 2,
          'padding': 'VALID',
          'w_init': nets.initializer(self._initialization)
      }
      for i, kernel in enumerate(self._kernels):
        depth = 2**i * self._depth
        x = jnn.relu(hk.Conv2D(depth, kernel, **kwargs)(x))
      return x

    cnn = hk.BatchApply(cnn)
    return hk.Flatten(2)(cnn(observation))


class Decoder(hk.Module):

  def __init__(self, depth: int, kernels: Sequence[int],
               output_shape: Sequence[int], initialization: str):
    super(Decoder, self).__init__()
    self._depth = depth
    self._kernels = kernels
    self._output_shape = output_shape
    self._initialization = initialization

  def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
    x = hk.BatchApply(
        hk.Linear(
            32 * self._depth, w_init=nets.initializer(self._initialization)))(
                features)
    x = hk.Reshape((1, 1, 32 * self._depth), 2)(x)

    def transpose_cnn(x):
      kwargs = {
          'stride': 2,
          'padding': 'VALID',
          'w_init': nets.initializer(self._initialization)
      }
      for i, kernel in enumerate(self._kernels):
        if i != len(self._kernels) - 1:
          depth = 2**(len(self._kernels) - i - 2) * self._depth
          x = jnn.relu(hk.Conv2DTranspose(depth, kernel, **kwargs)(x))
        else:
          x = hk.Conv2DTranspose(self._output_shape[-1], kernel, **kwargs)(x)
      return x

    out = hk.BatchApply(transpose_cnn)(x)
    return tfd.Independent(tfd.Normal(out, 1.0), len(self._output_shape))
