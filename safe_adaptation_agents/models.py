from typing import Sequence, Optional, Callable, Union

import jax.random

import jax.numpy as jnp
import jax.nn as jnn

import haiku as hk

from tensorflow_probability.substrates import jax as tfp

from safe_adaptation_agents import nets
from safe_adaptation_agents import utils

tfd = tfp.distributions
tfb = tfp.bijectors


class Actor(hk.Module):

  def __init__(self,
               output_size: Sequence[int],
               layers: Sequence[int],
               min_stddev: float,
               max_stddev: float,
               initialization: str = 'glorot',
               activation: Union[str, Callable[[jnp.ndarray],
                                               jnp.ndarray]] = jnn.relu,
               squash: bool = False,
               heteroscedastic: bool = True):
    super().__init__()
    factor = 2 if heteroscedastic else 1
    self.output_size = tuple(map(lambda x: factor * x, output_size))
    self.layers = layers
    self._min_stddev = min_stddev
    self._max_stddev = max_stddev
    self._initialization = initialization
    self._activation = activation if callable(activation) else eval(activation)
    self._squash = squash
    self._heteroscedastic = heteroscedastic

  def __call__(self, observation: jnp.ndarray):
    x = nets.mlp(
        observation,
        output_sizes=tuple(self.layers) + tuple(self.output_size),
        initializer=nets.initializer(self._initialization),
        activation=self._activation)
    if self._heteroscedastic:
      mu, stddev = jnp.split(x, 2, -1)
    else:
      mu, stddev = x, hk.get_parameter('pi_stddev', (x.shape[-1],), x.dtype,
                                       hk.initializers.Constant(-0.5))
      if stddev.ndim == 1:
        stddev = jnp.expand_dims(stddev, 0)
    if self._squash:
      init_std = utils.inv_softplus(5.)
      stddev = jnn.softplus(stddev + init_std) + self._min_stddev
      multivariate_normal_diag = tfd.Normal(5.0 * jnn.tanh(mu / 5.0), stddev)
      multivariate_normal_diag = tfd.TransformedDistribution(
          multivariate_normal_diag, StableTanhBijector())
    else:
      stddev = jnp.exp(stddev)
      multivariate_normal_diag = tfd.Normal(mu, stddev)
    dist = tfd.Independent(multivariate_normal_diag, 1)
    if self._squash:
      dist = SampleDist(dist)
    return dist


class DenseDecoder(hk.Module):

  def __init__(self,
               output_size: Sequence[int],
               layers: Sequence[int],
               dist: str,
               initialization: str = 'glorot',
               activation: Union[str, Callable[[jnp.ndarray],
                                               jnp.ndarray]] = jnn.relu,
               name: Optional[str] = None):
    super(DenseDecoder, self).__init__(name)
    self.output_size = output_size
    self.layers = layers
    self._dist = dist
    self._initialization = initialization
    self._activation = activation if callable(activation) else eval(activation)

  def __call__(self, x: jnp.ndarray):
    x = nets.mlp(
        x,
        output_sizes=tuple(self.layers) + tuple(self.output_size),
        initializer=nets.initializer(self._initialization),
        activation=self._activation)
    x = jnp.squeeze(x, axis=-1)
    dist = dict(
        normal=lambda mu: tfd.Normal(mu, 1.0),
        bernoulli=lambda p: tfd.Bernoulli(p, dtype=jnp.float32))[self._dist]
    return tfd.Independent(dist(x), 0)


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
class StableTanhBijector(tfb.Tanh):

  def __init__(self, validate_args=False, name='tanh_stable_bijector'):
    super(StableTanhBijector, self).__init__(
        validate_args=validate_args, name=name)

  def _inverse(self, y):
    dtype = y.dtype
    y = y.astype(jnp.float32)
    y = jnp.clip(y, -0.99999997, 0.99999997)
    y = jnp.arctanh(y)
    return y.astype(dtype)


class SampleDist:

  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    samples = self._dist.sample(self._samples, seed=jax.random.PRNGKey(666))
    return jnp.mean(samples, 0)

  def mode(self):
    sample = self._dist.sample(self._samples, seed=jax.random.PRNGKey(666))
    logprob = self._dist.log_prob(sample)
    return jnp.take_along_axis(sample,
                               jnp.argmax(logprob, 0)[None, :, None],
                               0).squeeze(0)

  def entropy(self):
    sample = self._dist.sample(self._samples, seed=jax.random.PRNGKey(666))
    logprob = self.log_prob(sample)
    return -jnp.mean(logprob, 0)
