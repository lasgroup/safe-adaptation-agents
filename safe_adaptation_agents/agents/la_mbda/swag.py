from copy import deepcopy
from functools import partial
from typing import Union, Dict, Any, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax

from safe_adaptation_agents import utils as u


class SWAGLearningState(NamedTuple):
  learning_state: u.LearningState
  mu: hk.Params
  variance: hk.Params
  covariance: hk.Params

  @property
  def params(self):
    return self.learning_state.params

  @property
  def opt_state(self):
    return self.learning_state.opt_state

  @property
  def iterations(self):
    return self.learning_state.opt_state[1][1].count


def cyclic_learning_rate(initial_value: float, peak_value: float,
                         cycle_steps: int):

  def schedule(count):
    cycle = jnp.floor(1 + count / (2. * cycle_steps))
    progress = jnp.abs(count / cycle_steps - 2 * cycle + 1)
    y = jnp.maximum(0, (1. - progress))
    return peak_value - (peak_value - initial_value) * y

  return schedule


class SWAG(u.Learner):

  def __init__(self,
               model: Union[hk.Transformed, hk.MultiTransformed,
                            chex.ArrayTree],
               seed: u.PRNGKey,
               optimizer_config: Dict,
               precision: jmp.Policy,
               *input_example: Any,
               start_averaging: int,
               average_period: int,
               max_num_models: int,
               decay: float,
               learning_rate_factor: float,
               scale: float = 1.):
    super(SWAG, self).__init__(model, seed, optimizer_config, precision,
                               *input_example)
    # base_lr = optimizer_config.get('lr', 1e-3)
    # schedule = cyclic_learning_rate(base_lr, base_lr * learning_rate_factor,
    #                                 average_period)
    # self.optimizer = optax.flatten(
    #     optax.chain(
    #         optax.clip_by_global_norm(
    #             optimizer_config.get('clip', float('inf'))),
    #         optax.adam(
    #             learning_rate=schedule, eps=optimizer_config.get('eps', 1e-8)),
    #     ))
    # self.opt_state = self.optimizer.init(self.params)
    self._start_averaging = start_averaging
    self._average_period = average_period
    self._max_num_models = max_num_models
    self._decay = decay
    self.scale = scale
    self.mu = jax.tree_map(jnp.zeros_like, self.params)
    self.variance = deepcopy(self.mu)
    var = [
        jax.tree_map(jnp.zeros_like, self.params) for _ in range(max_num_models)
    ]
    self.covariance_mat = u.pytrees_stack(var)

  @property
  def state(self):
    learning_state = super(SWAG, self).state
    return SWAGLearningState(learning_state, self.mu, self.variance,
                             self.covariance_mat)

  @state.setter
  def state(self, state: SWAGLearningState):
    # Set Learner's `learning_state` in a hacky way.
    u.Learner.state.fset(self, state.learning_state)
    self.mu = state.mu
    self.variance = state.variance
    self.covariance_mat = state.covariance

  @property
  def warm(self):
    return False
    # count = self.state.iterations
    # average_period = self._average_period
    # num_snapshots = max(0, (count - self._start_averaging) // average_period)
    # max_num_models = self._max_num_models
    # return count >= self._start_averaging and num_snapshots >= max_num_models

  def grad_step(self, grads, state: SWAGLearningState) -> SWAGLearningState:
    learning_state = super(SWAG, self).grad_step(grads, state.learning_state)
    # mu, variance, covariance = self._update_stats(learning_state, state.mu,
    #                                               state.variance,
    #                                               state.covariance)
    return SWAGLearningState(learning_state, state.mu, state.variance, state.covariance)

  def _update_stats(self, updated_state: u.LearningState, mu: hk.Params,
                    variance: hk.Params,
                    covariance: hk.Params) -> [hk.Params, hk.Params, hk.Params]:
    decay = self._decay
    # number of times snapshots of weights have been taken (using max to
    # avoid negative values of num_snapshots).
    iterations = updated_state.opt_state[1][1].count
    num_snapshots = jnp.maximum(0, (iterations - self._start_averaging) //
                                self._average_period)

    def compute_stats():
      # https://www.tensorflow.org/probability/api_docs/python/tfp/stats/assign_moving_mean_variance
      t = num_snapshots + 1
      bias_correction = 1. - decay**t

      def compute_mu(old_mean, value):
        new_mean = old_mean + (1. - decay) * (value - old_mean)
        new_mean /= bias_correction
        return new_mean

      def compute_var(old_mean, old_var, value):
        old_count = jnp.where(t > 1., t - 1., jnp.inf)
        old_bias_correction = 1. - decay**old_count
        old_mean /= old_bias_correction
        sq_diff = (value - old_mean)**2
        new_var = old_var + (1. - decay) * (decay * sq_diff - old_var)
        new_var /= bias_correction
        return new_var

      new_mean = jax.tree_map(compute_mu, mu, updated_state.params)
      new_var = jax.tree_map(compute_var, mu, variance, updated_state.params)

      def compute_cov(old_cov, old_param, new_mean):
        # Shift old covariances one step to the right. Update the leftmost
        # element with new covariance.
        old_cov = jnp.roll(old_cov, 1, 0)
        new_cov = old_cov.at[0].set(old_param - new_mean)
        return new_cov

      new_cov = jax.tree_map(compute_cov, covariance, updated_state.params,
                             new_mean)
      return new_mean, new_var, new_cov

    # The mean update should happen iff two conditions are met:
    # 1. A min number of iterations (start_averaging) have taken place.
    # 2. Iteration is one in which snapshot should be taken.
    checkpoint = self._start_averaging + num_snapshots * self._average_period
    mu, variance, covariance = jax.lax.cond(
        (iterations >= self._start_averaging) & (iterations == checkpoint),
        compute_stats, lambda: (mu, variance, covariance))
    return mu, variance, covariance

  def posterior_samples(self, num_samples: int, key: u.PRNGKey):
    state = self.state
    return _sample(
        state.mu,
        state.variance,
        state.covariance,
        self.scale,
        jnp.asarray(jax.random.split(key, num_samples)),
    )


@jax.jit
@partial(jax.vmap, in_axes=(None, None, None, None, 0))
def _sample(mean: hk.Params, variance: hk.Params, covariance: hk.Params,
            scale: float, key: u.PRNGKey) -> hk.Params:
  key, subkey = jax.random.split(key)
  sample_var = lambda p, k: jnp.sqrt(p / 2.) * jax.random.normal(k, p.shape)
  leaves, flat = jax.tree_flatten(variance)
  num_leaves = len(leaves)
  keys = jax.random.split(key, num_leaves + 1)
  var_keys = jax.tree_unflatten(flat, keys[1:])
  var_sample = jax.tree_map(sample_var, variance, var_keys)
  sample_cov = lambda p, k: jnp.matmul(p.T, jax.random.normal(
      k, (p.shape[0], 1))) / ((2. * (p.shape[0] - 1.))**0.5)
  keys = jax.random.split(keys[0], num_leaves + 1)
  cov_keys = jax.tree_unflatten(flat, keys[1:])
  cov_sample = jax.tree_map(sample_cov, covariance, cov_keys)
  rand_sample = lambda v, c: v + c.reshape(v.shape)
  sample_rand = jax.tree_map(rand_sample, var_sample, cov_sample)
  shift_scale = lambda m, r: m + (scale**0.5) * r
  sample = jax.tree_map(shift_scale, mean, sample_rand)
  return sample
