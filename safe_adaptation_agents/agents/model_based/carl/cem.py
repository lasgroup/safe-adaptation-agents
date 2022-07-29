from typing import Callable

import jax.numpy as jnp
import jax


def cross_entropy_method(objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
                         initial_guess: jnp.ndarray,
                         key: jnp.ndarray,
                         num_particles: int,
                         num_iters: int,
                         num_elite: int,
                         stop_cond: float = 0.1):
  mu = initial_guess
  stddev = jnp.ones_like(initial_guess)

  def cond(val):
    _, iters, _, stddev = val
    return (stddev.mean() > stop_cond) & (iters < num_iters)

  def body(val):
    key, iter, mu, stddev = val
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (num_particles,) + mu.shape)
    sample = eps * stddev + mu
    scores = objective_fn(sample)
    elite_ids = jnp.argsort(scores)[-num_elite:]
    elite = sample[elite_ids]
    mu, stddev = elite.mean(), elite.stddev()
    return key, iter + 1, mu, stddev

  _, _, mu, _ = jax.lax.while_loop(cond, body, (key, 0, mu, stddev))
  return mu
