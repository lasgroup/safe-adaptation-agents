from typing import Callable

import jax.numpy as jnp
import jax


def solve(objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
          initial_guess: jnp.ndarray,
          key: jnp.ndarray,
          num_particles: int,
          num_iters: int,
          num_elite: int,
          stop_cond: float = 0.1):
  mu = initial_guess
  stddev = jnp.ones_like(initial_guess)

  def cond(val):
    _, iters, _, stddev, *_ = val
    return (stddev.mean() > stop_cond) & (iters < num_iters)

  def body(val):
    key, iter, mu, stddev, best_score, best = val
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (num_particles,) + mu.shape)
    sample = eps * stddev[None] + mu[None]
    scores = objective_fn(sample)
    elite_ids = jnp.argsort(scores)[-num_elite:]
    best = jnp.where(scores[elite_ids[-1]] > best_score,
                     sample[elite_ids[-1]], best)
    best_score = jnp.maximum(best_score, scores[elite_ids[-1]])
    elite = sample[elite_ids]
    # Moment matching on the `particles` axis
    mu, stddev = elite.mean(0), elite.std(0)
    return key, iter + 1, mu, stddev, best_score, best

  *_, best = jax.lax.while_loop(
      cond, body, (key, 0, mu, stddev, -jnp.inf, initial_guess))
  return best
