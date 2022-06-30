import jax.numpy as jnp

import scipy.signal

from safe_adaptation_agents import utils


def test_discounted_cumsum():
  gamma = 0.99
  x = jnp.arange(100)
  y_hat = utils.discounted_cumsum(x, gamma)
  y = discount_cumsum(x, gamma)
  assert jnp.isclose(y, y_hat).all()


def discount_cumsum(x, discount):
  """
  magic from rllab for computing discounted cumulative sums of vectors.

  input:
      vector x,
      [x0,
       x1,
       x2]

  output:
      [x0 + discount * x1 + discount^2 * x2,
       x1 + discount * x2,
       x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def test_td_lambda():
  values = jnp.array([
      -0.2996, 0.31, -0.02774, -0.05212, 0.2573, -0.04715, -0.199, 0.10864,
      0.1254, -0.07367, -0.0657, -0.1157, -0.3335, -0.2048, 0.1471
  ])
  rewards = jnp.array([
      -0.2603, 0.07007, 0.2067, 0.253, 0.1948, 0.2153, 0.1962, 0.3525,
      -0.002075, -0.3308, 0.3296, -0.3474, -0.2815, -0.10004
  ])
  terminals = jnp.zeros_like(rewards)
  lambda_ = 0.95
  discount = 0.99
  tds = rewards + (1. - terminals) * (1. - lambda_) * discount * values[1:]
  tds = tds.at[-1].add(lambda_ * discount * values[-1])
  # Ground truth values were computed offline in a for-loop:
  # https://github.com/yardenas/jax-dreamer/blob/b3f3945b389cc9153f8e06ad416252977bda488a/dreamer/utils.py
  gt_values = jnp.array([
      0.5723, 0.868, 0.8496, 0.686, 0.4468, 0.27, 0.0687, -0.1411, -0.5312,
      -0.5586, -0.239, -0.598, -0.2488, 0.04565
  ])
  assert jnp.isclose(
      utils.discounted_cumsum(tds, lambda_ * discount), gt_values, 1e-3,
      1e-3).all()
