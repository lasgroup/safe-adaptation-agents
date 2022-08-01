import haiku as hk
import jax.numpy as jnp
from jax import lax


class AugmentedLagrangian(hk.Module):

  def __init__(self, initial_lagrangian: float, initial_penalty: float):
    super(AugmentedLagrangian, self).__init__()
    self._initial_lagrangian = initial_lagrangian
    self._initial_penalty = initial_penalty

  def __call__(self, cost_value: jnp.ndarray,
               cost_threshold: jnp.ndarray) -> [jnp.ndarray, jnp.ndarray]:
    # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    # Taking the mean value since E[V_c(s)]_p(s) ~= J_c
    g = cost_value - cost_threshold
    c = hk.get_parameter(
        'penalty',
        (),
        init=hk.initializers.Constant(self._initial_penalty),
    )
    lagrangian = hk.get_parameter(
        'lagrangian',
        (),
        init=hk.initializers.Constant(self._initial_lagrangian),
    )
    cond = lagrangian + c * g
    psi = lax.cond(
        cond > 0.,
        lambda: lagrangian * g + c / 2. * g**2,
        lambda: -1. / (2. * c) * lagrangian**2,
    )
    return psi, cond
