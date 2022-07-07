import numpy as np


class AugmentedLagrangian:

  def __init__(self, initial_lagrangian: float, initial_penalty: float,
               penalty_multiplier_factor: float):
    super(AugmentedLagrangian, self).__init__()
    self.lagrangian = initial_lagrangian
    self.penalty = initial_penalty
    self.penalty_multiplier_factor = penalty_multiplier_factor

  def __call__(self, cost_value: np.ndarray, cost_threshold: np.ndarray):
    # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    # Taking the mean value since E[V_c(s)]_p(s) ~= J_c
    g = cost_value - cost_threshold
    c = self.penalty
    cond = self.lagrangian + c * g
    old_lagrangian = self.lagrangian
    self.lagrangian = max(0., cond)
    if cond < 0.:
      psi = old_lagrangian * g + c / 2. * g**2
    else:
      psi = -1. / (2. * c) * old_lagrangian**2
    # Clip to make sure that c is non-decreasing.
    self.penalty = np.clip(c * (self.penalty_multiplier_factor + 1.), c, 1.)
    return psi, old_lagrangian, c
