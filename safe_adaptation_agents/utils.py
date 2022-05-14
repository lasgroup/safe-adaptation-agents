from typing import Callable, Tuple, Union, Any, Dict, NamedTuple

import haiku as hk
import jax.numpy as jnp
import jmp
import optax

PRNGKey = jnp.ndarray


class LearningState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState


class Learner:

  def __init__(self, model: Union[hk.Transformed, hk.MultiTransformed],
               seed: PRNGKey, optimizer_config: Dict, precision: jmp.Policy,
               *input_example: Any):
    self.optimizer = optax.flatten(
        optax.chain(
            optax.clip_by_global_norm(optimizer_config['clip']),
            optax.scale_by_adam(eps=optimizer_config['eps']),
            optax.scale(-optimizer_config['lr'])))
    self.model = model
    self.params = self.model.init(seed, *input_example)
    self.opt_state = self.optimizer.init(self.params)
    self.precision = precision

  @property
  def apply(self) -> Union[Callable, Tuple[Callable]]:
    return self.model.apply

  @property
  def state(self):
    return LearningState(self.params, self.opt_state)

  @state.setter
  def state(self, state: LearningState):
    self.params = state.params
    self.opt_state = state.opt_state

  def grad_step(self, grads, state: LearningState):
    params, opt_state = state
    grads = self.precision.cast_to_param(grads)
    updates, new_opt_state = self.optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    grads_finite = jmp.all_finite(grads)
    new_params, new_opt_state = jmp.select_tree(grads_finite,
                                                (new_params, new_opt_state),
                                                (params, opt_state))
    return LearningState(new_params, new_opt_state)


def get_mixed_precision_policy(precision):
  policy = ('params=float32,compute=float' + str(precision) + ',output=float' +
            str(precision))
  return jmp.get_policy(policy)


def discounted_cumsum(x: jnp.ndarray, discount: float) -> jnp.ndarray:
  """
  Compute a discounted cummulative sum of vector x. [x0, x1, x2] ->
  [x0 + discount * x1 + discount^2 * x2,
  x1 + discount * x2,
  x2]
  """
  # Divide by discount to have the first discount value from 1: [1, discount,
  # discount^2 ...]
  scales = jnp.cumprod(jnp.ones_like(x) * discount) / discount
  # Flip scales since jnp.convolve flips it as default.
  return jnp.convolve(x, scales[::-1])[-x.shape[0]:]
