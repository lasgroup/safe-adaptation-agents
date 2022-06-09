from types import SimpleNamespace
from typing import Callable
from functools import partial

import numpy as np
from gym.spaces import Space

import jax
import jax.numpy as jnp
from jax.scipy import sparse
import haiku as hk

from tensorflow_probability.substrates import jax as tfp

from safe_adaptation_agents.agents.on_policy import safe_vpg
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState

tfd = tfp.distributions


class Cpo(safe_vpg.SafeVanillaPolicyGradients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(Cpo, self).__init__(observation_space, action_space, config, logger,
                              actor, critic, safety_critic)

  def train(self, trajectory_data):
    pass

  @partial(jax.jit, static_argnums=0)
  def update_actor(self, state: LearningState, *args) -> [LearningState, dict]:
    observation, action, advantage, cost_advantage, constraint = args
    old_pi = self.actor.apply(state.params, observation)
    old_pi_logprob = old_pi.log_prob(action)
    (
        direction,
        unravel_params,
        old_pi_loss,
        old_surrogate_cost,
        optim_case,
        c,
    ) = self.step_direction(state.params, old_pi, observation, action,
                            advantage, cost_advantage, old_pi_logprob,
                            constraint)
    new_params, info = self.backtracking(direction, old_pi_loss,
                                         old_surrogate_cost, state.params,
                                         observation, c, old_pi)
    return new_params, info

  def step_direction(
      self, pi_params: hk.Params, old_pi, observation: jnp.ndarray,
      action: jnp.ndarray, advantage: jnp.ndarray, cost_advantage: jnp.ndarray,
      old_pi_logprob: jnp.ndarray, constraint: jnp.ndarray
  ) -> [jnp.ndarray, Callable, jnp.ndarray, jnp.ndarray, int]:
    # Implementation of CPO step direction is based on the implementation @
    # https://github.com/openai/safety-starter-agents
    out, jac = value_and_jacrev(self.policy_loss)(pi_params, observation,
                                                  action, advantage,
                                                  cost_advantage,
                                                  old_pi_logprob)
    old_pi_loss, surrogate_cost_old = out
    g, b = jac
    g, unravel_grads = jax.flatten_util.ravel_pytree(g)
    b, _ = jax.flatten_util.ravel_pytree(b)
    # https://github.com/openai/safety-starter-agents/blob
    # /4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L260
    c = (constraint - self.config.cost_limit) / (self.config.time_limit + 1e-8)

    # Computing grad d_kl w.r.t pi_params (inside hvp) could have been
    # implemented outside to save computation time, but here the code is
    # clearer and closer to the actual math.
    def d_kl_hvp(x):
      return hvp(
          lambda pi_params: old_pi.kl_divergence(
              self.actor(pi_params, observation)).mean(), pi_params, x)

    v = sparse.linalg.cg(d_kl_hvp, g)
    approx_g = d_kl_hvp(v)
    q = jnp.dot(v, approx_g)

    def trpo_case():
      w, r, s, A, B = 0, 0, 0, 0, 0
      optim_case = 4
      return optim_case, w, r, s, A, B

    def cpo_case():
      w = sparse.linalg.cg(d_kl_hvp, b)
      r = jnp.dot(w, approx_g)
      s = jnp.dot(w, d_kl_hvp(w))
      A = q - r**2 / s
      B = 2. * self.config.target_kl - c**2 / s
      optim_case = jax.lax.cond(
          jax.lax.bitwise_and(c < 0, B < 0), lambda: 3, lambda: 0)
      optim_case = jax.lax.cond(
          jax.lax.bitwise_and(
              jax.lax.bitwise_and(optim_case == 0, c < 0.), B >= 0.), lambda: 2,
          lambda: 0)
      optim_case = jax.lax.cond(
          jax.lax.bitwise_and(
              jax.lax.bitwise_and(optim_case == 0, c >= 0), B >= 0), lambda: 1,
          lambda: 0)
      return optim_case, w, r, s, A, B

    if self.config.safe:
      optim_case, w, r, s, A, B = jax.lax.cond(
          jax.lax.bitwise_and(jnp.dot(b, b) <= 1e-8, c < 0), trpo_case(),
          cpo_case())
    else:
      optim_case, w, r, s, A, B = trpo_case()

    def no_recovery():

      def feasible_cases():
        lam = jnp.sqrt(q / (2. * self.config.target_kl))
        nu = 0.
        return lam, nu

      def non_feasible_cases():
        LA, LB = [0, r / c], [r / c, np.inf]
        LA, LB = jax.lax.cond(c < 0, lambda: LA, LB, lambda: LB, LA)
        proj = lambda x, L: jnp.max(L[0], jnp.min(L[1], x))
        lam_a = proj(jnp.sqrt(A / B), LA)
        lam_b = proj(jnp.sqrt(q / (2 * self.config.target_kl)), LB)
        f_a = lambda lam: -0.5 * (A / (lam + 1e-8) + B * lam) - r * c / (
            s + 1e-8)
        f_b = lambda lam: -0.5 * (
            q / (lam + 1e-8) + 2 * self.config.target_kl * lam)
        lam = jnp.where(f_a(lam_a) >= f_b(lam_b), lam_a, lam_b)
        nu = jnp.max(0, lam * c - r) / (s + 1e-8)
        return lam, nu

      return jax.lax.cond(optim_case > 2, feasible_cases(),
                          non_feasible_cases())

    def recovery():
      lam = 0
      nu = jnp.sqrt(2. * self.config.target_kl / (s + 1e-8))
      return lam, nu

    lam, nu = jax.lax.cond(optim_case == 0, recovery(), no_recovery())
    direction = jax.lax.cond(optim_case > 0, lambda: (v + nu * w) /
                             (lam + 1e-8), lambda: nu * w)
    return (
        direction,
        unravel_grads,
        old_pi_loss,
        surrogate_cost_old,
        optim_case,
        c,
    )

  def backtracking(self, direction: jnp.ndarray, old_pi_loss: jnp.ndarray,
                   old_surrogate_cost: jnp.ndarray, old_params: hk.Params,
                   observation: jnp.ndarray, c: jnp.ndarray,
                   old_pi: tfd.Distribution):

    def cond(val):
      iter_, _, info = val
      kl = info['agent/actor/delta_kl']
      new_pi_loss = info['agent/actor/new_pi_loss']
      new_surrogate_cost = info['agent/actor/new_surrogate_cost']
      optim_case = info['agent/actor/optim_case']
      loss_cond = jax.lax.cond(optim_case > 1,
                               lambda: new_pi_loss <= old_pi_loss, lambda: True)
      cost_cond = new_surrogate_cost - old_surrogate_cost <= jnp.max(-c, 0)
      kl_cond = kl <= self.config.target_kl
      iters_cond = iter_ == self.config.backtrack_iters
      performance_cond = loss_cond & cost_cond & kl_cond
      return jax.lax.bitwise_not(
          jax.lax.bitwise_or(performance_cond, iters_cond))

    def body(val):
      iter_, *_ = val
      step_size = self.config.backtrack_coefficient**iter_
      p, unravel_params = jax.tree_util.ravel_pytree(old_params)
      new_params = unravel_params(p - step_size * direction)
      pi = self.actor.apply(new_params, observation)
      kl_d = old_pi.kl_divergence(pi).mean()
      new_pi_loss, new_surrogate_cost = self.policy_loss(new_params)
      return iter_ + 1, new_params, {
          'agent/actor/entropy': pi.entropy().mean(),
          'agent/actor/delta_kl': kl_d,
          'agent/actor/new_pi_loss': new_pi_loss,
          'agent/actor/new_surrogate_cost': new_surrogate_cost
      }

    init_state = (0, old_params, {
        'agent/actor/entropy': 0.,
        'agent/actor/delta_kl': 0.,
        'agent/actor/new_pi_loss': 0.,
        'agent/actor/new_surrogate_cost': 0.
    })
    iters, new_actor_params, info = jax.lax.while_loop(cond, body,
                                                       init_state)  # noqa
    # If used all backtracking iterations, fall back to the old policy.
    new_actor_params = jax.lax.cond(iters == self.config.backtrack_iters,
                                    lambda: old_params,
                                    lambda: new_actor_params)
    info['agent/actor/update_iters'] = iters
    return new_actor_params, info

  def policy_loss(self, params: hk.Params, *args):
    observation, action, advantage, cost_advantage, old_pi_logprob = args
    pi = self.actor.apply(params, observation)
    log_prob = pi.log_prob(action)
    ratio = jnp.exp(log_prob - old_pi_logprob)
    surr_advantage = ratio * advantage
    objective = (
        surr_advantage + self.config.entropy_regularization * pi.entropy())
    surrogate_cost = ratio * cost_advantage
    return jnp.stack([-objective.mean(), surrogate_cost.mean()])


# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
def hvp(f, primals, tangents):
  return jax.jvp(jax.grad(f), primals, tangents)[1]


# https://github.com/google/jax/discussions/9743
def value_and_jacrev(fun: Callable):

  def value_and_jacrev_fun(x):
    y, pullback = jax.vjp(fun, x)
    y: jnp.ndarray  # y should be an array
    assert y.ndim == 1
    jac = jax.vmap(pullback)(jnp.eye(y.shape[0], dtype=y.dtype))[0]
    return y, jac

  return value_and_jacrev_fun
