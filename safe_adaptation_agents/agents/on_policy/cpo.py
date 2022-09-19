from functools import partial
from types import SimpleNamespace
from typing import Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Space
from jax.scipy import sparse
from tensorflow_probability.substrates import jax as tfp

from safe_adaptation_agents.agents.on_policy import safe_vpg
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState

tfd = tfp.distributions


class CPO(safe_vpg.SafeVanillaPolicyGradients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(CPO, self).__init__(observation_space, action_space, config, logger,
                              actor, critic, safety_critic)
    self.margin = 0.

  def train(self, trajectory_data):
    eval_ = self.evaluate_with_safety(self.critic.params,
                                      self.safety_critic.params,
                                      trajectory_data.o, trajectory_data.r,
                                      trajectory_data.c)
    constraint = trajectory_data.c.sum(1).mean()
    # https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L260
    c = constraint - self.config.cost_limit
    self.margin = max(0, self.margin + self.config.margin_lr * c)
    c += self.margin
    c /= (self.config.time_limit + 1e-8)
    self.actor.state, actor_report = self.update_actor(
        self.actor.state, trajectory_data.o[:, :-1], trajectory_data.a,
        eval_.advantage, eval_.cost_advantage, c)
    self.critic.state, critic_report = self.update_critic(
        self.critic.state, trajectory_data.o[:, :-1], eval_.return_)
    if self.safe:
      self.safety_critic.state, safety_report = self.update_safety_critic(
          self.safety_critic.state, trajectory_data.o[:, :-1],
          eval_.cost_return)
      critic_report.update(safety_report)
    actor_report['agent/margin'] = self.margin
    for k, v in {**actor_report, **critic_report}.items():
      self.logger[k] = np.asarray(v).mean()

  @partial(jax.jit, static_argnums=0)
  def update_actor(self, state: LearningState, *args) -> [LearningState, dict]:
    observation, action, advantage, cost_advantage, c = args
    old_pi = self.actor.apply(state.params, observation)
    old_pi_logprob = old_pi.log_prob(action)
    g, b, old_pi_loss, old_surrogate_cost = self._cpo_grads(
        state.params, observation, action, advantage, cost_advantage,
        old_pi_logprob)
    p, unravel_tree = jax.flatten_util.ravel_pytree(state.params)

    def d_kl_hvp(x):
      d_kl = (lambda p: self.actor.apply(unravel_tree(p), observation).
              kl_divergence(old_pi).mean())
      # Ravel the params so every computation from now on is made on actual
      # vectors.
      return hvp(d_kl, (p,), (x,))

    direction, optim_case = step_direction(g, b, c, d_kl_hvp,
                                           self.config.target_kl,
                                           self.config.safe,
                                           self.config.damping_coeff)

    def evaluate_policy(params):
      new_pi_loss, new_surrogate_cost = self.policy_loss(
          params, observation, action, advantage, cost_advantage,
          old_pi_logprob)
      pi = self.actor.apply(params, observation)
      kl_d = pi.kl_divergence(old_pi).mean()
      return new_pi_loss, new_surrogate_cost, kl_d

    new_params, info = backtracking(direction, evaluate_policy, old_pi_loss,
                                    old_surrogate_cost, optim_case, c,
                                    state.params, self.config.safe,
                                    self.config.backtrack_iters,
                                    self.config.backtrack_coeff,
                                    self.config.target_kl)
    return LearningState(new_params, self.actor.opt_state), info

  def _cpo_grads(self, pi_params: hk.Params, observation: jnp.ndarray,
                 action: jnp.ndarray, advantage: jnp.ndarray,
                 cost_advantage: jnp.ndarray, old_pi_logprob: jnp.ndarray):
    # Take gradients of the objective and surrogate cost w.r.t. pi_params.
    jac = jax.jacobian(self.policy_loss)(pi_params, observation, action,
                                         advantage, cost_advantage,
                                         old_pi_logprob)
    out = self.policy_loss(pi_params, observation, action, advantage,
                           cost_advantage, old_pi_logprob)
    old_pi_loss, surrogate_cost_old = out
    g, b = jac
    return g, b, old_pi_loss, surrogate_cost_old

  def policy_loss(self, params: hk.Params, *args):
    observation, action, advantage, cost_advantage, old_pi_logprob = args
    pi = self.actor.apply(params, observation)
    logprob = pi.log_prob(action)
    ratio = jnp.exp(logprob - old_pi_logprob)
    surr_advantage = ratio * advantage
    objective = (
        surr_advantage + self.config.entropy_regularization * pi.entropy())
    surrogate_cost = ratio * cost_advantage
    return -objective.mean(), surrogate_cost.mean()


def step_direction(g: chex.ArrayTree,
                   b: chex.ArrayTree,
                   c: jnp.ndarray,
                   d_kl_hvp: Callable,
                   target_kl: float,
                   safe: bool,
                   damping_coeff: float = 0.):
  # Implementation of CPO step direction is based on the implementation @
  # https://github.com/openai/safety-starter-agents
  # Take gradients of the objective and surrogate cost w.r.t. pi_params.
  g, unravel_tree = jax.flatten_util.ravel_pytree(g)
  b, _ = jax.flatten_util.ravel_pytree(b)
  # Add damping to hvp, as in TRPO.
  damped_d_kl_hvp = lambda v: d_kl_hvp(v) + damping_coeff * v
  v = sparse.linalg.cg(damped_d_kl_hvp, g, maxiter=10)[0]
  approx_g = damped_d_kl_hvp(v)
  q = jnp.dot(v, approx_g)

  def trpo():
    w, r, s, A, B = jnp.zeros_like(v), 0., 0., 0., 0.
    optim_case = 4
    return optim_case, w, r, s, A, B

  def cpo():
    w = sparse.linalg.cg(damped_d_kl_hvp, b, maxiter=10)[0]
    r = jnp.dot(w, approx_g)
    s = jnp.dot(w, damped_d_kl_hvp(w))
    A = q - r**2 / s
    B = 2. * target_kl - c**2 / s
    # Implementation of the elif conditions in https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L282
    optim_case = jax.lax.cond((c < 0.) & (B < 0.), lambda: 3, lambda: 0)
    optim_case = _maybe_update_case(optim_case, (c < 0.) & (B >= 0.), 2, 0)
    optim_case = _maybe_update_case(optim_case, (c >= 0.) & (B >= 0.), 1, 0)
    return optim_case, w, r, s, A, B

  if safe:
    optim_case, w, r, s, A, B = jax.lax.cond(
        jax.lax.bitwise_and(jnp.dot(b, b) <= 1e-8, c < 0.), trpo, cpo)
  else:
    optim_case, w, r, s, A, B = trpo()

  def no_recovery():

    def feasible_cases():
      lam = jnp.sqrt(q / (2. * target_kl))
      nu = 0.
      return lam, nu

    def non_feasible_cases():
      LA, LB = [0., r / c], [r / c, np.inf]
      LA, LB = jax.lax.cond(c < 0, lambda: (LA, LB), lambda: (LB, LA))
      proj = lambda x, L: jnp.maximum(L[0], jnp.minimum(L[1], x))
      lam_a = proj(jnp.sqrt(A / (B + 1e-8)), LA)
      lam_b = proj(jnp.sqrt(q / (2 * target_kl)), LB)
      f_a = lambda lam: -0.5 * (A / (lam + 1e-8) + B * lam) - r * c / (s + 1e-8)
      f_b = lambda lam: -0.5 * (q / (lam + 1e-8) + 2. * target_kl * lam)
      lam = jnp.where(f_a(lam_a) >= f_b(lam_b), lam_a, lam_b)
      nu = jnp.maximum(0, lam * c - r) / (s + 1e-8)
      return lam, nu

    return jax.lax.cond(optim_case > 2, feasible_cases, non_feasible_cases)

  def recovery():
    lam = 0.
    nu = jnp.sqrt(2. * target_kl / (s + 1e-8))
    return lam, nu

  lam, nu = jax.lax.cond(optim_case == 0, recovery, no_recovery)
  direction = jax.lax.cond(optim_case > 0, lambda: (v + nu * w) / (lam + 1e-8),
                           lambda: nu * w)
  return direction, optim_case


def backtracking(direction: jnp.ndarray, evaluate_policy: Callable,
                 old_pi_loss: jnp.ndarray, old_surrogate_cost: jnp.ndarray,
                 optim_case: int, c: jnp.ndarray, old_params: hk.Params,
                 safe: bool, backtrack_iters: int, backtrack_coeff: float,
                 target_kl: float):

  def cond(val):
    iter_, _, info = val
    kl = info['agent/actor/delta_kl']
    new_pi_loss = info['agent/actor/new_pi_loss']
    new_surrogate_cost = info['agent/actor/new_surrogate_cost']
    loss_improve = jax.lax.cond(optim_case > 1,
                                lambda: new_pi_loss <= old_pi_loss,
                                lambda: True)
    if safe:
      cost_improve = (
          new_surrogate_cost - old_surrogate_cost <= jnp.maximum(-c, 0))
    else:
      cost_improve = True
    kl_cond = kl <= target_kl
    improve = loss_improve & cost_improve & kl_cond
    return (~improve) & (iter_ < backtrack_iters)

  def body(val):
    iter_, *_ = val
    step_size = backtrack_coeff**iter_
    p, unravel_params = jax.flatten_util.ravel_pytree(old_params)
    new_params = unravel_params(p - step_size * direction)
    new_pi_loss, new_surrogate_cost, kl_d = evaluate_policy(new_params)
    return iter_ + 1, new_params, {
        'agent/actor/delta_kl': kl_d,
        'agent/actor/new_pi_loss': new_pi_loss,
        'agent/actor/new_surrogate_cost': new_surrogate_cost
    }

  init_state = (0, old_params, {
      'agent/actor/delta_kl': target_kl + 1e-8,
      'agent/actor/new_pi_loss': old_pi_loss + 1e-8,
      'agent/actor/new_surrogate_cost': old_surrogate_cost - 1e-8
  })
  iters, new_actor_params, info = jax.lax.while_loop(cond, body, init_state)
  # If used all backtracking iterations, fall back to the old policy.
  new_actor_params = jax.lax.cond(iters == backtrack_iters, lambda: old_params,
                                  lambda: new_actor_params)
  info['agent/actor/line_search_step'] = iters
  info['agent/actor/optim_case'] = optim_case
  return new_actor_params, info


# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
def hvp(f, primals, tangents):
  return jax.jvp(jax.grad(f), primals, tangents)[1]


def _maybe_update_case(optim_case, pred, true_val, false_val):
  return jax.lax.cond(
      optim_case != 0, lambda: optim_case,
      lambda: jax.lax.cond(pred, lambda: true_val, lambda: false_val))
