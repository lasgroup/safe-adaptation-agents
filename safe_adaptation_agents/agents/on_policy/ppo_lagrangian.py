from types import SimpleNamespace
import functools

import numpy as np
from gym.spaces import Space

import optax
import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk

from safe_adaptation_agents.agents.on_policy import safe_vpg
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents import utils
from safe_adaptation_agents.utils import LearningState
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData


class PpoLagrangian(safe_vpg.SafeVanillaPolicyGradients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(PpoLagrangian, self).__init__(observation_space, action_space, config,
                                        logger, actor, critic, safety_critic)
    lagrangian = hk.without_apply_rng(
        hk.transform(lambda: Lagrangian(self.config.initial_lagrangian)()))
    self.lagrangian = utils.Learner(
        lagrangian, next(self.rng_seq), config.lagrangian_opt,
        utils.get_mixed_precision_policy(config.precision))

  def train(self, trajectory_data: TrajectoryData):
    (
        advantage,
        return_,
        cost_advantage,
        cost_return,
    ) = self.evaluate_with_safety(self.critic.params, self.safety_critic.params,
                                  trajectory_data.o, trajectory_data.r,
                                  trajectory_data.c)
    if self.safe:
      constraint = trajectory_data.c.sum(1).mean()
      self.lagrangian.state, lagrangian_report = self.lagrangian_update_step(
          self.lagrangian.state, constraint)
      lagrangian = jnn.softplus(self.lagrangian.apply(self.lagrangian.params))
    else:
      lagrangian = 0.
      lagrangian_report = {}
    self.actor.state, actor_report = self.update_actor(
        self.actor.state, trajectory_data.o[:, :-1], trajectory_data.a,
        advantage, cost_advantage, lagrangian)
    self.critic.state, critic_report = self.update_critic(
        self.critic.state, trajectory_data.o[:, :-1], return_)
    if self.safe:
      self.safety_critic.state, safety_report = self.update_safety_critic(
          self.safety_critic.state, trajectory_data.o[:, :-1], cost_return)
      critic_report.update({**safety_report, **lagrangian_report})
    for k, v in {**actor_report, **critic_report}.items():
      self.logger[k] = v.mean()

  @functools.partial(jax.jit, static_argnums=0)
  def update_actor(self, state: LearningState, *args) -> [LearningState, dict]:
    observation, action, advantage, cost_advantage, lagrangian = args
    old_pi = self.actor.apply(state.params, observation)
    old_pi_logprob = old_pi.log_prob(action)

    def cond(val):
      iter_, _, info = val
      kl = info['agent/actor/delta_kl']
      # Returns Truthy if iter is smaller than pi_iters and kl smaller than
      # kl threshold to continue iterating
      return jax.lax.bitwise_not(
          jax.lax.bitwise_or(kl > self.config.kl_margin * self.config.target_kl,
                             iter_ == self.config.pi_iters))

    def body(val):
      iter_, actor_state, _ = val
      loss, grads = jax.value_and_grad(
          self.policy_loss)(actor_state.params, observation, action, advantage,
                            cost_advantage, lagrangian, old_pi_logprob)
      new_actor_state = self.actor.grad_step(grads, actor_state)
      pi = self.actor.apply(new_actor_state.params, observation)
      kl_d = old_pi.kl_divergence(pi).mean()
      return iter_ + 1, new_actor_state, {
          'agent/actor/loss': loss,
          'agent/actor/grad': optax.global_norm(grads),
          'agent/actor/entropy': pi.entropy().mean(),
          'agent/actor/delta_kl': kl_d
      }

    # Implements a for-loop (through iter_) with an early-break condition (ppo's
    # too big kl)
    init_state = (0, state, {
        'agent/actor/loss': 0.,
        'agent/actor/grad': 0.,
        'agent/actor/entropy': 0.,
        'agent/actor/delta_kl': 0.
    })
    iters, new_actor_state, info = jax.lax.while_loop(cond, body,
                                                      init_state)  # noqa
    info['agent/actor/update_iters'] = iters
    return new_actor_state, info

  def policy_loss(self, params: hk.Params, *args, clip=True) -> jnp.ndarray:
    (
        observation,
        action,
        advantage,
        cost_advantage,
        lagrangian,
        old_pi_logprob,
    ) = args
    pi = self.actor.apply(params, observation)
    log_prob = pi.log_prob(action)
    ratio = jnp.exp(log_prob - old_pi_logprob)
    if clip:
      min_adv = jnp.where(advantage >= 0.,
                          (1. + self.config.clip_ratio) * advantage,
                          (1. - self.config.clip_ratio) * advantage)
      surr_advantage = jnp.minimum(ratio * advantage, min_adv)
    else:
      surr_advantage = ratio * advantage
    objective = (
        surr_advantage + self.config.entropy_regularization * pi.entropy())
    if self.safe:
      # https: // github.com / openai / safety - starter - agents / blob / 4151
      # a283967520ee000f03b3a79bf35262ff3509 / safe_rl / pg / run_agent.py  #
      # L178
      objective -= lagrangian * ratio * cost_advantage
      objective /= (1. + lagrangian)
    return -objective.mean()

  @functools.partial(jax.jit, static_argnums=0)
  def lagrangian_update_step(self, lagrangian: LearningState,
                             constraint: jnp.ndarray) -> [LearningState, dict]:

    def loss(params):
      return -(self.lagrangian.apply(params) *
               (constraint - self.config.cost_limit))[0]

    loss, grad = jax.value_and_grad(loss)(lagrangian.params)
    new_lagrangian_state = self.lagrangian.grad_step(grad, lagrangian)
    return new_lagrangian_state, {
        'agent/lagrangian/loss': loss,
        'agent/lagrangian/grad': optax.global_norm(grad)
    }


class Lagrangian(hk.Module):

  def __init__(self, initial_lagrangian):
    super(Lagrangian, self).__init__()
    self._initial_lagrangian = initial_lagrangian

  def __call__(self):
    init_lagrangian = np.log(max(np.exp(self._initial_lagrangian) - 1., 1e-8))
    lagrangian = hk.get_parameter('lagrangian', (1,), jnp.float32,
                                  hk.initializers.Constant(init_lagrangian))
    return lagrangian
