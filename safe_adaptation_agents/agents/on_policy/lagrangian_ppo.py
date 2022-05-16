from typing import Optional
from types import SimpleNamespace
import functools

import numpy as np
from gym.spaces import Space

import optax
import jax
import jax.numpy as jnp
import haiku as hk

from safe_adaptation_agents.agents.on_policy import safe_vpg
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents import utils
from safe_adaptation_agents.utils import LearningState


class LagrangianPPO(safe_vpg.SafeVanillaPolicyGradients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(LagrangianPPO, self).__init__(observation_space, action_space, config,
                                        logger, actor, critic)
    self.safety_critic = utils.Learner(
        safety_critic, next(self.rng_seq), config.critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
    self.lagrangian = utils.Learner(
        Lagrangian(self.config.initial_lagrangian),
        next(self.rng_seq), config.lagrangian_opt,
        utils.get_mixed_precision_policy(config.precision))

  def train(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, running_cost: np.ndarray):
    (advantage, return_, cost_advantage, cost_return,
     logprob_pi) = self.evaluate_with_safety(self.critic.params,
                                             self.safety_critic.params,
                                             self.actor.params, observation,
                                             action, reward, cost)
    self.lagrangian.state, lagrangian_report = self.lagrangian_update_step(
        self.lagrangian.state, running_cost, self.config.cost_limit)
    lagrangian = jnn.softplus(self.lagrangian.apply(self.lagrangian.params))
    self.actor.state, actor_report = self.actor_update_step(
        self.actor.state,
        observation[:, :-1],
        action=action,
        old_pi_logprob=logprob_pi,
        advantage=advantage,
        lagrangian=lagrangian,
        cost_advantage=cost_advantage)
    self.critic.state, critic_report = self.critic_update_step(
        self.critic.state, observation[:, -1], return_)
    if self.safe:
      self.safety_critic.state, safety_report = self.safe_critic_update_step(
          self.safety_critic.state, observation[:, :-1], cost_return)
      critic_report.update(safety_report)
    for k, v in {**actor_report, **critic_report, **lagrangian_report}.items():
      self.logger[k] = v.mean()

  @functools.partial(jax.jit, static_argnums=0)
  def actor_update_step(self, state: LearningState, *args,
                        **kwargs) -> [LearningState, dict]:
    observation, *_ = args
    old_pi = self.actor.apply(state.params, observation)

    def cond(val):
      iter_, _, info = val
      kl = info['agent/actor/delta_kl']
      if iter_ == self.config.pi_iter or kl > self.kl_margin * target_kl:
        return False
      return True

    def body(val):
      iter_, actor_state, _ = val
      loss, grads = jax.value_and_grad(self.policy_loss)(actor_state.params,
                                                         *args, **kwargs)
      new_actor_state = self.actor.grad_step(grads, actor_state)
      pi = self.actor.apply(actor_state.params, observation)
      kl_d = old_pi.kl_divergence(pi).mean()
      return iter_ + 1, new_actor_state, {
          'agent/actor/loss': loss,
          'agent/actor/grad': optax.global_norm(grads),
          'agent/actor/entropy': pi.entropy().mean(),
          'agent/actor/delta_kl': kl_d
      }

    iters, new_actor_state, info = jax.lax.while_loop(cond, body, (0, state, {
        'agent/actor/delta_kl': 0.
    }))
    info['agent/actor/update_iters'] = iters
    return new_actor_state, info

  @functools.partial(jax.jit, static_argnums=0)
  def lagrangian_update_step(self, lagrangian: LearningState,
                             running_cost: jnp.ndarray,
                             cost_limits: jnp.ndarray):

    def loss(lagrangian):
      return -self.lagrangian.apply(lagrangian) * (running_cost - cost_limits)

    loss, grad = jax.value_and_grad(loss)(lagrangian.params)
    new_lagrangian_state = self.lagrangian.grad_step(grad, lagrangian)
    return new_lagrangian_state, {
        'agent/lagrangian/loss': loss,
        'agent/lagrangian/grad': optax.global_norm(grad)
    }

  def policy_loss(self, params: hk.Params, *args, **kwargs) -> float:
    observation, *_ = args
    pi = self.actor.apply(params, observation)
    log_prob = pi.log_prob(kwargs['action'])
    ratio = jnp.exp(log_prob - kwargs['old_pi_logprob'])
    advantage = kwargs['advantage']
    min_adv = jnp.where(advantage > 0.,
                        (1. + self.config.clip_ratio) * advantage,
                        (1. - self.config.clip_ratio) * advantage)
    surr_advantage = jnp.minimum(ratio * advantage, min_adv)
    objective = (
        log_prob * surr_advantage +
        self.config.entropy_regularization * pi.entropy())
    if self.safe:
      # https: // github.com / openai / safety - starter - agents / blob / 4151
      # a283967520ee000f03b3a79bf35262ff3509 / safe_rl / pg / run_agent.py  #
      # L178
      lagrangian = kwargs['lagrangian']
      objective -= lagrangian * ratio * kwargs['cost_advantage']
      objective /= (1. + lagrangian)
    return -objective.mean()


class Lagrangian(hk.Module):

  def __init__(self, initial_lagrangian):
    super(Lagrangian, self).__init__()
    self._initial_lagrangian = initial_lagrangian

  def __call__(self):
    init_lagrangian = np.log(
        max(np.exp(self.config.init_lagrangian) - 1., 1e-8))
    lagrangian = hk.get_parameter('lagrangian', running_cost.shape, jnp.float32,
                                  hk.initializers.Constant(init_lagrangian))
    return lagrangian
