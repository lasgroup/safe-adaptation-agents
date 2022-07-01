from typing import Optional, NamedTuple
from types import SimpleNamespace
from functools import partial

import jax.lax
import numpy as np
from gym.spaces import Space

import jax.numpy as jnp
import haiku as hk

from safe_adaptation_agents import models
from safe_adaptation_agents import nets
from safe_adaptation_agents.agents import Transition
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.agents.on_policy import safe_vpg
from safe_adaptation_agents.agents.on_policy import vpg
from safe_adaptation_agents.agents.on_policy import cpo
from safe_adaptation_agents import utils
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData


def _initial_hidden_state(batch_size: int, hidden_size: int):
  return jnp.zeros((batch_size, hidden_size))


class State(NamedTuple):
  hidden: jnp.ndarray
  prev_action: jnp.ndarray
  prev_reward: jnp.ndarray
  prev_cost: jnp.ndarray
  done: jnp.ndarray

  @property
  def vec(self):
    return jnp.concatenate(
        [self.prev_action, self.prev_reward, self.prev_cost, self.done],
        -1), self.hidden


class GruPolicy(hk.Module):

  def __init__(self,
               hidden_size: int,
               actor_config: dict,
               initialization: str = 'glorot'):
    super(GruPolicy, self).__init__()
    self._cell = hk.GRU(
        hidden_size,
        w_i_init=nets.initializer(initialization),
        w_h_init=hk.initializers.Orthogonal())
    self._head = models.Actor(**actor_config)

  def __call__(self, observation: jnp.ndarray, state: State):
    embeddings, hidden = state.vec
    ins = jnp.concatenate([observation, embeddings], -1)
    outs, hidden = self._cell(ins, hidden)
    return self._head(outs), hidden


class Rl2Cpo(safe_vpg.SafeVanillaPolicyGradients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(Rl2Cpo, self).__init__(observation_space, action_space, config,
                                 logger, actor, critic, safety_critic)
    parallel_envs = self.config.parallel_envs
    hidden_state = _initial_hidden_state(parallel_envs, self.config.hidden_size)
    self.state = State(hidden_state,
                       jnp.zeros((parallel_envs,) + action_space.shape),
                       jnp.zeros((parallel_envs,)), jnp.zeros(parallel_envs,),
                       jnp.zeros((parallel_envs,)))
    self.task_id = -1
    self.margin = 0.

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if train and self.buffer.full:
      assert adapt, (
          'Should train at the first step of adaptation (after filling up the '
          'buffer with adaptation and query data)')
      self.train(self.buffer.dump())
      self.logger.log_metrics(self.training_step)
    action, hidden = self.statefull_policy(observation,
                                           self.actor.params, self.state,
                                           next(self.rng_seq), train, adapt)
    # Update only hidden here, update the rest of the attributes
    # in 'self.observe(...)'
    self.state = State(hidden, self.state.prev_action, self.state.prev_reward,
                       self.state.prev_cost, self.state.done)
    return action

  def statefull_policy(self, observation: jnp.ndarray, params: hk.Params,
                       state: State, key: jnp.ndarray, train: bool,
                       adapt: bool) -> [jnp.ndarray, jnp.ndarray]:
    policy, hidden = self.actor.apply(params, observation, state)
    # Take the mode only on query episodes in which we evaluate the agent.
    if not adapt and not train:
      action = policy.mode()
    else:
      action = policy.sample(seed=key)
    return action, hidden

  def observe(self, transition: Transition, adapt: bool):
    self.buffer.add(transition)
    self.training_step += sum(transition.steps)
    # Keep prev_hidden after computing a forward pass in `self.policy(...)`
    hidden = self.state.hidden
    self.state = State(hidden, transition.action, transition.reward,
                       transition.cost, transition.done)

  def observe_task_id(self, task_id: Optional[str] = None):
    self.task_id = (self.task_id + 1) % self.config.task_batch_size
    self.buffer.set_task(self.task_id)
    self.state = State(*map(jnp.zeros_like, self.state))

  def train(self, trajectory_data: TrajectoryData):
    eval_ = self.adapt_critics_and_evaluate(self.critic.state,
                                            self.safety_critic.state,
                                            trajectory_data.o)
    constraint = trajectory_data.c.sum(1).mean()
    self.actor.state, actor_report = self.update_actor(
        self.actor.state, trajectory_data.o[:, :-1], trajectory_data.a,
        eval_.advantage, eval_.cost_advantage, constraint)
    self.critic.state, critic_report = self.update_critic(
        self.critic.state, trajectory_data.o[:, :-1], eval_.return_)
    if self.safe:
      self.safety_critic.state, safety_report = self.update_safety_critic(
          self.safety_critic.state, trajectory_data.o[:, :-1],
          eval_.cost_return)
      critic_report.update(safety_report)
    for k, v in {**actor_report, **critic_report}.items():
      self.logger[k] = v.mean()

  def update_actor(self, state: utils.LearningState,
                   *args) -> [utils.LearningState, dict]:
    trajectory_data, advantage, cost_advantage, constraint = args
    old_pis = self._rollout_policy(state.params, trajectory_data)
    old_pi_logprob = old_pis.log_prob(trajectory_data.a)
    c = (constraint - self.config.cost_limit)
    self.margin = max(0, self.margin + self.config.margin_lr * c)
    c += self.margin
    c /= (self.config.time_limit + 1e-8)
    g, b, old_pi_loss, old_surrogate_cost = self._cpo_grads(
        state.params, trajectory_data, advantage, cost_advantage,
        old_pi_logprob)
    # Ravel the params so every computation from now on is made on actual
    # vectors.
    p, unravel_tree = jax.flatten_util.ravel_pytree(state.params)

    def d_kl_hvp(x):

      def d_kl(p):
        pis, _ = self._rollout_policy(unravel_tree(p), trajectory_data.o)
        return pis.kl_divergence(old_pis).mean()

      return cpo.hvp(d_kl, (p,), (x,))

    direction, optim_case = cpo.step_direction(g, b, c, d_kl_hvp,
                                               self.config.target_kl,
                                               self.config.safe,
                                               self.config.damping_coeff)

    def evaluate_policy(params):
      (new_pi_loss, new_surrogate_cost), (*_, pis) = self.policy_loss(
          params, trajectory_data, advantage, cost_advantage, old_pi_logprob)
      kl_d = pis.kl_divergence(old_pis).mean()
      return new_pi_loss, new_surrogate_cost, kl_d

    new_params, info = cpo.backtracking(direction, evaluate_policy, old_pi_loss,
                                        old_surrogate_cost, optim_case, c,
                                        state.params, self.config.safe,
                                        self.config.backtrack_iters,
                                        self.config.backtrack_coeff,
                                        self.config.target_kl)
    info['agent/margin'] = self.margin
    return utils.LearningState(new_params, self.actor.opt_state), info

  @partial(jax.jit, static_argnums=0)
  def _cpo_grads(self, pi_params: hk.Params, trajectory_data: TrajectoryData,
                 advantage: jnp.ndarray, cost_advantage: jnp.ndarray,
                 old_pi_logprob: jnp.ndarray):
    # Take gradients of the objective and surrogate cost w.r.t. pi_params.
    jac_fn = jax.jacobian(self.policy_loss, has_aux=True)
    jac, outs = jac_fn(pi_params, trajectory_data, advantage, cost_advantage,
                       old_pi_logprob)
    old_pi_loss, surrogate_cost_old, pis = outs
    g, b = jac
    return g, b, old_pi_loss, surrogate_cost_old, pis

  def policy_loss(self, params: hk.Params, *args):
    trajectory_data, advantage, cost_advantage, old_pi_logprob = args
    pis = self._rollout_policy(params, trajectory_data)
    log_prob = pis.log_prob(trajectory_data.a)
    ratio = jnp.exp(log_prob - old_pi_logprob)
    surr_advantage = ratio * advantage
    objective = (
        surr_advantage + self.config.entropy_regularization * pis.entropy())
    surrogate_cost = ratio * cost_advantage
    loss = -objective.mean()
    surrogate_cost = surrogate_cost.mean()
    outs = loss, surrogate_cost, pis
    return (loss, surrogate_cost), outs

  def _rollout_policy(self, params: hk.Params, trajectory_data: TrajectoryData):
    num_tasks = trajectory_data.o.shape[0]
    # Concatente episodes to a single long episodes.
    reshape = lambda x: x.reshape(num_tasks, -1, x.shape[-1])
    trajectory_data = TrajectoryData(*map(reshape, trajectory_data))
    # We don't keep track of dones in the regular setting, since fixed-length
    # episodes are assumed, we know the timestep of the last step.
    dones = jnp.zeros_like(trajectory_data.r, bool)
    time_limit = self.config.time_limit
    dones = dones.at[::time_limit].set(True)
    # Set the time axis as the leading axis, as required by scan.
    swapaxes = lambda x: x.swapexes(0, 1)
    trajectory_data = TrajectoryData(*map(swapaxes, trajectory_data))
    dones = swapaxes(dones)
    init = State(*map(jnp.zeros_like, self.state))

    def step(carry, xs):
      o, a, r, c, d = xs
      pi, hidden = self.actor.apply(params, o, carry)
      carry = State(hidden, a, r, c, d)
      return carry, pi

    _, pis = jax.lax.scan(
        step,
        init,
        (trajectory_data.o[:-1], trajectory_data.a, trajectory_data.r,
         trajectory_data.a, dones),
    )
    return pis

  @partial(jax.jit, static_argnums=0)
  def adapt_critics_and_evaluate(self, critic_state: utils.LearningState,
                                 safety_critic_state: utils.LearningState,
                                 observation: jnp.ndarray, reward: jnp.ndarray,
                                 cost: jnp.ndarray) -> safe_vpg.Evaluation:

    # Fit the critics to the returns of the given task and return the
    # evaluation.
    def per_batch_evaluation(observation: jnp.ndarray, reward: jnp.ndarray,
                             cost: jnp.ndarray):
      reward_return = vpg.discounted_cumsum(reward, self.config.discount)
      cost_return = vpg.discounted_cumsum(cost, self.config.cost_discount)
      adapted_critic_state, _ = self.update_critic(critic_state,
                                                   observation[:, :-1],
                                                   reward_return)
      if self.safe:
        adapted_safety_critic_state, _ = self.update_safety_critic(
            safety_critic_state, observation[:, :-1], cost_return)
      else:
        adapted_safety_critic_state = critic_state
      return self.evaluate_with_safety(adapted_critic_state.params,
                                       adapted_safety_critic_state.params,
                                       observation, reward, cost)

    evaluation = jax.vmap(per_batch_evaluation)
    eval_ = evaluation(observation, reward, cost)
    num_tasks = observation.shape[0]
    # Concatente episodes to a single long episodes.
    reshape = lambda x: x.reshape(num_tasks, -1, x.shape[-1])
    eval_ = safe_vpg.Evaluation(*map(reshape, eval_))
    return eval_
