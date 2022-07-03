from typing import Optional, NamedTuple, Sequence
from types import SimpleNamespace
from functools import partial

import jax.lax
import numpy as np
import optax
from gym.spaces import Space

import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk

from tensorflow_probability.substrates import jax as tfp

from safe_adaptation_agents import models
from safe_adaptation_agents import nets
from safe_adaptation_agents.agents import Transition
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.agents.on_policy import safe_vpg
from safe_adaptation_agents.agents.on_policy import vpg
from safe_adaptation_agents.agents.on_policy import cpo
from safe_adaptation_agents import utils
from safe_adaptation_agents import episodic_trajectory_buffer as etb
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData

tfd = tfp.distributions


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


class GRUPolicy(hk.Module):

  def __init__(self,
               output_size: Sequence[int],
               hidden_size: int,
               actor_config: dict,
               initialization: str = 'glorot'):
    super(GRUPolicy, self).__init__()
    self._cell = hk.GRU(
        hidden_size,
        w_i_init=nets.initializer(initialization),
        w_h_init=hk.initializers.Orthogonal())
    self._head = models.Actor(output_size, **actor_config)

  def __call__(self, observation: jnp.ndarray, state: State):
    embeddings, hidden = state.vec
    ins = jnp.concatenate([observation, embeddings], -1)
    ins = jnn.elu(hk.Linear(self._cell.hidden_size)(ins))
    outs, hidden = self._cell(ins, hidden)
    return self._head(outs), hidden


class RL2CPO(safe_vpg.SafeVanillaPolicyGradients):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    self.config = config
    self.logger = logger
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.training_step = 0
    num_steps = self.config.time_limit // self.config.action_repeat
    self.buffer = etb.EpisodicTrajectoryBuffer(self.config.num_trajectories,
                                               num_steps,
                                               observation_space.shape,
                                               action_space.shape,
                                               self.config.task_batch_size)
    parallel_envs = self.config.parallel_envs
    hidden_state = _initial_hidden_state(parallel_envs, self.config.hidden_size)
    zeros = jnp.zeros((parallel_envs, 1))
    self.state = State(hidden_state,
                       jnp.zeros((parallel_envs,) + action_space.shape), zeros,
                       zeros, zeros)
    self.actor = utils.Learner(
        actor, next(self.rng_seq), config.actor_opt,
        utils.get_mixed_precision_policy(config.precision),
        np.tile(observation_space.sample(), (parallel_envs, 1)), self.state)
    self.critic = utils.Learner(
        critic, next(self.rng_seq), config.critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
    self.safety_critic = utils.Learner(
        safety_critic, next(self.rng_seq), config.critic_opt,
        utils.get_mixed_precision_policy(config.precision),
        observation_space.sample())
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
    action, hidden = self.stateful_policy(observation,
                                          self.actor.params, self.state,
                                          next(self.rng_seq), train, adapt)
    # Update only hidden here, update the rest of the attributes
    # in 'self.observe(...)'
    self.state = State(hidden, self.state.prev_action, self.state.prev_reward,
                       self.state.prev_cost, self.state.done)
    return action

  @partial(jax.jit, static_argnums=(0, 5, 6))
  def stateful_policy(self, observation: jnp.ndarray, params: hk.Params,
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
    self.state = State(hidden, transition.action, transition.reward[:, None],
                       transition.cost[:, None], transition.done[:, None])

  def observe_task_id(self, task_id: Optional[str] = None):
    self.task_id = (self.task_id + 1) % self.config.task_batch_size
    self.buffer.set_task(self.task_id)
    self.state = State(*map(jnp.zeros_like, self.state))

  def train(self, trajectory_data: TrajectoryData):
    eval_ = self.adapt_critics_and_evaluate(self.critic.state,
                                            self.safety_critic.state,
                                            trajectory_data.o,
                                            trajectory_data.r,
                                            trajectory_data.c)
    if self.safe:
      # Assuming that the mean cost return across different MDPs bounded.
      constraint = trajectory_data.c.sum(2).mean()
      c = (constraint - self.config.cost_limit)
      self.margin = max(0., self.margin + self.config.margin_lr * c)
      c += self.margin
      c /= (self.config.time_limit + 1e-8)
    else:
      c = 0.
    self.actor.state, info = self.update_actor(self.actor.state,
                                               trajectory_data, eval_.advantage,
                                               eval_.cost_advantage, c)
    info['agent/margin'] = self.margin
    for k, v in info.items():
      self.logger[k] = float(v)

  @partial(jax.jit, static_argnums=0)
  def update_actor(self, state: utils.LearningState,
                   *args) -> [utils.LearningState, dict]:
    trajectory_data, advantage, cost_advantage, c = args
    old_pis = self._rollout_policy(state.params, trajectory_data)
    old_pi_logprob = old_pis.log_prob(trajectory_data.a)
    g, b, old_pi_loss, old_surrogate_cost = self._cpo_grads(
        state.params, trajectory_data, advantage, cost_advantage,
        old_pi_logprob)
    # Ravel the params so every computation from now on is made on actual
    # vectors.
    p, unravel_tree = jax.flatten_util.ravel_pytree(state.params)

    def d_kl_hvp(x):

      def d_kl(p):
        pis = self._rollout_policy(unravel_tree(p), trajectory_data)
        kl = tfd.kl_divergence(pis.distribution, old_pis.distribution).mean()
        return kl

      return cpo.hvp(d_kl, (p,), (x,))

    direction, optim_case = cpo.step_direction(g, b, c, d_kl_hvp,
                                               self.config.target_kl,
                                               self.config.safe,
                                               self.config.damping_coeff)

    def evaluate_policy(params):
      (new_pi_loss, new_surrogate_cost), (*_, pis) = self.policy_loss(
          params, trajectory_data, advantage, cost_advantage, old_pi_logprob)
      kl_d = pis.distribution.kl_divergence(old_pis.distribution).mean()
      return new_pi_loss, new_surrogate_cost, kl_d

    new_params, info = cpo.backtracking(direction, evaluate_policy, old_pi_loss,
                                        old_surrogate_cost, optim_case, c,
                                        state.params, self.config.safe,
                                        self.config.backtrack_iters,
                                        self.config.backtrack_coeff,
                                        self.config.target_kl)
    info['agent/actor/objective_grad'] = optax.global_norm(g)
    info['agent/actor/surrogate_cost_grad'] = optax.global_norm(b)
    return utils.LearningState(new_params, self.actor.opt_state), info

  def _cpo_grads(self, pi_params: hk.Params, trajectory_data: TrajectoryData,
                 advantage: jnp.ndarray, cost_advantage: jnp.ndarray,
                 old_pi_logprob: jnp.ndarray):
    # Take gradients of the objective and surrogate cost w.r.t. pi_params.
    jac_fn = jax.jacobian(self.policy_loss, has_aux=True)
    jac, outs = jac_fn(pi_params, trajectory_data, advantage, cost_advantage,
                       old_pi_logprob)
    old_pi_loss, surrogate_cost_old, pis = outs
    g, b = jac
    return g, b, old_pi_loss, surrogate_cost_old

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
    num_episodes = trajectory_data.o.shape[1]
    # Drop last observations. Standardize dimensions.
    trajectory_data = TrajectoryData(
        trajectory_data.o[:, :, :-1],
        trajectory_data.a,
        trajectory_data.r[..., None],
        trajectory_data.c[..., None],
    )
    dones = jnp.zeros_like(trajectory_data.r)
    # We don't keep track of dones in the regular setting, since fixed-length
    # episodes are assumed, we know the timestep of the last step.
    dones = dones.at[:, :, -1].set(1.)
    # Concatente episodes to have "meta-episode" per task.
    reshape = lambda x: x.reshape(num_tasks, -1, x.shape[-1])
    trajectory_data = TrajectoryData(*map(reshape, trajectory_data))
    dones = reshape(dones)
    # Set the time axis as the leading axis, as required by scan.
    swapaxes = lambda x: x.swapaxes(0, 1)
    trajectory_data = TrajectoryData(*map(swapaxes, trajectory_data))
    dones = swapaxes(dones)
    make_init = lambda x: jnp.zeros((num_tasks,) + x.shape[1:])
    init = State(*map(make_init, self.state))

    def step(carry, xs):
      o, a, r, c, d = xs
      pi, hidden = self.actor.apply(params, o, carry)
      carry = State(hidden, a, r, c, d)
      return carry, pi

    _, pis = jax.lax.scan(
        step,
        init,
        (trajectory_data.o, trajectory_data.a, trajectory_data.r,
         trajectory_data.c, dones),
    )
    time_limit = self.config.time_limit
    # Reshape back to the input dimension and shape
    pis = tfd.BatchReshape(pis, (num_tasks, num_episodes, time_limit))
    return pis

  @partial(jax.jit, static_argnums=0)
  def adapt_critics_and_evaluate(self, critic_state: utils.LearningState,
                                 safety_critic_state: utils.LearningState,
                                 observation: jnp.ndarray, reward: jnp.ndarray,
                                 cost: jnp.ndarray) -> safe_vpg.Evaluation:

    # Fit the critics to the returns of the given task and return the
    # evaluation.
    def per_task_evaluation(observation: jnp.ndarray, reward: jnp.ndarray,
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

    evaluation = jax.vmap(per_task_evaluation)
    eval_ = evaluation(observation, reward, cost)
    return eval_
