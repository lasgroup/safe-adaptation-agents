from functools import partial
from types import SimpleNamespace
from typing import Tuple, Optional

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

import safe_adaptation_agents.utils as utils
from safe_adaptation_agents.agents import agent
from safe_adaptation_agents.agents.la_mbda import replay_buffer as rb
from safe_adaptation_agents.agents.la_mbda.rssm import init_state
from safe_adaptation_agents.logging import TrainingLogger

PRNGKey = jnp.ndarray
State = Tuple[jnp.ndarray, jnp.ndarray]
Action = jnp.ndarray
Observation = np.ndarray
tfd = tfp.distributions
LearningState = utils.LearningState


def compute_lambda_values(next_values: jnp.ndarray, rewards: jnp.ndarray,
                          discount: float, lambda_: float) -> jnp.ndarray:
  tds = rewards + (1. - lambda_) * discount * next_values
  tds = tds.at[-1].add(lambda_ * discount * next_values[-1])
  return utils.discounted_cumsum(tds, lambda_ * discount)


def discount_sequence(factor, length):
  d = np.cumprod(factor * np.ones((length - 1,)))
  d = np.concatenate([np.ones((1,)), d])
  return d


class LaMBDA(agent.Agent):

  def __init__(self, observation_space: gym.Space, action_space: gym.Space,
               model: hk.MultiTransformed, actor: hk.Transformed,
               critic: hk.Transformed, safety_critic: hk.Transformed,
               augmented_lagrangian: hk.Transformed,
               replay_buffer: rb.ReplayBuffer, logger: TrainingLogger,
               config: SimpleNamespace):
    super(LaMBDA, self).__init__(config, logger)
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.precision = utils.get_mixed_precision_policy(self.config.precision)
    dtype = self.precision.compute_dtype
    self.model = utils.Learner(
        model, next(self.rng_seq), config.model_opt, self.precision,
        observation_space.sample()[None, None].astype(dtype),
        action_space.sample()[None, None].astype(dtype))
    features_example = jnp.concatenate(self.init_state, -1)[None]
    self.actor = utils.Learner(actor, next(self.rng_seq), config.actor_opt,
                               self.precision, features_example.astype(dtype))
    self.critic = utils.Learner(critic, next(self.rng_seq), config.critic_opt,
                                self.precision,
                                features_example[None].astype(dtype))
    self.safety_critic = utils.Learner(safety_critic, next(self.rng_seq),
                                       config.critic_opt, self.precision,
                                       features_example[None].astype(dtype))
    self.lagrangian = utils.Learner(augmented_lagrangian, next(self.rng_seq),
                                    {}, self.precision, 1., 0.)
    self.replay_buffer = replay_buffer
    self.state = (self.init_state,
                  jnp.zeros(action_space.shape, self.precision.compute_dtype))
    self.training_step = 0
    self._prefill_policy = lambda x: action_space.sample()

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if self.training_step <= self.config.prefill and train:
      return self._prefill_policy(observation)
    if self.time_to_update and train:
      self.train()
    action, current_state = self.policy(self.state[0], self.state[1],
                                        observation,
                                        self.model.params, self.actor.params,
                                        next(self.rng_seq), train)
    self.state = (current_state, action)
    return np.clip(action.astype(np.float32), -1.0, 1.0)

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    pass

  @partial(jax.jit, static_argnums=(0, 7))
  def policy(self,
             prev_state: State,
             prev_action: Action,
             observation: Observation,
             model_params: hk.Params,
             actor_params: hk.Params,
             key: PRNGKey,
             training=True) -> Tuple[jnp.ndarray, State]:
    filter_, *_ = self.model.apply
    key, subkey = jax.random.split(key)
    observation = observation.astype(self.precision.compute_dtype)
    _, current_state = filter_(model_params, subkey, prev_state, prev_action,
                               observation)
    features = jnp.concatenate(current_state, -1)[None]
    policy = self.actor.apply(actor_params, features)
    key, subkey = jax.random.split(key)
    action = policy.sample(seed=subkey) if training else policy.mode()
    return action, current_state

  def observe(self, transition: agent.Transition, adapt: bool):
    self.training_step += sum(transition.steps)
    self.replay_buffer.add(transition)
    if transition.last:
      self.state = (self.init_state, jnp.zeros_like(self.state[-1]))

  @property
  def init_state(self):
    state = init_state(1, self.config.rssm['stochastic_size'],
                       self.config.rssm['deterministic_size'],
                       self.precision.compute_dtype)
    return jax.tree_map(lambda x: x.squeeze(0), state)

  def train(self):
    for batch in tqdm(
        self.replay_buffer.sample(self.config.update_steps),
        leave=False,
        total=self.config.update_steps):
      self.model.state, model_report, features = self.update_model(
          batch, self.model.state, next(self.rng_seq))
      self.actor.state, actor_report, aux = self.update_actor(
          self.actor.state,
          features,
          self.model.params,
          self.critic.params,
          self.safety_critic.params,
          self.lagrangian.params,
          next(self.rng_seq),
      )
      trajectoris, reward_lambdas, cost_lambdas, cond = aux
      self.critic.state, critic_report = self.update_critic(
          self.critic.state, features, reward_lambdas)
      if self.safe:
        self.lagrangian.state, lagrangian_report = self.update_lagrangian(
            self.lagrangian.state, cond)
        self.safety_critic.state, s_critic_report = self.update_safety_critic(
            self.safety_critic.state, features, cost_lambdas)
        critic_report.update({**s_critic_report, **lagrangian_report})
      reports = {**model_report, **actor_report, **critic_report}
      # Average training metrics across update steps.
      for k, v in reports.items():
        self.logger[k] = v.mean() / self.config.update_steps
    self.logger.log_metrics(self.training_step)

  @partial(jax.jit, static_argnums=0)
  def update_model(self, batch: rb.etb.TrajectoryData, state: LearningState,
                   key: PRNGKey) -> Tuple[LearningState, dict, jnp.ndarray]:
    params, opt_state = state

    def loss(params: hk.Params) -> Tuple[float, dict]:
      _, _, infer, _ = self.model.apply
      outputs_infer = infer(params, key, batch.o[:, 1:], batch.a)

      (prior, posterior), features, decoded, reward, cost = outputs_infer
      kl = jnp.maximum(
          tfd.kl_divergence(posterior, prior).mean(), self.config.free_kl)
      log_p_obs = decoded.log_prob(batch.o[:, 1:]).mean()
      log_p_rews = reward.log_prob(batch.r).mean()
      log_p_cost = cost.log_prob(batch.c).mean()
      loss_ = self.config.kl_scale * kl - log_p_obs - log_p_rews - log_p_cost
      return loss_, {
          'agent/model/kl': kl,
          'agent/model/post_entropy': posterior.entropy().mean(),
          'agent/model/prior_entropy': prior.entropy().mean(),
          'agent/model/log_p_observation': -log_p_obs,
          'agent/model/log_p_reward': -log_p_rews,
          'agent/model/log_p_cost': -cost,
          'features': features
      }

    grads, report = jax.grad(loss, has_aux=True)(params)
    new_state = self.model.grad_step(grads, state)
    report['agent/model/grads'] = optax.global_norm(grads)
    return new_state, report, report.pop('features')

  @partial(jax.jit, static_argnums=0)
  def update_actor(
      self, state: LearningState, features: jnp.ndarray,
      model_params: hk.Params, critic_params: hk.Params,
      safety_critic_params: hk.Params, lagrangian_params: hk.Params,
      key: PRNGKey) -> [LearningState, dict, Tuple[jnp.ndarray, ...]]:
    _, generate_experience, *_ = self.model.apply
    generate_experience = partial(generate_experience, model_params)
    reward_critic = partial(self.critic.apply, critic_params)
    cost_critic = partial(self.safety_critic.apply, safety_critic_params)
    lagrangian = partial(self.lagrangian.apply, lagrangian_params)
    policy = self.actor
    # Flatten the features so that trajectory sampling starts from every
    # state in the model-inferred features.
    flattened_features = features.reshape((-1, features.shape[-1]))

    def loss(params: hk.Params):
      trajectories, reward, cost = generate_experience(key, flattened_features,
                                                       policy, params)
      reward_values = reward_critic(trajectories[:, 1:]).mean()
      reward_lambdas = compute_lambda_values(reward_values, reward.mean(),
                                             self.config.discount,
                                             self.config.lambda_)
      discount = discount_sequence(self.config.discount,
                                   self.config.imag_horizon - 1)
      loss_ = (-reward_lambdas * discount).mean()
      if self.safe:
        cost_values = cost_critic(trajectories[:, 1:]).mean()
        cost_lambdas = compute_lambda_values(cost_values, cost.mean(),
                                             self.config.safety_discount,
                                             self.config.lambda_)
        penalty, cond = lagrangian(cost_lambdas.mean(), self.config.cost_limit)
        loss_ += penalty
      else:
        cost_lambdas = jnp.zeros_like(reward_lambdas)
        cond = np.array([0.])
      return loss_, (trajectories, reward_lambdas, cost_lambdas, cond)

    (loss_, aux), grads = jax.value_and_grad(loss, has_aux=True)(state.params)
    new_state = self.actor.grad_step(grads, state)
    return new_state, {
        'agent/actor/loss': loss_,
        'agent/actor/grads': optax.global_norm(grads)
    }, aux

  @partial(jax.jit, static_argnums=0)
  def update_critic(self, state: LearningState, features: jnp.ndarray,
                    lambda_values: jnp.ndarray) -> Tuple[LearningState, dict]:
    params, opt_state = state

    def loss(params: hk.Params) -> float:
      values = self.critic.apply(params, features[:, :-1])
      discount = discount_sequence(self.config.discount,
                                   self.config.imag_horizon - 1)
      return -(values.log_prob(lambda_values) * discount).mean()

    (loss_, grads) = jax.value_and_grad(loss)(params)
    new_state = self.critic.grad_step(grads, state)
    return new_state, {
        'agent/critic/loss': loss_,
        'agent/critic/grads': optax.global_norm(grads)
    }

  def update_safety_critic(
      self, state: LearningState, features: jnp.ndarray,
      lambda_values: jnp.ndarray) -> Tuple[LearningState, dict]:
    params, opt_state = state

    def loss(params: hk.Params) -> float:
      values = self.safety_critic.apply(params, features[:, :-1])
      discount = discount_sequence(self.config.discount,
                                   self.config.imag_horizon - 1)
      return -(values.log_prob(lambda_values) * discount).mean()

    (loss_, grads) = jax.value_and_grad(loss)(params)
    new_state = self.safety_critic.grad_step(grads, state)
    return new_state, {
        'agent/safety_critic/loss': loss_,
        'agent/safety_critic/grads': optax.global_norm(grads)
    }

  @partial(jax.jit, static_argnums=0)
  def update_lagrangian(self, state: LearningState,
                        cond: jnp.ndarray) -> [LearningState, dict]:
    lagrangian, penalty_multiplier = state.params
    new_lagrangian = jnn.relu(cond)
    new_penalty = jnp.clip(
        penalty_multiplier * (self.config.penalty_power_factor + 1.),
        penalty_multiplier,
        1.,
    )
    new_params = {'lagrangian': new_lagrangian, 'penalty': new_penalty}
    report = {'agent/lagrangian': new_lagrangian, 'agent/penatly': new_penalty}
    return LearningState(new_params, state.opt_state), report

  @property
  def time_to_update(self):
    return self.training_step > self.config.prefill and \
           self.training_step % self.config.train_every == 0

  @property
  def learning_states(self):
    return self.model.state, self.actor.state, self.critic.state

  @learning_states.setter
  def learning_states(self, states):
    self.model.state, self.actor.state, self.critic.state = states

  @property
  def safe(self):
    return self.config.safe
