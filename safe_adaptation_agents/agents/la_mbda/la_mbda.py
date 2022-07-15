from functools import partial
from types import SimpleNamespace
from typing import Tuple, Optional, NamedTuple

import gym
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

import safe_adaptation_agents.utils as utils
from safe_adaptation_agents.agents import agent
from safe_adaptation_agents.agents.la_mbda import replay_buffer as rb
from safe_adaptation_agents.agents.la_mbda import swag
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


# First vmap the time horizon axis.
compute_lambda_values = jax.vmap(compute_lambda_values, (0, 0, None, None))
# Then merge the posterior samples axis and batch axis.
compute_lambda_values = hk.BatchApply(compute_lambda_values)


class UpdateActorResult(NamedTuple):
  optimistic_sample: jnp.ndarray
  pessimistic_sample: jnp.ndarray
  reward_lambdas: jnp.ndarray
  cost_lambdas: jnp.ndarray
  cond: jnp.ndarray


class LaMBDA(agent.Agent):

  def __init__(self, observation_space: gym.Space, action_space: gym.Space,
               logger: TrainingLogger, config: SimpleNamespace,
               model: hk.MultiTransformed, actor: hk.Transformed,
               critic: hk.Transformed, safety_critic: hk.Transformed,
               augmented_lagrangian: hk.Transformed,
               replay_buffer: rb.ReplayBuffer):
    super(LaMBDA, self).__init__(config, logger)
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.precision = utils.get_mixed_precision_policy(self.config.precision)
    dtype = self.precision.compute_dtype
    self.model = swag.SWAG(model, next(self.rng_seq), config.model_opt,
                           self.precision,
                           observation_space.sample()[None, None].astype(dtype),
                           action_space.sample()[None, None].astype(dtype),
                           **config.swag)
    features_example = jnp.concatenate(self.init_state, -1)
    self.actor = utils.Learner(actor, next(self.rng_seq), config.actor_opt,
                               self.precision, features_example.astype(dtype))
    self.critic = utils.Learner(critic, next(self.rng_seq), config.critic_opt,
                                self.precision,
                                features_example[None].astype(dtype))
    self.safety_critic = utils.Learner(safety_critic, next(self.rng_seq),
                                       config.safety_critic_opt, self.precision,
                                       features_example[None].astype(dtype))
    self.lagrangian = utils.Learner(augmented_lagrangian, next(self.rng_seq),
                                    {}, self.precision, 1., 0.)
    self.replay_buffer = replay_buffer
    self.state = (self.init_state,
                  jnp.zeros((self.config.parallel_envs,) + action_space.shape,
                            self.precision.compute_dtype))
    self.training_step = 0
    self._prefill_policy = lambda x: jax.device_put(
        jax.random.uniform(
            next(self.rng_seq), (x.shape[0],) + action_space.shape,
            minval=-1.,
            maxval=1.),
        device=jax.devices("cpu")[0])

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if self.training_step < self.config.prefill and train:
      return self._prefill_policy(observation)
    if self.time_to_update and train:
      self.train()
    action, current_state = self.policy(self.state[0], self.state[1],
                                        observation,
                                        self.model.params, self.actor.params,
                                        next(self.rng_seq), train)
    self.state = (current_state, action)
    return action

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
    batch_shape = observation.shape[0]
    prev_state = tuple(map(lambda x: x[:batch_shape], prev_state))
    observation = observation.astype(self.precision.compute_dtype)
    observation = rb.preprocess(observation)
    _, current_state = filter_(model_params, subkey, prev_state, prev_action,
                               observation)
    features = jnp.concatenate(current_state, -1)
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
    state = init_state(self.config.parallel_envs,
                       self.config.rssm['stochastic_size'],
                       self.config.rssm['deterministic_size'],
                       self.precision.compute_dtype)
    return state

  def train(self):
    print("Updating world model and actor-critic.")
    # Use the critic params for the previous iteration to update to actor.
    critic_params = self.critic.params
    safety_critic_params = self.safety_critic.params
    for batch in tqdm(
        self.replay_buffer.sample(self.config.update_steps),
        leave=False,
        total=self.config.update_steps):
      self.model.state, model_report, features = self.update_model(
          self.model.state, batch, next(self.rng_seq))
      if self.model.warm:
        model_posteriors = self.model.posterior_samples(
            self.config.posterior_samples, next(self.rng_seq))
      else:
        model_posteriors = utils.pytrees_stack([self.model.params])
      self.actor.state, actor_report, aux = self.update_actor(
          self.actor.state, features, model_posteriors, critic_params,
          safety_critic_params, self.lagrangian.params, next(self.rng_seq))
      self.critic.state, critic_report = self.update_critic(
          self.critic.state, aux.optimistic_sample, aux.reward_lambdas)
      if self.safe:
        self.lagrangian.state, lagrangian_report = self.update_lagrangian(
            self.lagrangian.state, aux.cond)
        self.safety_critic.state, s_critic_report = self.update_safety_critic(
            self.safety_critic.state, aux.pessimistic_sample, aux.cost_lambdas)
        critic_report.update({**s_critic_report, **lagrangian_report})
      reports = {**model_report, **actor_report, **critic_report}
      # Average training metrics across update steps.
      for k, v in reports.items():
        self.logger[k] = v.mean()
    if self.config.evaluate_model:
      self.evaluate_model()
    self.logger.log_metrics(self.training_step)

  @partial(jax.jit, static_argnums=0)
  def update_model(
      self, state: swag.SWAGLearningState, batch: rb.etb.TrajectoryData,
      key: PRNGKey) -> Tuple[swag.SWAGLearningState, dict, jnp.ndarray]:
    config = self.config

    def loss(params: hk.Params) -> Tuple[float, dict]:
      _, _, infer, _ = self.model.apply
      outputs_infer = infer(params, key, batch.o[:, 1:], batch.a)
      (prior, posterior), features, decoded, reward, cost = outputs_infer
      kl_loss, kl = balanced_kl_loss(posterior, prior, config.free_kl,
                                     config.kl_mix)
      log_p_obs = decoded.log_prob(batch.o[:, 1:]).astype(jnp.float32).mean()
      log_p_rews = reward.log_prob(batch.r).mean()
      # Generally costs can be greater than 1. (especially if we use
      # ActionRepeat). However, since the cost is modeled as an indicator,
      # we transform it back to a continuous variable.
      log_p_cost = cost.log_prob(batch.c > 0.)
      log_p_cost = jnp.where(batch.c > 0., log_p_cost * self.config.cost_weight,
                             log_p_cost).mean()
      loss_ = config.kl_scale * kl_loss - log_p_obs - log_p_rews - log_p_cost
      posterior_entropy = posterior.entropy().astype(jnp.float32).mean()
      prior_entropy = prior.entropy().astype(jnp.float32).mean()
      return loss_, {
          'agent/model/kl': kl,
          'agent/model/post_entropy': posterior_entropy,
          'agent/model/prior_entropy': prior_entropy,
          'agent/model/log_p_observation': -log_p_obs,
          'agent/model/log_p_reward': -log_p_rews,
          'agent/model/log_p_cost': -log_p_cost,
          'features': features
      }

    grads, report = jax.grad(loss, has_aux=True)(state.params)
    new_state = self.model.grad_step(grads, state)
    report['agent/model/grads'] = optax.global_norm(grads)
    return new_state, report, report.pop('features')

  @partial(jax.jit, static_argnums=0)
  def update_actor(self, state: LearningState, features: jnp.ndarray,
                   model_params: hk.Params, critic_params: hk.Params,
                   safety_critic_params: hk.Params,
                   lagrangian_params: hk.Params,
                   key: PRNGKey) -> [LearningState, dict, UpdateActorResult]:
    # Prepare `generate_trajectory` to sample with different posteriors.
    # Note that we do not vmap the key, so that under equal posterior samples
    # we get equal trajectories.
    generate_trajectories = jax.vmap(self._generate_trajectories,
                                     (0, None, None, None))
    generate_trajectories = partial(generate_trajectories, model_params)
    # Prepare all other modules.
    reward_critic = partial(self.critic.apply, critic_params)
    cost_critic = partial(self.safety_critic.apply, safety_critic_params)
    lagrangian = partial(self.lagrangian.apply, lagrangian_params)
    # Flatten the features so that trajectory sampling starts from every
    # state in the model-inferred features.
    flattened_features = features.reshape((-1, features.shape[-1]))
    discount = discount_sequence(self.config.discount,
                                 self.config.sample_horizon - 1)

    def loss(params: hk.Params):
      trajectories, reward, cost = generate_trajectories(
          params, flattened_features, key)
      reward_values = reward_critic(trajectories[:, :, 1:]).mean()
      reward_lambdas = compute_lambda_values(reward_values, reward[:, :, :-1],
                                             self.config.discount,
                                             self.config.lambda_)
      optimistic_sample, reward_lambdas = estimate_upper_bound(
          trajectories, reward_lambdas)
      loss_ = (-reward_lambdas * discount).mean()
      if self.safe:
        cost_values = cost_critic(trajectories[:, :, 1:]).mean()
        cost_lambdas = compute_lambda_values(cost_values, cost[:, :, :-1],
                                             self.config.cost_discount,
                                             self.config.lambda_)
        pessimistic_sample, cost_lambdas = estimate_upper_bound(
            trajectories, cost_lambdas)
        penalty, cond = lagrangian(cost_lambdas.mean(), self.config.cost_limit)
        loss_ += penalty
      else:
        cost_lambdas = jnp.zeros_like(reward_lambdas)
        cond = np.array([0.])
        pessimistic_sample = optimistic_sample
      return loss_, UpdateActorResult(optimistic_sample, pessimistic_sample,
                                      reward_lambdas, cost_lambdas, cond)

    (loss_, aux), grads = jax.value_and_grad(loss, has_aux=True)(state.params)
    new_state = self.actor.grad_step(grads, state)
    return new_state, {
        'agent/actor/loss': loss_,
        'agent/actor/grads': optax.global_norm(grads),
        'agent/actor/pred_constraint': aux.cost_lambdas.mean()
    }, aux

  @partial(jax.jit, static_argnums=0)
  def update_critic(self, state: LearningState, features: jnp.ndarray,
                    lambda_values: jnp.ndarray) -> Tuple[LearningState, dict]:

    def loss(params: hk.Params) -> float:
      values = self.critic.apply(params, features[:, :-1])
      return -(values.log_prob(lambda_values)).mean()

    (loss_, grads) = jax.value_and_grad(loss)(state.params)
    new_state = self.critic.grad_step(grads, state)
    return new_state, {
        'agent/critic/loss': loss_,
        'agent/critic/grads': optax.global_norm(grads)
    }

  @partial(jax.jit, static_argnums=0)
  def update_safety_critic(
      self, state: LearningState, features: jnp.ndarray,
      lambda_values: jnp.ndarray) -> Tuple[LearningState, dict]:

    def loss(params: hk.Params) -> float:
      values = self.safety_critic.apply(params, features[:, :-1])
      return -(values.log_prob(lambda_values)).mean()

    (loss_, grads) = jax.value_and_grad(loss)(state.params)
    new_state = self.safety_critic.grad_step(grads, state)
    return new_state, {
        'agent/safety_critic/loss': loss_,
        'agent/safety_critic/grads': optax.global_norm(grads)
    }

  @partial(jax.jit, static_argnums=0)
  def update_lagrangian(self, state: LearningState,
                        cond: jnp.ndarray) -> [LearningState, dict]:
    params = state.params['augmented_lagrangian'].values()
    lagrangian, penalty_multiplier = params
    new_lagrangian = jnn.relu(cond)
    new_penalty = jnp.clip(
        penalty_multiplier * (self.config.penalty_power_factor + 1.),
        penalty_multiplier,
        1.,
    )
    new_params = dict(augmented_lagrangian={
        'lagrangian': new_lagrangian,
        'penalty': new_penalty
    })
    finite = jmp.all_finite(new_params)
    new_params = jmp.select_tree(finite, new_params, state.params)
    report = {'agent/lagrangian': new_lagrangian, 'agent/penatly': new_penalty}
    return LearningState(new_params, state.opt_state), report

  def _generate_trajectories(
      self, model_params: hk.Params, policy_params: hk.Params,
      features: jnp.ndarray,
      key: PRNGKey) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, generate_trajectory, *_ = self.model.apply
    generate_trajectory = partial(generate_trajectory, model_params)
    policy = self.actor
    trajectories, reward, cost = generate_trajectory(key, features, policy,
                                                     policy_params)
    # The cost decoder predicts an indicator ({0, 1}) but the total cost
    # is summed if `action_repeat` > 1
    return trajectories, reward.mean(), cost.mode() * self.config.action_repeat

  @property
  def time_to_update(self):
    return self.training_step >= self.config.prefill and \
           self.training_step % self.config.train_every == 0

  @property
  def safe(self):
    return self.config.safe

  def evaluate_model(self):
    if self.replay_buffer.empty:
      return
    batch = next(self.replay_buffer.sample(1))
    eval_video = evaluate_model(batch.o, batch.a, next(self.rng_seq),
                                self.model, self.model.params, self.precision)
    self.logger.log_video(
        eval_video, step=self.training_step, name='agent/model/error')


# https://github.com/danijar/dreamerv2/blob/259e3faa0e01099533e29b0efafdf240adeda4b5/common/nets.py#L130
def balanced_kl_loss(posterior: tfd.Distribution, prior: tfd.Distribution,
                     free_nats: float,
                     mix: float) -> [jnp.ndarray, jnp.ndarray]:
  sg = lambda x: jax.tree_map(jax.lax.stop_gradient, x)
  lhs = tfd.kl_divergence(posterior, sg(prior)).mean()
  rhs = tfd.kl_divergence(sg(posterior), prior).mean()
  return (1. - mix) * jnp.maximum(lhs, free_nats) + mix * jnp.maximum(
      rhs, free_nats), lhs


def estimate_upper_bound(trajectories: jnp.ndarray,
                         values: jnp.ndarray) -> [jnp.ndarray, jnp.ndarray]:
  ids = jnp.argmax(values.mean(2), axis=0)
  value_upper_bound = jnp.take_along_axis(
      values,
      ids[None, :, None],
      0,
  ).squeeze(0)
  trajectories_upper_bound = jnp.take_along_axis(
      trajectories,
      ids[None, :, None, None],
      0,
  ).squeeze(0)
  return trajectories_upper_bound, value_upper_bound


@partial(jax.jit, static_argnums=(3, 5))
def evaluate_model(observations, actions, key, model, model_params, precision):
  length = min(len(observations) + 1, 50)
  observations, actions = jax.tree_map(
      lambda x: x.astype(precision.compute_dtype), (observations, actions))
  _, generate_sequence, infer, decode = model.apply
  key, subkey = jax.random.split(key)
  _, features, infered_decoded, *_ = infer(model_params, subkey,
                                           observations[:1, 1:length + 1],
                                           actions[:1, :length])
  conditioning_length = length // 5
  key, subkey = jax.random.split(key)
  generated, *_ = generate_sequence(
      model_params,
      subkey,
      features[:, conditioning_length],
      None,
      None,
      actions=actions[:1, conditioning_length:])
  key, subkey = jax.random.split(key)
  generated_decoded = decode(model_params, subkey, generated)
  prediction = generated_decoded.mean()
  out = jnp.abs(observations[:1, conditioning_length + 1:] - prediction)
  out = ((out + 0.5) * 255).astype(jnp.uint8)
  return out
