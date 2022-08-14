import copy
from typing import Optional, Union, Dict, Any
from types import SimpleNamespace
from functools import partial
from tqdm import tqdm

import numpy as np

from gym import spaces
from gym.wrappers.normalize import RunningMeanStd

import jax
from jax import numpy as jnp
import haiku as hk
import jmp
import chex

from safe_adaptation_agents.agents import agent, Transition
from safe_adaptation_agents import logging, utils
from safe_adaptation_agents.agents.model_based import replay_buffer as rb
from safe_adaptation_agents.agents.model_based.carl import model
from safe_adaptation_agents.agents.model_based.carl import cem


class EnsembleLearner(utils.Learner):

  def __init__(self, model: Union[hk.Transformed, hk.MultiTransformed,
                                  chex.ArrayTree], seed: utils.PRNGKey,
               optimizer_config: Dict, precision: jmp.Policy, num_models: int,
               *input_example: Any):
    super(EnsembleLearner, self).__init__(model, seed, optimizer_config,
                                          precision, *input_example)
    if isinstance(model, (hk.Transformed, hk.MultiTransformed)):
      ensemble_init = jax.vmap(self.model.init,
                               (0,) + (None,) * len(input_example))
      seeds = jax.random.split(seed, num_models)
      self.params = ensemble_init(jnp.asarray(seeds), *input_example)
      self.opt_state = self.optimizer.init(self.params)


class CARL(agent.Agent):

  def __init__(self, observation_space: spaces.Box, action_space: spaces.Box,
               config: SimpleNamespace, logger: logging.TrainingLogger,
               model: hk.Transformed, replay_buffer: rb.ReplayBuffer):
    super(CARL, self).__init__(config, logger)
    self.rng_seq = hk.PRNGSequence(config.seed)
    self.model = EnsembleLearner(
        model,
        next(self.rng_seq),
        config.model_opt,
        utils.get_mixed_precision_policy(config.precision),
        config.num_models,
        observation_space.sample()[None],
        action_space.sample()[None],
    )
    self.replay_buffer = replay_buffer
    self._prefill_policy = lambda x: jax.device_put(
        jax.random.uniform(
            next(self.rng_seq), (x.shape[0],) + action_space.shape,
            minval=-1.,
            maxval=1.),
        device=jax.devices("cpu")[0])
    self.action_space = action_space
    self.adapted_model_params = None
    self.obs_moments_tracker = RunningMeanStd(shape=observation_space.shape)

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if self.training_step < self.config.prefill and train:
      return self._prefill_policy(observation)
    if self.time_to_update and train:
      self.train()
    params = self.model.params if adapt else self.adapted_model_params
    normalized_obs = self.normalize(observation)
    action = self.policy(normalized_obs, params, next(self.rng_seq))
    return action

  @partial(jax.jit, static_argnums=0)
  @partial(jax.vmap, in_axes=[None, 0, None, None])
  def policy(self, observation: jnp.ndarray, model_params: hk.Params,
             key: jnp.ndarray) -> jnp.ndarray:

    def rollout_action_sequence(model_params, action_sequence):
      action_sequence = jnp.clip(action_sequence, self.action_space.low,
                                 self.action_space.high)
      model_ = lambda o, a: self.model.apply(model_params, o, a)
      tiled_obs = jnp.tile(observation, [self.config.num_particles, 1])
      _, reward, cost = model.sample_trajectories(model_, tiled_obs,
                                                  action_sequence, key)
      return reward.mean(), cost.mean()

    # vmap over the ensemble axis
    rollout_action_sequence = jax.vmap(rollout_action_sequence, [0, None])

    def objective(sample: jnp.ndarray) -> jnp.ndarray:
      reward, cost = rollout_action_sequence(model_params, sample)
      # Average sum along the time horizon, average along the ensemble axis
      objective_, constraint = [x.sum(-1).mean(0) for x in (reward, cost)]
      constraint *= self.config.action_repeat
      constraint -= self.config.cost_limit
      if self.safe:
        return objective_ - jnp.maximum(self.config.lambda_ * constraint, 0.)
      else:
        return objective_

    horizon = self.config.plan_horizon
    low, high = self.action_space.low, self.action_space.high
    initial_guess = jnp.zeros((horizon, np.prod(self.action_space.shape)))
    key, subkey = jax.random.split(key)
    optimized_action_sequence = cem.solve(
        objective,
        initial_guess,
        subkey,
        self.config.num_particles,
        self.config.num_iters,
        self.config.num_elite,
        initial_stddev=(high - low))
    return optimized_action_sequence[0]

  def observe(self, transition: Transition, adapt: bool):
    self.training_step += sum(transition.steps)
    self.obs_moments_tracker.update(transition.observation)
    normalized_transition = Transition(
        self.normalize(transition.observation),
        self.normalize(transition.next_observation), transition.action,
        transition.reward, transition.cost, transition.done, transition.info)
    self.replay_buffer.add(normalized_transition)

  def normalize(self, observation: np.ndarray):
    diff = observation - self.obs_moments_tracker.mean
    return diff / np.sqrt(self.obs_moments_tracker.var + 1e-8)

  def train(self):
    print('Updating model')
    for batch in self.replay_buffer.sample(self.config.update_steps):
      self.model.state, report = self.update_model(self.model.state, batch)
      for k, v in report.items():
        self.logger[k] = v.mean()
    self.logger.log_metrics(self.training_step)

  @partial(jax.jit, static_argnums=0)
  def update_model(
      self, state: utils.LearningState,
      batch: rb.etb.TrajectoryData) -> [utils.LearningState, dict, jnp.ndarray]:

    def loss(params):
      preds = self.model.apply(params, batch.o[:, :-1], batch.a)
      log_probs = tuple(
          map(
              lambda d, x: d.log_prob(x),
              preds,
              (batch.o[:, 1:], batch.r, batch.c > 0.),
          ))
      log_probs = sum(log_probs)
      return -log_probs.mean()

    # vmap to get the loss of each model in the ensemble.
    loss = jax.vmap(loss)
    # Taking mean along the ensemble.
    loss, grads = jax.value_and_grad(lambda p: loss(p).mean())(state.params)
    return self.model.grad_step(grads, state), {'agent/model/loss': loss}

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    reshape = lambda x: x.reshape(-1, x.shape[-1])
    o = reshape(observation[:, :, :-1])
    next_o = reshape(observation[:, :, 1:])
    a, r, c = reshape(action), reshape(reward), reshape(cost)
    state = copy.deepcopy(self.model.state)
    # TODO (yarden): find a better parameter
    for _ in range(max(self.config.update_steps // 10, 1)):
      ids = self.replay_buffer.rs.choice(
          a.shape[0], self.config.replay_buffer['batch_size'])
      batch = rb.etb.TrajectoryData(
          np.stack([o[ids], next_o[ids]], 1), a[ids], r[ids], c[ids])
      state, report = self.update_model(self.model.state, batch)
    self.adapted_model_params = state.params

  @property
  def time_to_update(self):
    return self.training_step >= self.config.prefill and \
           self.training_step % self.config.train_every == 0

  @property
  def safe(self):
    return self.config.safe
