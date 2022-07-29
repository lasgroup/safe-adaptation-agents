from typing import Optional, Union, Dict, Any
from types import SimpleNamespace
from functools import partial
from tqdm import tqdm

import numpy as np

from gym import spaces

import jax
from jax import numpy as jnp
import haiku as hk
import jmp
import chex

from safe_adaptation_agents.agents import agent, Transition
from safe_adaptation_agents import logging, utils
from safe_adaptation_agents.agents.model_based import replay_buffer as rb
from safe_adaptation_agents.agents.model_based.carl import model


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

  def __init__(self, observation_space: spaces.Space,
               action_space: spaces.Space, config: SimpleNamespace,
               model: hk.Transformed, logger: logging.TrainingLogger,
               replay_buffer: rb.ReplayBuffer):
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
    return 0.

  def observe(self, transition: Transition, adapt: bool):
    self.training_step += sum(transition.steps)
    self.replay_buffer.add(transition)

  def train(self):
    for batch in tqdm(
        self.replay_buffer.sample(self.config.update_steps),
        leave=False,
        total=self.config.update_steps):
      self.model.state, report = self.update_model(self.model.state, batch,
                                                   next(self.rng_seq))
      for k, v in report.items():
        self.logger[k] = v.mean()
    self.logger.log_metrics(self.training_step)

  @partial(jax.jit, static_argnums=0)
  def update_model(
      self, state: utils.LearningState,
      batch: rb.etb.TrajectoryData) -> [utils.LearningState, dict, jnp.ndarray]:

    def loss(params):
      preds = self.model.apply(params, batch.o[:, :1], batch.a)
      log_probs = tuple(
          map(
              lambda d, x: d.log_prob(x),
              preds,
              (batch.o[:, 1:], batch.r, batch.c),
          ))
      log_probs = sum(log_probs)
      return -log_probs.mean()

    # vmap to get the loss of each model in the ensemble.
    loss = jax.vmap(loss)
    # Summing along the ensemble axis would give each member of ther ensemble
    # its correct gradients.
    loss, grads = jax.value_and_grad(lambda p: loss(p).mean())(state.params)
    return self.model.grad_step(grads, state), {'agent/model/loss': loss}

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    pass

  @property
  def time_to_update(self):
    return self.training_step >= self.config.prefill and \
           self.training_step % self.config.train_every == 0

  @property
  def safe(self):
    return self.config.safe
