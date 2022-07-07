from typing import Tuple, Iterator

import numpy as np

from tensorflow import data as tfd
import tensorflow_datasets as tfds

from safe_adaptation_agents import agents
from safe_adaptation_agents import episodic_trajectory_buffer as etb


class ReplayBuffer:

  def __init__(self, capacity: int, max_length: int, observation_shape: Tuple,
               action_shape: Tuple, seed: int, batch_size: int,
               sequence_length: int):
    self.idx = 0
    self.episode_id = 0
    self.obs_dtype = np.uint8 if len(observation_shape) == 3 else np.float32
    self.observation = np.zeros(
        (
            capacity,
            max_length + 1,
        ) + observation_shape, dtype=self.obs_dtype)
    self.action = np.zeros(
        (
            capacity,
            max_length,
        ) + action_shape, dtype=np.float32)
    self.reward = np.zeros((
        capacity,
        max_length,
    ), dtype=np.float32)
    self.cost = np.zeros((
        capacity,
        max_length,
    ), dtype=np.float32)
    self._valid_episodes = 0
    self.rs = np.random.RandomState(seed)
    example = next(iter(self._sample_batch(batch_size, sequence_length)))
    dataset = tfd.Dataset.from_generator(
        self._sample_batch(batch_size, sequence_length),
        *zip(*tuple((v.dtype, v.shape) for v in example)))
    dataset = dataset.prefetch(10)
    self._dataset = dataset

  def add(self, transition: agents.Transition):
    """
    Adds transitions to the current running trajectory.
    """
    if self.obs_dtype == np.uint8:
      quantized = _quantize(transition.observation)
      next_quantized = _quantize(transition.next_observation)
      transition = agents.Transition(quantized, next_quantized,
                                     transition.action, transition.reward,
                                     transition.cost, transition.done,
                                     transition.info)
    capacity, episode_length = self.reward.shape[1:2]
    batch_size = min(transition.observation.shape[0], capacity)
    slice = np.arange(self.episode_id, self.episode_id + batch_size)
    for data, val in zip(
        (self.observation, self.action, self.reward, self.cost),
        transition.observation, transition.action, transition.reward,
        transition.cost):
      # Put new values in data, wrap if needed.
      np.put(data[:, self.idx], slice, val[:batch_size].copy(), mode='wrap')
    if transition.last:
      assert self.idx == episode_length - 1
      np.put(self.observation[slice, self.idx + 1],
             transition.next_observation[:batch_size].copy())
      self.episode_id = (self.episode_id + batch_size) % capacity
      self._valid_episodes = min(self._valid_episodes + 1, capacity)
    self.idx = (self.idx + 1) % episode_length

  def _sample_batch(self, batch_size: int, sequence_length: int):
    while True:
      low = self.rs.randint(0, self.observation.shape[1] - sequence_length,
                            batch_size)
      timestep_ids = low[:, None] + np.tile(
          np.arange(sequence_length),
          (batch_size, 1),
      )
      episode_ids = self.rs.permutation(self._valid_episodes)[:batch_size]
      o, a, r, c = [
          x[episode_ids[:, None], timestep_ids] for x in (
              self.observation,
              self.action,
              self.reward,
              self.cost,
          )
      ]
      if self.obs_dtype == np.uint8:
        o = (o / 255. - 0.5).astype(np.float32)
      a = a[:, :-1]
      r = r[:, :-1]
      c = c[:, :-1]
      yield o, a, r, c

  def sample(self, n_batches: int) -> Iterator[etb.TrajectoryData]:
    for batch in tfds.as_numpy(self._dataset.take(n_batches)):
      yield etb.TrajectoryData(*batch)


def _quantize(vec):
  return ((vec + 0.5) * 255).astype(np.uint8)
