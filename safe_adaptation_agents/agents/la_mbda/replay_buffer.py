from typing import Tuple, Iterator, Optional

import numpy as np
from tensorflow import data as tfd

from safe_adaptation_agents import agents
from safe_adaptation_agents import episodic_trajectory_buffer as etb


class ReplayBuffer:

  def __init__(self, observation_shape: Tuple, action_shape: Tuple,
               max_length: int, seed: int, capacity: int, batch_size: int,
               sequence_length: int, precision: int):
    self.idx = 0
    self.episode_id = 0
    self.dtype = {16: np.float16, 32: np.float32}[precision]
    self.obs_dtype = np.uint8 if len(observation_shape) == 3 else self.dtype
    self.observation = np.zeros(
      (
        capacity,
        max_length + 1,
      ) + observation_shape, dtype=self.obs_dtype)
    self.action = np.zeros(
      (
        capacity,
        max_length,
      ) + action_shape, dtype=self.dtype)
    self.reward = np.zeros((
      capacity,
      max_length,
    ), dtype=self.dtype)
    self.cost = np.zeros((
      capacity,
      max_length,
    ), dtype=self.dtype)
    self._valid_episodes = 0
    self.rs = np.random.RandomState(seed)
    example = next(
      iter(self._sample_batch(batch_size, sequence_length, capacity)))
    generator = lambda: self._sample_batch(batch_size, sequence_length)
    dataset = tfd.Dataset.from_generator(
      generator,
      *zip(*tuple((v.dtype, v.shape) for v in example)),
    )
    dataset = dataset.prefetch(10)
    self._dataset = dataset

  def add(self, transition: agents.Transition):
    """
    Adds transitions to the current running trajectory.
    """
    if self.obs_dtype == np.uint8:
      quantized = _quantize(transition.observation)
      transition = agents.Transition(quantized, transition.next_observation,
                                     transition.action, transition.reward,
                                     transition.cost, transition.done,
                                     transition.info)
    capacity, episode_length = self.reward.shape
    batch_size = min(transition.observation.shape[0], capacity)
    # Discard data if batch size overflows capacity.
    end = min(self.episode_id + batch_size, capacity)
    episode_slice = slice(self.episode_id, end)
    for data, val in zip(
        (self.observation, self.action, self.reward, self.cost),
        (transition.observation, transition.action, transition.reward,
         transition.cost)):
      data[episode_slice, self.idx] = val[:batch_size].astype(self.dtype)
    if transition.last:
      assert self.idx == episode_length - 1
      next_obs = _quantize(transition.next_observation[:batch_size])
      self.observation[episode_slice,
                       self.idx + 1] = next_obs.astype(self.dtype)
      self.episode_id = (self.episode_id + batch_size) % capacity
      self._valid_episodes = min(self._valid_episodes + 1, capacity)
    self.idx = (self.idx + 1) % episode_length

  def _sample_batch(self,
                    batch_size: int,
                    sequence_length: int,
                    valid_episodes: Optional[int] = None):
    if valid_episodes is not None:
      valid_episodes = valid_episodes
    else:
      valid_episodes = self._valid_episodes
    time_limit = self.observation.shape[1]
    assert time_limit > sequence_length
    while True:
      low = self.rs.choice(time_limit - sequence_length - 1, batch_size)
      timestep_ids = low[:, None] + np.tile(
        np.arange(sequence_length + 1),
        (batch_size, 1),
      )
      episode_ids = self.rs.choice(valid_episodes, size=batch_size)
      # Sample a sequence of length H for the actions, rewards and costs,
      # and a length of H + 1 for the observations (which is needed for
      # bootstrapping)
      a, r, c = [
        x[episode_ids[:, None], timestep_ids[:, :-1]] for x in (
          self.action,
          self.reward,
          self.cost,
        )
      ]
      o = self.observation[episode_ids[:, None], timestep_ids]
      if self.obs_dtype == np.uint8:
        o = (o / 255. - 0.5).astype(self.dtype)
      yield o, a, r, c

  def sample(self, n_batches: int) -> Iterator[etb.TrajectoryData]:
    for batch in self._dataset.take(n_batches):
      yield etb.TrajectoryData(*map(lambda x: x.numpy(), batch))


def _quantize(vec):
  return ((vec + 0.5) * 255).astype(np.uint8)
