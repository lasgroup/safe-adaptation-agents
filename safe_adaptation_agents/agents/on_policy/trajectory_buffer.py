from typing import Tuple

import numpy as np

from safe_adaptation_agents.agents import Transition


class TrajectoryBuffer:

  def __init__(self,
               batch_size: int,
               max_length: int,
               observation_shape: Tuple,
               action_shape: Tuple,
               n_tasks: int = 1):
    self.length = 0
    self.episode_id = 0
    self.task_id = 0
    self._full = False
    self.observation = np.zeros(
        (
            n_tasks,
            batch_size,
            max_length + 1,
        ) + observation_shape,
        dtype=np.float32)
    self.action = np.zeros(
        (
            n_tasks,
            batch_size,
            max_length,
        ) + action_shape, dtype=np.float32)
    self.reward = np.zeros((
        n_tasks,
        batch_size,
        max_length,
    ), dtype=np.float32)
    self.cost = np.zeros((
        n_tasks,
        batch_size,
        max_length,
    ), dtype=np.float32)

  def set_task(self, task_id: int):
    """
    Sets the current task id.
    """
    self.task_id = task_id
    self.episode_id = 0
    self.length = 0

  def add(self, transition: Transition):
    """
    Adds transitions to the current running trajectory.
    """
    transition_batch_size = min(transition.observation.shape[0],
                                self.observation.shape[1])
    episode_slice = slice(self.episode_id,
                          self.episode_id + transition_batch_size)
    self.observation[
        self.task_id, episode_slice,
        self.length] = transition.observation[:transition_batch_size].copy()
    self.action[self.task_id, episode_slice,
                self.length] = transition.action[:transition_batch_size].copy()
    self.reward[self.task_id, episode_slice,
                self.length] = transition.reward[:transition_batch_size].copy()
    self.cost[self.task_id, episode_slice,
              self.length] = transition.cost[:transition_batch_size].copy()
    if transition.last:
      self.observation[
          self.task_id, episode_slice, self.length +
          1] = transition.info[0]['last_observation'][:transition_batch_size]
      if self.episode_id + transition_batch_size == self.observation.shape[
          1] and self.task_id + 1 == self.observation.shape[0]:
        self._full = True
      self.episode_id += transition_batch_size
      self.length = -1
    self.length += 1

  def dump(
      self,
  ) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns all trajectories from all tasks (with shape [N_tasks, K_episodes,
    T_steps, ...]).
    """
    self.length = 0
    self.episode_id = 0
    self.task_id = 0
    self._full = False
    o = self.observation
    a = self.action
    r = self.reward
    c = self.cost
    if self.observation.shape[0] == 1:
      o, a, r, c = map(lambda x: x.squeeze(0), (o, a, r, c))
    return o, a, r, c

  @property
  def full(self):
    return self._full
