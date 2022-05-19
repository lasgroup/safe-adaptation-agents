from typing import Tuple
from dataclasses import dataclass

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
    self.running_cost = MovingAverage(np.zeros((n_tasks,), dtype=np.float32))

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
    batch_size = min(transition.observation.shape[0], self.observation.shape[1])
    episode_slice = slice(self.episode_id, self.episode_id + batch_size)
    self.observation[self.task_id, episode_slice,
                     self.length] = transition.observation[:batch_size].copy()
    self.action[self.task_id, episode_slice,
                self.length] = transition.action[:batch_size].copy()
    self.reward[self.task_id, episode_slice,
                self.length] = transition.reward[:batch_size].copy()
    self.cost[self.task_id, episode_slice,
              self.length] = transition.cost[:batch_size].copy()
    if transition.last:
      self.observation[self.task_id, episode_slice, self.length +
                       1] = transition.next_observation[:batch_size].copy()
      # Following https://github.com/openai/safety-starter-agents/blob
      # /4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/run_agent.py
      # #L274 but making the computation task-wise.
      episodic_cost_average = self.cost[self.task_id].sum(1).mean()
      self.running_cost.update(episodic_cost_average, self.task_id)
      if self.episode_id + batch_size == self.observation.shape[
          1] and self.task_id + 1 == self.observation.shape[0]:
        self._full = True
      self.episode_id += batch_size
    self.length += 1

  def dump(
      self,) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns all trajectories from all tasks (with shape [N_tasks, K_episodes,
    T_steps, ...]).
    """
    o = self.observation[:, :, :self.length + 1]
    a = self.action[:, :, :self.length]
    r = self.reward[:, :, :self.length]
    c = self.cost[:, :, :self.length]
    rc = self.running_cost.value
    # Reset the on-policy running cost.
    self.length = 0
    self.episode_id = 0
    self.task_id = 0
    self._full = False
    self.running_cost.reset()
    if self.observation.shape[0] == 1:
      o, a, r, c, rc = map(lambda x: x.squeeze(0), (o, a, r, c, rc))
    return o, a, r, c, rc

  @property
  def full(self):
    return self._full


@dataclass
class MovingAverage:
  value: np.ndarray
  n: int = 0

  def update(self, new_val, at=None):
    if at is None:
      self.value = (self.value * self.n + new_val) / (self.n + 1.)
    else:
      self.value[at] = (self.value[at] * self.n + new_val) / (self.n + 1.)

  def reset(self):
    self.value = np.zeros_like(self.value)
    self.n = 0
