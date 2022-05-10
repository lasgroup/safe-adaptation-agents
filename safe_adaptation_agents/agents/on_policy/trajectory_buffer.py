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
    self.terminal = np.zeros((
        n_tasks,
        batch_size,
        max_length,
    ), dtype=bool)

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
    self.observation[self.task_id, self.episode_id,
                     self.length] = transition.observation
    self.action[self.task_id, self.episode_id, self.length] = transition.action
    self.reward[self.task_id, self.episode_id, self.length] = transition.reward
    self.cost[self.task_id, self.episode_id, self.length] = transition.cost
    self.terminal[self.task_id, self.episode_id,
                  self.length] = transition.terminal
    if transition.last:
      self.observation[self.task_id, self.episode_id,
                       self.length + 1] = transition.next_observation
      self.terminal[self.task_id, self.episode_id, self.length + 1:] = True
      self.episode_id += 1
      self.length = -1
    self.length += 1

  def dump(
      self) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns all trajectories from all tasks (with shape [N_tasks, K_episodes,
    T_steps, ...]).
    """
    self.length = 0
    self.episode_id = 0
    self.task_id = 0
    if self.observation.shape[0] == 1:
      self.observation.squeeze(0)
      self.action.squeeze(0)
      self.reward.squeeze(0)
      self.cost.squeeze(0)
      self.terminal.squeeze(0)
    return self.observation, self.action, self.reward, self.cost, self.terminal
