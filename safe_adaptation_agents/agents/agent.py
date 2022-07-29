import abc
from types import SimpleNamespace
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

from safe_adaptation_agents.logging import TrainingLogger


class Transition(NamedTuple):
  observation: np.ndarray
  next_observation: np.ndarray
  action: np.ndarray
  reward: np.ndarray
  cost: np.ndarray
  done: np.ndarray
  info: Tuple[Dict]

  @property
  def last(self):
    return self.done.all()

  @property
  def steps(self):
    return [info.get('steps', 1) for info in self.info]


class Agent(abc.ABC):

  def __init__(self, config: SimpleNamespace, logger: TrainingLogger):
    self.config = config
    self.logger = logger

  @abc.abstractmethod
  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    """
    Compute the next action based on the observation, update internal state
    as needed.
    """

  @abc.abstractmethod
  def observe(self, transition: Transition, adapt: bool):
    """
    Observe a transition, update internal state as needed.
    """

  @abc.abstractmethod
  def observe_task_id(self, task_id: Optional[str] = None):
    """
    Lets the agent know that a new task was sampled, possibly giving it the
    task's id.
    """

  @abc.abstractmethod
  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    """
    Adapts to new tasks based on their trajectories.
    """

  def __getstate__(self):
    """
    Define how the agent should be pickled.
    """
    state = self.__dict__.copy()
    del state['logger']
    return state

  def __setstate__(self, state):
    """
    Define how the agent should be loaded.
    """
    self.__dict__.update(state)
    self.logger = TrainingLogger(self.config.log_dir)
