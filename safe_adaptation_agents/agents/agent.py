import abc

from typing import Dict, NamedTuple, Optional
from types import SimpleNamespace

import numpy as np

from safe_adaptation_agents.logger import TrainingLogger


class Transition(NamedTuple):
  observation: np.ndarray
  next_observation: np.ndarray
  action: np.ndarray
  reward: float
  cost: float
  terminal: bool
  info: Dict

  @property
  def last(self):
    return self.terminal or self.info.get('TimeLimit.truncated', False)


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
  def observe(self, transition: Transition):
    """
    Observe a transition, update internal state as needed.
    """

  @abc.abstractmethod
  def observe_task_id(self, task_id: Optional[str] = None):
    """
    Lets the agent know that a new task was sampled, possibly giving it the
    task's id.
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
