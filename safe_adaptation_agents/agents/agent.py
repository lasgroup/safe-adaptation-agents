import abc

from typing import Dict, NamedTuple, Optional, Tuple
from types import SimpleNamespace

import numpy as np

from safe_adaptation_agents.logging import TrainingLogger


class Transition(NamedTuple):
  observation: np.ndarray
  next_observation: np.ndarray
  action: np.ndarray
  reward: np.ndarray
  cost: np.ndarray
  last: np.ndarray
  info: Tuple[Dict]

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
