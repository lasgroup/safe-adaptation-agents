import abc

from typing import Dict, NamedTuple, Optional
from types import SimpleNamespace

import numpy as np


class Transition(NamedTuple):
  observation: np.ndarray
  next_observation: np.ndarray
  action: np.ndarray
  reward: float
  terminal: bool
  info: Dict


class Agent(abc.ABC):

  @abc.abstractmethod
  def __call__(self, observation: np.ndarray, traininig: bool,
               exploration: bool, *args, **kwargs) -> np.ndarray:
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
