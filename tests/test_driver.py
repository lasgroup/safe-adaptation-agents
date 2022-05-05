from typing import Optional

import pytest

import numpy as np

from gym.wrappers import TimeLimit

from safe_adaptation_gym import benchmark

from safe_adaptation_agents import train, agents
from safe_adaptation_agents.agents.agent import Transition


class DummyAgent(agents.Agent):

  def __init__(self):
    self.rs = np.random.RandomState(0)

  def __call__(self, observation: np.ndarray, traininig: bool,
               exploration: bool, *args, **kwargs) -> np.ndarray:
    return self.rs.uniform(-1., 1., (2,))

  def observe(self, transition: Transition):
    pass

  def observe_task_id(self, task_id: Optional[int] = None):
    pass


@pytest.fixture
def driver():

  def on_epoch(adaptation_episodes, test_episodes):
    # Check number of tasks
    assert len(adaptation_episodes) == len(test_episodes) == len(
        benchmark.TASKS)
    # Check the amount of episodes
    assert all(
        len(adaptation_episodes[key]) == 4
        for key in adaptation_episodes.keys())
    # Check the length of each episode
    assert all(
        len(adaptation_episodes[key][0]['reward']) == 25
        for key in adaptation_episodes.keys())
    assert all(
      len(test_episodes[key]) == 2
      for key in test_episodes.keys())
    assert all(
      len(test_episodes[key][0]['reward']) == 25
      for key in test_episodes.keys())

  def on_episode(episode_summary):
    pass

  return train.Driver(100, 50, on_episode_end=on_episode, on_iter_end=on_epoch)


@pytest.fixture(params=['no_adaptation'])
def tasks(request):

  def wrappers(gym):
    return TimeLimit(gym, 25)

  return benchmark.make(request.param, 'point', wrappers=wrappers)


def test_number_episodes(driver, tasks):
  driver.run(DummyAgent(), tasks.train_tasks, False)
