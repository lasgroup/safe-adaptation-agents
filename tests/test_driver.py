from typing import Optional

import pytest

import numpy as np

from gym.wrappers import TimeLimit

from safe_adaptation_gym import benchmark

from safe_adaptation_agents import episodic_async_env
from safe_adaptation_agents import agents
from safe_adaptation_agents import driver as d
from safe_adaptation_agents.agents import Transition

NUM_EPISODES = 2


class DummyAgent(agents.Agent):

  def __init__(self):
    super(DummyAgent, self).__init__(None, None)  # noqa
    self.rs = np.random.RandomState(0)

  def __call__(self, observation: np.ndarray, train: bool, *args,
               **kwargs) -> np.ndarray:
    return self.rs.uniform(-1., 1., (2,))

  def observe(self, transition: Transition):
    pass

  def observe_task_id(self, task_id: Optional[int] = None):
    pass

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray):
    # Make sure that all needed episodes were filled.
    assert all((reward[:, i] != 0).all() for i in range(NUM_EPISODES))


@pytest.fixture(params=['domain_randomization'])
def tasks(request):
  return benchmark.make(request.param, batch_size=1)


def test_run(tasks):

  def on_iter(adaptation_episodes, test_episodes):
    # Check number of tasks
    assert len(adaptation_episodes) == len(test_episodes) == 1
    # Check the amount of episodes
    assert all(
        len(adaptation_episodes[key]) == NUM_EPISODES
        for key in adaptation_episodes.keys())
    # Check the length of each episode
    assert all(
        len(adaptation_episodes[key][0]['reward']) == 100
        for key in adaptation_episodes.keys())
    assert all(len(test_episodes[key]) == 1 for key in test_episodes.keys())
    assert all(
        len(test_episodes[key][0]['reward']) == 100
        for key in test_episodes.keys())

  def make_env():
    import safe_adaptation_gym
    env = TimeLimit(safe_adaptation_gym.make('point', 'go_to_goal'), 100)
    return env

  env = episodic_async_env.EpisodicAsync(make_env, vector_size=1)
  driver = d.Driver(
      NUM_EPISODES * 100,
      100,
      100,
      env.observation_space.shape,
      env.action_space.shape,
      1,
  )
  adaptation_summary, query_summary = driver.run(DummyAgent(), env,
                                                 tasks.train_tasks, False)
  on_iter(adaptation_summary, query_summary)
