from typing import Optional

import pytest

import numpy as np

from gym.wrappers import TimeLimit

from safe_adaptation_agents.episodic_async_env import EpisodicAsync
from safe_adaptation_agents.agents import Agent, Transition
from safe_adaptation_agents.episodic_trajectory_buffer import (
    EpisodicTrajectoryBuffer)
from safe_adaptation_agents import driver

N_TASKS = 5
EPISODE_LENGTH = 100


class DummyAgent(Agent):

  def __init__(self, buffer: EpisodicTrajectoryBuffer):
    self.rs = np.random.RandomState(0)
    self.buffer = buffer

  def __call__(self, observation: np.ndarray, train: bool, *args,
               **kwargs) -> np.ndarray:
    return self.rs.uniform(-1., 1., (2,))

  def observe(self, transition: Transition):
    self.buffer.add(transition)

  def observe_task_id(self, task_id: Optional[int] = None):
    self.buffer.set_task(task_id)

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray):
    pass


@pytest.fixture
def buffer_and_env():

  def make_env():
    import safe_adaptation_gym
    safe_adaptation_gym.make('point', 'go_to_goal')
    return env

  env = EpisodicAsync(make_env, 1, EPISODE_LENGTH)
  return EpisodicTrajectoryBuffer(
      2,
      EPISODE_LENGTH,
      env.observation_space.shape,
      env.action_space.shape,
      n_tasks=N_TASKS), env


def test_fill(buffer_and_env):
  buffer, env = buffer_and_env
  agent = DummyAgent(buffer)
  for i in range(N_TASKS):
    agent.observe_task_id(i)
    driver.interact(agent, env, EPISODE_LENGTH * 2, True)
  assert agent.buffer.full
  observation, action, reward, cost = agent.buffer.dump()
  assert observation.shape == (N_TASKS, 2,
                               EPISODE_LENGTH + 1) + env.observation_space.shape
  # Make sure that all needed episodes were filled.
  assert all((reward[:, i] != 0).all() for i in range(2))

