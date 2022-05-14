from typing import Optional

import pytest

import numpy as np

from gym.wrappers import TimeLimit

import safe_adaptation_gym

from safe_adaptation_agents.agents import Agent, Transition
from safe_adaptation_agents.agents.on_policy.trajectory_buffer import (
    TrajectoryBuffer)
from safe_adaptation_agents import driver

N_TASKS = 5
EPISODE_LENGTH = 100


class DummyAgent(Agent):

  def __init__(self, buffer: TrajectoryBuffer):
    self.rs = np.random.RandomState(0)
    self.buffer = buffer

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    return self.rs.uniform(-1., 1., (2,))

  def observe(self, transition: Transition):
    self.buffer.add(transition)

  def observe_task_id(self, task_id: Optional[int] = None):
    self.buffer.set_task(task_id)


@pytest.fixture
def buffer_and_env():
  env = TimeLimit(
      safe_adaptation_gym.make('go_to_goal', 'point'), EPISODE_LENGTH)
  return TrajectoryBuffer(
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
    train.interact(agent, env, EPISODE_LENGTH * 2, True, True)
  observation, action, reward, cost, terminal = agent.buffer.dump()
  assert observation.shape == (N_TASKS, 2,
                               EPISODE_LENGTH + 1) + env.observation_space.shape
  # Make sure that all needed episodes were filled.
  assert all((reward[:, i] != 0).all() for i in range(2))
