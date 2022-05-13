import pytest

import safe_adaptation_gym

from safe_adaptation_agents import episodic_async_env

NUM_ENVS = 1


@pytest.fixture
def env():
  return safe_adaptation_gym.make('go_to_goal', 'point')


def test_reset(env):
  env = episodic_async_env.EpisodicAsync(lambda: env, vector_size=NUM_ENVS)
  observation = env.reset()
  assert len(observation) == NUM_ENVS


def test_render(env):
  pass
