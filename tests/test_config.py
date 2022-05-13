import pytest

from safe_adaptation_agents import config as c


@pytest.fixture
def config():
  return c.load_config([])


def test_config(config):
  assert config.log_dir == 'results'
  assert config.seed == 0
  assert config.time_limit == 1000
  assert config.train_driver == {
      'adaptation_steps': 5000,
      'query_steps': 2000,
      'iters': 10
  }
  assert config.test_driver == {
      'adaptation_steps': 5000,
      'query_steps': 2000,
      'iters': 5
  }
  assert config.action_repeat == 1
  assert config.render_episodes == 0
  assert config.initialization == 'glorot'
  assert config.jit
