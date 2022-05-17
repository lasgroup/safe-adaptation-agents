import pytest
import os

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def make_env(config):
  import gym
  env = gym.make('HalfCheetah-v2')
  env._max_episode_steps = config.time_limit
  return env


def test_score():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  with Trainer.from_pickle(os.path.join(
      config.log_dir, 'state.pkl')) if os.path.exists(
          os.path.join(config.log_dir, 'state.pkl')) else Trainer(
              config, agents.make, lambda: make_env(config)) as trainer:
    objective, cost = trainer.train()
  assert objective > 150.
  assert cost == 0.
