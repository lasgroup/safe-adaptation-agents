import os
import pytest

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


@pytest.mark.not_safe
def test_not_safe():

  def make_env(config):
    import gym
    env = gym.make('HalfCheetah-v2')
    env._max_episode_steps = config.time_limit
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'ppo_lagrangian',
      '--num_trajectories', '300', '--time_limit', '150', '--vf_iters', '5',
      '--pi_iters', '5', '--eval_trials', '1',
      '--train_driver.adaptation_steps', '45000',
      '--test_driver.adaptation_steps', '1500', '--lambda_', '0.95', '--epochs',
      '50', '--safe', 'False', '--log_dir',
      'results/test_ppo_lagrangian_not_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, cost = trainer.train()
  assert objective > 185.
  assert cost == 0.


@pytest.mark.safe
def test_safe():

  def make_env(config):
    import safe_adaptation_gym
    from gym.wrappers import TimeLimit
    env = safe_adaptation_gym.make(config.robot, config.task)
    env = TimeLimit(env, config.time_limit)
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'ppo_lagrangian',
      '--num_trajectories', '30', '--time_limit', '1000', '--vf_iters', '80',
      '--pi_iters', '80', '--eval_trials', '1',
      '--train_driver.adaptation_steps', '30000', '--lambda_', '0.95',
      '--epochs', '100', '--safe', 'True', '--log_dir',
      'results/test_ppo_lagrangian_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, cost = trainer.train()
  assert objective > 14.
  assert cost < 25.
