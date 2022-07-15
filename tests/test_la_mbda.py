import os
import pytest

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def test_run():

  def make_env(config):
    import safe_adaptation_gym
    from safe_adaptation_agents import wrapppers
    env = safe_adaptation_gym.make(
        config.robot,
        config.task,
        rgb_observation=True,
        config={
            'obstacles_size_noise_scale': 0.,
            'robot_ctrl_range_scale': 0.
        })
    env = wrapppers.ActionRepeat(env, config.action_repeat)
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'la_mbda',
      '--eval_trials', '1', '--render_episodes', '0',
      '--train_driver.adaptation_steps', '200', '--test_driver.query_steps',
      '200', '--epochs', '1', '--safe', 'True', '--log_dir',
      'results/test_lambda_safe', '--time_limit', '100', '--replay_buffer',
      '{capacity: 10, batch_size: 5, sequence_length: 8}', '--prefill', '100',
      '--update_steps', '2', '--train_every', '100', '--parallel_envs', '1',
      '--swag',
      '{start_averaging: 1, average_period: 1, max_num_models: 1, decay: 0.8, scale: 1., learning_rate_factor: 5.}',
      '--action_repeat', '1', '--posterior_samples', '2'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] >= 15.
  assert constraint[config.task] == 0.


@pytest.mark.not_safe
def test_not_safe():

  def make_env(config):
    import safe_adaptation_gym
    from safe_adaptation_agents import wrapppers
    env = safe_adaptation_gym.make(
        config.robot,
        config.task,
        rgb_observation=True,
        config={
            'obstacles_size_noise_scale': 0.,
            'robot_ctrl_range_scale': 0.
        })
    env = wrapppers.ActionRepeat(env, config.action_repeat)
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'la_mbda',
      '--eval_trials', '1', '--render_episodes', '0',
      '--train_driver.adaptation_steps', '30000', '--epochs', '33', '--safe',
      'False', '--log_dir', 'results/test_lambda_not_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] >= 15.
  assert constraint[config.task] == 0.


@pytest.mark.safe
def test_safe():

  def make_env(config):
    import safe_adaptation_gym
    from safe_adaptation_agents import wrapppers
    env = safe_adaptation_gym.make(
        config.robot,
        config.task,
        rgb_observation=True,
        config={
            'obstacles_size_noise_scale': 0.,
            'robot_ctrl_range_scale': 0.
        })
    env = wrapppers.ActionRepeat(env, config.action_repeat)
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'la_mbda',
      '--eval_trials', '1', '--render_episodes', '0',
      '--train_driver.adaptation_steps', '30000', '--epochs', '33', '--safe',
      'True', '--log_dir', 'results/test_lambda_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] >= 7.
  assert constraint[config.task] < config.cost_limit
