import os
import pytest

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


@pytest.mark.not_safe
def test_not_safe():

  def make_env(config):
    import gym
    from safe_adaptation_agents import wrapppers
    env = gym.make('InvertedPendulum-v2')
    old_step = env.step

    def non_terminating_step(action):
      ob, reward, terminated, info = old_step(action)
      if terminated:
        reward = 0.
      return ob, reward, False, info

    env.step = non_terminating_step
    env._max_episode_steps = config.time_limit
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    env = gym.wrappers.ClipAction(env)
    env = wrapppers.ActionRepeat(env, config.action_repeat)
    return env

  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'carl',
      '--time_limit', '150', '--eval_trials', '1', '--train_every', '1500',
      '--train_driver.adaptation_steps', '45000', '--render_episodes', '0',
      '--test_driver.query_steps', '1500', '--epochs', '100', '--safe', 'False',
      '--log_dir', 'results/test_carl_not_safe', '--action_repeat', '5'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] >= 100.
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
      '--configs', 'defaults', 'no_adaptation', '--agent', 'carl',
      '--eval_trials', '1', '--render_episodes', '0',
      '--train_driver.adaptation_steps', '30000', '--epochs', '33', '--safe',
      'True', '--log_dir', 'results/test_carl_safe'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_env=lambda: make_env(config)) as trainer:
    objective, constraint = trainer.train()
  assert objective[config.task] >= 7.
  assert constraint[config.task] < config.cost_limit
