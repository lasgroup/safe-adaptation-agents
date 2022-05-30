import os
import pytest

import numpy as np

from safe_adaptation_gym import benchmark

from safe_adaptation_agents import agents
from safe_adaptation_agents.agents.on_policy import maml_ppo_lagrangian
from safe_adaptation_agents import config as options
from safe_adaptation_agents import logging
from safe_adaptation_agents.trainer import Trainer


@pytest.fixture
def agent_env_config():
  config = options.load_config([
      '--configs', 'defaults', 'domain_randomization', '--agent',
      'maml_ppo_lagrangian', '--num_trajectories', '30', '--time_limit', '1000',
      '--vf_iters', '80', '--pi_iters', '80', '--eval_trials', '1',
      '--render_episodes', '0', '--train_driver.adaptation_steps', '1000',
      '--train_driver.query_steps', '100', '--safe', 'True',
      '--task_batch_size', '2'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)

  def make_env(config):
    import safe_adaptation_gym
    from gym.wrappers import TimeLimit
    env = safe_adaptation_gym.make(config.robot)
    env = TimeLimit(env, config.time_limit)
    return env

  env = make_env(config)
  agent: maml_ppo_lagrangian.MamlPpoLagrangian = agents.make(
      config, env.observation_space, env.action_space,
      logging.TrainingLogger(config.log_dir))
  return agent, env, config


def test_adapt(agent_env_config):
  agent, env, config = agent_env_config
  obs = np.ones((
      config.task_batch_size,
      config.num_trajectories,
      config.time_limit + 1,
  ) + env.observation_space.shape)
  rs = np.random.RandomState(666)
  act = rs.random((config.task_batch_size, config.num_trajectories,
                   config.time_limit) + env.action_space.shape)
  reward = rs.random(
      (config.task_batch_size, config.num_trajectories, config.time_limit))
  cost = rs.random(
      (config.task_batch_size, config.num_trajectories, config.time_limit))
  agent.adapt(obs, act, reward, cost)
  assert all(
      task_posterior is not None for task_posterior in agent.pi_posterior)


@pytest.mark.safe
def test_safe():

  def make_env(config):
    import safe_adaptation_gym
    env = safe_adaptation_gym.make(config.robot)
    return env

  config = options.load_config([
      '--configs', 'defaults', 'domain_randomization', '--agent',
      'maml_ppo_lagrangian', '--eval_trials', '0', '--epochs', '334', '--safe',
      'True', '--log_dir', 'results/test_ppo_lagrangian_safe',
      '--task_batch_size', '2'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  task_sampler = benchmark.make(
      'domain_randomization', batch_size=config.task_batch_size)
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config,
      make_agent=agents.make,
      make_env=lambda: make_env(config),
      task_sampler=task_sampler) as trainer:
    objective, constraint = trainer.train()
  assert np.asarray(objective.values()).mean() > 5.
  assert all(value < config.cost_limit for value in constraint.values())
