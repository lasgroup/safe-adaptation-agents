import pytest

import numpy as np

from safe_adaptation_gym import benchmark

from safe_adaptation_agents import driver
from safe_adaptation_agents import agents
from safe_adaptation_agents.agents.on_policy import maml_ppo_lagrangian
from safe_adaptation_agents import config as options
from safe_adaptation_agents import logging
from safe_adaptation_agents.episodic_async_env import EpisodicAsync


@pytest.fixture
def agent_env_config():
  config = options.load_config([
      '--configs', 'defaults', 'domain_randomization', '--agent',
      'maml_ppo_lagrangian', '--num_trajectories', '30', '--time_limit', '1000',
      '--vf_iters', '80', '--pi_iters', '80', '--eval_trials', '1',
      '--render_episodes', '0', '--train_driver.adaptation_steps', '1000',
      '--train_driver.query_steps', '100', '--safe', 'False',
      '--task_batch_size', '2', '--jit', 'True'
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

  env = EpisodicAsync(lambda: make_env(config), config.parallel_envs)
  agent: maml_ppo_lagrangian.MamlPpoLagrangian = agents.make(
      config, env.observation_space, env.action_space,
      logging.TrainingLogger(config.log_dir))
  return agent, env, config


def test_adaptation_buffer_empty(agent_env_config):
  agent, env, config = agent_env_config
  tasks_gen = benchmark.make(
      'domain_randomization', batch_size=config.task_batch_size)
  train_driver = driver.Driver(**config.train_driver)
  train_driver.run(agent, env, tasks_gen.train_tasks, True)
  assert not agent.adaptation_buffer.full


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
  running_costs = rs.random((config.task_batch_size,))
  agent.adapt(obs, act, reward, cost, running_costs)
  assert len(agent.pi_posterior) == config.task_batch_size
