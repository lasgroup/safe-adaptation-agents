import os
from itertools import repeat

import safe_adaptation_gym.benchmark
from safe_adaptation_gym import benchmark

from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def make_env(config):
  import safe_adaptation_gym
  from safe_adaptation_agents import wrapppers
  env = safe_adaptation_gym.make(
      config.robot,
      rgb_observation=config.rgb_observation,
      render_options=config.render_options,
      render_lidar_and_collision=config.render_lidar_and_collision)
  env = wrapppers.ActionRepeat(env, config.action_repeat)
  return env


class OneTaskDummySampler:

  def __init__(self, task, task_batch_size):
    self.task = task
    self.task_batch_size = task_batch_size

  @property
  def train_tasks(self):
    return repeat(self.task, self.task_batch_size)

  @property
  def test_tasks(self):
    return repeat(self.task, self.task_batch_size)


def main():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'first_phase', 'state.pkl')
  task_sampler = benchmark.make(
      'task_adaptation', batch_size=config.task_batch_size, seed=config.seed)
  with Trainer.from_pickle(config,
                           'first_phase') if os.path.exists(path) else Trainer(
                               config=config,
                               make_env=lambda: make_env(config),
                               task_sampler=task_sampler,
                               namespace='first_phase') as trainer:
    trainer.train()
  path = os.path.join(config.log_dir, 'second_phase', 'state.pkl')
  task_sampler = OneTaskDummySampler(
      next(iter(task_sampler.test_tasks)), config.task_batch_size)
  with Trainer.from_pickle(config,
                           'second_phase') if os.path.exists(path) else Trainer(
                               agent=trainer.agent,
                               config=config,
                               make_env=lambda: make_env(config),
                               task_sampler=task_sampler,
                               namespace='second_phase') as trainer:
    trainer.train(config.second_phase_epochs)


if __name__ == '__main__':
  main()
