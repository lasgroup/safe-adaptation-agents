import os

from gym.wrappers import TimeLimit

from safe_adaptation_gym import benchmark

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def make_env(config):
  import safe_adaptation_gym
  env = safe_adaptation_gym.make(
      config.robot,
      render_options=config.render_options,
      render_lidar_and_collision=config.render_lidar_and_collision)
  env = TimeLimit(env, config.time_limit)
  return env


def main():
  config = options.load_config()
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
    trainer.train()


if __name__ == '__main__':
  main()
