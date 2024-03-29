import os

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


def main():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  task_sampler = benchmark.make(
      config.benchmark, batch_size=config.task_batch_size, seed=config.seed)
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config,
      make_env=lambda: make_env(config),
      task_sampler=task_sampler) as trainer:
    trainer.train()


if __name__ == '__main__':
  main()
