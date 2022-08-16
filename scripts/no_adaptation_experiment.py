import os

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def make_env(config):
  import safe_adaptation_gym
  env = safe_adaptation_gym.make(
      config.robot,
      config.task,
      config={
          'obstacles_size_noise_scale': 0.,
          'robot_ctrl_range_scale': 0.
      },
      render_options=config.render_options)
  return env


def main():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    trainer.train()


if __name__ == '__main__':
  main()
