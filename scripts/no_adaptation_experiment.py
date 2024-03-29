import os

from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def make_env(config):
  import safe_adaptation_gym
  from safe_adaptation_agents import wrapppers
  env = safe_adaptation_gym.make(
      config.robot,
      config.task,
      rgb_observation=config.rgb_observation,
      config={
          'obstacles_size_noise_scale': 0.,
          'robot_ctrl_range_scale': 0.
      },
      render_options=config.render_options)
  env = wrapppers.ActionRepeat(env, config.action_repeat)
  return env


def main():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_env=lambda: make_env(config)) as trainer:
    trainer.train()


if __name__ == '__main__':
  main()
