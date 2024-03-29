import os

import gym
import numpy as np
import pytest
from gym.envs.mujoco.mujoco_env import MujocoEnv

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


class DummyBenchmark:

  def __init__(self, batch_size: int):
    self._batch_size = batch_size
    self.rs = np.random.RandomState(666)

  def _sample_tasks(self):
    return self.rs.choice((
        -1.,
        1.,
    ), (self._batch_size,))

  @property
  def train_tasks(self):
    return self.test_tasks

  @property
  def test_tasks(self):
    tasks = self._sample_tasks()
    for task in tasks:
      yield str(task), task


# Acknowledgment: https://github.com/jonasrothfuss/ProMP/blob
# https://github.com/jonasrothfuss/ProMP/blob
# /93ae339e23dfc6e1133f9538f2c7cc0ccee89d19/meta_policy_search/envs
# /mujoco_envs/half_cheetah_rand_direc.py
class HalfCheetahRandDirecEnv(MujocoEnv):

  def __init__(self):
    self.goal_direction = 1.
    MujocoEnv.__init__(self, 'half_cheetah.xml', 5)

  def set_task(self, task):
    """
        Args:
            task: task of the meta-learning environment
        """
    self.goal_direction = task

  def get_task(self):
    """
        Returns:
            task: task of the meta-learning environment
        """
    return self.goal_direction

  def step(self, action):
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    ob = self._get_obs()
    reward_ctrl = -0.5 * 0.1 * np.square(action).sum()
    reward_run = self.goal_direction * (xposafter - xposbefore) / self.dt
    reward = reward_ctrl + reward_run
    done = False
    return ob, reward, done, dict(
        reward_run=reward_run, reward_ctrl=reward_ctrl)

  def _get_obs(self):
    return np.concatenate([
        self.sim.data.qpos.flat[1:],
        self.sim.data.qvel.flat,
    ])

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        low=-.1, high=.1, size=self.model.nq)
    qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    self.set_state(qpos, qvel)
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5

  def reset(
      self,
      *,
      seed=None,
      return_info=False,
      options=None,
  ):
    super().reset(seed=seed)
    self.sim.reset()
    ob = self.reset_model()
    if options is not None and 'task' in options:
      self.set_task(options['task'])
    if not return_info:
      return ob
    else:
      return ob, {}


@pytest.mark.not_safe
def test_cheetah():

  def make_env(config):
    env = HalfCheetahRandDirecEnv()
    env = gym.wrappers.TimeLimit(env, config.time_limit)
    env = gym.wrappers.RescaleAction(env, -10.0, 10.0)
    env = gym.wrappers.ClipAction(env)
    return env

  config = options.load_config([
      '--configs', 'defaults', '--agent', 'rl2_cpo', '--eval_trials', '0',
      '--epochs', '1000', '--render_episodes', '0', '--log_dir',
      'results/test_rl2_cpo_half_cheetah', '--task_batch_size', '16', '--safe',
      'False', '--actor.layers', '[128, 128]', '--critic.layers', '[64, 64]',
      '--time_limit', '200', '--num_trajectories', '10', '--train_driver',
      '{\'adaptation_steps\': 2000, \'query_steps\': 2000}', '--test_driver',
      '{\'adaptation_steps\': 2000, \'query_steps\': 2000}', '--jit', 'True',
      '--target_kl', '0.1', '--damping_coeff', '1e-5', '--vf_iters', '10',
      '--episodes_per_task', '2'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  task_sampler = DummyBenchmark(config.task_batch_size)
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config,
      make_env=lambda: make_env(config),
      task_sampler=task_sampler) as trainer:
    objective, constraint = trainer.train()
  assert objective['average'] > 150.0
