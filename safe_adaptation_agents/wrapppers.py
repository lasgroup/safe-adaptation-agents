from typing import Iterable, Callable

import gym
from gym import Wrapper

# def make_env(name, episode_length, action_repeat, seed):
#   domain, task = name.rsplit('.', 1)
#   env = suite.load(domain, task,
#                    environment_kwargs={'flat_observation': True})
#   env = DeepMindBridge(env)
#   env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
#   render_kwargs = {'height': 64,
#                    'width': 64,
#                    'camera_id': 0}
#   env = ActionRepeat(env, action_repeat)
#   env = RescaleAction(env, -1.0, 1.0)
#   env = RenderedObservation(env, (64, 64), render_kwargs)
#   env.seed(seed)
#   return env


class ActionRepeat(Wrapper):

  def __init__(self, env, repeat):
    assert repeat >= 1, 'Expects at least one repeat.'
    super(ActionRepeat, self).__init__(env)
    self.repeat = repeat

  def step(self, action):
    done = False
    total_reward = 0.0
    current_step = 0
    info = {'steps': 0}
    while current_step < self.repeat and not done:
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      current_step += 1
    info['steps'] = current_step
    return obs, total_reward, done, info  # noqa


class AsyncVector(Wrapper):

  def __init__(self,
               env_fns,
               observation_space=None,
               action_space=None,
               shared_memory=True,
               copy=True,
               context=None,
               daemon=True,
               worker=None):
    env = gym.vector.AsyncVectorEnv(env_fns, observation_space, action_space,
                                    shared_memory, copy, context, daemon,
                                    worker)
    super(AsyncVector, self).__init__(env)

  def step(self, action):
    return self.env.step(action)

  def render(self, mode="human", **kwargs):
    return self.env.call('render', mode, **kwargs)
