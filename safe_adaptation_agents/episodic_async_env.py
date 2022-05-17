import sys
import functools
from typing import Optional, Union, List, Callable

import atexit
import traceback

import cloudpickle

import multiprocessing as mp

from enum import Enum

import numpy as np

from gym.vector import VectorEnv
from gym import Env


class Protocol(Enum):
  ACCESS = 0
  SET = 1
  CALL = 2
  RESULT = 3
  EXCEPTION = 4
  CLOSE = 5


# Based on https://github.com/danijar/dreamerv2/blob
# /07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/common/envs.py#L522 as
# OpenAI gym's AsynVectorEnv fails to render nicely together with dm-control.
# (The main issue is with creating a dm-control
# https://github.com/openai/gym/blob/9a5db3b77a0c880ffed96ece1ab76eeff92c85e1
# /gym/vector/async_vector_env.py#L127 which loads all the rendering handler
# in the main process.)
class EpisodicAsync(VectorEnv):

  def __init__(self, ctor: Callable[[], Env], vector_size: int = 1):
    self.env_fn = cloudpickle.dumps(ctor)
    if vector_size < 1:
      self._env = ctor()
      self.observation_space = self._env.observation_space
      self.action_space = self._env.action_space
    else:
      self._env = None
      self.parents, self.processes = zip(
          *[self._make_worker() for _ in range(vector_size)])
      atexit.register(self.close)
      for process in self.processes:
        process.start()
      self.observation_space = self.get_attr('observation_space')[0]
      self.action_space = self.get_attr('action_space')[0]
      self.num_envs = len(self.parents)

  def _make_worker(self):
    parent, child = mp.Pipe()
    process = mp.Process(target=_worker, args=(self.env_fn, child))
    return parent, process

  @functools.lru_cache
  def get_attr(self, name):
    if self._env is not None:
      return getattr(self._env, name)
    for parent in self.parents:
      parent.send((Protocol.ACCESS, name))
    return self._receive()

  def set_attr(self, name, values):
    if self._env is not None:
      setattr(self._env, name, values)
    else:
      for parent, value in zip(self.parents, values):
        payload = name, value
        parent.send((Protocol.SET, payload))
    return self._receive()

  def close(self):
    if self._env is not None:
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      for parent in self.parents:
        parent.send((Protocol.CLOSE, None))
        parent.close()
    except IOError:
      # The connection was already closed.
      pass
    for process in self.processes:
      process.join()

  def _receive(self):
    payloads = []
    for parent in self.parents:
      try:
        message, payload = parent.recv()
      except ConnectionResetError:
        raise RuntimeError('Environment worker crashed.')
      # Re-raise exceptions in the main process.
      if message == Protocol.EXCEPTION:
        stacktrace = payload
        raise Exception(stacktrace)
      if message == Protocol.RESULT:
        payloads.append(payload)
      else:
        raise KeyError(f'Received message of unexpected type {message}')
    assert len(payloads) == len(self.parents)
    return payloads

  def step_async(self, actions):
    for parent, action in zip(self.parents, actions):
      payload = 'step', (action,), {}
      parent.send((Protocol.CALL, payload))

  def step_wait(self, **kwargs):
    observations, rewards, dones, infos = zip(*self._receive())
    return np.asarray(observations), np.asarray(rewards), np.asarray(
        dones, dtype=bool), infos

  def call_async(self, name, *args, **kwargs):
    if self._env is not None:
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    for parent in self.parents:
      parent.send((Protocol.CALL, payload))

  def call_wait(self, **kwargs):
    return self._receive()

  def render(self, mode="human"):
    self.call_async('render', mode)
    return np.asarray(self.call_wait())

  def reset(self,
            seed: Optional[Union[int, List[int]]] = None,
            return_info: bool = False,
            options: Optional[dict] = None):
    return self.reset_wait(seed, return_info, options)

  def reset_wait(self,
                 seed: Optional[Union[int, List[int]]] = None,
                 return_info: bool = False,
                 options: Optional[dict] = None):
    if seed is None:
      seed = [None for _ in range(self.num_envs)]
    if isinstance(seed, int):
      seed = [seed + i for i in range(self.num_envs)]
    assert len(seed) == self.num_envs
    for parent, s in zip(self.parents, seed):
      payload = 'reset', (), {
          'seed': s,
          'return_info': return_info,
          'options': options
      }
      parent.send((Protocol.CALL, payload))
    return np.asarray(self.call_wait())


def _worker(ctor, conn):
  try:
    env = cloudpickle.loads(ctor)()
    while True:
      try:
        # Only block for short times to have keyboard exceptions be raised.
        if not conn.poll(0.1):
          continue
        message, payload = conn.recv()
      except (EOFError, KeyboardInterrupt):
        break
      if message == Protocol.ACCESS:
        name = payload
        result = getattr(env, name)
        conn.send((Protocol.RESULT, result))
        continue
      if message == Protocol.SET:
        name, value = payload
        setattr(env, name, value)
        result = True
        conn.send((Protocol.RESULT, result))
        continue
      if message == Protocol.CALL:
        name, args, kwargs = payload
        result = getattr(env, name)(*args, **kwargs)
        conn.send((Protocol.RESULT, result))
        continue
      if message == Protocol.CLOSE:
        assert payload is None
        break
      raise KeyError(f'Received message of unknown type {message}')
  except Exception:
    stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
    print(f'Error in environment process: {stacktrace}')
    conn.send((Protocol.EXCEPTION, stacktrace))
  finally:
    env.close()  # noqa
    conn.close()
