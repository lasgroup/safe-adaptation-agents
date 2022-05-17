import os
from typing import Optional, List, Dict, Iterable, Tuple, Callable
from types import SimpleNamespace
from functools import partial

import cloudpickle

import numpy as np

from gym.vector import VectorEnv
from gym import Env
from gym.spaces import Space
from gym.utils import seeding

from safe_adaptation_gym import benchmark
from safe_adaptation_gym import tasks as sagt
from safe_adaptation_agents import agents, logging, driver, episodic_async_env


def evaluate(agent: agents.Agent, env: VectorEnv,
             tasks: Iterable[Tuple[str, sagt.Task]], test_driver: driver.Driver,
             trials: int):
  # Taking only the query set results as support set is less relevant for
  # evaluation.
  results = [
      test_driver.run(agent, env, tasks, False)[1] for _ in range(trials)
  ]
  return results


def evaluation_summary(runs: List[driver.IterationSummary]) -> [Dict, Dict]:
  all_runs = []
  task_vids = {}

  def return_(arr):
    return np.asarray(arr).sum(0).mean()

  for i, run in enumerate(runs):
    all_tasks = []
    for task_name, task in run.items():
      episode_return_ = return_([episode['reward'] for episode in task])
      cost_return_ = return_([episode['cost'] for episode in task])
      if i == 0:
        task_vids[task_name] = task[0].get('frames', [])
      all_tasks.append((episode_return_, cost_return_))
    all_runs.append(all_tasks)
  total_return, total_cost = np.split(np.asarray(all_runs), 2, axis=-1)
  return {
      'evaluation/return': total_return.mean(),
      'evaluation/cost_return': total_cost.mean()
  }, task_vids


def on_episode_end(episode: driver.EpisodeSummary,
                   logger: logging.TrainingLogger, train: bool):

  def return_(arr):
    return np.asarray(arr).sum(0).mean()

  episode_return = return_(episode['reward'])
  cost_return = return_(episode['cost'])
  print("\nReward return: {} -- Cost return: {}".format(episode_return,
                                                        cost_return))
  if train:
    summary = {
        'training/episode_return': episode_return,
        'training/episode_cost_return': cost_return
    }
    logger.log_summary(summary)
    logger.step += np.asarray(episode['reward']).size


class Trainer:

  def __init__(self,
               config: SimpleNamespace,
               make_env: Callable[[], Env],
               make_agent: Optional[Callable[
                   [SimpleNamespace, Space, Space, logging.TrainingLogger],
                   agents.Agent]] = None,
               agent: Optional[agents.Agent] = None,
               task_generator: Optional[benchmark.Benchmark] = None,
               start_epoch: int = 0,
               seeds: Optional[List[int]] = None):
    self.config = config
    assert not (agent is not None and make_agent is not None), (
        'agent and make_agent parameters are mutually exclusice.')
    self.make_agent = make_agent
    self.agent = agent
    self.make_env = make_env
    self.tasks_gen = task_generator
    self.epoch = start_epoch
    self.seeds = seeds
    self.logger = None
    self.state_writer = None
    self.env = None

  def __enter__(self):
    self.state_writer = logging.StateWriter(self.config.log_dir)
    self.logger = logging.TrainingLogger(self.config.log_dir)
    self.env = episodic_async_env.EpisodicAsync(self.make_env,
                                                self.config.parallel_envs)
    if self.seeds is not None:
      self.env.reset(seed=self.seeds)
    if self.make_agent is not None:
      self.agent = self.make_agent(self.config, self.env.observation_space,
                                   self.env.action_space, self.logger)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.state_writer.write(self.state)
    self.state_writer.close()
    self.logger.flush()

  def train(self, epochs: Optional[int] = None) -> [float, float]:
    config, agent, env = self.config, self.agent, self.env
    epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
    train_driver = driver.Driver(
        **config.train_driver,
        on_episode_end=partial(on_episode_end, train=True, logger=logger))
    test_driver = driver.Driver(
        **config.test_driver,
        on_episode_end=partial(on_episode_end, train=False, logger=logger),
        render_episodes=config.render_episodes)
    objective, cost = 0, 0
    for epoch in range(epoch, epochs or config.epochs):
      print('Training epoch #{}'.format(epoch))
      train_driver.run(agent, env, self.tasks(train=True), True)
      if epoch % config.eval_every == 0 and config.eval_trials:
        print('Evaluating...')
        results = evaluate(agent, env, self.tasks(train=False), test_driver,
                           config.eval_trials)
        summary, videos = evaluation_summary(results)
        objective = max(objective, summary['evaluation/return'])
        cost = min(cost, summary['evaluation/cost_return'])
        logger.log_summary(summary, epoch)
        for task_name, video in videos.items():
          logger.log_video(
              np.asarray(video).transpose([1, 0, 2, 3, 4])[:1],
              task_name + '_video',
              step=epoch)
      self.epoch = epoch
      state_writer.write(self.state)
    state_writer.close()
    logger.flush()
    return objective, cost

  def get_env_random_state(self):
    rs = []
    for state in self.env.get_attr('np_random'):
      if isinstance(state, np.random.RandomState):
        s = state.get_state()[1]
      elif isinstance(state, seeding.RandomNumberGenerator):
        s = state.get_state()['state']['state']
      else:
        raise ValueError('Cannot handle this random number generator.')
      rs.append(s)
    return rs

  def tasks(self, train=True):
    if self.tasks_gen is None:
      return [(self.config.task, benchmark.TASKS[self.config.task]())]
    if train:
      return self.tasks_gen.train_tasks
    else:
      return self.tasks_gen.test_tasks

  @classmethod
  def from_pickle(cls, config: SimpleNamespace):
    with open(os.path.join(config.log_dir, 'state.pkl'), 'rb') as f:
      make_env, env_rs, agent, epoch, task_gen = cloudpickle.load(f).values()
    print('Resuming experiment from {}'.format(config.log_dir))
    assert agent.config == config, 'Loaded different hyperparameters.'
    return cls(
        config=agent.config,
        make_env=make_env,
        task_generator=task_gen,
        start_epoch=epoch,
        seeds=env_rs,
        agent=agent)

  @property
  def state(self):
    return {
        'make_env': self.make_env,
        'env_rs': self.get_env_random_state(),
        'agent': self.agent,
        'epoch': self.epoch,
        'task_gen': self.tasks_gen
    }