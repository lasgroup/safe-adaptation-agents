import os
from collections import defaultdict
from itertools import repeat
from types import SimpleNamespace
from typing import Optional, List, Dict, Callable

import cloudpickle
import numpy as np
from gym import Env
from gym.spaces import Space
from safe_adaptation_gym import benchmark

from safe_adaptation_agents import agents, logging, driver, episodic_async_env


def evaluation_summary(runs: List[driver.IterationSummary],
                       prefix: str = 'evaluation') -> [Dict, Dict, Dict, Dict]:
  reward_returns = defaultdict(float)
  cost_returns = defaultdict(float)
  summary = defaultdict(float)
  task_vids = {}

  def return_(arr):
    return np.asarray(arr).sum(1).mean()

  def average(old_val, new_val, i):
    return (old_val * i + new_val) / (i + 1)

  for i, run in enumerate(runs):
    for task_name, task in run.items():
      reward_return = return_([episode['reward'] for episode in task])
      cost_return = return_([episode['cost'] for episode in task])
      reward_returns[task_name] = average(reward_returns[task_name],
                                          reward_return, i)
      cost_returns[task_name] = average(cost_returns[task_name], cost_return, i)
      reward_id = f'{prefix}/{task_name}/reward_return'
      cost_id = f'{prefix}/{task_name}/cost_return'
      summary[reward_id] = reward_returns[task_name]
      summary[cost_id] = cost_returns[task_name]
      if i == 0:
        if frames := task[0].get('frames', []):
          task_vids[f'{prefix}/{task_name}'] = frames
  task_average_reward_return = np.asarray(list(reward_returns.values())).mean()
  task_average_cost_retrun = np.asarray(list(cost_returns.values())).mean()
  summary[f'{prefix}/average_reward_return'] = task_average_reward_return
  summary[f'{prefix}/average_cost_return'] = task_average_cost_retrun
  reward_returns['average'] = task_average_reward_return
  cost_returns['average'] = task_average_reward_return
  return summary, reward_returns, cost_returns, task_vids


def on_episode_end(episode: driver.EpisodeSummary, task_name: str,
                   logger: logging.TrainingLogger, train: bool, adapt: bool,
                   episode_steps):

  def return_(arr):
    return np.asarray(arr).sum(0).mean()

  episode_return = return_(episode['reward'])
  cost_return = return_(episode['cost'])
  print("\ntask: {} / reward return: {:.4f} / cost return: {:.4f}".format(
      task_name, episode_return, cost_return))
  if train:
    adapt_str = 'adapt' if adapt else 'query'
    summary = {
        f'training/{adapt_str}/{task_name}/episode_return': episode_return,
        f'training/{adapt_str}/{task_name}/episode_cost_return': cost_return
    }
    logger.log_summary(summary)
    logger.step += episode_steps


def log_videos(logger: logging.TrainingLogger, videos: Dict, epoch: int):
  for task_name, video in videos.items():
    logger.log_video(
        np.asarray(video).transpose([1, 0, 2, 3, 4])[:1],
        task_name + '_video',
        step=epoch)


class Trainer:

  def __init__(self,
               config: SimpleNamespace,
               make_env: Callable[[], Env],
               make_agent: Optional[Callable[
                   [SimpleNamespace, Space, Space, logging.TrainingLogger],
                   agents.Agent]] = None,
               agent: Optional[agents.Agent] = None,
               task_sampler: Optional[benchmark.Benchmark] = None,
               start_epoch: int = 0,
               seeds: Optional[List[int]] = None):
    self.config = config
    assert not (agent is not None and make_agent is not None), (
        'agent and make_agent parameters are mutually exclusive.')
    self.make_agent = make_agent
    self.agent = agent
    self.make_env = make_env
    self.tasks_sampler = task_sampler
    self.epoch = start_epoch
    self.seeds = seeds
    self.logger = None
    self.state_writer = None
    self.env = None

  def __enter__(self):
    self.state_writer = logging.StateWriter(self.config.log_dir)
    time_limit = self.config.time_limit // self.config.action_repeat
    self.env = episodic_async_env.EpisodicAsync(self.make_env,
                                                self.config.parallel_envs,
                                                time_limit)
    _, task = next(self.tasks())
    if self.seeds is not None:
      self.env.reset(seed=self.seeds, options={'task': task})
    else:
      self.env.reset(seed=self.config.seed, options={'task': task})
    if self.make_agent is not None:
      self.logger = logging.TrainingLogger(self.config.log_dir)
      self.agent = self.make_agent(self.config, self.env.observation_space,
                                   self.env.action_space, self.logger)
    else:
      self.logger = self.agent.logger
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.state_writer.close()
    self.logger.close()

  def make_drivers(self):
    env = self.env
    config = self.config
    logger = self.logger
    train_driver = driver.Driver(
        **config.train_driver,
        time_limit=config.time_limit,
        action_repeat=config.action_repeat,
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        on_episode_end=lambda episode, task_name, adapt, steps: on_episode_end(
            episode, task_name, logger, True, adapt, steps))
    test_driver = driver.Driver(
        **config.test_driver,
        time_limit=config.time_limit,
        action_repeat=config.action_repeat,
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        on_episode_end=lambda episode, task_name, adapt, steps: on_episode_end(
            episode, task_name, logger, False, adapt, steps),
        render_episodes=config.render_episodes)
    return train_driver, test_driver

  def train(self, epochs: Optional[int] = None) -> [float, float]:
    config, agent, env = self.config, self.agent, self.env
    epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
    objective, constraint = defaultdict(float), defaultdict(float)
    train_driver, test_driver = self.make_drivers()
    for epoch in range(epoch, epochs or config.epochs):
      print('Training epoch #{}'.format(epoch))
      _, results = train_driver.run(agent, env, self.tasks(train=True), True)
      if results:
        summary, *_ = evaluation_summary([results], 'on_policy_evaluation')
        logger.log_summary(summary, epoch)
      if config.eval_trials and epoch % config.eval_every == 0:
        print('Evaluating...')
        results = self.evaluate(test_driver)
        summary, reward_returns, cost_returns, videos = evaluation_summary(
            results)
        for (_, reward), (task_name, cost) in zip(reward_returns.items(),
                                                  cost_returns.items()):
          objective[task_name] = max(objective[task_name], reward)
          constraint[task_name] = min(constraint[task_name], cost)
        logger.log_summary(summary, epoch)
        log_videos(logger, videos, epochs)
      self.epoch = epoch + 1
      state_writer.write(self.state)
    logger.flush()
    return objective, constraint

  def evaluate(self, test_driver: driver.Driver):
    # Taking only the query set results as support set is less relevant for
    # evaluation.
    results = []
    for _ in range(self.config.eval_trials):
      results.append(
          test_driver.run(self.agent, self.env, self.tasks(False), False)[1])
    return results

  def get_env_random_state(self):
    rs = [
        state.get_state()[1]
        for state in self.env.get_attr('rs')
        if state is not None
    ]
    if not rs:
      rs = [
          state.get_state()['state']['state']
          for state in self.env.get_attr('np_random')
      ]
    return rs

  def tasks(self, train=True):
    if self.tasks_sampler is None:
      return repeat((self.config.task, benchmark.TASKS[self.config.task]()),
                    self.config.task_batch_size)
    if train:
      return self.tasks_sampler.train_tasks
    else:
      return self.tasks_sampler.test_tasks

  @classmethod
  def from_pickle(cls, config: SimpleNamespace):
    with open(os.path.join(config.log_dir, 'state.pkl'), 'rb') as f:
      make_env, env_rs, agent, epoch, task_sampler = cloudpickle.load(
          f).values()
    print('Resuming experiment from: {}...'.format(config.log_dir))
    assert agent.config == config, 'Loaded different hyperparameters.'
    return cls(
        config=agent.config,
        make_env=make_env,
        task_sampler=task_sampler,
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
        'task_sampler': self.tasks_sampler
    }
