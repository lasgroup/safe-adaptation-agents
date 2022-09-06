import os
from collections import defaultdict
from itertools import repeat, zip_longest
from types import SimpleNamespace
from typing import Optional, List, Dict, Callable, Tuple

import cloudpickle
import numpy as np
from gym import Env
from safe_adaptation_gym import benchmark

from safe_adaptation_agents import agents, logging, driver, episodic_async_env


def evaluation_summary(runs: List[Tuple[driver.IterationSummary,
                                        driver.IterationSummary]],
                       prefix: str = 'evaluation') -> [Dict, Dict, Dict, Dict]:
  reward_returns = defaultdict(float)
  cost_returns = defaultdict(float)
  summary = defaultdict(float)
  task_count = defaultdict(int)
  task_vids = {}

  def return_(arr):
    return np.asarray(arr).sum(1).mean()

  def rate(arr):
    return np.asarray(arr).mean()

  def average(old_val, new_val, i):
    return (old_val * i + new_val) / (i + 1)

  objective, cost, cost_rate, feasibility = 0., 0., 0., 0.
  total_count = 0
  for i, run in enumerate(runs):
    for (_, adaptation_task), (task_name,
                               query_task) in zip_longest(*run, ('', [])):
      task_bound = query_task[0]['info'][0][0]['bound']
      reward_return = return_([episode['reward'] for episode in query_task])
      cost_return = return_([episode['cost'] for episode in query_task])
      cost_return -= task_bound
      cost_rate += rate([episode['cost'] for episode in query_task] +
                        [episode['cost'] for episode in adaptation_task])
      count = task_count[task_name]
      reward_returns[task_name] = average(reward_returns[task_name],
                                          reward_return, count)
      cost_returns[task_name] = average(cost_returns[task_name], cost_return,
                                        count)
      reward_id = f'{prefix}/{task_name}/reward_return'
      cost_id = f'{prefix}/{task_name}/cost_return'
      summary[reward_id] = reward_returns[task_name]
      summary[cost_id] = cost_returns[task_name]
      task_count[task_name] += 1
      if i == 0:
        if frames := query_task[0].get('frames', []):
          task_vids[f'{prefix}/{task_name}'] = frames
      objective += reward_return
      cost += cost_return
      feasibility += (cost_return <= 0.)
      total_count += 1
  if total_count > 0:
    summary[f'{prefix}/average_reward_return'] = objective / total_count
    summary[f'{prefix}/average_cost_return'] = cost / total_count
    summary[f'{prefix}/average_feasibilty'] = feasibility / total_count
    summary[f'{prefix}/cost_rate'] = cost_rate / total_count
    reward_returns['average'] = objective / total_count
    cost_returns['average'] = cost / total_count
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
    logger.step += episode_steps
    logger.log_summary(summary)


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
               agent: Optional[agents.Agent] = None,
               task_sampler: Optional[benchmark.Benchmark] = None,
               start_epoch: int = 0,
               seeds: Optional[List[int]] = None):
    self.config = config
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
    agents.set_precision(self.config.agent, self.config.precision)
    if self.seeds is not None:
      self.env.reset(seed=self.seeds, options={'task': task})
    else:
      self.env.reset(seed=self.config.seed, options={'task': task})
    if self.agent is None:
      self.logger = logging.TrainingLogger(self.config.log_dir)
      self.agent = agents.make(self.config, self.env.observation_space,
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
      adaptation_results, query_results = train_driver.run(
          agent, env, self.tasks(train=True), True)
      if query_results:
        summary, *_ = evaluation_summary([(adaptation_results, query_results)],
                                         'on_policy_evaluation')
        logger.log_summary(summary, epoch + 1)
      if config.eval_trials and (epoch + 1) % config.eval_every == 0:
        print('Evaluating...')
        results = self.evaluate(test_driver)
        summary, reward_returns, cost_returns, videos = evaluation_summary(
            results)
        for (_, reward), (task_name, cost) in zip(reward_returns.items(),
                                                  cost_returns.items()):
          objective[task_name] = max(objective[task_name], reward)
          constraint[task_name] = min(constraint[task_name], cost)
        logger.log_summary(summary, epoch + 1)
        log_videos(logger, videos, epochs + 1)
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
          test_driver.run(self.agent, self.env, self.tasks(False), False))
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
