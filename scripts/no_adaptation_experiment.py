import os
from functools import partial
import concurrent.futures
from itertools import repeat
from typing import List, Dict

import multiprocessing as mp

import cloudpickle

import numpy as np

from gym import Env
from gym.wrappers import TimeLimit

import safe_adaptation_gym

from safe_adaptation_agents import agents, logging, train
from safe_adaptation_agents import config as options


def run_one(agent_bytes: bytes, env: Env, task_name: str, seed: np.ndarray,
            driver: train.Driver):
  # Haiku functions are not pickleable, so first encode to bytes with
  # cloudpickle and then decode back to an agent instance.
  agent = cloudpickle.loads(agent_bytes)
  env.seed(seed)
  episodes, _ = driver.run(agent, [(task_name, env)], False)
  return episodes


def evaluate(agent: agents.Agent, env: Env, task_name: str,
             test_driver: train.Driver, trials: int,
             seed_sequence: np.random.SeedSequence):

  # Running evaluation in parallel as the agent should not be stateful during
  # evaluation. Even if there's a bug the agent is not returned from the
  # different processes running it, so anything that could have been saved
  # during evaluation is left out.
  agent_bytes = cloudpickle.dumps(agent)
  with concurrent.futures.ProcessPoolExecutor(
      mp_context=mp.get_context('forkserver')) as executor:
    results = [
        result for result in executor.map(
            run_one, repeat(agent_bytes), repeat(env), repeat(task_name),
            list(seed_sequence.generate_state(trials)), repeat(test_driver))
    ]
  return results


def evaluation_summary(runs: List[train.IterationSummary]) -> [Dict, Dict]:
  all_runs = []
  task_vids = {}
  for i, run in enumerate(runs):
    all_tasks = []
    for task_name, task in run.items():
      return_ = np.asarray([sum(episode['reward']) for episode in task]).mean()
      cost_return_ = np.asarray([
          sum(list(map(lambda info: info['cost'], episode['info'])))
          for episode in task
      ]).mean()
      if i == 0:
        task_vids['task_name'] = [episode.get('frames', []) for episode in task]
      all_tasks.append((return_, cost_return_))
    all_runs.append(all_tasks)
  total_return, total_cost = np.split(np.asarray(all_runs), 2, axis=-1)
  return {
      'evaluation/return': total_return.mean(),
      'evaluation/cost_return': total_cost.mean()
  }, task_vids


def on_episode_end(episode: train.EpisodeSummary,
                   logger: logging.TrainingLogger, train: bool):
  episode_return = sum(episode['reward'])
  summary = {'training/episode_return': episode_return}
  sum_costs = sum(list(map(lambda info: info['cost'], episode['info'])))
  summary['training/episode_cost_return'] = sum_costs
  print("\nReward return: {} -- Cost return: {}".format(episode_return,
                                                        sum_costs))
  if train:
    logger.log_summary(summary)
    logger.step += len(episode['reward'])


def resume_experiment(log_dir):
  with open(os.path.join(log_dir, 'state.pkl'), 'rb') as f:
    env, agent, epoch = cloudpickle.load(f).values()
  return env, agent, agent.logger, agent.config, epoch


def main():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  seed_sequence = np.random.SeedSequence(config.seed)
  if os.path.exists(os.path.join(config.log_dir, 'state.pkl')):
    env, agent, logger, config, epoch = resume_experiment(config.log_dir)
  else:
    env = safe_adaptation_gym.make(config.task, config.robot)
    env = TimeLimit(env, config.time_limit)
    env.seed(config.seed)
    logger = logging.TrainingLogger(config.log_dir)
    agent = agents.make(config, env, logger)
    epoch = 0
  state_writer = logging.StateWriter(config.log_dir)
  train_driver = train.Driver(
      **config.train_driver,
      on_episode_end=partial(on_episode_end, train=True, logger=logger))
  test_driver = train.Driver(
      **config.test_driver,
      on_episode_end=partial(on_episode_end, train=False, logger=logger),
      render_options=config.render_options,
      render_episodes=config.render_episodes)
  for epoch in range(epoch, config.epochs):
    print('Training epoch #{}'.format(epoch))
    episodes, _ = train_driver.run(agent, [(config.task, env)], True)
    if epoch % config.eval_every == 0 and config.eval_trials:
      print('Evaluating...')
      results = evaluate(agent, env, config.task, test_driver,
                         config.eval_trials, seed_sequence)
      summary, videos = evaluation_summary(results)
      logger.log_summary(summary, epoch)
      for task_name, video in videos.items():
        logger.log_video(video, task_name + 'video', step=epoch)
    state_writer.write({'env': env, 'agent': agent, 'epoch': epoch})


if __name__ == '__main__':
  main()
