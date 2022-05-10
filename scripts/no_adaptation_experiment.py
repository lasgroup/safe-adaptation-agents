import os
from functools import partial
import concurrent.futures
from itertools import repeat
from typing import List, Dict

import cloudpickle

import numpy as np

from gym import Env
from gym.wrappers import TimeLimit

import safe_adaptation_gym

from safe_adaptation_agents import agents, training_logger, train
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
  with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [
        result for result in executor.map(
            run_one, repeat(agent_bytes), repeat(env), repeat(task_name),
            list(seed_sequence.generate_state(trials)), repeat(test_driver))
    ]
  return results


def evaluation_summary(runs: List[train.IterationSummary]) -> Dict:
  all_runs = []
  for run in runs:
    all_tasks = []
    for task_name, task in run.items():
      return_ = np.asarray([sum(episode['reward']) for episode in task]).mean()
      cost_return_ = np.asarray([
          sum(list(map(lambda info: info['cost'], episode['info'])))
          for episode in task
      ]).mean()
      all_tasks.append((return_, cost_return_))
    all_runs.append(all_tasks)
  total_return, total_cost = np.split(np.asarray(all_runs), 2, axis=-1)
  return {
      'evaluation/return': total_return.mean(),
      'evaluation/cost_return': total_cost.mean()
  }


def on_episode_end(episode: train.EpisodeSummary,
                   logger: training_logger.TrainingLogger, train: bool):
  episode_return = sum(episode['reward'])
  summary = {'training/episode_return': episode_return}
  sum_costs = sum(list(map(lambda info: info['cost'], episode['info'])))
  summary['training/episode_cost_return'] = sum_costs
  print("\nReward return: {} -- Cost return: {}".format(episode_return,
                                                        sum_costs))
  logger.log_summary(summary)
  if train:
    logger.step += len(episode['reward'])


def main():
  config = options.load_config()
  seed_sequence = np.random.SeedSequence(config.seed)
  env = safe_adaptation_gym.make(config.task, config.robot)
  env = TimeLimit(env, config.time_limit)
  env.seed(config.seed)
  logger = training_logger.TrainingLogger(config.log_dir)
  agent = agents.make(config, env, logger)
  train_driver = train.Driver(
      **config.train_driver,
      on_episode_end=partial(on_episode_end, train=True, logger=logger))
  test_driver = train.Driver(
      **config.test_driver,
      on_episode_end=partial(on_episode_end, train=False, logger=logger))
  for epoch in range(config.epochs):
    # episodes, _ = train_driver.run(agent, [(config.task, env)], True)
    if epoch % config.eval_every == 0:
      results = evaluate(agent, env, config.task, test_driver,
                         config.eval_trials, seed_sequence)
      logger.log_summary(evaluation_summary(results))
    with open(os.path.join(config.log_dir, 'state.pkl'), 'wb') as f:
      cloudpickle.dump({'env': env, 'agent': agent}, f)


if __name__ == '__main__':
  main()
