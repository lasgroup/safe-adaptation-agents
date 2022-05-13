import os
from functools import partial
from typing import List, Dict

import cloudpickle

import numpy as np

from gym.vector import VectorEnv
from gym.wrappers import TimeLimit

from safe_adaptation_agents import agents, logging, train, episodic_async_env
from safe_adaptation_agents import config as options


def evaluate(agent: agents.Agent, env: VectorEnv, task_name: str,
             test_driver: train.Driver, trials: int):
  results = [
      test_driver.run(agent, [(task_name, env)], False)[0]
      for _ in range(trials)
  ]
  return results


def evaluation_summary(runs: List[train.IterationSummary]) -> [Dict, Dict]:
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


def on_episode_end(episode: train.EpisodeSummary,
                   logger: logging.TrainingLogger, train: bool):

  def return_(arr):
    return np.asarray(arr).sum(0).mean()

  episode_return = return_(episode['reward'])
  cost_return = return_(episode['cost'])
  print("\nReward return: {} -- Cost return: {}".format(episode_return,
                                                        cost_return))
  if train:
    summary = {'training/episode_return': episode_return}
    summary['training/episode_cost_return'] = cost_return
    logger.log_summary(summary)
    logger.step += np.asarray(episode['reward']).size


def resume_experiment(log_dir):
  with open(os.path.join(log_dir, 'state.pkl'), 'rb') as f:
    env_rs, agent, epoch = cloudpickle.load(f).values()

  return env_rs, agent, agent.logger, agent.config, epoch


def make_env(config):
  # Importing safe-adaptation-gym inside the function to allow the process to
  # load it internally.
  import safe_adaptation_gym
  env = safe_adaptation_gym.make(config.task, config.robot)
  env = TimeLimit(env, config.time_limit)
  return env


def main():
  config = options.load_config()
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  env = episodic_async_env.EpisodicAsync(lambda: make_env(config),
                                         config.parallel_envs)
  if os.path.exists(os.path.join(config.log_dir, 'state.pkl')):
    seeds, agent, logger, config, epoch = resume_experiment(config.log_dir)
    env.seed([rs.get_state()[1] for rs in env_rs])
  else:
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
      render_episodes=config.render_episodes)
  for epoch in range(epoch, config.epochs):
    print('Training epoch #{}'.format(epoch))
    episodes, _ = train_driver.run(agent, [(config.task, env)], True)
    if epoch % config.eval_every == 0 and config.eval_trials:
      print('Evaluating...')
      results = evaluate(agent, env, config.task, test_driver,
                         config.eval_trials)
      summary, videos = evaluation_summary(results)
      logger.log_summary(summary, epoch)
      for task_name, video in videos.items():
        logger.log_video(video, task_name + '_video', step=epoch, fps=60)
    state_writer.write({
        'env_rs': [rs.get_state()[1] for rs in env.get_attr('rs')],
        'agent': agent,
        'epoch': epoch
    })


if __name__ == '__main__':
  main()
