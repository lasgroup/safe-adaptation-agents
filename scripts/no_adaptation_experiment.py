import concurrent.futures

import numpy as np

from gym import Env
from gym.wrappers import TimeLimit

import safe_adaptation_gym

from safe_adaptation_agents import agents, logger, train
from safe_adaptation_agents import config as options


def evaluate(agent: agents.Agent, env: Env, test_driver: train.Driver,
             trials: int, seed_sequence: np.random.SeedSequence):
  envs = (env.seed(seed) for seed in seed_sequence.generate_state(trials))
  with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(lambda env: test_driver.run(agent, [env], False),
                           envs)
  return results


def main():
  config = options.load_config()
  seed_sequence = np.random.SeedSequence(config.seed)
  env = safe_adaptation_gym.make(config.task, config.robot)
  # TODO (yarden): what about action repeat?
  env = TimeLimit(env, config.time_limit)
  env.seed(config.seed)
  agent = agents.make(config, env, logger.TrainingLogger(config.log_dir))
  train_driver = train.Driver(**config.train_driver)
  test_driver = train.Driver(**config.test_driver)
  for epoch in range(config.epochs):
    iter_adaptation_episodes, iter_query_episodes = train_driver.run(
        agent, [env], True)
    if epoch % config.eval_every == 0:
      evaluate(agent, env, test_driver, config.eval_trials, seed_sequence)


if __name__ == '__main__':
  main()
