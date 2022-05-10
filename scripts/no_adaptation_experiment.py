import os
import concurrent.futures
from itertools import repeat

import cloudpickle

import numpy as np

from gym import Env
from gym.wrappers import TimeLimit

import safe_adaptation_gym

from safe_adaptation_agents import agents, logger, train
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
    episodes, _ = train_driver.run(agent, [(config.task, env)], True)
    if epoch % config.eval_every == 0:
      evaluate(agent, env, config.task, test_driver, config.eval_trials,
               seed_sequence)
    with open(os.path.join(config.log_dir, 'state.pkl'), 'wb') as f:
      cloudpickle.dump({'env': env, 'agent': agent}, f)


if __name__ == '__main__':
  main()
