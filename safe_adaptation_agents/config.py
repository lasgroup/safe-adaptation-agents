import os
from typing import Optional, List, AnyStr

import safe_adaptation_agents

BASE_PATH = os.path.join(os.path.dirname(safe_adaptation_agents.__file__))


def validate_config(config):
  assert config.time_limit % config.action_repeat == 0, ('Action repeat '
                                                         'should '
                                                         ''
                                                         'be a factor of time '
                                                         ''
                                                         'limit')
  assert config.train_driver['adaptation_steps'] % config.time_limit == 0, (
      'Time limit should be a factor of adaptation steps')
  assert config.test_driver['adaptation_steps'] % config.time_limit == 0, (
      'Time limit should be a factor of adaptation steps')
  assert config.train_driver['query_steps'] % config.time_limit == 0, (
      'Time limit should be a factor of query steps')
  assert config.test_driver['query_steps'] % config.time_limit == 0, (
      'Time limit should be a factor of query steps')
  assert config.eval_every > 0, 'Eval every should be a positive number.'
  return config


def resolve_agent(remaining, config_names, configs):
  if '--agent' in remaining:
    idx = remaining.index('--agent')
    agent_name = remaining[idx + 1]
    return agent_name
  else:
    from safe_adaptation_agents import agents
    agent_name = configs['defaults']['agent']
    for name in reversed(config_names):
      if name in configs and name in agents.AGENTS:
        agent_name = name
        break
    return agent_name


def resolve_benchmark(remaining, config_names, configs):
  if '--benchmark' in remaining:
    idx = remaining.index('--benchmark')
    benchmark_name = remaining[idx + 1]
    return benchmark_name
  else:
    from safe_adaptation_gym.benchmark import BENCHMARKS
    benchmark_name = configs['defaults']['benchmark']
    for name in reversed(config_names):
      if name in configs and name in BENCHMARKS:
        benchmark_name = name
        break
    return benchmark_name


# Acknowledgement: https://github.com/danijar
def load_config(args: Optional[List[AnyStr]] = None):
  import argparse
  import ruamel.yaml as yaml
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', default=['defaults'])
  # Load all config parameters and infer their types and default values.
  args, remaining = parser.parse_known_args(args)
  with open(os.path.join(BASE_PATH, 'configs.yaml')) as file:
    configs = yaml.safe_load(file)
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  agent_name = resolve_agent(remaining, args.configs, configs)
  defaults['agent'] = agent_name
  defaults.update(configs[agent_name])
  benchmark_name = resolve_benchmark(remaining, args.configs, configs)
  defaults['benchmark'] = benchmark_name
  defaults.update(configs[benchmark_name])
  updated_remaining = []
  # Collect all the user inputs that override the default parameters.
  for idx in range(0, len(remaining), 2):
    stripped = remaining[idx].strip('-')
    # Allow the user to override specific values within dictionaries.
    if '.' in stripped:
      params_group, key = stripped.split('.')
      # Override the default value within a dictionary.
      defaults[params_group][key] = yaml.safe_load(remaining[idx + 1])
    else:
      updated_remaining.append(remaining[idx])
      updated_remaining.append(remaining[idx + 1])
  remaining = updated_remaining
  parser = argparse.ArgumentParser()
  # Add arguments from the defaults to create the default parameters namespace.
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    parser.add_argument(f'--{key}', type=yaml.safe_load, default=value)
  # Parse the remaining arguments into the parameters' namespace.
  return validate_config(parser.parse_args(remaining))
