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
  return config


def resolve_agent(remaining, config_names, configs):
  if '--agent' in remaining:
    idx = remaining.index('--agent')
    agent_name = remaining[idx + 1]
    return configs[agent_name]
  else:
    for name in reversed(config_names):
      if 'agent' in configs[name]:
        agent_name = configs[name]['agent']
        return configs[agent_name]
  raise ValueError('No agent provided')


# Acknowledgement: https://github.com/danijar
def load_config(args: Optional[List[AnyStr]] = None):
  import argparse
  import ruamel.yaml as yaml

  def args_type(default):

    def parse_string(x):
      if default is None:
        return x
      if isinstance(default, bool):
        return bool(['False', 'True'].index(x))
      if isinstance(default, int):
        return float(x) if ('e' in x or '.' in x) else int(x)
      if isinstance(default, (list, tuple)):
        return tuple(args_type(default[0])(y) for y in x.split(','))
      return type(default)(x)

    def parse_object(x):
      if isinstance(default, (list, tuple)):
        return tuple(x)
      return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', default=['defaults'])
  args, remaining = parser.parse_known_args(args)
  with open(os.path.join(BASE_PATH, 'configs.yaml')) as file:
    configs = yaml.safe_load(file)
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  defaults.update(resolve_agent(remaining, args.configs, configs))
  updated_remaining = []
  for idx in range(0, len(remaining), 2):
    stripped = remaining[idx].strip('-')
    if '.' in stripped:
      params_group, key = stripped.split('.')
      orig_value = defaults[params_group][key]
      arg_type = args_type(orig_value)
      defaults[params_group][key] = arg_type(remaining[idx + 1])
    else:
      updated_remaining.append(remaining[idx])
      updated_remaining.append(remaining[idx + 1])
  remaining = updated_remaining
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  return validate_config(parser.parse_args(remaining))
