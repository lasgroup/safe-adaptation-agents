import argparse
import ruamel.yaml as yaml


# Acknowledgement: https://github.com/danijar
def load_config(config_path):

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
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--steps_per_train_task', default=5000, type=int)
  parser.add_argument('--train_steps_per_epoch', default=25000, type=int)
  parser.add_argument(
      '--adaptation_steps', default=5000, type=int)
  parser.add_argument(
      '--evaluation_steps', default=10000, type=int)
  parser.add_argument('--test_n_tasks', default=5, type=int)
  args, remaining = parser.parse_known_args()
  with open(config_path) as file:
    configs = yaml.safe_load(file)
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
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
  return parser.parse_args(remaining)
