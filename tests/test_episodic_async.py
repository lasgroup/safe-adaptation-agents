from safe_adaptation_agents import episodic_async_env

NUM_ENVS = 5


def make_env():
  import safe_adaptation_gym
  return safe_adaptation_gym.make('go_to_goal', 'point')


def test_reset():
  env = episodic_async_env.EpisodicAsync(make_env, vector_size=NUM_ENVS)
  observation = env.reset()
  assert len(observation) == NUM_ENVS


def test_render():
  env = episodic_async_env.EpisodicAsync(make_env, vector_size=NUM_ENVS)
  env.reset()
  image = env.render()
  # env.step(env.action_space.sample())
  # env.render()
  # env.reset()
  # image = env.render()
  assert len(image) == NUM_ENVS
