from collections import defaultdict

from typing import Callable, Optional, Tuple, Dict, List, DefaultDict
from tqdm import tqdm

from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym
from safe_adaptation_agents.agents.agent import Agent, Transition


def interact(
    agent: Agent,
    environment: SafeAdaptationGym,
    steps: int,
    render_episodes: int = 0,
    training: bool = True,
    on_episode_end: Optional[Callable[[DefaultDict[List]], None]] = None,
    render_kwargs: Optional[Dict] = None
) -> Tuple[Agent, SafeAdaptationGym, List[DefaultDict]]:
  observation = environment.reset()
  episodes = []
  episode = defaultdict(list)
  for _ in tqdm(range(steps)):
    if render_episodes:
      frame = environment.render(**render_kwargs)
      episode['frames'].append(frame)
    # TODO (yarden): how to adapt this loop to adaptation/exploration?
    action = agent(observation, training)
    next_observation, reward, done, info = environment.step(action)
    terminal = done and not info.get('TimeLimit.truncated', False)
    transition = Transition(observation, next_observation, action, reward,
                            terminal, info)
    episode = _append(transition, episode)
    if training:
      agent.observe(transition)
    if done:
      on_episode_end(episode)
      episodes.append(episode)
      episode.clear()
  return agent, environment, episodes


def _append(transition: Transition, episode: DefaultDict) -> DefaultDict:
  episode['obesrvation'].append(transition.observation)
  episode['reward'].append(transition.reward)
  episode['terminal'].append(transition.terminal)
  episode['info'].append(transition.info)
  return episode
