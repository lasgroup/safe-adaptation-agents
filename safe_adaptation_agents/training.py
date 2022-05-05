from collections import defaultdict

from typing import Callable, Optional, Dict, List, DefaultDict, Iterable

from tqdm import tqdm

from gym import Env

from safe_adaptation_agents.agents.agent import Agent, Transition

EpisodeSummary = DefaultDict[List]
EpochSummary = List[List[EpisodeSummary]]


def interact(
    agent: Agent,
    environment: Env,
    steps: int,
    train: bool,
    adapt: bool,
    on_episode_end: Optional[Callable[[EpisodeSummary], None]] = None,
    render_episodes: int = 0,
    render_options: Optional[Dict] = None) -> [Agent, List[EpisodeSummary]]:
  observation = environment.reset()
  episodes = []
  episode = defaultdict(list)
  for _ in tqdm(range(steps)):
    if render_episodes:
      frame = environment.render(**render_options)
      episode['frames'].append(frame)
    action = agent(observation, train, adapt)
    next_observation, reward, done, info = environment.step(action)
    terminal = done and not info.get('TimeLimit.truncated', False)
    transition = Transition(observation, next_observation, action, reward,
                            terminal, info)
    episode = _append(transition, episode)
    if train:
      agent.observe(transition)
    if done:
      if on_episode_end:
        on_episode_end(episode)
      episodes.append(episode)
      episode.clear()
  return agent, episodes


def _append(transition: Transition, episode: DefaultDict) -> DefaultDict:
  episode['obesrvation'].append(transition.observation)
  episode['reward'].append(transition.reward)
  episode['terminal'].append(transition.terminal)
  episode['info'].append(transition.info)
  return episode


class Driver:

  def __init__(self,
               adaptation_steps: int,
               test_steps: int,
               on_episode_end: Optional[Callable[[EpisodeSummary],
                                                 None]] = None,
               on_epoch_end: Optional[Callable[[EpochSummary, EpochSummary],
                                               None]] = None,
               render_episodes: int = 0,
               render_options: Optional[Dict] = None):
    self.adaptation_steps = adaptation_steps
    self.test_steps = test_steps
    self.episode_callback = on_episode_end
    self.epoch_callback = on_epoch_end
    self.render_episodes = render_episodes
    self.render_options = render_options

  def run(self, agent: Agent, tasks: Iterable[Env], n_iter: int, train: bool):
    for _ in range(n_iter):
      agent, adaptation_episodes, test_episodes = self._epoch(
        agent, tasks, train)
      if self.epoch_callback:
        self.epoch_callback(adaptation_episodes, test_episodes)

  def _epoch(self, agent: Agent, tasks: Iterable[Env],
             train: bool) -> [Agent, EpochSummary, EpochSummary]:
    epoch_adaptation_episodes = epoch_test_episodes = []
    for task in tasks:
      agent.observe_task_id()
      agent, adaptation_episodes = interact(
        agent,
        task,
        self.adaptation_steps,
        train=train,
        adapt=True,
        on_episode_end=self.episode_callback,
        render_episodes=self.render_episodes,
        render_options=self.render_options)
      epoch_adaptation_episodes.append(adaptation_episodes)
      agent, test_episodes = interact(
        agent,
        task,
        self.test_steps,
        train=train,
        adapt=False,
        on_episode_end=self.episode_callback,
        render_episodes=self.render_episodes,
        render_options=self.render_options)
      epoch_test_episodes.append(test_episodes)
    return agent, epoch_adaptation_episodes, epoch_test_episodes
