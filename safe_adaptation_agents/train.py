from itertools import tee

from collections import defaultdict

from typing import Callable, Optional, Dict, List, DefaultDict, Iterable

from tqdm import tqdm

from gym import Env

from safe_adaptation_agents.agents import Agent, Transition

EpisodeSummary = DefaultDict[str, List]
IterationSummary = Dict[str, List[EpisodeSummary]]


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
  episodes = [defaultdict(list, {'observation': [observation]})]
  for _ in tqdm(range(steps)):
    if render_episodes:
      frame = environment.render(**render_options)
      episodes[-1]['frames'].append(frame)
    action = agent(observation, train, adapt)
    next_observation, reward, done, info = environment.step(action)
    cost = info.get('cost', 0)
    terminal = done and not info.get('TimeLimit.truncated', False)
    transition = Transition(observation, next_observation, action, reward, cost,
                            terminal, info)
    episodes[-1] = _append(transition, episodes[-1])
    if train:
      agent.observe(transition)
    if done:
      if on_episode_end:
        on_episode_end(episodes[-1])
      observation = environment.reset()
      episodes.append(defaultdict(list, {'observation': [observation]}))
  if 'reward' not in episodes[-1].keys():
    episodes.pop()
  return agent, episodes


def _append(transition: Transition, episode: DefaultDict) -> DefaultDict:
  episode['observation'].append(transition.observation)
  episode['reward'].append(transition.reward)
  episode['cost'].append(transition.cost)
  episode['terminal'].append(transition.terminal)
  episode['info'].append(transition.info)
  return episode


class Driver:

  def __init__(self,
               adaptation_steps: int,
               query_steps: int,
               expose_task_id: bool = False,
               on_episode_end: Optional[Callable[[EpisodeSummary],
                                                 None]] = None,
               render_episodes: int = 0,
               render_options: Optional[Dict] = None):
    self.adaptation_steps = adaptation_steps
    self.query_steps = query_steps
    self.episode_callback = on_episode_end
    self.render_episodes = render_episodes
    self.render_options = render_options
    self.expose_task_id = expose_task_id

  def run(self, agent: Agent, tasks: Iterable[Env],
          train: bool) -> [Agent, IterationSummary, IterationSummary]:
    iter_adaptation_episodes, iter_query_episodes = {}, {}
    adaptation_tasks, query_tasks = tee(tasks)
    for task_name, task in adaptation_tasks:
      agent.observe_task_id(task_name if self.expose_task_id else None)
      agent, adaptation_episodes = interact(
          agent,
          task,
          self.adaptation_steps,
          train=train,
          adapt=True,
          on_episode_end=self.episode_callback,
          render_episodes=self.render_episodes,
          render_options=self.render_options)
      iter_adaptation_episodes[task_name] = adaptation_episodes
    for task_name, task in query_tasks:
      agent.observe_task_id(task_name if self.expose_task_id else None)
      agent, query_episodes = interact(
          agent,
          task,
          self.query_steps,
          train=train,
          adapt=False,
          on_episode_end=self.episode_callback,
          render_episodes=self.render_episodes,
          render_options=self.render_options)
      iter_query_episodes[task_name] = query_episodes
    return agent, iter_adaptation_episodes, iter_query_episodes
