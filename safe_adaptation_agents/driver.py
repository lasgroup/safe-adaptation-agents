from itertools import tee

from collections import defaultdict

from typing import (Callable, Optional, Dict, List, DefaultDict, Iterable,
                    Tuple)

from gym.vector import VectorEnv
import numpy as np
from tqdm import tqdm

from safe_adaptation_gym import tasks as sagt

from safe_adaptation_agents.agents import Agent, Transition

EpisodeSummary = Dict[str, List]
IterationSummary = Dict[str, List[EpisodeSummary]]


def interact(agent: Agent,
             environment: VectorEnv,
             steps: int,
             train: bool,
             adapt: bool,
             on_episode_end: Optional[Callable[[EpisodeSummary], None]] = None,
             render_episodes: int = 0,
             render_mode: str = 'rgb_array') -> [Agent, List[EpisodeSummary]]:
  observations = environment.reset()
  step = 0
  pbar = tqdm(total=steps, leave=True, position=0)
  episodes = [defaultdict(list, {'observation': [observations]})]
  while step < steps:
    if render_episodes:
      frames = environment.render(render_mode)
      episodes[-1]['frames'].append(frames)
    actions = agent(observations, train, adapt)
    next_observations, rewards, dones, infos = environment.step(actions)
    costs = np.array([info.get('cost', 0) for info in infos])
    transition = Transition(observations, next_observations, actions, rewards,
                            costs, dones, infos)
    episodes[-1] = _append(transition, episodes[-1])
    if train:
      agent.observe(transition)
    observations = next_observations
    if transition.last:
      if on_episode_end:
        on_episode_end(episodes[-1])
      episodes.append(defaultdict(list))
      observations = environment.reset()
    transition_steps = sum(transition.steps)
    step += transition_steps
    pbar.update(transition_steps)
  if not episodes[-1]:
    episodes.pop()
  return agent, episodes


def _append(transition: Transition, episode: DefaultDict) -> DefaultDict:
  episode['observation'].append(transition.observation)
  episode['action'].append(transition.action)
  episode['reward'].append(transition.reward)
  episode['cost'].append(transition.cost)
  episode['done'].append(transition.done)
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
               render_mode: str = 'rgb_array'):
    self.adaptation_steps = adaptation_steps
    self.query_steps = query_steps
    self.episode_callback = on_episode_end
    self.render_episodes = render_episodes
    self.render_mode = render_mode
    self.expose_task_id = expose_task_id

  def run(self, agent: Agent, env: VectorEnv, tasks: Iterable[Tuple[str,
                                                                    sagt.Task]],
          train: bool) -> [IterationSummary, IterationSummary]:
    iter_adaptation_episodes, iter_query_episodes = {}, {}
    adaptation_tasks, query_tasks = tee(tasks)
    for task_name, task in adaptation_tasks:
      env.reset(options={'task': task})
      agent.observe_task_id(task_name if self.expose_task_id else None)
      agent, adaptation_episodes = interact(
          agent,
          env,
          self.adaptation_steps,
          train=train,
          adapt=True,
          on_episode_end=self.episode_callback,
          render_episodes=self.render_episodes,
          render_mode=self.render_mode)
      iter_adaptation_episodes[task_name] = adaptation_episodes
    for task_name, task in query_tasks:
      env.reset(options={'task': task})
      agent.observe_task_id(task_name if self.expose_task_id else None)
      agent, query_episodes = interact(
          agent,
          env,
          self.query_steps,
          train=train,
          adapt=False,
          on_episode_end=self.episode_callback,
          render_episodes=self.render_episodes,
          render_mode=self.render_mode)
      iter_query_episodes[task_name] = query_episodes
    return iter_adaptation_episodes, iter_query_episodes
