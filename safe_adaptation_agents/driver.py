from collections import defaultdict
from typing import (Callable, Optional, Dict, List, DefaultDict, Iterable,
                    Tuple)

import numpy as np
from safe_adaptation_gym import tasks as sagt
from tqdm import tqdm

from safe_adaptation_agents import episodic_trajectory_buffer as etb
from safe_adaptation_agents.agents import Agent, Transition
from safe_adaptation_agents.episodic_async_env import EpisodicAsync

EpisodeSummary = Dict[str, List]
IterationSummary = Dict[str, List[EpisodeSummary]]


def interact(agent: Agent,
             environment: EpisodicAsync,
             num_episodes: int,
             train: bool,
             adaptation_buffer: Optional[etb.EpisodicTrajectoryBuffer] = None,
             on_episode_end: Optional[Callable[[EpisodeSummary, bool, int],
                                               None]] = None,
             render_episodes: int = 0,
             render_mode: str = 'rgb_array') -> [Agent, List[EpisodeSummary]]:
  observations = environment.reset()
  step = 0
  episodes = [defaultdict(list, {'observation': [observations]})]
  adapt = adaptation_buffer is not None
  episode_steps = 0
  steps = num_episodes * environment.time_limit
  with tqdm(total=steps) as pbar:
    while len(episodes) < num_episodes + 1:
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
        agent.observe(transition, adapt)
      # Append adaptation data if needed.
      if adaptation_buffer is not None:
        adaptation_buffer.add(transition)
      observations = next_observations
      if transition.last:
        render_episodes = max(render_episodes - 1, 0)
        if on_episode_end:
          on_episode_end(episodes[-1], adapt, episode_steps)
        episode_steps = 0
        observations = environment.reset()
        episodes.append(defaultdict(list, {'observation': [observations]}))
      transition_steps = sum(transition.steps)
      step += transition_steps
      episode_steps += transition_steps
      pbar.update(transition_steps)
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

  def __init__(
      self,
      adaptation_episodes: int,
      query_episodes: int,
      time_limit: int,
      observation_shape: Tuple,
      action_shape: Tuple,
      expose_task_id: bool = False,
      on_episode_end: Optional[Callable[[EpisodeSummary, str, bool, int],
                                        None]] = None,
      render_episodes: int = 0,
      render_mode: str = 'rgb_array'):
    self.adaptation_buffer = etb.EpisodicTrajectoryBuffer(
        adaptation_episodes, time_limit, observation_shape, action_shape)
    self.num_adaptation_episodes = adaptation_episodes
    self.num_query_episodes = query_episodes
    self.episode_callback = on_episode_end
    self.render_episodes = render_episodes
    self.render_mode = render_mode
    self.expose_task_id = expose_task_id

  def run(self, agent: Agent, env: EpisodicAsync,
          tasks: Iterable[Tuple[str, sagt.Task]],
          train: bool) -> [IterationSummary, IterationSummary]:
    iter_adaptation_episodes, iter_query_episodes = {}, {}
    for i, (task_name, task) in enumerate(tasks):
      if self.episode_callback is not None:
        callback = lambda summary, adapt, steps: self.episode_callback(
            summary, task_name, adapt, steps)
      else:
        callback = None
      if self.num_adaptation_episodes > 0:
        env.reset(options={'task': task})
        agent.observe_task_id(task_name if self.expose_task_id else None)
        agent, adaptation_episodes = interact(
            agent,
            env,
            self.num_adaptation_episodes,
            train=train,
            adaptation_buffer=self.adaptation_buffer,
            on_episode_end=callback,
            render_episodes=self.render_episodes,
            render_mode=self.render_mode)
        iter_adaptation_episodes[task_name] = adaptation_episodes
        assert self.adaptation_buffer.full, (
            'Adaptation buffer should be full at this point. Episode id: {}, '
            'transition idx: {}'.format(self.adaptation_buffer.episode_id,
                                        self.adaptation_buffer.idx))
        agent.adapt(*self.adaptation_buffer.dump(), train)
      if self.num_query_episodes > 0:
        agent, query_episodes = interact(
            agent,
            env,
            self.num_query_episodes,
            train=train,
            on_episode_end=callback,
            render_episodes=self.render_episodes,
            render_mode=self.render_mode)
        iter_query_episodes[task_name] = query_episodes
    return iter_adaptation_episodes, iter_query_episodes
