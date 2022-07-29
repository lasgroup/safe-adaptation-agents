from typing import Optional
from types import SimpleNamespace
import numpy as np

from gym import spaces

from safe_adaptation_agents.agents.on_policy import cpo
from safe_adaptation_agents.agents import agent, Transition
from safe_adaptation_agents.logging import TrainingLogger


class Alternate:

  def __init__(self, protagonist_iters: int, adversary_iters: int):
    self.protagonist_iters = protagonist_iters
    self.adversary_iters = adversary_iters
    self._iters = 0

  def tick(self):
    self._iters += 1

  @property
  def protagonist_turn(self):
    x = self._iters % (self.protagonist_iters + self.adversary_iters)
    return x < self.protagonist_iters


class RARLCPO(agent.Agent):

  def __init__(self, config: SimpleNamespace, logger: TrainingLogger,
               protagonist: cpo.CPO, adversary: cpo.CPO,
               action_space: spaces.Box):
    super(RARLCPO, self).__init__(config, logger)
    self.protagonist = protagonist
    self.adversary = adversary
    # Keep actions for training as the transition in `observe` holds the sum of
    # the protagonist and adversary.
    self._protagonist_acs = None
    self._adversary_acs = None
    self._alternate = Alternate(config.protagonist_iters,
                                config.adversary_iters)
    self._env_action_space = action_space

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if self.protagonist.time_to_update and train:
      self.protagonist.train(self.protagonist.buffer.dump())
      self._alternate.tick()
    elif self.adversary.time_to_update and train:
      self.adversary.train(self.adversary.buffer.dump())
      self._alternate.tick()
    protagonist_acs = self.protagonist(observation, train, adapt)
    self._protagonist_acs = protagonist_acs
    if not train:
      self._adversary_acs = None
      return protagonist_acs
    adversary_acs = self.adversary(observation, train, adapt)
    self._adversary_acs = adversary_acs
    scale = self.config.adversary_scale
    adversary_acs *= scale
    np.clip(adversary_acs, self._env_action_space.low * scale,
            self._env_action_space.high * scale)
    return protagonist_acs + adversary_acs

  def observe(self, transition: Transition, adapt: bool):
    if self._alternate.protagonist_turn:
      transition = Transition(transition.observation,
                              transition.next_observation,
                              self._protagonist_acs, transition.reward,
                              transition.cost, transition.done, transition.info)
      self.protagonist.observe(transition, adapt)
    else:
      # The adversary tries to maximize the cost return, thus making the
      # protagonist unsafe.
      reward = transition.cost if self.config.safe else -transition.reward
      transition = Transition(transition.observation,
                              transition.next_observation, self._adversary_acs,
                              reward, transition.cost, transition.done,
                              transition.info)
      self.adversary.observe(transition, adapt)

  def observe_task_id(self, task_id: Optional[str] = None):
    pass

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray, train: bool):
    pass
