from safe_adaptation_agents.agents.agent import Agent, Transition

__all__ = ['Agent', 'Transition', 'make']

from types import SimpleNamespace

from gym import Space

import haiku as hk

from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.agents.on_policy import vpg

from safe_adaptation_agents import models


def make(config: SimpleNamespace, observation_space: Space, action_space: Space,
         logger: TrainingLogger):
  if config.agent == 'vanilla_policy_gradients':
    print('Creating VPG...\n{}'.format(config))
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    return vpg.VanillaPolicyGrandients(observation_space, action_space, config,
                                       logger, actor, critic)
