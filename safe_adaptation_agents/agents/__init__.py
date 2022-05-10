from safe_adaptation_agents.agents.agent import Agent, Transition

__all__ = ['Agent', 'Transition', 'make']

from types import SimpleNamespace

from gym import Env

import haiku as hk

from safe_adaptation_agents.training_logger import TrainingLogger
from safe_adaptation_agents.agents.on_policy import vanilla_policy_gradients

from safe_adaptation_agents import models


def make(config: SimpleNamespace, env: Env, logger: TrainingLogger):
  if config.agent == 'vanilla_policy_gradients':
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=env.action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    return vanilla_policy_gradients.VanillaPolicyGrandients(
        env.observation_space, env.action_space, config, logger, actor, critic)
