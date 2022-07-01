from types import SimpleNamespace
from copy import deepcopy
from safe_adaptation_agents.agents.agent import Agent, Transition

__all__ = ['Agent', 'Transition', 'make']

from gym import Space

import haiku as hk

from safe_adaptation_agents.logging import TrainingLogger

from safe_adaptation_agents import models


def make(config: SimpleNamespace, observation_space: Space, action_space: Space,
         logger: TrainingLogger):
  # TODO (yarden): all VPG agents share the same factory, so this should be
  #  refactored.
  if config.agent == 'vanilla_policy_gradients':
    from safe_adaptation_agents.agents.on_policy import vpg
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    return vpg.VanillaPolicyGrandients(observation_space, action_space, config,
                                       logger, actor, critic)
  elif config.agent == 'ppo_lagrangian':
    from safe_adaptation_agents.agents.on_policy import ppo_lagrangian
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    return ppo_lagrangian.PpoLagrangian(observation_space, action_space, config,
                                        logger, actor, critic, safety_critic)
  elif config.agent == 'cpo':
    from safe_adaptation_agents.agents.on_policy import cpo
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    return cpo.Cpo(observation_space, action_space, config, logger, actor,
                   critic, safety_critic)
  elif config.agent == 'maml_ppo_lagrangian':
    from safe_adaptation_agents.agents.on_policy import maml_ppo_lagrangian
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    return maml_ppo_lagrangian.MamlPpoLagrangian(observation_space,
                                                 action_space, config, logger,
                                                 actor, critic, safety_critic)
  elif config.agent == 'maml_cpo':
    from safe_adaptation_agents.agents.on_policy import maml_cpo
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    return maml_cpo.MamlCpo(observation_space, action_space, config, logger,
                            actor, critic, safety_critic)
  elif config.agent == 'rl2_cpo':
    from safe_adaptation_agents.agents.on_policy import rl2_cpo
    actor = hk.without_apply_rng(
        hk.transform(lambda x, s: rl2_cpo.GruPolicy(
            action_space.shape, config.hidden_size, config.actor)(x, s)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    return rl2_cpo.Rl2Cpo(observation_space, action_space, config, logger,
                          actor, critic, safety_critic)
  else:
    raise ValueError('Could not find the requested agent.')
