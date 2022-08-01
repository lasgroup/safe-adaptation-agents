from copy import deepcopy
from types import SimpleNamespace

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
    return ppo_lagrangian.PPOLagrangian(observation_space, action_space, config,
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
    return cpo.CPO(observation_space, action_space, config, logger, actor,
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
    return maml_ppo_lagrangian.MamlPPOLagrangian(observation_space,
                                                 action_space, config, logger,
                                                 actor, critic, safety_critic)
  elif config.agent == 'rl2_cpo':
    from safe_adaptation_agents.agents.on_policy import rl2_cpo
    actor = hk.without_apply_rng(
        hk.transform(lambda x, s: rl2_cpo.GRUPolicy(
            action_space.shape, config.hidden_size, config.actor)(x, s)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    return rl2_cpo.RL2CPO(observation_space, action_space, config, logger,
                          actor, critic, safety_critic)
  elif config.agent == 'rarl_cpo':
    from types import SimpleNamespace
    from safe_adaptation_agents.agents.on_policy import rarl_cpo

    def make_cpo(config):
      from safe_adaptation_agents.agents.on_policy import cpo
      actor = hk.without_apply_rng(
          hk.transform(lambda x: models.Actor(
              **config.actor, output_size=action_space.shape)(x)))
      critic = hk.without_apply_rng(
          hk.transform(lambda x: models.DenseDecoder(
              **config.critic, output_size=(1,))(x)))
      safety_critic = deepcopy(critic)
      return cpo.CPO(observation_space, action_space, config, logger, actor,
                     critic, safety_critic)

    protagonist = make_cpo(config)
    adversary_config = vars(config)
    adversary_config['safe'] = False
    adversary_config = SimpleNamespace(**adversary_config)
    adversary = make_cpo(adversary_config)
    return rarl_cpo.RARLCPO(config, logger, protagonist, adversary,
                            action_space)
  elif config.agent == 'la_mbda':
    from safe_adaptation_agents import utils
    from safe_adaptation_agents.agents.model_based.la_mbda import world_model
    from safe_adaptation_agents.agents.model_based.la_mbda import \
      augmented_lagrangian as al
    from safe_adaptation_agents.agents.model_based import replay_buffer as rb
    from safe_adaptation_agents.agents.model_based.la_mbda import la_mbda
    model = world_model.create_model(config, observation_space)
    actor = hk.without_apply_rng(
        hk.transform(lambda x: models.Actor(
            **config.actor, output_size=action_space.shape)(x)))
    critic = hk.without_apply_rng(
        hk.transform(lambda x: models.DenseDecoder(
            **config.critic, output_size=(1,))(x)))
    safety_critic = deepcopy(critic)
    augmented_lagrangian = hk.without_apply_rng(
        hk.transform(lambda cost, limit: al.AugmentedLagrangian(
            **config.augmented_lagrangian)(cost, limit)))
    replay_buffer = rb.ReplayBuffer(
        observation_space.shape,
        action_space.shape,
        config.time_limit // config.action_repeat,
        config.seed,
        **config.replay_buffer,
        precision=config.precision)
    policy_config = utils.get_mixed_precision_policy(config.precision)
    policy_32 = utils.get_mixed_precision_policy(32)
    hk.mixed_precision.set_policy(world_model.WorldModel, policy_config)
    hk.mixed_precision.set_policy(models.Actor, policy_config)
    # Set the critics' policy to 32 to stabilize their learning.
    hk.mixed_precision.set_policy(models.DenseDecoder, policy_32)
    hk.mixed_precision.set_policy(world_model.Decoder, policy_config)
    return la_mbda.LaMBDA(observation_space, action_space, logger, config,
                          model, actor, critic, safety_critic,
                          augmented_lagrangian, replay_buffer)
  else:
    raise ValueError('Could not find the requested agent.')
