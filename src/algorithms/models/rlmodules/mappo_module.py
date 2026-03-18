from src.algorithms.models.rlmodules.base import ActorCriticRLModule


class MAPPORLModule(ActorCriticRLModule):
    """
    MAPPO (Multi-Agent PPO) RLModule implementing CTDE.
    
    - Actor: Uses observations determined by actor_obs_type (default: local)
    - Critic: Uses observations determined by critic_obs_type (default: global)
    
    The base class handles obs routing automatically via actor_obs_type and
    critic_obs_type in model_config.
    """
    
    # No need to override anything - base class handles obs routing automatically
    # via the actor_obs_type / critic_obs_type flags in model_config.
