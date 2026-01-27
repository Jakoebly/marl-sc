from src.algorithms.models.rlmodules.base import BaseActorCriticRLModule


class MAPPORLModule(BaseActorCriticRLModule):
    """
    MAPPO (Multi-Agent PPO) RLModule implementing CTDE.
    
    - Actor: Always uses LOCAL observations (decentralized execution)
    - Critic: Uses GLOBAL state during TRAINING (centralized training)
    - Critic: Uses LOCAL observations during INFERENCE (decentralized execution)
    
    The base class handles CTDE automatically when use_centralized_critic=True
    is set in model_config. During inference, the base class will use local
    observations for the critic (since global state is not available).
    """
    
    # No need to override anything - base class handles CTDE automatically!
    # The setup() method builds critic with global_state_dim when use_centralized_critic=True
    # The _forward_train() method extracts global state from batch["infos"] for critic
    # The _forward_inference() method uses local obs for critic (decentralized execution)
