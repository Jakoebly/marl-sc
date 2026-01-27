from typing import Dict, Any, Optional
import torch
from gymnasium.spaces import Box, Discrete

from src.algorithms.models.rlmodules.base import BaseActorCriticRLModule


class IPPORLModule(BaseActorCriticRLModule):
    """
    IPPO (Independent PPO) RLModule.
    
    Both actor and critic use LOCAL observations only.
    No information sharing between agents - each agent learns independently.
    """
    
