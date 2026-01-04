"""Base classes for environment components."""

from abc import ABC, abstractmethod
from typing import Optional

STOCHASTIC_COMPONENT_REGISTRY = (
    'demand_sampler',
    'lead_time_sampler'
)

class StochasticComponent(ABC):
    """
    Implements a base class for components that use randomness. All stochastic 
    components should inherit from this class and implement a reset() method to 
    manage their random number generators.
    """
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """
        Abstract method to reset the stochastic component's random state with a given seed.

        Args:
            seed (Optional[int]): Random seed for reproducibility. If None, component 
                is reset without an explicit seed. Defaults to None.
        """
        pass

