from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .base import StochasticComponent
from src.environment.context import EnvironmentContext
from src.config.schema import LeadTimeSamplerConfig


class BaseLeadTimeSampler(StochasticComponent):
    """
    Implements a base class for lead time sampling components. Lead time samplers generate 
    delivery lead times for replenishment orders. All lead time samplers must inherit from this 
    class and implement the sample() as well as the reset() method from the stochastic component 
    base class (if stochastic).
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LeadTimeSamplerConfig):
        """
        Initializes common attributes for all lead time samplers.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LeadTimeSamplerConfig): Lead time sampler configuration.
        """

        # Store general environment parameters
        self.n_skus = context.n_skus

        # Initialize the component's own RNG for reproducibility
        self._rng = np.random.default_rng()
    
    @abstractmethod
    def sample(self) -> np.ndarray:
        """
        Abstract method to sample delivery lead times for all SKUs.
        
        Returns:
            lead_times (np.ndarray): Lead times for all SKUs. Shape: (n_skus,).
        """
        pass
    
    def reset(self, seed: Optional[int] = None):
        """
        Resets the lead time sampler's random state with a given seed. If no seed is provided,
        the component is reset without an explicit seed.

        Args:
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        """

        # Reset RNG
        self._rng = np.random.default_rng(seed)


class UniformLeadTimeSampler(BaseLeadTimeSampler):
    """
    Implements a uniform lead time sampler that samples lead times independently for each SKU 
    from a uniform distribution over [min_lead_time, max_lead_time].
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LeadTimeSamplerConfig):
        """
        Initializes the uniform lead time sampler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LeadTimeSamplerConfig): Lead time sampler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)
        
        # Extract uniform distribution parameters
        self.min_lead_time = component_config.params["min"]
        self.max_lead_time = component_config.params["max"]
    
    def sample(self) -> np.ndarray:
        """
        Samples independent lead times for all SKUs from a uniform distribution. Each SKU gets 
        an independent lead time sampled uniformly from [min_lead_time, max_lead_time].
        
        Returns:
            lead_times (np.ndarray): Lead times for all SKUs. Shape: (n_skus,).
        """
        # Sample n_skus independent lead times
        lead_times = self._rng.integers(self.min_lead_time, self.max_lead_time + 1, size=self.n_skus) # Shape: (n_skus,)
        
        return lead_times

