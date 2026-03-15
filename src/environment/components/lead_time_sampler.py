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
    class and must implement sample(), get_expected(), and get_max_expected().
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
        self.n_warehouses = context.n_warehouses

        # Initialize the component's own RNG for reproducibility
        self._rng = np.random.default_rng()
    
    @abstractmethod
    def sample(self) -> np.ndarray:
        """
        Samples delivery lead times for all (warehouse, SKU) pairs.
        
        Returns:
            lead_times (np.ndarray): Shape (n_warehouses, n_skus).
        """
        pass

    @abstractmethod
    def get_expected(self) -> np.ndarray:
        """
        Returns expected (config-specified) lead times for all (warehouse, SKU) pairs.
        
        Returns:
            expected_lead_times (np.ndarray): Shape (n_warehouses, n_skus).
        """
        pass

    @abstractmethod
    def get_max_expected(self) -> int:
        """
        Returns the maximum expected lead time across all (warehouse, SKU) pairs.
        
        Returns:
            max_expected (int): Maximum expected lead time.
        """
        pass
    
    def reset(self, rng: Optional[np.random.Generator] = None):
        """
        Resets the lead time sampler's random state.

        Args:
            rng (Optional[np.random.Generator]): Generator from SeedManager. Defaults to None.
        """
        self._rng = rng if rng is not None else np.random.default_rng()


class FixedLeadTimeSampler(BaseLeadTimeSampler):
    """
    Implements a deterministic lead time sampler. Returns the config-specified expected lead
    times on every call to sample().
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LeadTimeSamplerConfig):
        """
        Initializes the deterministic lead time sampler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LeadTimeSamplerConfig): Lead time sampler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)

        # Store expected lead times from config
        self.expected_lead_times = np.array(
            component_config.params.expected_lead_times, dtype=int
        )  # Shape: (n_warehouses, n_skus)
    
    def sample(self) -> np.ndarray:
        """
        Samples delivery lead times for all (warehouse, SKU) pairs.
        
        Returns:
            actual_lead_times (np.ndarray): Shape (n_warehouses, n_skus).
        """

        # Get expected lead times
        actual_lead_times = self.expected_lead_times.copy()

        return actual_lead_times
    
    def get_expected(self) -> np.ndarray:
        """
        Returns expected (config-specified) lead times for all (warehouse, SKU) pairs.
        
        Returns:
            expected_lead_times (np.ndarray): Shape (n_warehouses, n_skus).
        """

        # Get expected lead times
        expected_lead_times = self.expected_lead_times.copy()
        return expected_lead_times
    
    def get_max_expected(self) -> int:
        """
        Returns the maximum expected lead time across all (warehouse, SKU) pairs.
        
        Returns:
            max_expected (int): Maximum expected lead time.
        """

        # Get maximum expected lead time
        max_expected = int(self.expected_lead_times.max())
        
        return max_expected


class StochasticLeadTimeSampler(BaseLeadTimeSampler):
    """
    Implements a stochastic lead time sampler. While the expected lead times are
    config-specified structural parameters, the actual lead times are computed as
    expected + random deviation. The deviation is sampled from a uniform
    distribution [-max_deviation, +max_deviation] independently per (warehouse, SKU).
    Actual lead times are clipped to a minimum of 1.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LeadTimeSamplerConfig):
        """
        Initializes the stochastic lead time sampler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LeadTimeSamplerConfig): Lead time sampler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)

        # Store expected lead times from config
        self.expected_lead_times = np.array(
            component_config.params.expected_lead_times, dtype=int
        )  # Shape: (n_warehouses, n_skus)

        # Store maximum deviation from config (either scalar or per-SKU)
        max_deviation = component_config.params.deviation.max_deviation
        if isinstance(max_deviation, list):
            self.max_deviation = np.array(max_deviation, dtype=int)  # Shape: (n_skus,)
        else:
            self.max_deviation = int(max_deviation) # Shape: scalar
    
    def sample(self) -> np.ndarray:
        """
        Samples delivery lead times for all (warehouse, SKU) pairs by adding a random deviation
        from a uniform distribution [-max_deviation, +max_deviation] to the expected lead times.
        
        Returns:
            actual_lead_times (np.ndarray): Shape (n_warehouses, n_skus).
        """

        # If max_deviation is an array, sample deviation using SKU-specific maximum deviations
        if isinstance(self.max_deviation, np.ndarray):
            
            deviation = np.column_stack([
                self._rng.integers(-self.max_deviation[s], self.max_deviation[s] + 1,
                                   size=self.n_warehouses)
                for s in range(self.n_skus)
            ])  # Shape: (n_warehouses, n_skus)

        # If max_deviation is a scalar, sample deviation using a single maximum deviation for all (warehouse, SKU) pairs
        else:
            deviation = self._rng.integers(
                -self.max_deviation, self.max_deviation + 1,
                size=(self.n_warehouses, self.n_skus),
            )  # Shape: (n_warehouses, n_skus)

        # Clip lead times to a minimum of 1
        actual_lead_times = np.maximum(1, self.expected_lead_times + deviation)

        return actual_lead_times
    
    def get_expected(self) -> np.ndarray:
        """
        Returns expected (config-specified) lead times for all (warehouse, SKU) pairs.
        
        Returns:
            expected_lead_times (np.ndarray): Shape (n_warehouses, n_skus).
        """

        # Get expected lead times
        expected_lead_times = self.expected_lead_times.copy()

        return expected_lead_times
    
    def get_max_expected(self) -> int:
        """
        Returns the maximum expected lead time across all (warehouse, SKU) pairs.

        Returns:
            max_expected (int): Maximum expected lead time.
        """

        # Get maximum expected lead time
        max_expected = int(self.expected_lead_times.max())
        
        return max_expected
