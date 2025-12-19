from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import StochasticComponent
from src.environment.context import EnvironmentContext
from src.config.schema import DemandSamplerConfig


@dataclass
class Order:
    """
    Implements a single customer order with demand for multiple SKUs.
    
    Attributes:
        region_id (int): Demand region identifier (0-indexed).
        sku_demands (np.ndarray): Demand quantities for each SKU. Shape: (n_skus,)
    """

    # Order parameters
    region_id: int
    sku_demands: np.ndarray  # Shape: (n_skus,)


class BaseDemandSampler(StochasticComponent):
    """
    Implements a base class for demand sampler components. Demand samplers generate a list of 
    Order objects for each timestep Each order contains a region_id and its sku_demands. All
	demand samplers must inherit from this class and implement the sample() as well as the reset()
    method from the stochastic component base class (if stochastic).
    """
    
    def __init__(self, context: EnvironmentContext, component_config: DemandSamplerConfig):
        """
        Initializes common attributes for all demand samplers.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (DemandSamplerConfig): Demand sampler configuration.
        """

        # Store general environment parameters
        self.n_regions = context.n_regions
        self.n_skus = context.n_skus

        # Initialize the component's own RNG for reproducibility
        self._rng = np.random.default_rng()
    
    @abstractmethod
    def sample(self, timestep: int) -> List[Order]:
        """
        Abstract method to sample demand orders for a timestep.
        
        Args:
            timestep (int): Current timestep in the episode.
            
        Returns:
            orders (List[Order]): List of Order objects which contain the region_id and the sku_demands for each order.
        """
        pass
    
    def reset(self, seed: Optional[int] = None):
        """
        Resets the demand sampler's random state with a given seed. If no seed is provided,
        the component is reset without an explicit seed.

        Args:
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        """

        # Reset RNG
        self._rng = np.random.default_rng(seed)


class PoissonDemandSampler(BaseDemandSampler):
    """
    Implements a Poisson-based demand sampler that generates orders using Poisson processes for
    the number of orders per region, the number of SKUs per order, and the quantity per SKU.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: DemandSamplerConfig):
        """
        Initializes the Poisson demand sampler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (DemandSamplerConfig): Demand sampler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config) 
        
        # Extract Poisson rate parameters
        self.lambda_orders = component_config.params["lambda_orders"]  # Orders per region
        self.lambda_skus = component_config.params["lambda_skus"]  # SKUs per order
        self.lambda_quantity = component_config.params["lambda_quantity"]  # Quantity per SKU
    
    def sample(self, timestep: int) -> List[Order]:
        """
        Samples orders using Poisson processes by performing the following steps:
        
        For each region:
            1. Sample the number of orders for the region (~Poisson(lambda_orders))
            2. For each order:
                2.1 Sample the number of SKUs in the order(~ Poisson(lambda_skus), capped at n_skus)
                2.2 Sample which SKUs are in the order (equal probability, without replacement)
                2.3 Sample quantities for each SKU (~ Poisson(lambda_quantity), minimum 1 unit)

        Args:
            timestep (int): Current timestep in the episode (unused for the Poisson sampler).
            
        Returns:
            orders (List[Order]): List of Order objects for this timestep. Shape: (n_regions, n_orders).
        """
        orders = []
        
        # Generate orders for each region independently
        for region_id in range(self.n_regions):
            # Sample number of orders
            n_orders = self._rng.poisson(self.lambda_orders)
            
            # Fill each order with SKUs and quantities
            for _ in range(n_orders):
                # Sample number of SKUs and cap at n_skus
                n_skus_in_order = self._rng.poisson(self.lambda_skus)
                n_skus_in_order = min(n_skus_in_order, self.n_skus) 
                
                # Initialize SKU demands array
                sku_demands = np.zeros(self.n_skus, dtype=float) # Shape: (n_skus,)
                
                # Sample SKUs and quantities for this order
                if n_skus_in_order > 0:
                    # Sample SKUs types (without replacement)
                    sku_indices = self._rng.choice(self.n_skus, size=n_skus_in_order, replace=False) # Shape: (n_skus_in_order,)
                    
                    # Sample SKU quantities and ensure at least 1 unit
                    quantities = np.maximum(1, self._rng.poisson(self.lambda_quantity, size=n_skus_in_order)) # Shape: (n_skus_in_order,)
                    
                    # Assign quantities to selected SKUs
                    sku_demands[sku_indices] = quantities
                
                # Create Order object and add to list
                orders.append(Order(region_id=region_id, sku_demands=sku_demands))
        
        return orders


class EmpiricalDemandSampler(BaseDemandSampler):
    """
    Implements a demand sampler that generates orders based on real-world demand data from a
    CSV file. Samples a random time window of orders from the data based on the actual timestep.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: DemandSamplerConfig):
        """

        Initializes the empirical demand sampler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (DemandSamplerConfig): Demand sampler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config) 
        
        # Extract episode length and data path
        self.episode_length = component_config.params["episode_length"]
        self.data_path = context.data_path
        
        # Load empirical data from CSV file
        from src.data.loader import load_empirical_data
        self.data = load_empirical_data(self.data_path)

        # Validate data length vs episode_length
        if len(self.data) < self.episode_length:
            raise ValueError(
                f"Data length ({len(self.data)}) must be >= episode_length ({self.episode_length})"
            )
        
        # Initialize start timestep for the random time window
        self._start_timestep = None
    
    def sample(self, timestep: int) -> List[Order]:
        """
        Samples orders from historical data by performing the following steps:
        
            1. Selects a random time window from the data of length episode_length
            2. Select the orders from the time window based on the given episode timestep

        Args:
            timestep (int): Current timestep in the episode.
            
        Returns:
            orders (List[Order]): List of Order objects for this timestep. Shape: (n_regions, n_orders).
        """

        # Sample random window start timestep if not set
        if self._start_timestep is None:
            max_start = len(self.data) - self.episode_length # Maximum start timestep
            self._start_timestep = self._rng.integers(0, max_start)
        
        # Compute relative timestep within the window based on the given episode timestep
        relative_timestep = timestep % self.episode_length

        # Compute actual timestep in the data
        actual_timestep = self._start_timestep + relative_timestep
        
        # Extract all data rows corresponding to the actual timestep
        timestep_data = self.data[self.data['timestep'] == actual_timestep]
        
        # Initialize list of orders
        orders = []
        
        # Group by region and order_id to create Order objects
        for (region_id, order_id), group in timestep_data.groupby(['region', 'order_id']):
            # Initialize SKU demands array
            sku_demands = np.zeros(self.n_skus, dtype=float) # Shape: (n_skus,)
            
            # Compute total quantities for each SKU in the current order
            for _, row in group.iterrows():
                sku_id = row['sku_id']
                quantity = row['quantity']
                if 0 <= sku_id < self.n_skus: # Validate SKU ID
                    sku_demands[sku_id] += quantity
            
            # Create Order object and add to list
            orders.append(Order(region_id=int(region_id), sku_demands=sku_demands))
        
        return orders
    
    def reset(self, seed: Optional[int] = None):
        """
        Resets the demand sampler's random state with a given seed. If no seed is provided,
        the component is reset without an explicit seed.

        Args:
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        """
        
        # Reset RNG
        super().reset(seed)

        # Reset start timestep
        self._start_timestep = None

