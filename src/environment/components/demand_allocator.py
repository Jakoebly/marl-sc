from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from .demand_sampler import Order
from src.environment.context import EnvironmentContext
from src.config.schema import DemandAllocatorConfig


@dataclass
class AllocationResult:
    """
    Implements the result of the demand allocation process.
    
    Attributes:
        fulfillment_matrix (np.ndarray): Quantity of each SKU that each warhouse fulfills for each order. 
            Shape: (n_orders, n_warehouses, n_skus).
        unfulfilled_demands (np.ndarray): Remaining unfulfilled demand per region. 
            Shape: (n_regions, n_skus).
        shipment_counts (np.ndarray): Number of shipments per warehouse-region pair. 
            Shape: (n_warehouses, n_regions).
        shipment_quantities (np.ndarray): Total units shipped per warehouse-region pair. 
            Shape: (n_warehouses, n_regions).
    """

    # Result parameters
    fulfillment_matrix: np.ndarray  # Shape: (n_orders, n_warehouses, n_skus)
    unfulfilled_demands: np.ndarray  # Shape: (n_regions, n_skus)
    shipment_counts: np.ndarray  # Shape: (n_warehouses, n_regions)
    shipment_quantities: np.ndarray  # Shape: (n_warehouses, n_regions)


class BaseDemandAllocator(ABC):
    """
    Implements a base class for demand allocator components. Demand allocators take orders and available inventories  
    to determine how to fulfill each order across warehouses (including transshipment). All demand allocators must inherit 
    from this class and implement the allocate() method.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: DemandAllocatorConfig):
        """
        Initializes common attributes for all demand allocators.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (DemandAllocatorConfig): Demand allocator configuration.
        """

        # Store general environment parameters
        self.n_warehouses = context.n_warehouses
        self.n_skus = context.n_skus
        self.n_regions = context.n_regions
        self.shipment_costs = context.shipment_cost
    
    @abstractmethod
    def allocate(self, orders: List[Order], available_inventories: np.ndarray,) -> AllocationResult:
        """
        Abstract method to allocate orders across warehouses.
        
        Args:
            orders (List[Order]): List of Order objects to allocate.
            available_inventories (np.ndarray): Current inventory levels at each warehouse.
                Shape: (n_warehouses, n_skus)
            
        Returns:
            AllocationResult containing:
                - fulfillment_matrix (np.ndarray): Quantity of each SKU that each warhouse fulfills for each order. 
                    Shape: (n_orders, n_warehouses, n_skus).
                - unfulfilled_demands (np.ndarray): Remaining unfulfilled demand per region. 
                    Shape: (n_regions, n_skus).
                - shipment_counts (np.ndarray): Number of shipments per warehouse-region pair. 
                    Shape: (n_warehouses, n_regions).
                - shipment_quantities (np.ndarray): Total units shipped per warehouse-region pair. 
                    Shape: (n_warehouses, n_regions).
        """
        pass


class GreedyDemandAllocator(BaseDemandAllocator):
    """
    Implements a greedy demand allocator that allocates orders greedily by trying warehouses 
    in order of shipment cost (cheapest first). Supports order splitting across multiple warehouses 
    up to a limit of max_splits.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: DemandAllocatorConfig):
        """
        Initializes the greedy demand allocator.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (DemandAllocatorGreedy): Demand allocator configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)
        
        # Handle default value for max_splits parameter
        max_splits = component_config.params["max_splits"]
        if max_splits == "default":
            self.max_splits = self.n_warehouses - 1 # Default: allow splitting across all warehouses
        else:
            self.max_splits = max_splits
        
        # Pre-compute warehouse orderings by shipment cost for each region
        self.warehouse_orderings = np.argsort(self.shipment_costs, axis=0) # Shape: (n_warehouses, n_regions)
    
    def allocate(self, orders: List[Order], available_inventories: np.ndarray) -> AllocationResult:
        """
        Allocates orders greedily across warehouses by trying warehouses in ascending order of shipment cost. 
        The allocator performs the following steps until a stopping condition is met:
        
        For each order:
            1. Select the cheapest warehouse for the order's region.
            2. Fulfill as much as possible from the selected warehouse.
            3. Update the remaining demand and warehouse's available inventory.
            4. If the order is not fully fulfilled, select the next warehouse by shipment cost.
            5. Repeat steps 2-4 until the order is fulfilled or the max_splits limit is reached.

        Args:
            orders (List[Order]): List of Order objects to allocate.
            available_inventories (np.ndarray): Current inventory levels of all warehouses.
                Shape: (n_warehouses, n_skus)

        Returns:
            AllocationResult containing:
                - fulfillment_matrix (np.ndarray): Quantity of each SKU that each warhouse fulfills for each order. 
                    Shape: (n_orders, n_warehouses, n_skus).
                - unfulfilled_demands (np.ndarray): Remaining unfulfilled demand per region. 
                    Shape: (n_regions, n_skus).
                - shipment_counts (np.ndarray): Number of shipments per warehouse-region pair. 
                    Shape: (n_warehouses, n_regions).
                - shipment_quantities (np.ndarray): Total units shipped per warehouse-region pair. 
                    Shape: (n_warehouses, n_regions).
        """

        # Initialize variables
        n_orders = len(orders)
        available_inventories = available_inventories.copy() # Copy to avoid modifying original inventory
        max_warehouses_per_order = self.max_splits + 1
        
        # Initialize result arrays
        fulfillment_matrix = np.zeros((n_orders, self.n_warehouses, self.n_skus), dtype=float) # Shape: (n_orders, n_warehouses, n_skus)
        shipment_counts = np.zeros((self.n_warehouses, self.n_regions), dtype=int) # Shape: (n_warehouses, n_regions)
        shipment_quantities = np.zeros((self.n_warehouses, self.n_regions), dtype=float) # Shape: (n_warehouses, n_regions)
        unfulfilled_demands = np.zeros((self.n_regions, self.n_skus), dtype=float) # Shape: (n_regions, n_skus)
        
        # Process each order sequentially
        for order_idx, order in enumerate(orders):
            # Get the current order's region ID and the corresponding warehouse ordering
            region_id = order.region_id
            warehouse_ordering = self.warehouse_orderings[:, region_id] # Shape: (n_warehouses,)
            
            # Track remaining demand for this order and the warehouses used
            remaining_demand = order.sku_demands.copy()
            warehouses_used = [] 
            
            # Try warehouses in order of cost (cheapest first)
            for wh_idx in warehouse_ordering:
                # Check if the maximum number of warehouses per order has been reached
                if len(warehouses_used) >= max_warehouses_per_order:
                    break
                
                # Try to fulfill from the current warehouse
                fulfillment = np.minimum(remaining_demand, available_inventories[wh_idx]) # Shape: (n_skus,)
                
                # Update only if the warehouse can (partially) fulfill the order
                if np.any(fulfillment > 0):	
                    # Update the result matrices
                    fulfillment_matrix[order_idx, wh_idx, :] += fulfillment 
                    shipment_counts[wh_idx, region_id] += 1 
                    shipment_quantities[wh_idx, region_id] += fulfillment.sum() 

                    # Update the remaining demand and the warehouse's inventory
                    remaining_demand -= fulfillment 
                    available_inventories[wh_idx, :] -= fulfillment
                    warehouses_used.append(wh_idx)
                    
                    # Early exit if the order is fully fulfilled
                    if np.all(remaining_demand <= 0):
                        break
            
            # Track any unfulfilled demand for this order
            if np.any(remaining_demand > 0):
                unfulfilled_demands[region_id, :] += remaining_demand
        
        return AllocationResult(
            fulfillment_matrix=fulfillment_matrix,
            unfulfilled_demands=unfulfilled_demands,
            shipment_counts=shipment_counts,
            shipment_quantities=shipment_quantities
        )

