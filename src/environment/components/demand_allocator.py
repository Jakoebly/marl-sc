from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import StochasticComponent
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
        shipment_quantities_by_sku (np.ndarray): Units shipped per warehouse-region-SKU combination.
            Shape: (n_warehouses, n_regions, n_skus).
        lost_order_counts (np.ndarray): Number of orders with unfulfilled demand per region.
            Shape: (n_regions,).
    """

    # Result parameters
    fulfillment_matrix: np.ndarray  # Shape: (n_orders, n_warehouses, n_skus)
    unfulfilled_demands: np.ndarray  # Shape: (n_regions, n_skus)
    shipment_counts: np.ndarray  # Shape: (n_warehouses, n_regions)
    shipment_quantities: np.ndarray  # Shape: (n_warehouses, n_regions)
    shipment_quantities_by_sku: np.ndarray  # Shape: (n_warehouses, n_regions, n_skus)
    lost_order_counts: np.ndarray  # Shape: (n_regions,)


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
        self.fixed_cost_per_order = context.shipment_cost.outbound_fixed
        self.variable_cost_per_weight = context.shipment_cost.outbound_variable
        self.sku_weights = context.sku_weights
    
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
                - lost_order_counts (np.ndarray): Number of orders with unfulfilled demand per region.
                    Shape: (n_regions,).
        """
        pass


class GreedyDemandAllocator(BaseDemandAllocator, StochasticComponent):
    """
    Implements a greedy demand allocator that serves each region from its home warehouse first and
    then hands out the leftover demand with the regions taking turns in a random order. A region's
    home warehouse is its closest warehouse by distance. Leftover demand goes to the other warehouses
    in ascending order of shipment cost, with ties between equally expensive warehouses broken at
    random. Supports order splitting across multiple warehouses up to a limit of max_splits.

    The random region order and the random tie-breaks keep the allocation fair: no region is always
    served before the others, and no warehouse is always preferred over an equally expensive one.
    The random generator is passed in through reset() by the environment's SeedManager.
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

        # Map each region to its home warehouse (its closest warehouse by distance)
        self.home_wh = np.argmin(context.distances, axis=0) # Shape: (n_regions,)

        # Initialize the component's own RNG for reproducibility
        self._rng = np.random.default_rng()

    def reset(self, rng: Optional[np.random.Generator] = None):
        """
        Resets the demand allocator's random state.

        Args:
            rng (Optional[np.random.Generator]): Generator from SeedManager. Defaults to None.
        """
        self._rng = rng if rng is not None else np.random.default_rng()

    def allocate(self, orders: List[Order], available_inventories: np.ndarray) -> AllocationResult:
        """
        Allocates orders across warehouses in two passes. The regions take turns in a random order
        that is drawn once per call and used in both passes.

        Pass 1 (home warehouse):
            For each order, fulfill as much as possible from the home warehouse of the order's region.

        Pass 2 (leftover demand):
            For each region in the random order and each of its orders with remaining demand:
            1. Rank the other warehouses by the shipment cost of the remaining demand
               (fixed + variable * remaining weight, cheapest first). Ties are broken at random with
               one random permutation of the warehouses per region and call.
            2. Fulfill as much as possible from the next warehouse in the ranking.
            3. Update the remaining demand and the warehouse's available inventory.
            4. Repeat steps 2-3 until the order is fulfilled, the ranking is exhausted, or the order
               has been served by max_splits + 1 warehouses (counting the home warehouse).

        Demand that is still unfulfilled after both passes becomes lost sales of the order's region.

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
                - shipment_quantities_by_sku (np.ndarray): Units shipped per warehouse-region-SKU combination.
                    Shape: (n_warehouses, n_regions, n_skus).
                - lost_order_counts (np.ndarray): Number of orders with unfulfilled demand per region.
                    Shape: (n_regions,).
        """

        # Initialize variables
        n_orders = len(orders)
        available_inventories = available_inventories.copy() # Copy to avoid modifying original inventory
        max_warehouses_per_order = self.max_splits + 1
        
        # Initialize result arrays
        fulfillment_matrix = np.zeros((n_orders, self.n_warehouses, self.n_skus), dtype=float) # Shape: (n_orders, n_warehouses, n_skus)
        shipment_counts = np.zeros((self.n_warehouses, self.n_regions), dtype=int) # Shape: (n_warehouses, n_regions)
        shipment_quantities = np.zeros((self.n_warehouses, self.n_regions), dtype=float) # Shape: (n_warehouses, n_regions)
        shipment_quantities_by_sku = np.zeros((self.n_warehouses, self.n_regions, self.n_skus), dtype=float) # Shape: (n_warehouses, n_regions, n_skus)
        unfulfilled_demands = np.zeros((self.n_regions, self.n_skus), dtype=float) # Shape: (n_regions, n_skus)
        lost_order_counts = np.zeros(self.n_regions, dtype=int) # Shape: (n_regions,)
        
        # Track the remaining demand of each order and the number of warehouses that served it
        remaining_demands = [order.sku_demands.copy() for order in orders] # Each of shape: (n_skus,)
        warehouses_used = np.zeros(n_orders, dtype=int) # Shape: (n_orders,)

        # Group the order indices by region (orders of one region keep their sampled order)
        orders_by_region = [[] for _ in range(self.n_regions)]
        for order_idx, order in enumerate(orders):
            orders_by_region[order.region_id].append(order_idx)

        # Draw the order in which the regions take turns
        region_order = self._rng.permutation(self.n_regions) # Shape: (n_regions,)

        def fulfill(order_idx: int, wh_idx: int, region_id: int) -> None:
            """Ships as much of the order's remaining demand as the warehouse holds and records it."""

            # Try to fulfill from the current warehouse
            fulfillment = np.minimum(remaining_demands[order_idx], available_inventories[wh_idx]) # Shape: (n_skus,)

            # Update only if the warehouse can (partially) fulfill the order
            if np.any(fulfillment > 0):
                # Update the result matrices
                fulfillment_matrix[order_idx, wh_idx, :] += fulfillment
                shipment_counts[wh_idx, region_id] += 1
                shipment_quantities[wh_idx, region_id] += fulfillment.sum()
                shipment_quantities_by_sku[wh_idx, region_id, :] += fulfillment

                # Update the remaining demand and the warehouse's inventory
                remaining_demands[order_idx] -= fulfillment
                available_inventories[wh_idx, :] -= fulfillment
                warehouses_used[order_idx] += 1

        # Pass 1: serve every order from the home warehouse of its region
        for region_id in region_order:
            home_wh = self.home_wh[region_id]
            for order_idx in orders_by_region[region_id]:
                fulfill(order_idx, home_wh, region_id)

        # Pass 2: hand out the leftover demand, one region after the other in the random order
        for region_id in region_order:
            # Collect the region's orders that still have demand and may use another warehouse
            open_orders = [
                order_idx for order_idx in orders_by_region[region_id]
                if np.any(remaining_demands[order_idx] > 0)
                and warehouses_used[order_idx] < max_warehouses_per_order
            ]
            if not open_orders:
                continue

            # Shuffle the non-home warehouses once per region so that cost ties are broken at random
            shuffled_wh = self._rng.permutation(self.n_warehouses)
            shuffled_wh = shuffled_wh[shuffled_wh != self.home_wh[region_id]] # Shape: (n_warehouses - 1,)

            for order_idx in open_orders:
                # Rank the other warehouses by the shipment cost of the remaining demand (cheapest first).
                # The stable sort keeps the shuffled order among equally expensive warehouses.
                remaining_weight = remaining_demands[order_idx].dot(self.sku_weights)
                total_costs = (
                    self.fixed_cost_per_order[shuffled_wh, region_id] +
                    self.variable_cost_per_weight[shuffled_wh, region_id] * remaining_weight
                ) # Shape: (n_warehouses - 1,)
                warehouse_ranking = shuffled_wh[np.argsort(total_costs, kind="stable")] # Shape: (n_warehouses - 1,)

                # Try warehouses in order of cost (cheapest first)
                for wh_idx in warehouse_ranking:
                    # Check if the maximum number of warehouses per order has been reached
                    if warehouses_used[order_idx] >= max_warehouses_per_order:
                        break

                    # Try to fulfill from the current warehouse
                    fulfill(order_idx, wh_idx, region_id)

                    # Early exit if the order is fully fulfilled
                    if np.all(remaining_demands[order_idx] <= 0):
                        break

        # Track any unfulfilled demand per region
        for order_idx, order in enumerate(orders):
            if np.any(remaining_demands[order_idx] > 0):
                unfulfilled_demands[order.region_id, :] += remaining_demands[order_idx]
                lost_order_counts[order.region_id] += 1
        
        return AllocationResult(
            fulfillment_matrix=fulfillment_matrix,
            unfulfilled_demands=unfulfilled_demands,
            shipment_counts=shipment_counts,
            shipment_quantities=shipment_quantities,
            shipment_quantities_by_sku=shipment_quantities_by_sku,
            lost_order_counts=lost_order_counts
        )

