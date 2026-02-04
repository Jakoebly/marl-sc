from abc import ABC, abstractmethod

import numpy as np
from scipy.special import softmax

from src.environment.context import EnvironmentContext
from src.config.schema import LostSalesHandlerConfig


class BaseLostSalesHandler(ABC):
    """
    Implements a base class for lost sales handler components. Lost sales handlers assign 
    unfulfilled demand to warehouses for penalty cost calculation. All lost sales handlers must 
    inherit from this class and implement the calculate_lost_sales() method.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LostSalesHandlerConfig):
        """
        Initializes common attributes for all lost sales handlers.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LostSalesHandlerConfig): Lost sales handler configuration.
        """

        # Store general environment parameters
        self.n_warehouses = context.n_warehouses
        self.n_skus = context.n_skus
        self.n_regions = context.n_regions
        self.distances = context.distances
        self.fixed_cost_per_order = context.shipment_cost.outbound_fixed
        self.variable_cost_per_weight = context.shipment_cost.outbound_variable
        self.sku_weights = context.sku_weights
        
        # Pre-compute closest warehouses for each region based on distances
        self.closest_warehouses = np.argmin(self.distances, axis=0) # Shape: (n_regions,)
    
    @abstractmethod
    def calculate_lost_sales(self, lost_order_counts: np.ndarray, unfulfilled_demand: np.ndarray, shipments: np.ndarray) -> np.ndarray:
        """
        Abstract method to calculate lost sales assignments to warehouses.
        
        Args:
            unfulfilled_demand (np.ndarray): Unfulfilled demand per region. Shape: (n_regions, n_skus).
            shipments (np.ndarray): Shipment information per warehouse-region pair. Shape: (n_warehouses, n_regions).
            
        Returns:
            lost_sales_assigned (np.ndarray): Lost sales assignment matrix. Shape: (n_warehouses, n_skus).
        """
        pass


class ClosestLostSalesHandler(BaseLostSalesHandler):
    """
    Implements a closest lost sales handler that assigns all lost sales of a region to the closest
    warehouse only.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LostSalesHandlerConfig):
        """
        Initializes the cheapest lost sales handler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LostSalesHandlerConfig): Lost sales handler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)
    
    def calculate_lost_sales(self, lost_order_counts: np.ndarray, unfulfilled_demand: np.ndarray, shipments: np.ndarray) -> np.ndarray:
        """
        Assigns the lost sales of each region to its closest warehouse only.
        
        Args:
            unfulfilled_demand (np.ndarray): Unfulfilled demand per region. Shape: (n_regions, n_skus).
            shipments (np.ndarray): Shipment quantities per warehouse-region pair (unused for the cheapest handler).
                Shape: (n_warehouses, n_regions)
            
        Returns:
            lost_sales_assigned (np.ndarray): Lost sales assignment matrix. Shape: (n_warehouses, n_skus).
        """

        # Initialize lost sales assignment matrix
        lost_sales_assigned = np.zeros((self.n_warehouses, self.n_skus), dtype=float) # Shape: (n_warehouses, n_skus)
        
        # For each region, assign all lost sales to its closest warehouse
        for region_id in range(self.n_regions):
            closest_wh = self.closest_warehouses[region_id]
            lost_sales_assigned[closest_wh, :] += unfulfilled_demand[region_id, :] # Shape: (n_skus,)
        
        return lost_sales_assigned


class ShipmentLostSalesHandler(BaseLostSalesHandler):
    """
    Implements a shipment-based lost sales handler that assigns lost sales proportionally
    based on the quantity of units each warehouse shipped to each region.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LostSalesHandlerConfig):
        """
        Initializes the shipment-based lost sales handler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LostSalesHandlerConfig): Lost sales handler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)
    
    def calculate_lost_sales(self, lost_order_counts: np.ndarray, unfulfilled_demand: np.ndarray, shipments: np.ndarray) -> np.ndarray:
        """
        Assigns lost sales proportionally to warehouses based on the quantity of units they shipped 
        to each region. If no shipments occurred, falls back to assigning to the closest warehouse only.
        
        Args:
            unfulfilled_demand (np.ndarray): Unfulfilled demand per region. Shape: (n_regions, n_skus).
            shipments (np.ndarray): Shipment quantities per warehouse-region pair.
                Shape: (n_warehouses, n_regions)
            
        Returns:
            lost_sales_assigned (np.ndarray): Lost sales assignment matrix. Shape: (n_warehouses, n_skus).
        """

        # Initialize lost sales assignment matrix
        lost_sales_assigned = np.zeros((self.n_warehouses, self.n_skus), dtype=float) # Shape: (n_warehouses, n_skus)
        
        # For each region, assign lost sales proportionally based on units shipped
        for region_id in range(self.n_regions):
            # Compute total shipped units of all warehouses to the current region
            total_shipped = shipments[:, region_id].sum()
            
            # If shipments occurred, compute proportional weights based on shipment quantities
            if total_shipped > 0:
                weights = shipments[:, region_id] / total_shipped # Shape: (n_warehouses,)

            # If no shipments occurred, fallback to closest warehouse only
            else:
                weights = np.zeros(self.n_warehouses)
                weights[self.closest_warehouses[region_id]] = 1.0 # Shape: (n_warehouses,)
            
            # Assign lost sales for each SKU according to the weights
            for sku_id in range(self.n_skus):
                lost_sales_assigned[:, sku_id] += weights * unfulfilled_demand[region_id, sku_id] # Shape: (n_warehouses,)
        
        return lost_sales_assigned


class CostLostSalesHandler(BaseLostSalesHandler):
    """
    Implements a cost-based lost sales handler that assigns lost sales to all warehouses based on
    their shipment costs to each region using softmax.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: LostSalesHandlerConfig):
        """
        Initializes the cost-based lost sales handler.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (LostSalesHandlerConfig): Lost sales handler configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)
        
        # Extract softmax temperature parameter
        self.alpha = component_config.params["alpha"]
    
    def calculate_lost_sales(self, lost_order_counts: np.ndarray, unfulfilled_demand: np.ndarray, shipments: np.ndarray) -> np.ndarray:
        """
        Assigns lost sales proportionally to warehouses based on their shipment costs to each region. Uses 
        a softmax over inverse shipment costs to assign lost sales such that lower cost warehouses get higher
        weights. The alpha parameter in the softmax controls the temperature (higher alpha = more uniform 
        distribution, lower alpha = more concentrated distribution).
        
        Args:
            unfulfilled_demand (np.ndarray): Unfulfilled demand per region. Shape: (n_regions, n_skus).
            shipments (np.ndarray): Shipment quantities per warehouse-region pair (unused for the cost-based handler).
                Shape: (n_warehouses, n_regions)
            
        Returns:
            lost_sales_assigned (np.ndarray): Lost sales assignment matrix. Shape: (n_warehouses, n_skus).
        """

        # Initialize lost sales assignment matrix
        lost_sales_assigned = np.zeros((self.n_warehouses, self.n_skus), dtype=float) # Shape: (n_warehouses, n_skus)
        
        # For each region, assign lost sales proportionally based on shipment costs
        for region_id in range(self.n_regions):
            # Compute total lost weight and orders of the region
            lost_orders = float(lost_order_counts[region_id])
            lost_weight = float(unfulfilled_demand[region_id, :].dot(self.sku_weights))

            # Get shipment costs of each warehouse to the current region
            costs = (
                self.fixed_cost_per_order[:, region_id] * lost_orders
                + self.variable_cost_per_weight[:, region_id] * lost_weight
            ) # Shape: (n_warehouses,)
            
            # Compute softmax weights based on inverse shipment costs
            logits = -costs / self.alpha 
            weights = softmax(logits) # Shape: (n_warehouses,)
            
            # Assign lost sales for each SKU according to the weights
            lost_sales_assigned += weights[:, None] * unfulfilled_demand[region_id, :] # Shape: (n_warehouses, n_skus)  

        return lost_sales_assigned

