from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np

from src.environment.context import EnvironmentContext
from src.config.schema import RewardCalculatorConfig


class BaseRewardCalculator(ABC):
    """
    Implements a base class for reward calculator components. Reward calculators compute rewards 
    for each warehouse based on given environment parameters. All reward calculators must inherit 
    from this class and implement the calculate() methods.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: RewardCalculatorConfig):
        """
        Initializes common attributes for all reward calculators.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (RewardCalculatorConfig): Reward calculator configuration.
        """

        # Store general environment parameters
        self.n_warehouses = context.n_warehouses
        self.n_skus = context.n_skus
        self.n_regions = context.n_regions
        self.holding_cost = context.holding_cost
        self.penalty_cost = context.penalty_cost
        self.outbound_fixed_cost_per_order = context.shipment_cost.outbound_fixed
        self.outbound_variable_cost_per_weight = context.shipment_cost.outbound_variable
        self.inbound_fixed_cost_per_order = context.shipment_cost.inbound_fixed
        self.inbound_variable_cost_per_weight = context.shipment_cost.inbound_variable
        self.sku_weights = context.sku_weights
    
    @abstractmethod
    def calculate(self,
                  inventory: np.ndarray,  
                  lost_sales: np.ndarray,
                  shipment_counts: np.ndarray,
                  shipment_quantities_by_sku: np.ndarray,
                  ) -> np.ndarray:
        """
        Abstract method to calculate rewards for each warehouse.
        
        Args:
            inventory (np.ndarray): Current inventory levels. Shape: (n_warehouses, n_skus)
            lost_sales (np.ndarray): Lost sales assigned to each warehouse. Shape: (n_warehouses, n_skus)
            shipment_counts (np.ndarray): Number of shipments per warehouse-region pair.
                Shape: (n_warehouses, n_regions)
            shipment_quantities_by_sku (np.ndarray): Units shipped per warehouse-region-SKU combination.
                Shape: (n_warehouses, n_regions, n_skus)
        
        Returns:
            rewards (np.ndarray): Reward array. Shape: (n_warehouses,).
        """
        pass


class CostRewardCalculator(BaseRewardCalculator):
    """
    Implements a cost-based reward calculator that calculates rewards as the sum of negative weighted 
    supply chain costs. Supports team/agent scope and optional normalization/scaling.
    """
    
    def __init__(self, context: EnvironmentContext, component_config: RewardCalculatorConfig):
        """
        Initializes the cost-based reward calculator.
        
        Args:
            context (EnvironmentContext): Shared environment context.
            component_config (RewardCalculatorConfig): Reward calculator configuration.
        """

        # Initialize base class
        super().__init__(context, component_config)
        
        # Normalize SKU weights for reward calculation
        self.sku_weights_norm = self.sku_weights / self.sku_weights.mean() # Shape: (n_skus,)

        # Extract reward parameters from configuration
        params = component_config.params

        # Handle both dict and Pydantic model access
        if hasattr(params, "scope"):
            self.scope = params.scope
            self.scale_factor = params.scale_factor
            self.normalize = params.normalize
            self.cost_weights = np.array(params.cost_weights, dtype=float)
        else:
            self.scope = params["scope"]
            self.scale_factor = params["scale_factor"]
            self.normalize = params["normalize"]
            self.cost_weights = np.array(params["cost_weights"], dtype=float)
    
    def calculate(
        self, 
        inventory: np.ndarray, 
        ordered_skus: np.ndarray, 
        lost_sales: np.ndarray, 
        shipment_counts: np.ndarray, 
        shipment_quantities_by_sku: np.ndarray,
        ) -> np.ndarray:
        """
        Calculates rewards as the weighted sum of negative supply chain costs. The calculator computes the reward by 
        performing the following steps:

            1. Compute total holding, penalty, and shipment costs for each warehouse.
            2. Compute the weighted sum of the costs for each warehouse.
            3. If enabled, normalize and/or scale weighted total costs (both not implemented yet).
            4. Convert weighted total costs to rewards (negative costs). 
            5. If shared reward is enabled, sum rewards across all warehouses and broadcast to all warehouses.
        
        Args:
            inventory (np.ndarray): Current inventory levels. Shape: (n_warehouses, n_skus)
            ordered_skus (np.ndarray): Ordered SKUs by each warehouse. Shape: (n_warehouses, n_skus)
            lost_sales (np.ndarray): Lost sales assigned to each warehouse. Shape: (n_warehouses, n_skus)
            shipment_counts (np.ndarray): Number of shipments per warehouse-region pair. 
                Shape: (n_warehouses, n_regions)
            shipment_quantities_by_sku (np.ndarray): Units shipped per warehouse-region-SKU combination.
                Shape: (n_warehouses, n_regions, n_skus)
            
        Returns:
            rewards (np.ndarray): Reward array. Shape: (n_warehouses,).
        """

        # Compute holding costs for each warehouse
        if isinstance(self.holding_cost, np.ndarray):  # Per-SKU holding costs
            holding_costs_total = (inventory * self.holding_cost[np.newaxis, :]).sum(axis=1) # Shape: (n_warehouses,)
        else:  # Scalar holding costs
            holding_costs_total = (inventory * self.sku_weights[np.newaxis, :] * self.holding_cost).sum(axis=1) # Shape: (n_warehouses,)
        
        # Compute penalty costs for each warehouse
        if isinstance(self.penalty_cost, np.ndarray): # Per-SKU penalty costs
            penalty_costs_total = (lost_sales * self.penalty_cost[np.newaxis, :]).sum(axis=1) # Shape: (n_warehouses,)
        else: # Scalar penalty costs
            penalty_costs_total = (lost_sales * self.sku_weights[np.newaxis, :] * self.penalty_cost).sum(axis=1) # Shape: (n_warehouses,)
        
        # Compute outbound shipment costs for each warehouse
        outbound_fixed_costs_total = (shipment_counts * self.outbound_fixed_cost_per_order).sum(axis=1) # Shape: (n_warehouses,)
        outbound_shipment_weights = (shipment_quantities_by_sku * self.sku_weights[np.newaxis, np.newaxis, :]).sum(axis=2) # Shape: (n_warehouses, n_regions)
        outbound_variable_costs_total = (outbound_shipment_weights * self.outbound_variable_cost_per_weight).sum(axis=1) # Shape: (n_warehouses,)

        # Compute inbound shipment costs for each warehouse
        order_counts = (ordered_skus > 0).astype(int) # Shape: (n_warehouses, n_skus)
        inbound_fixed_costs_total = (order_counts * self.inbound_fixed_cost_per_order.T).sum(axis=1) # Shape: (n_warehouses,)
        inbound_shipment_weights = (ordered_skus * self.sku_weights[np.newaxis, :])	 # Shape: (n_warehouses, n_skus)
        inbound_variable_costs_total = (inbound_shipment_weights * self.inbound_variable_cost_per_weight.T).sum(axis=1) # Shape: (n_warehouses,)
        
        # Total shipment costs
        outbound_shipment_costs_total = outbound_fixed_costs_total + outbound_variable_costs_total # Shape: (n_warehouses,)
        inbound_shipment_costs_total = inbound_fixed_costs_total + inbound_variable_costs_total # Shape: (n_warehouses,)
        
        # Compute weighted total costs for each warehouse
        #costs_per_warehouse = (
        #    self.cost_weights[0] * holding_costs_total +
        #    self.cost_weights[1] * penalty_costs_total +
        #    self.cost_weights[2] * outbound_shipment_costs_total +
        #    self.cost_weights[3] * inbound_shipment_costs_total
        #) # Shape: (n_warehouses,)
        
        # Compute unweighted total costs for each warehouse
        costs_per_warehouse = (
            holding_costs_total +
            penalty_costs_total +
            outbound_shipment_costs_total +
            inbound_shipment_costs_total
        ) # Shape: (n_warehouses,)

        # Normalize if enabled
        if self.normalize:
            pass # TODO: Implement normalization

        # Apply scale factor if enabled
        if self.scale_factor:
            pass # TODO: Implement scaling

        # Store individual cost components for visualization (read via env.reward_calculator._cost_breakdown)
        self._cost_breakdown = {
            "holding_cost": holding_costs_total.copy(),
            "penalty_cost": penalty_costs_total.copy(),
            "outbound_shipment_cost": outbound_shipment_costs_total.copy(),
            "inbound_shipment_cost": inbound_shipment_costs_total.copy(),
        }

        # Convert to rewards (negative costs)
        rewards = -costs_per_warehouse # Shape: (n_warehouses,)
        
        # If team scope is enabled, sum per-warehouse rewards and use as reward for all warehouses
        if self.scope == "team":
            total_reward = rewards.sum() 
            rewards = np.full(self.n_warehouses, total_reward)
        
        return rewards

