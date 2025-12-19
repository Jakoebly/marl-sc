from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from src.config.schema import EnvironmentConfig


@dataclass
class EnvironmentContext:
    """
    Implements a shared environment context that is passed to all components. It contains
    common environment-level values to avoid duplicating parameters and provide a unified 
    way to pass a shared state.
    
    Attributes:
        n_warehouses (int): Number of warehouses.
        n_skus (int): Number of stock-keeping units.
        n_regions (int): Number of demand regions.
        holding_cost (Union[float, np.ndarray]): Holding cost rate(s). Shape: scalar or (n_warehouses,).
        penalty_cost (Union[float, np.ndarray]): Penalty cost rate(s). Shape: scalar or (n_skus,).
        shipment_cost (np.ndarray): Shipment costs per warehouse-region pair. Shape: (n_warehouses, n_regions).
        data_path (Optional[str]): Path to empirical data file (if using real_world data source). None if using synthetic data.
    """

    # Environment-level parameters
    n_warehouses: int
    n_skus: int
    n_regions: int

    # Cost structure parameters
    holding_cost: Union[float, np.ndarray]  # Scalar or (n_warehouses,)
    penalty_cost: Union[float, np.ndarray]  # Scalar or (n_skus,)
    shipment_cost: np.ndarray  # Shape: (n_warehouses, n_regions)

    # Data source parameters
    data_path: Optional[str] = None

def create_environment_context(env_config: EnvironmentConfig) -> EnvironmentContext:
    """
    Builds the EnvironmentContext from the EnvironmentConfig by converting the cost 
    structure to NumPy arrays/floats and extracting the data path if using a real-world 
    data source.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        
    Returns:
        environment_context (EnvironmentContext): Instantiated EnvironmentContext.
    """

    # Convert holding costs to NumPy array or float depending on its type 
    holding_cost = env_config.cost_structure.holding_cost
    if isinstance(holding_cost, list):
        holding_cost = np.array(holding_cost, dtype=float)  # Shape: (n_warehouses,)
    else:
        holding_cost = float(holding_cost)
    
    # Convert penalty costs to NumPy array or float depending on its type 
    penalty_cost = env_config.cost_structure.penalty_cost
    if isinstance(penalty_cost, list):
        penalty_cost = np.array(penalty_cost, dtype=float)  # Shape: (n_skus,)
    else:
        penalty_cost = float(penalty_cost)
    
    # Convert shipment costs to NumPy array
    shipment_cost = np.array(env_config.cost_structure.shipment_cost, dtype=float) # Shape: (n_warehouses, n_regions)
    
    # Extract data path if using a real-world data source
    data_path = None
    if env_config.data_source.type == "real_world":
        data_path = str(env_config.data_source.path)
    
    # Create the EnvironmentContext
    environment_context = EnvironmentContext(
        n_warehouses=env_config.n_warehouses,
        n_skus=env_config.n_skus,
        n_regions=env_config.n_regions,
        holding_cost=holding_cost,
        penalty_cost=penalty_cost,
        shipment_cost=shipment_cost,
        data_path=data_path
    )

    return environment_context

