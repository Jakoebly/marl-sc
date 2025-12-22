from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.random import SeedSequence

from src.config.schema import EnvironmentConfig

if TYPE_CHECKING:
    from src.data.preprocessor import PreprocessedData


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
        episode_length (int): Maximum number of timesteps per episode.
        holding_cost (Union[float, np.ndarray]): Holding cost rate(s). Shape: scalar or (n_warehouses,).
        penalty_cost (Union[float, np.ndarray]): Penalty cost rate(s). Shape: scalar or (n_skus,).
        shipment_cost (np.ndarray): Shipment costs per warehouse-region pair. Shape: (n_warehouses, n_regions).
        preprocessed_data (Optional[PreprocessedData]): Preprocessed data if using real_world data source. None if using synthetic data.
    """

    # Environment-level parameters
    n_warehouses: int
    n_skus: int
    n_regions: int
    episode_length: int

    # Cost structure parameters
    holding_cost: Union[float, np.ndarray]  # Scalar or (n_warehouses,)
    penalty_cost: Union[float, np.ndarray]  # Scalar or (n_skus,)
    shipment_cost: np.ndarray  # Shape: (n_warehouses, n_regions)

    # Data parameters
    preprocessed_data: Optional['PreprocessedData'] = None

def create_environment_context(
    env_config: EnvironmentConfig, 
    preprocessing_seed: Optional[SeedSequence] = None
) -> EnvironmentContext:
    """
    Builds the EnvironmentContext from the EnvironmentConfig by converting the cost 
    structure to NumPy arrays/floats and preprocessing data if using a real-world 
    data source.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        preprocessing_seed (Optional[SeedSequence]): Pre-spawned seed for preprocessing.
        
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
    
    # Initialize shipment_cost and preprocessed_data
    shipment_cost = None
    preprocessed_data = None
    
    # Handle and preprocess real-world data
    if env_config.data_source.type == "real_world":
        # Get raw data path
        raw_data_path = str(env_config.data_source.path.parent if env_config.data_source.path.is_file() else env_config.data_source.path)
        
        # Create preprocessor and run preprocessing
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(
            raw_data_path=raw_data_path,
            n_skus=env_config.n_skus,
            n_warehouses=env_config.n_warehouses,
            n_regions=env_config.n_regions
        )
        seed = preprocessing_seed.entropy if preprocessing_seed is not None else None
        preprocessed_data = preprocessor.preprocess(seed=seed)
        
        # Extract shipment costs from preprocessed data (override config)
        shipment_cost = preprocessed_data.shipment_costs
   
    # Handle synthetic data
    else:
        shipment_cost = np.array(env_config.cost_structure.shipment_cost, dtype=float)  # Shape: (n_warehouses, n_regions)
    
    # Create the EnvironmentContext
    environment_context = EnvironmentContext(
        n_warehouses=env_config.n_warehouses,
        n_skus=env_config.n_skus,
        n_regions=env_config.n_regions,
        episode_length=env_config.episode_length,
        holding_cost=holding_cost,
        penalty_cost=penalty_cost,
        shipment_cost=shipment_cost,
        preprocessed_data=preprocessed_data
    )

    return environment_context

