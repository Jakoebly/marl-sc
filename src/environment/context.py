from dataclasses import dataclass
from typing import Optional, Union, Tuple, TYPE_CHECKING

import numpy as np
from numpy.random import SeedSequence

from src.config.schema import EnvironmentConfig, CostStructureConfig

if TYPE_CHECKING:
    from src.data.preprocessor import PreprocessedData
    from src.utils.seed_manager import SeedManager


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
    data_mode: str = "train" 

def preprocess_real_world_data(
    env_config: EnvironmentConfig,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Optional['PreprocessedData']]:
    """
    Preprocesses real-world data and extracts shipment costs.
    Handles data splitting if data_split config is provided.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        seed (Optional[int]): Pre-spawned seed for preprocessing.
        
    Returns:
        shipment_cost (np.ndarray): Shipment costs matrix. Shape: (n_warehouses, n_regions).
        preprocessed_data (Optional['PreprocessedData']): PreprocessedData instance containing both train and val DataFrames.
    """
       
    # Get raw data path
    raw_data_path = str(
        env_config.data_source.path.parent 
        if env_config.data_source.path.is_file() 
        else env_config.data_source.path
    )
    
    # Create preprocessor and run preprocessing
    from src.data.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor(
        raw_data_path=raw_data_path,
        n_skus=env_config.n_skus,
        n_warehouses=env_config.n_warehouses,
        n_regions=env_config.n_regions
    )

    # Extract data_split from data_source if it is real_world
    from src.config.schema import DataSourceRealWorld
    data_split = None
    if isinstance(env_config.data_source, DataSourceRealWorld):
        data_split = env_config.data_source.data_split
    
    preprocessed_data, shipment_cost = preprocessor.preprocess(
        data_split=data_split,
        seed=seed
    )
    
    return shipment_cost, preprocessed_data

def convert_cost_structure(
    cost_structure: CostStructureConfig
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts cost structure from config format to runtime format.
    
    Args:
        cost_structure (CostStructureConfig): Cost structure configuration.
        
    Returns:
        holding_cost (Union[float, np.ndarray]): Holding cost rate(s). Shape: scalar or (n_warehouses,).
        penalty_cost (Union[float, np.ndarray]): Penalty cost rate(s). Shape: scalar or (n_skus,).
        Tuple of (holding_cost, penalty_cost)
    """

    # Convert holding costs to NumPy array or float depending on its type 
    holding_cost = cost_structure.holding_cost
    if isinstance(holding_cost, list):
        holding_cost = np.array(holding_cost, dtype=float) # Shape: (n_warehouses,)
    else:
        holding_cost = float(holding_cost)
    
    # Convert penalty costs to NumPy array or float depending on its type 
    penalty_cost = cost_structure.penalty_cost
    if isinstance(penalty_cost, list):
        penalty_cost = np.array(penalty_cost, dtype=float) # Shape: (n_skus,)
    else:
        penalty_cost = float(penalty_cost)
    
    return holding_cost, penalty_cost

def create_environment_context(
    env_config: EnvironmentConfig, 
    seed_manager: Optional['SeedManager'] = None,
    data_mode: str = "train"
) -> EnvironmentContext:
    """
    Builds the EnvironmentContext from the EnvironmentConfig by converting the cost 
    structure to NumPy arrays/floats and preprocessing data if using a real-world 
    data source.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        seed_manager (Optional['SeedManager']): Seed manager to get preprocessing seed from.
        data_mode (str): Data mode that determines which dataset to use. Defaults to "train".
        
    Returns:
        environment_context (EnvironmentContext): Instantiated EnvironmentContext.
    """

    # Extract preprocessing seed from seed_manager if provided
    preprocessing_seed = None
    if seed_manager is not None:
        preprocessing_seed = seed_manager.get_seed_int('preprocessing')

    # Convert costs to NumPy array or float depending on its type 
    holding_cost, penalty_cost = convert_cost_structure(env_config.cost_structure)
    
    # Handle real-world data
    if env_config.data_source.type == "real_world":
        shipment_cost, preprocessed_data = preprocess_real_world_data(env_config, seed=preprocessing_seed)
    
    # Handle synthetic data
    else:
        shipment_cost = np.array(env_config.cost_structure.shipment_cost, dtype=float) # Shape: (n_warehouses, n_regions)
        preprocessed_data = None
    
    # Create the EnvironmentContext
    environment_context = EnvironmentContext(
        n_warehouses=env_config.n_warehouses,
        n_skus=env_config.n_skus,
        n_regions=env_config.n_regions,
        episode_length=env_config.episode_length,
        holding_cost=holding_cost,
        penalty_cost=penalty_cost,
        shipment_cost=shipment_cost,
        preprocessed_data=preprocessed_data,
        data_mode=data_mode
    )

    return environment_context

