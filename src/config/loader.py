"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Type, Dict, Any, Union, Annotated, get_origin
from pydantic import BaseModel, ValidationError, TypeAdapter

from .schema import EnvironmentConfig, IPPOConfig, MAPPOConfig, AlgorithmConfig, TuneConfig


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigFileError(ConfigError):
    """Exception raised when config file cannot be loaded."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when config validation fails."""
    pass


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Loads a YAML file and returns its contents as a dictionary.
    
    Args:
        path (str): Path to YAML file
        
    Returns:
        data (dict): Dictionary containing YAML contents
    """

    # Convert path to Path object
    path_obj = Path(path)

    # Check if path exists
    if not path_obj.exists():
        raise ConfigFileError(f"Config file not found: {path}")
    
    # Check if file exists
    if not path_obj.is_file():
        raise ConfigFileError(f"Path is not a file: {path}")
    
    # load the YAML file
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except yaml.YAMLError as e:
        raise ConfigFileError(f"Error parsing YAML file {path}: {e}")
    except Exception as e:
        raise ConfigFileError(f"Error reading config file {path}: {e}")


def validate_config(config_dict: Dict[str, Any], schema: Union[Type[BaseModel], Type[Any]]) -> BaseModel:
    """
    Validates a configuration dictionary against a Pydantic schema.

    Args:
        config_dict (dict): Configuration dictionary to validate.
        schema (Union[Type[BaseModel], Type[Any]]): Pydantic model class or Annotated type to validate against.
        
    Returns:
        validated_config (BaseModel): Validated Pydantic model instance.
    """

    # Validate the configuration dictionary against the schema
    try:
        adapter = TypeAdapter(schema)
        return adapter.validate_python(config_dict)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_messages.append(f"{field_path}: {error['msg']}")
        raise ConfigValidationError(
            f"Configuration validation failed:\n" + "\n".join(error_messages)
        )


def load_environment_config(path: str) -> EnvironmentConfig:
    """
    Loads and validates environment configuration from a YAML file.
    
    When data_source.type is "synthetic", cost structures (penalty_cost, sku_weights,
    distances, shipment costs) are automatically generated to match the environment
    dimensions (n_warehouses, n_skus, n_regions) before validation.
    
    Args:
        path (str): Path to environment config YAML file
        
    Returns:
        validated_config (EnvironmentConfig): Validated EnvironmentConfig instance.
    """
    
    # Load the configuration dictionary from the YAML file
    config_dict = load_yaml(path)

    # Extract 'environment' key if present
    if 'environment' in config_dict:
        config_dict = config_dict['environment']
    
    # Auto-generate costs for synthetic data mode
    config_dict = _apply_synthetic_costs(config_dict)

    # Validate the configuration dictionary against the EnvironmentConfig schema
    validated_config = validate_config(config_dict, EnvironmentConfig)

    return validated_config


def _apply_synthetic_costs(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates cost structures matching the environment dimensions and 
    injects them into the config dictionary if data_source.type is "synthetic".
    
    Args:
        config_dict (Dict[str, Any]): Raw environment config dictionary.
        
    Returns:
        config_dict (Dict[str, Any]): Config dictionary with generated costs applied.
    """
    from src.utils.cost_generator import generate_synthetic_costs

    # Check if data source is synthetic
    data_source = config_dict.get("data_source", {})
    if not isinstance(data_source, dict) or data_source.get("type") != "synthetic":
        return config_dict

    # Get environment dimensions
    n_warehouses = config_dict.get("n_warehouses", 3)
    n_skus = config_dict.get("n_skus", 5)
    n_regions = config_dict.get("n_regions", n_warehouses)

    # Generate costs matching dimensions
    costs = generate_synthetic_costs(n_warehouses, n_skus, n_regions)

    # Ensure cost_structure and shipment_cost dicts exist
    config_dict.setdefault("cost_structure", {})
    config_dict["cost_structure"].setdefault("shipment_cost", {})

    # Apply generated costs
    config_dict["cost_structure"]["penalty_cost"] = costs["penalty_cost"]
    config_dict["cost_structure"]["sku_weights"] = costs["sku_weights"]
    config_dict["cost_structure"]["distances"] = costs["distances"]
    config_dict["cost_structure"]["shipment_cost"]["outbound_fixed"] = costs["outbound_fixed"]
    config_dict["cost_structure"]["shipment_cost"]["outbound_variable"] = costs["outbound_variable"]
    config_dict["cost_structure"]["shipment_cost"]["inbound_fixed"] = costs["inbound_fixed"]
    config_dict["cost_structure"]["shipment_cost"]["inbound_variable"] = costs["inbound_variable"]

    return config_dict


def load_algorithm_config(path: str) -> Union[IPPOConfig, MAPPOConfig]:
    """
    Loads and validates algorithm configuration from a YAML file.
    
    Args:
        path (str): Path to algorithm config YAML file
        
    Returns:
        validated_config: Validated algorithm config.
    """

    # Load the configuration dictionary from the YAML file
    config_dict = load_yaml(path)
    
    # Extract 'algorithm' key if present
    if 'algorithm' in config_dict:
        config_dict = config_dict['algorithm']
    
    # Validate the configuration dictionary against the AlgorithmConfig discriminated union
    validated_config = validate_config(config_dict, AlgorithmConfig)
    
    return validated_config


def load_tune_config(path: str) -> TuneConfig:
    """
    Loads and validates tune configuration from a YAML file.
    
    Args:
        path (str): Path to tune config YAML file
        
    Returns:
        validated_config (TuneConfig): Validated TuneConfig instance.
    """

    # Load the configuration dictionary from the YAML file
    config_dict = load_yaml(path)
    
    # Extract 'search_space' key if present (for backward compatibility)
    if 'search_space' in config_dict:
        config_dict = config_dict['search_space']
    
    # Validate the configuration dictionary against the TuneConfig schema
    validated_config = validate_config(config_dict, TuneConfig)
    
    return validated_config

