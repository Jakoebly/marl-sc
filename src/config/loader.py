"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Type, Dict, Any, Union, Annotated, get_origin
from pydantic import BaseModel, ValidationError, TypeAdapter

from .schema import EnvironmentConfig, IPPOConfig, MAPPOConfig, HAPPOConfig, AlgorithmConfig


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
    
    # Validate the configuration dictionary against the EnvironmentConfig schema
    validated_config = validate_config(config_dict, EnvironmentConfig)
    
    return validated_config


def load_algorithm_config(path: str) -> Union[IPPOConfig, MAPPOConfig, HAPPOConfig]:
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

