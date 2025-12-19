import pytest
import tempfile
import os
from pathlib import Path

from src.config.loader import (
    load_yaml,
    validate_config,
    load_environment_config,
    load_algorithm_config,
    ConfigFileError,
    ConfigValidationError
)
from src.config.schema import (
    EnvironmentConfig,
    AlgorithmConfig,
    ComponentConfig,
    InitialInventoryConfig,
    CostStructureConfig
)


class TestYAMLLoader:
    """Tests for YAML loading functionality."""
    
    def test_load_valid_yaml(self):
        """Test loading a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            temp_path = f.name
        
        try:
            result = load_yaml(temp_path)
            assert result == {"key": "value", "number": 42}
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(ConfigFileError):
            load_yaml("nonexistent_file.yaml")
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigFileError):
                load_yaml(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_empty_yaml(self):
        """Test loading empty YAML file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            result = load_yaml(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_validate_valid_component_config(self):
        """Test validating a valid component config."""
        config_dict = {
            "type": "poisson",
            "params": {"lambda": 5.0}
        }
        config = validate_config(config_dict, ComponentConfig)
        assert config.type == "poisson"
        assert config.params == {"lambda": 5.0}
    
    def test_validate_invalid_type(self):
        """Test validation fails with wrong type."""
        config_dict = {
            "type": 123,  # Should be string
            "params": {}
        }
        with pytest.raises(ConfigValidationError):
            validate_config(config_dict, ComponentConfig)
    
    def test_validate_missing_required_field(self):
        """Test validation fails with missing required field."""
        config_dict = {
            "params": {}  # Missing 'type'
        }
        with pytest.raises(ConfigValidationError):
            validate_config(config_dict, ComponentConfig)
    
    def test_validate_unknown_field(self):
        """Test validation rejects unknown fields."""
        config_dict = {
            "type": "poisson",
            "params": {},
            "unknown_field": "should_fail"  # Not allowed
        }
        with pytest.raises(ConfigValidationError):
            validate_config(config_dict, ComponentConfig)
    
    def test_validate_cost_structure_constraints(self):
        """Test cost structure validation with constraints."""
        # Valid config
        valid_config = {
            "holding_cost": 1.0,
            "penalty_cost": 10.0,
            "shipment_cost": 5.0
        }
        config = validate_config(valid_config, CostStructureConfig)
        assert config.holding_cost == 1.0
        
        # Invalid: negative cost
        invalid_config = {
            "holding_cost": -1.0,  # Should be > 0
            "penalty_cost": 10.0,
            "shipment_cost": 5.0
        }
        with pytest.raises(ConfigValidationError):
            validate_config(invalid_config, CostStructureConfig)


class TestEnvironmentConfigLoading:
    """Tests for environment config loading."""
    
    def test_load_valid_environment_config(self):
        """Test loading a valid environment config."""
        config_content = """
environment:
  n_warehouses: 3
  n_skus: 10
  n_regions: 3
  initial_inventory:
    method: "uniform"
    params:
      min: 0
      max: 100
  cost_structure:
    holding_cost: 1.0
    penalty_cost: 10.0
    shipment_cost: 5.0
  components:
    demand_sampler:
      type: "poisson"
      params: {}
    demand_allocator:
      type: "greedy"
      params: {}
    lead_time_sampler:
      type: "discrete"
      params: {}
    lost_sales_handler:
      type: "simple"
      params: {}
    reward_calculator:
      type: "standard"
      params: {}
  data_source:
    type: "synthetic"
    path: null
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = load_environment_config(temp_path)
            assert config.n_warehouses == 3
            assert config.n_skus == 10
            assert config.components.demand_sampler.type == "poisson"
        finally:
            os.unlink(temp_path)
    
    def test_validate_regions_constraint(self):
        """Test that n_regions >= n_warehouses constraint is enforced."""
        config_content = """
environment:
  n_warehouses: 5
  n_skus: 10
  n_regions: 3  # Less than n_warehouses, should fail
  initial_inventory:
    method: "uniform"
    params: {}
  cost_structure:
    holding_cost: 1.0
    penalty_cost: 10.0
    shipment_cost: 5.0
  components:
    demand_sampler:
      type: "poisson"
      params: {}
    demand_allocator:
      type: "greedy"
      params: {}
    lead_time_sampler:
      type: "discrete"
      params: {}
    lost_sales_handler:
      type: "simple"
      params: {}
    reward_calculator:
      type: "standard"
      params: {}
  data_source:
    type: "synthetic"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError):
                load_environment_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestAlgorithmConfigLoading:
    """Tests for algorithm config loading."""
    
    def test_load_valid_algorithm_config(self):
        """Test loading a valid algorithm config."""
        config_content = """
algorithm:
  name: "qmix"
  rllib_config:
    learning_rate: 0.0005
    batch_size: 32
    gamma: 0.99
    num_workers: 4
  training:
    num_iterations: 1000
    checkpoint_freq: 100
    evaluation_freq: 50
    evaluation_num_episodes: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = load_algorithm_config(temp_path)
            assert config.name == "qmix"
            assert config.rllib_config.learning_rate == 0.0005
            assert config.training.num_iterations == 1000
        finally:
            os.unlink(temp_path)
    

