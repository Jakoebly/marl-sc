"""
Test script for real-world data loading and integration.

This script tests:
1. Raw data loading and validation
2. Data preprocessing pipeline
3. Environment initialization with real-world data
4. Data flow from preprocessing → demand sampler → environment
5. Correctness of data at each stage
"""

import numpy as np
from pathlib import Path
from src.config.loader import load_environment_config
from src.environment.environment import InventoryEnvironment
from src.data.preprocessor import DataPreprocessor, RawDataLoader
from src.environment.components.demand_sampler import EmpiricalDemandSampler


def test_raw_data_loading():
    """Test that raw data files can be loaded and validated."""
    print("\n[TEST] Raw Data Loading")
    print("-" * 60)
    
    raw_data_path = Path("data_files/raw")
    loader = RawDataLoader(raw_data_path)
    
    # Load all files
    loader.load_all()
    print(f"[OK] All CSV files loaded")
    
    # Validate relationships
    loader.validate_relationships()
    print(f"[OK] Data relationships validated")
    
    # Check data shapes
    assert loader.orders_df is not None, "Orders DataFrame should be loaded"
    assert loader.order_sku_demand_df is not None, "Order-SKU demand DataFrame should be loaded"
    assert loader.skus_df is not None, "SKUs DataFrame should be loaded"
    assert loader.regions_df is not None, "Regions DataFrame should be loaded"
    assert loader.warehouses_df is not None, "Warehouses DataFrame should be loaded"
    
    print(f"  Orders: {len(loader.orders_df)} rows")
    print(f"  Order-SKU Demand: {len(loader.order_sku_demand_df)} rows")
    print(f"  SKUs: {len(loader.skus_df)} rows")
    print(f"  Regions: {len(loader.regions_df)} rows")
    print(f"  Warehouses: {len(loader.warehouses_df)} rows")
    
    # Check required columns
    assert 'regionid' in loader.orders_df.columns, "Orders should have 'regionid' column"
    assert 'sku_index' in loader.skus_df.columns, "SKUs should have 'sku_index' column"
    assert 'region_index' in loader.regions_df.columns, "Regions should have 'region_index' column"
    
    print(f"[OK] All required columns present")
    print("[PASS] Raw data loading test passed\n")


def test_data_preprocessing():
    """Test the data preprocessing pipeline."""
    print("\n[TEST] Data Preprocessing")
    print("-" * 60)
    
    raw_data_path = Path("data_files/raw")
    n_skus = 10
    n_warehouses = 5
    n_regions = 3
    seed = 42
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path=raw_data_path,
        n_skus=n_skus,
        n_warehouses=n_warehouses,
        n_regions=n_regions
    )
    
    # Run preprocessing
    preprocessed_data, shipment_costs, sku_weights, distances = preprocessor.preprocess(seed=seed)
    print(f"[OK] Preprocessing completed")
    
    # Check PreprocessedData structure
    assert preprocessed_data.demand_data is not None, "demand_data should be created"
    assert shipment_costs is not None, "shipment_costs should be created"
    assert sku_weights is not None, "sku_weights should be created"
    assert distances is not None, "distances should be created"
    
    # Check demand_data columns
    required_columns = ['timestep', 'region_id', 'order_id', 'sku_id', 'quantity']
    for col in required_columns:
        assert col in preprocessed_data.demand_data.columns, f"demand_data should have '{col}' column"
    print(f"[OK] demand_data has required columns: {required_columns}")
    
    # Check data types
    assert preprocessed_data.demand_data['timestep'].dtype == int, "timestep should be int"
    assert preprocessed_data.demand_data['region_id'].dtype == int, "region_id should be int"
    assert preprocessed_data.demand_data['sku_id'].dtype == int, "sku_id should be int"
    print(f"[OK] Data types are correct")
    
    # Check ID ranges (should be 0-indexed)
    assert preprocessed_data.demand_data['region_id'].min() >= 0, "region_id should be >= 0"
    assert preprocessed_data.demand_data['region_id'].max() < n_regions, f"region_id should be < {n_regions}"
    assert preprocessed_data.demand_data['sku_id'].min() >= 0, "sku_id should be >= 0"
    assert preprocessed_data.demand_data['sku_id'].max() < n_skus, f"sku_id should be < {n_skus}"
    print(f"[OK] ID ranges are correct (0-indexed, within selection)")
    
    # Check shipment_costs shape
    assert shipment_costs.fixed_per_order.shape == (n_warehouses, n_regions), \
        f"shipment_costs.fixed_per_order shape should be ({n_warehouses}, {n_regions}), got {shipment_costs.fixed_per_order.shape}"
    assert shipment_costs.variable_per_weight.shape == (n_warehouses, n_regions), \
        f"shipment_costs.variable_per_weight shape should be ({n_warehouses}, {n_regions}), got {shipment_costs.variable_per_weight.shape}"
    print(f"[OK] shipment_costs shape is correct: fixed={shipment_costs.fixed_per_order.shape}, variable={shipment_costs.variable_per_weight.shape}")
    
    # Check sku_weights shape
    assert sku_weights.shape == (n_skus,), \
        f"sku_weights shape should be ({n_skus},), got {sku_weights.shape}"
    print(f"[OK] sku_weights shape is correct: {sku_weights.shape}")
    
    # Check distances shape
    assert distances.shape == (n_warehouses, n_regions), \
        f"distances shape should be ({n_warehouses}, {n_regions}), got {distances.shape}"
    print(f"[OK] distances shape is correct: {distances.shape}")
    
    # Check that we have data
    assert len(preprocessed_data.demand_data) > 0, "demand_data should not be empty"
    unique_timesteps = preprocessed_data.demand_data['timestep'].nunique()
    print(f"  Unique timesteps: {unique_timesteps}")
    print(f"  Total demand rows: {len(preprocessed_data.demand_data)}")
    
    print("[PASS] Data preprocessing test passed\n")
    
    return preprocessed_data, shipment_costs, sku_weights, distances


def test_environment_with_real_world_data():
    """Test environment initialization with real-world data."""
    print("\n[TEST] Environment Initialization with Real-World Data")
    print("-" * 60)
    
    # Create a temporary config file for real-world data
    import tempfile
    import os
    import yaml
    
    config_content = {
        'environment': {
            'n_warehouses': 3,
            'n_skus': 10,
            'n_regions': 5,
            'episode_length': 2,
            'initial_inventory': {
                'type': 'uniform',
                'params': {
                    'min': 0,
                    'max': 100
                }
            },
            'cost_structure': {
                'holding_cost': 1.0,
                'penalty_cost': 10.0,
                'shipment_cost': [[5.0] * 5] * 3  # Will be overridden by preprocessed data
            },
            'components': {
                'demand_sampler': {
                    'type': 'empirical'
                },
                'demand_allocator': {
                    'type': 'greedy',
                    'params': {
                        'max_splits': 2
                    }
                },
                'lead_time_sampler': {
                    'type': 'uniform',
                    'params': {
                        'min': 1,
                        'max': 5
                    }
                },
                'lost_sales_handler': {
                    'type': 'cheapest',
                    'params': None
                },
                'reward_calculator': {
                    'type': 'cost',
                    'params': {
                        'scope': 'team',
                        'scale_factor': 1.0,
                        'normalize': False,
                        'cost_weights': [0.34, 0.33, 0.33]
                    }
                }
            },
            'data_source': {
                'type': 'real_world',
                'path': str(Path('data_files/raw').absolute())
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        # Load config
        env_config = load_environment_config(temp_config_path)
        print(f"[OK] Config loaded")
        
        # Create environment
        env = InventoryEnvironment(env_config, seed=42)
        print(f"[OK] Environment created")
        
        # Check that demand sampler is EmpiricalDemandSampler
        assert isinstance(env.demand_sampler, EmpiricalDemandSampler), \
            "Demand sampler should be EmpiricalDemandSampler for real-world data"
        print(f"[OK] Demand sampler is EmpiricalDemandSampler")
        
        # Check that preprocessed data exists in context
        assert env.demand_sampler.data is not None, "Demand sampler should have data"
        print(f"[OK] Demand sampler has data")
        
        # Check data structure
        assert 'timestep' in env.demand_sampler.data.columns, "Data should have 'timestep' column"
        assert 'region_id' in env.demand_sampler.data.columns, "Data should have 'region_id' column"
        assert 'sku_id' in env.demand_sampler.data.columns, "Data should have 'sku_id' column"
        assert 'quantity' in env.demand_sampler.data.columns, "Data should have 'quantity' column"
        print(f"[OK] Data has correct columns")
        
        # Check available timesteps
        assert len(env.demand_sampler.available_timesteps) > 0, "Should have available timesteps"
        assert len(env.demand_sampler.available_timesteps) >= env.episode_length, \
            f"Should have at least {env.episode_length} timesteps"
        print(f"[OK] Available timesteps: {len(env.demand_sampler.available_timesteps)}")
        
        print("[PASS] Environment initialization test passed\n")
        
        return env
        
    finally:
        os.unlink(temp_config_path)


def test_demand_sampler_data_flow():
    """Test that correct data flows from preprocessing to demand sampler."""
    print("\n[TEST] Demand Sampler Data Flow")
    print("-" * 60)
    
    # Create environment
    import tempfile
    import os
    import yaml
    
    config_content = {
        'environment': {
            'n_warehouses': 23,
            'n_skus': 500,
            'n_regions': 40,
            'episode_length': 10,
            'initial_inventory': {
                'type': 'uniform',
                'params': {'min': 0, 'max': 100}
            },
            'cost_structure': {
                'holding_cost': 1.0,
                'penalty_cost': 10.0,
                'shipment_cost': [[5.0] * 40] * 23
            },
            'components': {
                'demand_sampler': {
                    'type': 'empirical'
                },
                'demand_allocator': {
                    'type': 'greedy',
                    'params': {'max_splits': 2}
                },
                'lead_time_sampler': {
                    'type': 'uniform',
                    'params': {'min': 1, 'max': 5}
                },
                'lost_sales_handler': {
                    'type': 'cheapest',
                    'params': None
                },
                'reward_calculator': {
                    'type': 'cost',
                    'params': {
                        'scope': 'team',
                        'scale_factor': 1.0,
                        'normalize': False,
                        'cost_weights': [0.34, 0.33, 0.33]
                    }
                }
            },
            'data_source': {
                'type': 'real_world',
                'path': str(Path('data_files/raw').absolute())
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        env_config = load_environment_config(temp_config_path)
        env = InventoryEnvironment(env_config, seed=42)
        
        # Reset environment
        observations, infos = env.reset(seed=42)
        print(f"[OK] Environment reset")
        
        # Sample demand for multiple timesteps
        for timestep in range(20):
            orders = env.demand_sampler.sample(timestep)
            
            # Check orders structure
            assert isinstance(orders, list), "Orders should be a list"
            print(f"  Timestep {timestep}: {len(orders)} orders")
            
            for order in orders:
                # Check Order structure
                assert hasattr(order, 'region_id'), "Order should have region_id"
                assert hasattr(order, 'sku_demands'), "Order should have sku_demands"
                
                # Check region_id range
                assert 0 <= order.region_id < env.n_regions, \
                    f"region_id should be in [0, {env.n_regions}), got {order.region_id}"
                
                # Check sku_demands shape
                assert order.sku_demands.shape == (env.n_skus,), \
                    f"sku_demands shape should be ({env.n_skus},), got {order.sku_demands.shape}"
                
                # Check that quantities are non-negative
                assert np.all(order.sku_demands >= 0), "sku_demands should be non-negative"
        
        print(f"[OK] All orders have correct structure")
        print("[PASS] Demand sampler data flow test passed\n")
        
    finally:
        os.unlink(temp_config_path)


def test_environment_data_flow():
    """Test that correct data flows through the environment."""
    print("\n[TEST] Environment Data Flow")
    print("-" * 60)
    
    # Create environment
    import tempfile
    import os
    import yaml
    
    config_content = {
        'environment': {
            'n_warehouses': 3,
            'n_skus': 10,
            'n_regions': 5,
            'episode_length': 2,
            'initial_inventory': {
                'type': 'uniform',
                'params': {'min': 0, 'max': 100}
            },
            'cost_structure': {
                'holding_cost': 1.0,
                'penalty_cost': 10.0,
                'shipment_cost': [[5.0] * 5] * 3
            },
            'components': {
                'demand_sampler': {
                    'type': 'empirical'
                },
                'demand_allocator': {
                    'type': 'greedy',
                    'params': {'max_splits': 2}
                },
                'lead_time_sampler': {
                    'type': 'uniform',
                    'params': {'min': 1, 'max': 5}
                },
                'lost_sales_handler': {
                    'type': 'cheapest',
                    'params': None
                },
                'reward_calculator': {
                    'type': 'cost',
                    'params': {
                        'scope': 'team',
                        'scale_factor': 1.0,
                        'normalize': False,
                        'cost_weights': [0.34, 0.33, 0.33]
                    }
                }
            },
            'data_source': {
                'type': 'real_world',
                'path': str(Path('data_files/raw').absolute())
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        env_config = load_environment_config(temp_config_path)
        env = InventoryEnvironment(env_config, seed=42)
        
        # Reset environment
        observations, infos = env.reset(seed=42)
        print(f"[OK] Environment reset")
        
        # Check initial state
        assert env.inventory.shape == (env.n_warehouses, env.n_skus), \
            f"Inventory shape should be ({env.n_warehouses}, {env.n_skus})"
        assert env.timestep == 0, "Initial timestep should be 0"
        print(f"[OK] Initial state correct")
        
        # Run a few steps
        for step in range(5):
            # Sample actions
            actions = {
                agent_id: np.random.uniform(0, 10, env.n_skus) 
                for agent_id in env.agents
            }
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check observations
            assert len(observations) == env.n_warehouses, \
                f"Should have {env.n_warehouses} observations"
            
            for agent_id, obs in observations.items():
                assert obs.shape == (2 * env.n_skus,), \
                    f"Observation shape should be ({2 * env.n_skus},), got {obs.shape}"
            
            # Check rewards
            assert len(rewards) == env.n_warehouses, \
                f"Should have {env.n_warehouses} rewards"
            
            # Check that timestep incremented
            assert env.timestep == step + 1, \
                f"Timestep should be {step + 1}, got {env.timestep}"
            
            print(f"  Step {step + 1}: timestep={env.timestep}, rewards={rewards}")
        
        print(f"[OK] Environment steps executed correctly")
        print("[PASS] Environment data flow test passed\n")
        
    finally:
        os.unlink(temp_config_path)


def test_data_consistency():
    """Test data consistency across the pipeline."""
    print("\n[TEST] Data Consistency")
    print("-" * 60)
    
    # Preprocess data
    raw_data_path = Path("data_files/raw")
    preprocessor = DataPreprocessor(
        raw_data_path=raw_data_path,
        n_skus=10,
        n_warehouses=3,
        n_regions=5
    )
    preprocessed_data, shipment_costs, sku_weights, distances = preprocessor.preprocess(seed=42)
    
    # Create environment with same config
    import tempfile
    import os
    import yaml
    
    config_content = {
        'environment': {
            'n_warehouses': 3,
            'n_skus': 10,
            'n_regions': 5,
            'episode_length': 2,
            'initial_inventory': {
                'type': 'uniform',
                'params': {'min': 0, 'max': 100}
            },
            'cost_structure': {
                'holding_cost': 1.0,
                'penalty_cost': 10.0,
                'shipment_cost': [[5.0] * 5] * 3
            },
            'components': {
                'demand_sampler': {
                    'type': 'empirical'
                },
                'demand_allocator': {
                    'type': 'greedy',
                    'params': {'max_splits': 2}
                },
                'lead_time_sampler': {
                    'type': 'uniform',
                    'params': {'min': 1, 'max': 5}
                },
                'lost_sales_handler': {
                    'type': 'cheapest',
                    'params': None
                },
                'reward_calculator': {
                    'type': 'cost',
                    'params': {
                        'scope': 'team',
                        'scale_factor': 1.0,
                        'normalize': False,
                        'cost_weights': [0.34, 0.33, 0.33]
                    }
                }
            },
            'data_source': {
                'type': 'real_world',
                'path': str(Path('data_files/raw').absolute())
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        env_config = load_environment_config(temp_config_path)
        env = InventoryEnvironment(env_config, seed=42)
        
        # Check that data matches
        assert env.demand_sampler.data.equals(preprocessed_data.demand_data), \
            "Demand sampler data should match preprocessed data"
        print(f"[OK] Data matches between preprocessing and demand sampler")
        
        # Check shipment costs match (accessed via demand_allocator which has context)
        np.testing.assert_array_equal(
            env.demand_allocator.shipment_costs,
            shipment_costs.fixed_per_order
        ), "Shipment costs should match"
        print(f"[OK] Shipment costs match")
        
        print("[PASS] Data consistency test passed\n")
        
    finally:
        os.unlink(temp_config_path)


if __name__ == "__main__":
    print("=" * 60)
    print("REAL-WORLD DATA LOADING TESTS")
    print("=" * 60)
    
    try:
        test_raw_data_loading()
        preprocessed_data, shipment_costs, sku_weights, distances = test_data_preprocessing()
        env = test_environment_with_real_world_data()
        test_demand_sampler_data_flow()
        test_environment_data_flow()
        test_data_consistency()
        
        print("=" * 60)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

