"""Tests for algorithm wrappers and environment modifications."""

import numpy as np
import pytest
from gymnasium.spaces import Box

from src.config.loader import load_environment_config, load_algorithm_config
from src.config.schema import IPPOConfig, MAPPOConfig
from src.environment.environment import InventoryEnvironment
from src.algorithms.registry import get_algorithm
from src.algorithms.base import BaseAlgorithmWrapper


class TestEnvironmentActionRescaling:
    """Tests for environment action rescaling functionality."""
    
    def test_action_space_normalized(self):
        """Test that action space returns normalized [-1, 1] range."""
        config = load_environment_config("config_files/environments/base_env.yaml")
        env = InventoryEnvironment(config, seed=42)
        
        first_agent = env.agents[0]
        action_space = env.action_space(first_agent)

        assert isinstance(action_space, Box)
        assert action_space.low == pytest.approx(-1.0)
        assert action_space.high == pytest.approx(1.0)
        assert action_space.shape == (config.n_skus,)
        assert action_space.dtype == np.float32

        print(f"[OK] Action space shape: {action_space.shape}")
    
    def test_rescale_actions_to_quantities(self):
        """Test that actions are correctly rescaled from [-1, 1] to [0, max_order_quantity]."""
        config = load_environment_config("config_files/environments/base_env.yaml")
        env = InventoryEnvironment(config, seed=42)
        
        # Test with normalized actions
        actions = {
            agent_id: np.array([-1.0, 0.0, 1.0], dtype=np.float32)
            for agent_id in env.agents
        }
        print(f"[OK] Actions: {actions}")
        
        rescaled = env._rescale_actions_to_quantities(actions)
        print(f"[OK] Rescaled actions: {rescaled}")

        # Check that all agents have rescaled actions
        assert len(rescaled) == len(env.agents)
        
        # Check rescaling for first agent
        first_agent = env.agents[0]
        rescaled_action = rescaled[first_agent]
        
        # -1.0 should map to 0
        assert rescaled_action[0] == pytest.approx(0.0)
        # 0.0 should map to max_order_quantity / 2
        assert rescaled_action[1] == pytest.approx(config.max_order_quantity / 2.0)
        # 1.0 should map to max_order_quantity
        assert rescaled_action[2] == pytest.approx(float(config.max_order_quantity))
        
        # Check that values are integers (as float)
        assert all(isinstance(x, (int, float)) for x in rescaled_action)
        assert all(0 <= x <= config.max_order_quantity for x in rescaled_action)
    
    def test_rescale_actions_clipping(self):
        """Test that actions outside [-1, 1] are clipped correctly."""
        config = load_environment_config("config_files/environments/base_env.yaml")
        env = InventoryEnvironment(config, seed=42)
        
        # Test with actions outside [-1, 1] range
        actions = {
            agent_id: np.array([-2.0, 2.0], dtype=np.float32)
            for agent_id in env.agents
        }
        print(f"[OK] Actions: {actions}")
        
        rescaled = env._rescale_actions_to_quantities(actions)
        print(f"[OK] Rescaled actions: {rescaled}")
        
        first_agent = env.agents[0]
        rescaled_action = rescaled[first_agent]
        
        # Should be clipped to [0, max_order_quantity]
        assert all(0 <= x <= config.max_order_quantity for x in rescaled_action)


class TestEnvironmentGlobalState:
    """Tests for environment global state extraction."""
    
    def test_get_global_cstate(self):
        """Test that global state concatenates all agent observations."""
        config = load_environment_config("config_files/environments/base_env.yaml")
        env = InventoryEnvironment(config, seed=42)
        
        # Reset environment to get initial observations
        observations, _ = env.reset(seed=42)
        print(f"[OK] Observations: {observations}")
        print(f"[OK] Observations shape: {observations[env.agents[0]].shape}")
        
        # Get global state
        global_state = env.get_state()
        print(f"[OK] Global state: {global_state}")
        print(f"[OK] Global state shape: {global_state.shape}")
        
        # Check shape: should be (n_warehouses * obs_dim,)
        obs_dim = 2 * config.n_skus
        expected_shape = (config.n_warehouses * obs_dim,)
        assert global_state.shape == expected_shape
        
        # Check that global state contains all agent observations in order
        for i, agent_id in enumerate(env.agents):
            start_idx = i * obs_dim
            end_idx = (i + 1) * obs_dim
            agent_obs = global_state[start_idx:end_idx]
            assert np.allclose(agent_obs, observations[agent_id])
    
    def test_get_global_state_space(self):
        """Test that global state space has correct shape."""
        config = load_environment_config("config_files/environments/base_env.yaml")
        env = InventoryEnvironment(config, seed=42)
        
        global_observation_space = env.global_observation_space()
        
        assert isinstance(global_state_space, Box)
        obs_dim = 2 * config.n_skus
        expected_shape = (config.n_warehouses * obs_dim,)
        assert global_state_space.shape == expected_shape
        assert global_state_space.low == pytest.approx(0.0)
        assert global_state_space.high == pytest.approx(np.inf)
        assert global_state_space.dtype == np.float32


class TestAlgorithmConfigs:
    """Tests for algorithm configuration loading and validation."""
    
    def test_load_ippo_config(self):
        """Test loading IPPO config."""
        config = load_algorithm_config("config_files/algorithms/ippo.yaml")
        assert isinstance(config, IPPOConfig)
        assert config.name == "ippo"
        assert config.shared.learning_rate == 0.0003
        assert config.shared.batch_size == 4000
        assert config.algorithm_specific["vf_loss_coeff"] == 0.5
    
    def test_load_mappo_config(self):
        """Test loading MAPPO config."""
        config = load_algorithm_config("config_files/algorithms/mappo.yaml")
        assert isinstance(config, MAPPOConfig)
        assert config.name == "mappo"
        assert config.shared.learning_rate == 0.0003
        assert config.algorithm_specific["use_critic"] is True
    
    def test_load_happo_config(self):
        """Test loading HAPPO config."""
        config = load_algorithm_config("config_files/algorithms/happo.yaml")
        assert isinstance(config, HAPPOConfig)
        assert config.name == "happo"
        assert config.shared.learning_rate == 0.0003
        assert config.algorithm_specific["advantage_normalization"] is True


class TestAlgorithmWrappers:
    """Tests for algorithm wrapper initialization and basic functionality."""
    
    def test_ippo_wrapper_initialization(self):
        """Test IPPO wrapper can be initialized."""
        env_config = load_environment_config("config_files/environments/base_env.yaml")
        algorithm_config = load_algorithm_config("config_files/algorithms/ippo.yaml")
        
        env = InventoryEnvironment(env_config, seed=42)

        wrapper = get_algorithm("ippo", env, algorithm_config)
        
        assert isinstance(wrapper, BaseAlgorithmWrapper)
        assert wrapper.ippo_config.name == "ippo"
    
    def test_mappo_wrapper_initialization(self):
        """Test MAPPO wrapper can be initialized."""
        env_config = load_environment_config("config_files/environments/base_env.yaml")
        algorithm_config = load_algorithm_config("config_files/algorithms/mappo.yaml")
        
        env = InventoryEnvironment(env_config, seed=42)

        wrapper = get_algorithm("mappo", env, algorithm_config)
        
        assert isinstance(wrapper, BaseAlgorithmWrapper)
        assert wrapper.config.name == "mappo"
    
    def test_happo_wrapper_initialization(self):
        """Test HAPPO wrapper can be initialized."""
        env_config = load_environment_config("config_files/environments/base_env.yaml")
        algorithm_config = load_algorithm_config("config_files/algorithms/happo.yaml")
        
        env = InventoryEnvironment(env_config, seed=42)
        wrapper = get_algorithm("happo", env, algorithm_config)
        
        assert isinstance(wrapper, BaseAlgorithmWrapper)
        assert wrapper.config.name == "happo"
    
    def test_get_algorithm_unknown(self):
        """Test that unknown algorithm raises ValueError."""
        env_config = load_environment_config("config_files/environments/base_env.yaml")
        algorithm_config = load_algorithm_config("config_files/algorithms/ippo.yaml")
        
        env = InventoryEnvironment(env_config, seed=42)
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm("unknown_algorithm", env, algorithm_config)


class TestIntegration:
    """Integration tests for environment and algorithm interaction."""
    
    def test_env_step_with_normalized_actions(self):
        """Test that environment step works with normalized actions."""
        config = load_environment_config("config_files/environments/base_env.yaml")
        env = InventoryEnvironment(config, seed=42)
        
        # Reset environment
        observations, _ = env.reset(seed=42)
        print(f"[OK] Observations:\n {observations}\n")
        
        # Create normalized actions in [-1, 1] range
        actions = {
            agent_id: np.random.uniform(-1.0, 1.0, size=(config.n_skus,)).astype(np.float32)
            for agent_id in env.agents
        }
        print(f"[OK] Actions:\n {actions}\n")
        
        # Step environment
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"[OK] Next observations:\n {next_obs}\n")
        print(f"[OK] Rewards:\n {rewards}\n")
        print(f"[OK] Terminations:\n {terminations}\n")
        print(f"[OK] Truncations:\n {truncations}\n")
        print(f"[OK] Infos:\n {infos}\n")
        
        # Check that step completed successfully
        assert len(next_obs) == len(env.agents)
        assert len(rewards) == len(env.agents)
        assert len(terminations) == len(env.agents)
        assert len(truncations) == len(env.agents)
        assert len(infos) == len(env.agents)
        
        # Check that observations have correct shape
        for agent_id in env.agents:
            assert "local" in next_obs[agent_id]
            assert next_obs[agent_id]["local"].shape == (2 * config.n_skus,)
            assert "global" in next_obs[agent_id]
            assert next_obs[agent_id]["global"].shape == (config.n_warehouses * 2 * config.n_skus,)
    
    def test_algorithm_train_one_iteration(self):
        """Test that algorithm can train for one iteration (may require RLlib setup)."""
        env_config = load_environment_config("config_files/environments/base_env.yaml")
        algorithm_config = load_algorithm_config("config_files/algorithms/ippo.yaml")
        
        env = InventoryEnvironment(env_config, seed=42)
        wrapper = get_algorithm("ippo", env, algorithm_config)
        
        # Try to train for one iteration
        result = wrapper.train()
        print(f"[OK] Result: {result}")

        assert isinstance(result, dict)

        # Check that result contains expected keys
        assert "info" in result or "episode_reward_mean" in result or len(result) > 0
