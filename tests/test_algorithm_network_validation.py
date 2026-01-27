"""
Intensive tests for algorithm and network configuration validation.

This test suite validates:
- MLP network configurations
- GRU network configurations (including dropout and layer constraints)
- Shared algorithm parameter validation
- PPO parameter validation
- ActorCritic configuration validation
- IPPO/MAPPO algorithm configuration validation
- GRU-specific batch size and minibatch constraints
"""

import pytest
from pydantic import ValidationError, TypeAdapter

from src.config.schema import (
    MLPConfig,
    NetworkMLP,
    GRUConfig,
    NetworkGRU,
    NetworkConfig,
    OptionalSharedLayers,
    SharedAlgorithmConfig,
    ActorCriticConfig,
    PPOConfig,
    IPPOSpecificConfig,
    IPPOConfig,
    MAPPOSpecificConfig,
    MAPPOConfig,
    AlgorithmConfig,
    extract_gru_configs,
    extract_gru_configs_from_model,
    validate_gru_constraints,
)


# ============================================================================
# MLP Network Validation Tests
# ============================================================================

class TestMLPConfig:
    """Tests for MLP network configuration validation."""

    def test_valid_mlp_config(self):
        """Test valid MLP configuration."""
        config = MLPConfig(
            hidden_sizes=[128, 256, 128],
            activation="relu",
            output_activation="tanh",
            output_dim=64
        )
        assert config.hidden_sizes == [128, 256, 128]
        assert config.activation == "relu"
        assert config.output_activation == "tanh"
        assert config.output_dim == 64

    def test_valid_mlp_config_minimal(self):
        """Test valid MLP configuration with minimal required fields."""
        config = MLPConfig(
            hidden_sizes=[64],
            activation="relu"
        )
        assert config.hidden_sizes == [64]
        assert config.activation == "relu"
        assert config.output_activation is None
        assert config.output_dim is None

    def test_invalid_empty_hidden_sizes(self):
        """Test that empty hidden_sizes list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[], activation="relu")
        assert "too_short" in str(exc_info.value).lower() or "at least 1 item" in str(exc_info.value).lower()

    def test_invalid_negative_hidden_size(self):
        """Test that negative hidden sizes are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128, -64], activation="relu")
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_zero_hidden_size(self):
        """Test that zero hidden size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128, 0], activation="relu")
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_activation_name(self):
        """Test that invalid activation name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128], activation="invalid_activation")
        assert "literal" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

    def test_invalid_output_activation_name(self):
        """Test that invalid output activation name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128], activation="relu", output_activation="invalid")
        assert "literal" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

    def test_invalid_negative_output_dim(self):
        """Test that negative output_dim is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128], activation="relu", output_dim=-1)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_zero_output_dim(self):
        """Test that zero output_dim is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128], activation="relu", output_dim=0)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_valid_all_activation_names(self):
        """Test that all valid activation names are accepted."""
        valid_activations = ["relu", "tanh", "sigmoid", "elu", "selu", "gelu",
                            "swish", "mish", "hard_swish", "hard_sigmoid"]
        for activation in valid_activations:
            config = MLPConfig(hidden_sizes=[128], activation=activation)
            assert config.activation == activation

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MLPConfig(hidden_sizes=[128], activation="relu", extra_field="should_fail")
        assert "extra" in str(exc_info.value).lower() or "forbidden" in str(exc_info.value).lower()


class TestNetworkMLP:
    """Tests for NetworkMLP wrapper validation."""

    def test_valid_network_mlp(self):
        """Test valid NetworkMLP configuration."""
        network = NetworkMLP(
            type="mlp",
            config=MLPConfig(hidden_sizes=[128, 256], activation="relu")
        )
        assert network.type == "mlp"
        assert isinstance(network.config, MLPConfig)

    def test_invalid_type(self):
        """Test that wrong type discriminator is rejected."""
        with pytest.raises(ValidationError):
            NetworkMLP(type="gru", config=MLPConfig(hidden_sizes=[128], activation="relu"))

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            NetworkMLP(
                type="mlp",
                config=MLPConfig(hidden_sizes=[128], activation="relu"),
                extra_field="should_fail"
            )


# ============================================================================
# GRU Network Validation Tests
# ============================================================================

class TestGRUConfig:
    """Tests for GRU network configuration validation."""

    def test_valid_gru_config(self):
        """Test valid GRU configuration."""
        config = GRUConfig(
            num_layers=2,
            hidden_size=128,
            bidirectional=True,
            dropout=0.1,
            max_seq_len=20,
            activation="tanh",
            output_activation="relu",
            output_dim=64
        )
        assert config.num_layers == 2
        assert config.hidden_size == 128
        assert config.bidirectional is True
        assert config.dropout == 0.1
        assert config.max_seq_len == 20

    def test_valid_gru_config_minimal(self):
        """Test valid GRU configuration with minimal required fields."""
        config = GRUConfig(
            num_layers=1,
            hidden_size=64,
            max_seq_len=10
        )
        assert config.num_layers == 1
        assert config.hidden_size == 64
        assert config.max_seq_len == 10
        assert config.bidirectional is False
        assert config.dropout == 0.0

    def test_invalid_negative_num_layers(self):
        """Test that negative num_layers is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=-1, hidden_size=64, max_seq_len=10)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_zero_num_layers(self):
        """Test that zero num_layers is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=0, hidden_size=64, max_seq_len=10)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_negative_hidden_size(self):
        """Test that negative hidden_size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=1, hidden_size=-64, max_seq_len=10)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_negative_max_seq_len(self):
        """Test that negative max_seq_len is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=1, hidden_size=64, max_seq_len=-10)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_dropout_above_one(self):
        """Test that dropout > 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=2, hidden_size=64, max_seq_len=10, dropout=1.5)
        assert "dropout must be in [0.0, 1.0]" in str(exc_info.value)

    def test_valid_dropout_boundaries(self):
        """Test that dropout at boundaries (0.0, 1.0) is accepted."""
        config1 = GRUConfig(num_layers=2, hidden_size=64, max_seq_len=10, dropout=0.0)
        assert config1.dropout == 0.0
        
        config2 = GRUConfig(num_layers=2, hidden_size=64, max_seq_len=10, dropout=1.0)
        assert config2.dropout == 1.0

    def test_invalid_dropout_with_single_layer(self):
        """Test that dropout > 0 with num_layers=1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=1, hidden_size=64, max_seq_len=10, dropout=0.1)
        assert "dropout > 0 requires num_layers > 1" in str(exc_info.value)

    def test_valid_dropout_with_multiple_layers(self):
        """Test that dropout > 0 with num_layers > 1 is accepted."""
        config = GRUConfig(num_layers=2, hidden_size=64, max_seq_len=10, dropout=0.1)
        assert config.dropout == 0.1
        assert config.num_layers == 2

    def test_invalid_negative_output_dim(self):
        """Test that negative output_dim is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GRUConfig(num_layers=1, hidden_size=64, max_seq_len=10, output_dim=-1)
        assert "positive" in str(exc_info.value).lower() or "greater" in str(exc_info.value).lower()

    def test_invalid_activation_name(self):
        """Test that invalid activation name is rejected."""
        with pytest.raises(ValidationError):
            GRUConfig(num_layers=1, hidden_size=64, max_seq_len=10, activation="invalid")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            GRUConfig(
                num_layers=1,
                hidden_size=64,
                max_seq_len=10,
                extra_field="should_fail"
            )


class TestNetworkGRU:
    """Tests for NetworkGRU wrapper validation."""

    def test_valid_network_gru(self):
        """Test valid NetworkGRU configuration."""
        network = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        assert network.type == "gru"
        assert isinstance(network.config, GRUConfig)

    def test_invalid_type(self):
        """Test that wrong type discriminator is rejected."""
        with pytest.raises(ValidationError):
            NetworkGRU(type="mlp", config=GRUConfig(num_layers=1, hidden_size=64, max_seq_len=10))


# ============================================================================
# Shared Layers Validation Tests
# ============================================================================

class TestOptionalSharedLayers:
    """Tests for OptionalSharedLayers validation."""

    def test_valid_no_shared_layers(self):
        """Test that None shared_layers is valid."""
        config = OptionalSharedLayers(shared_layers=None)
        assert config.shared_layers is None

    def test_valid_shared_layers_with_output_dim(self):
        """Test that shared_layers with output_dim is valid."""
        shared_mlp = NetworkMLP(
            type="mlp",
            config=MLPConfig(hidden_sizes=[128], activation="relu", output_dim=64)
        )
        config = OptionalSharedLayers(shared_layers=shared_mlp)
        assert config.shared_layers == shared_mlp

    def test_invalid_shared_layers_without_output_dim(self):
        """Test that shared_layers without output_dim is rejected."""
        shared_mlp = NetworkMLP(
            type="mlp",
            config=MLPConfig(hidden_sizes=[128], activation="relu")  # No output_dim
        )
        with pytest.raises(ValidationError) as exc_info:
            OptionalSharedLayers(shared_layers=shared_mlp)
        assert "output_dim is required when shared_layers is provided" in str(exc_info.value)

    def test_valid_shared_gru_with_output_dim(self):
        """Test that shared GRU with output_dim is valid."""
        shared_gru = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20, output_dim=64)
        )
        config = OptionalSharedLayers(shared_layers=shared_gru)
        assert config.shared_layers == shared_gru


# ============================================================================
# Shared Algorithm Config Validation Tests
# ============================================================================

class TestSharedAlgorithmConfig:
    """Tests for SharedAlgorithmConfig validation."""

    def test_valid_shared_config(self):
        """Test valid SharedAlgorithmConfig."""
        config = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=128,
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        assert config.batch_size == 128
        assert config.num_minibatches == 4

    def test_invalid_batch_size_less_than_minibatches(self):
        """Test that batch_size < num_minibatches is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SharedAlgorithmConfig(
                num_iterations=1000,
                checkpoint_freq=100,
                batch_size=3,
                num_epochs=10,
                num_minibatches=4,
                learning_rate=0.0003
            )
        assert "batch_size (3) must be >= num_minibatches (4)" in str(exc_info.value)

    def test_invalid_batch_size_not_divisible_by_minibatches(self):
        """Test that batch_size not divisible by num_minibatches is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SharedAlgorithmConfig(
                num_iterations=1000,
                checkpoint_freq=100,
                batch_size=130,  # 130 / 4 = 32.5, not divisible
                num_epochs=10,
                num_minibatches=4,
                learning_rate=0.0003
            )
        assert "must be divisible by num_minibatches" in str(exc_info.value)

    def test_valid_batch_size_divisible_by_minibatches(self):
        """Test that batch_size divisible by num_minibatches is accepted."""
        config = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=128,  # 128 / 4 = 32, divisible
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        assert config.batch_size == 128

    def test_invalid_checkpoint_freq_exceeds_iterations(self):
        """Test that checkpoint_freq > num_iterations is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SharedAlgorithmConfig(
                num_iterations=100,
                checkpoint_freq=150,  # Exceeds num_iterations
                batch_size=128,
                num_epochs=10,
                num_minibatches=4,
                learning_rate=0.0003
            )
        assert "checkpoint_freq (150) must be <= num_iterations (100)" in str(exc_info.value)

    def test_valid_checkpoint_freq_equal_to_iterations(self):
        """Test that checkpoint_freq == num_iterations is accepted."""
        config = SharedAlgorithmConfig(
            num_iterations=100,
            checkpoint_freq=100,  # Equal to num_iterations
            batch_size=128,
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        assert config.checkpoint_freq == 100

    def test_invalid_negative_values(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            SharedAlgorithmConfig(
                num_iterations=-1,
                checkpoint_freq=100,
                batch_size=128,
                num_epochs=10,
                num_minibatches=4,
                learning_rate=0.0003
            )

    def test_invalid_zero_learning_rate(self):
        """Test that zero learning_rate is rejected."""
        with pytest.raises(ValidationError):
            SharedAlgorithmConfig(
                num_iterations=1000,
                checkpoint_freq=100,
                batch_size=128,
                num_epochs=10,
                num_minibatches=4,
                learning_rate=0.0
            )


# ============================================================================
# PPO Config Validation Tests
# ============================================================================

class TestPPOConfig:
    """Tests for PPO configuration validation."""

    def test_valid_ppo_config(self):
        """Test valid PPO configuration."""
        config = PPOConfig(
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            clip_param=0.2,
            use_gae=True,
            lam=0.95
        )
        assert config.clip_param == 0.2
        assert config.lam == 0.95

    def test_valid_ppo_config_defaults(self):
        """Test valid PPO configuration with defaults."""
        config = PPOConfig()
        assert config.vf_loss_coeff == 0.5
        assert config.entropy_coeff == 0.01
        assert config.clip_param == 0.2
        assert config.use_gae is True
        assert config.lam == 0.95

    def test_invalid_clip_param_above_one(self):
        """Test that clip_param > 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PPOConfig(clip_param=1.5)
        assert "clip_param should typically be <= 1.0" in str(exc_info.value)

    def test_valid_clip_param_at_one(self):
        """Test that clip_param == 1.0 is accepted."""
        config = PPOConfig(clip_param=1.0)
        assert config.clip_param == 1.0

    def test_invalid_lambda_below_zero(self):
        """Test that lam < 0.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PPOConfig(lam=-0.1)
        # The validation happens in model_validator, but Pydantic may reject negative NonNegativeFloat first
        error_str = str(exc_info.value).lower()
        assert "lam must be in [0.0, 1.0]" in str(exc_info.value) or "greater than or equal to 0" in error_str

    def test_invalid_lambda_above_one(self):
        """Test that lam > 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PPOConfig(lam=1.1)
        assert "lam must be in [0.0, 1.0]" in str(exc_info.value)

    def test_valid_lambda_boundaries(self):
        """Test that lam at boundaries (0.0, 1.0) is accepted."""
        config1 = PPOConfig(lam=0.0)
        assert config1.lam == 0.0
        
        config2 = PPOConfig(lam=1.0)
        assert config2.lam == 1.0

    def test_invalid_negative_coefficients(self):
        """Test that negative coefficients are rejected."""
        with pytest.raises(ValidationError):
            PPOConfig(vf_loss_coeff=-0.1)

        with pytest.raises(ValidationError):
            PPOConfig(entropy_coeff=-0.01)

    def test_valid_zero_coefficients(self):
        """Test that zero coefficients are accepted (non-negative)."""
        config = PPOConfig(vf_loss_coeff=0.0, entropy_coeff=0.0)
        assert config.vf_loss_coeff == 0.0
        assert config.entropy_coeff == 0.0


# ============================================================================
# ActorCritic Config Validation Tests
# ============================================================================

class TestActorCriticConfig:
    """Tests for ActorCriticConfig validation."""

    def test_valid_actor_critic_config(self):
        """Test valid ActorCriticConfig."""
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        config = ActorCriticConfig(actor=actor, critic=critic)
        assert config.actor == actor
        assert config.critic == critic

    def test_valid_actor_critic_with_shared_layers(self):
        """Test valid ActorCriticConfig with shared layers."""
        shared = NetworkMLP(
            type="mlp",
            config=MLPConfig(hidden_sizes=[128], activation="relu", output_dim=64)
        )
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        config = ActorCriticConfig(shared_layers=shared, actor=actor, critic=critic)
        assert config.shared_layers == shared

    def test_invalid_shared_layers_without_output_dim(self):
        """Test that shared_layers without output_dim is rejected."""
        shared = NetworkMLP(
            type="mlp",
            config=MLPConfig(hidden_sizes=[128], activation="relu")  # No output_dim
        )
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        with pytest.raises(ValidationError) as exc_info:
            ActorCriticConfig(shared_layers=shared, actor=actor, critic=critic)
        assert "output_dim is required when shared_layers is provided" in str(exc_info.value)

    def test_valid_mixed_mlp_gru(self):
        """Test valid ActorCriticConfig with MLP actor and GRU critic."""
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        critic = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=64, max_seq_len=20)
        )
        config = ActorCriticConfig(actor=actor, critic=critic)
        assert isinstance(config.actor, NetworkMLP)
        assert isinstance(config.critic, NetworkGRU)


# ============================================================================
# GRU Constraint Validation Tests
# ============================================================================

class TestGRUConstraints:
    """Tests for GRU-specific constraint validation."""

    def test_extract_gru_configs_from_mlp(self):
        """Test extracting GRU configs from MLP returns empty list."""
        mlp = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        gru_configs = extract_gru_configs(mlp)
        assert gru_configs == []

    def test_extract_gru_configs_from_gru(self):
        """Test extracting GRU configs from GRU returns the config."""
        gru = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        gru_configs = extract_gru_configs(gru)
        assert len(gru_configs) == 1
        assert gru_configs[0].max_seq_len == 20

    def test_extract_gru_configs_from_actor_critic(self):
        """Test extracting GRU configs from ActorCriticConfig."""
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        ac_config = ActorCriticConfig(actor=actor, critic=critic)
        
        gru_configs = extract_gru_configs_from_model(ac_config)
        assert len(gru_configs) == 1
        assert gru_configs[0].max_seq_len == 20

    def test_extract_gru_configs_from_actor_critic_both_gru(self):
        """Test extracting GRU configs when both actor and critic are GRU."""
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=1, hidden_size=64, max_seq_len=20)
        )
        ac_config = ActorCriticConfig(actor=actor, critic=critic)
        
        gru_configs = extract_gru_configs_from_model(ac_config)
        assert len(gru_configs) == 2
        assert all(gru.max_seq_len == 20 for gru in gru_configs)

    def test_validate_gru_constraints_no_gru(self):
        """Test that validation passes when no GRU networks."""
        # Should not raise any error
        validate_gru_constraints(batch_size=128, num_minibatches=4, gru_configs=[])

    def test_validate_gru_constraints_batch_size_too_small(self):
        """Test that batch_size < max_seq_len * num_minibatches is rejected."""
        gru_config = GRUConfig(num_layers=1, hidden_size=64, max_seq_len=20)
        with pytest.raises(ValueError) as exc_info:
            validate_gru_constraints(batch_size=50, num_minibatches=4, gru_configs=[gru_config])
        assert "batch_size (50) must be >= max_seq_len * num_minibatches (80)" in str(exc_info.value)

    def test_validate_gru_constraints_minibatch_size_too_small(self):
        """Test that minibatch_size < max_seq_len is rejected."""
        gru_config = GRUConfig(num_layers=1, hidden_size=64, max_seq_len=20)
        # batch_size=80, num_minibatches=4 -> minibatch_size=20, but need >= 20
        # Let's use batch_size=76, num_minibatches=4 -> minibatch_size=19 < 20
        # Note: This will fail the first check (batch_size < max_seq_len * num_minibatches)
        # before checking minibatch_size, which is expected behavior
        with pytest.raises(ValueError) as exc_info:
            validate_gru_constraints(batch_size=76, num_minibatches=4, gru_configs=[gru_config])
        # The error message checks batch_size first, so we check for that
        assert "batch_size (76)" in str(exc_info.value) and "max_seq_len * num_minibatches" in str(exc_info.value)

    def test_validate_gru_constraints_valid(self):
        """Test that valid GRU constraints pass."""
        gru_config = GRUConfig(num_layers=1, hidden_size=64, max_seq_len=20)
        # batch_size=80, num_minibatches=4 -> minibatch_size=20 >= 20 ✓
        validate_gru_constraints(batch_size=80, num_minibatches=4, gru_configs=[gru_config])

    def test_validate_gru_constraints_multiple_gru_different_max_seq_len(self):
        """Test that multiple GRUs with different max_seq_len are rejected."""
        gru1 = GRUConfig(num_layers=1, hidden_size=64, max_seq_len=20)
        gru2 = GRUConfig(num_layers=1, hidden_size=64, max_seq_len=25)
        with pytest.raises(ValueError) as exc_info:
            validate_gru_constraints(batch_size=100, num_minibatches=4, gru_configs=[gru1, gru2])
        assert "All GRU networks must have the same max_seq_len" in str(exc_info.value)

    def test_validate_gru_constraints_multiple_gru_same_max_seq_len(self):
        """Test that multiple GRUs with same max_seq_len pass."""
        gru1 = GRUConfig(num_layers=1, hidden_size=64, max_seq_len=20)
        gru2 = GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        # batch_size=80, num_minibatches=4 -> minibatch_size=20 >= 20 ✓
        validate_gru_constraints(batch_size=80, num_minibatches=4, gru_configs=[gru1, gru2])


# ============================================================================
# IPPO Config Validation Tests
# ============================================================================

class TestIPPOConfig:
    """Tests for IPPO configuration validation."""

    def create_valid_shared_config(self):
        """Helper to create valid SharedAlgorithmConfig."""
        return SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=128,
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )

    def create_valid_actor_critic_mlp(self):
        """Helper to create valid ActorCriticConfig with MLP networks."""
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        return ActorCriticConfig(actor=actor, critic=critic)

    def test_valid_ippo_config_mlp_only(self):
        """Test valid IPPO configuration with MLP networks only."""
        shared = self.create_valid_shared_config()
        networks = self.create_valid_actor_critic_mlp()
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config = IPPOConfig(
            name="ippo",
            shared=shared,
            algorithm_specific=algorithm_specific
        )
        assert config.name == "ippo"

    def test_valid_ippo_config_with_gru(self):
        """Test valid IPPO configuration with GRU networks."""
        # batch_size=80, num_minibatches=4 -> minibatch_size=20
        # max_seq_len=20, so 20 >= 20 ✓
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=80,  # Must be >= 20 * 4 = 80
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config = IPPOConfig(
            name="ippo",
            shared=shared,
            algorithm_specific=algorithm_specific
        )
        assert config.name == "ippo"

    def test_invalid_ippo_config_gru_batch_size_too_small(self):
        """Test that IPPO with GRU rejects batch_size < max_seq_len * num_minibatches."""
        # Use batch_size=80 which is divisible by 4, but still < 20 * 4 = 80
        # Actually, 80 == 80, so let's use 76 which is divisible by 4 but < 80
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=76,  # Divisible by 4, but 76 < 20 * 4 = 80
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        with pytest.raises(ValidationError) as exc_info:
            IPPOConfig(
                name="ippo",
                shared=shared,
                algorithm_specific=algorithm_specific
            )
        error_str = str(exc_info.value)
        assert "batch_size" in error_str and "max_seq_len" in error_str

    def test_invalid_ippo_config_gru_minibatch_size_too_small(self):
        """Test that IPPO with GRU rejects minibatch_size < max_seq_len."""
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=76,  # 76 / 4 = 19 < 20
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        with pytest.raises(ValidationError) as exc_info:
            IPPOConfig(
                name="ippo",
                shared=shared,
                algorithm_specific=algorithm_specific
            )
        error_str = str(exc_info.value)
        # The error checks batch_size first, so we check for that
        assert "batch_size" in error_str and "max_seq_len" in error_str

    def test_invalid_ippo_config_multiple_gru_different_max_seq_len(self):
        """Test that IPPO rejects multiple GRUs with different max_seq_len."""
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=100,  # Must accommodate max_seq_len=25
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=1, hidden_size=64, max_seq_len=25)  # Different max_seq_len
        )
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        with pytest.raises(ValidationError) as exc_info:
            IPPOConfig(
                name="ippo",
                shared=shared,
                algorithm_specific=algorithm_specific
            )
        assert "All GRU networks must have the same max_seq_len" in str(exc_info.value)

    def test_valid_ippo_config_with_shared_layers(self):
        """Test valid IPPO configuration with shared layers."""
        shared = self.create_valid_shared_config()
        shared_mlp = NetworkMLP(
            type="mlp",
            config=MLPConfig(hidden_sizes=[128], activation="relu", output_dim=64)
        )
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(shared_layers=shared_mlp, actor=actor, critic=critic)
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config = IPPOConfig(
            name="ippo",
            shared=shared,
            algorithm_specific=algorithm_specific
        )
        assert config.name == "ippo"


# ============================================================================
# MAPPO Config Validation Tests
# ============================================================================

class TestMAPPOConfig:
    """Tests for MAPPO configuration validation."""

    def create_valid_shared_config(self):
        """Helper to create valid SharedAlgorithmConfig."""
        return SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=128,
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )

    def create_valid_actor_critic_mlp(self):
        """Helper to create valid ActorCriticConfig with MLP networks."""
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        return ActorCriticConfig(actor=actor, critic=critic)

    def test_valid_mappo_config_mlp_only(self):
        """Test valid MAPPO configuration with MLP networks only."""
        shared = self.create_valid_shared_config()
        networks = self.create_valid_actor_critic_mlp()
        algorithm_specific = MAPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config = MAPPOConfig(
            name="mappo",
            shared=shared,
            algorithm_specific=algorithm_specific
        )
        assert config.name == "mappo"

    def test_valid_mappo_config_with_gru(self):
        """Test valid MAPPO configuration with GRU networks."""
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=80,  # Must be >= 20 * 4 = 80
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkGRU(
            type="gru",
            config=GRUConfig(num_layers=2, hidden_size=128, max_seq_len=20)
        )
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = MAPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config = MAPPOConfig(
            name="mappo",
            shared=shared,
            algorithm_specific=algorithm_specific
        )
        assert config.name == "mappo"


# ============================================================================
# Algorithm Config Discriminated Union Tests
# ============================================================================

class TestAlgorithmConfig:
    """Tests for AlgorithmConfig discriminated union."""

    def test_valid_ippo_via_algorithm_config(self):
        """Test that IPPO can be created via AlgorithmConfig."""
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=128,
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = IPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config_dict = {
            "name": "ippo",
            "shared": shared.model_dump(),
            "algorithm_specific": algorithm_specific.model_dump(by_alias=True)
        }
        adapter = TypeAdapter(AlgorithmConfig)
        config = adapter.validate_python(config_dict)
        assert isinstance(config, IPPOConfig)
        assert config.name == "ippo"

    def test_valid_mappo_via_algorithm_config(self):
        """Test that MAPPO can be created via AlgorithmConfig."""
        shared = SharedAlgorithmConfig(
            num_iterations=1000,
            checkpoint_freq=100,
            batch_size=128,
            num_epochs=10,
            num_minibatches=4,
            learning_rate=0.0003
        )
        actor = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[128], activation="relu"))
        critic = NetworkMLP(type="mlp", config=MLPConfig(hidden_sizes=[64], activation="relu"))
        networks = ActorCriticConfig(actor=actor, critic=critic)
        algorithm_specific = MAPPOSpecificConfig(
            parameter_sharing=False,
            networks=networks
        )
        config_dict = {
            "name": "mappo",
            "shared": shared.model_dump(),
            "algorithm_specific": algorithm_specific.model_dump(by_alias=True)
        }
        adapter = TypeAdapter(AlgorithmConfig)
        config = adapter.validate_python(config_dict)
        assert isinstance(config, MAPPOConfig)
        assert config.name == "mappo"

    def test_invalid_unknown_algorithm_name(self):
        """Test that unknown algorithm name is rejected."""
        config_dict = {
            "name": "unknown_algorithm",
            "shared": {},
            "algorithm_specific": {}
        }
        adapter = TypeAdapter(AlgorithmConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python(config_dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

