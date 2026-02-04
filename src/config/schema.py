# validation.py
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, NonNegativeInt, NonNegativeFloat, PositiveFloat, ValidationError, field_validator, model_validator


# ============================================================================
# Initial Inventory Config Schemas
# ============================================================================

# Uniform method
class InitialInventoryUniform(BaseModel):
    """Configuration for uniform initial inventory setup."""

    type: Literal["uniform"]
    params: Dict[Literal["min", "max"], NonNegativeInt]
    model_config = ConfigDict(extra="forbid")

    # Existence and shape of uniform parameters
    @model_validator(mode="after")
    def _check_uniform_keys(self):
        mn = self.params.get("min")
        mx = self.params.get("max")
        if mn is None or mx is None:
            raise ValueError("uniform params must contain min and max")
        if mn > mx:
            raise ValueError("uniform params must satisfy min <= max")
        return self

# Custom method
class InitialInventoryCustom(BaseModel):
    """Configuration for custom initial inventory setup."""    

    type: Literal["custom"]
    params: Dict[Literal["value"], Union[NonNegativeInt, List[List[NonNegativeInt]]]]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of value parameter
    @field_validator("params", mode="after")
    @classmethod
    def _check_custom_value_shape(cls, v: Dict[str, Any]):
        value = v["value"]
        # If value is a bool, raise an error (bool is a subclass of int)
        if isinstance(value, bool):
            raise ValueError("custom params.value must be an int, not a bool")
        # If value is an integer, return the value
        if isinstance(value, int):
            return v
        # If value is a list, check that it is a non-empty 2D list
        if not value or any(not row for row in value):
            raise ValueError("custom params.value must be a non-empty 2D list")
        # Check that all rows have the same length
        if len({len(row) for row in value}) != 1:
            raise ValueError("custom params.value must be a rectangular 2D list")

        return v

# Zero method
class InitialInventoryZero(BaseModel):
    """Configuration for zero initial inventory setup."""

    type: Literal["zero"]
    params: Optional[None] = None
    model_config = ConfigDict(extra="forbid")

# Union of initial inventory configurations
InitialInventoryConfig = Union[InitialInventoryUniform, InitialInventoryCustom, InitialInventoryZero]


# ============================================================================
# Cost Structure Config Schemas
# ============================================================================

# Shipment cost configuration
class ShipmentCostConfig(BaseModel):
    """Configuration for shipment costs."""
    
    outbound_fixed: Optional[List[List[NonNegativeFloat]]] = None
    outbound_variable: Optional[List[List[NonNegativeFloat]]] = None
    inbound_fixed: Optional[List[List[NonNegativeFloat]]] = None
    inbound_variable: Optional[List[List[NonNegativeFloat]]] = None
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("outbound_fixed")
    @classmethod
    def _check_outbound_fixed_is_rectangular(cls, v: Optional[List[List[NonNegativeFloat]]]):
        if v is not None:
            if not v or any(not row for row in v):
                raise ValueError("shipment_cost.outbound_fixed must be a non-empty 2D list")
            if len({len(row) for row in v}) != 1:
                raise ValueError("shipment_cost.outbound_fixed must be rectangular (all rows same length)")
        return v
    
    @field_validator("outbound_variable")
    @classmethod
    def _check_outbound_variable_is_rectangular(cls, v: Optional[List[List[NonNegativeFloat]]]):
        if v is not None:
            if not v or any(not row for row in v):
                raise ValueError("shipment_cost.outbound_variable must be a non-empty 2D list")
            if len({len(row) for row in v}) != 1:
                raise ValueError("shipment_cost.outbound_variable must be rectangular (all rows same length)")
        return v
    
    @field_validator("inbound_fixed")
    @classmethod
    def _check_inbound_fixed_is_rectangular(cls, v: Optional[List[List[NonNegativeFloat]]]):
        if v is not None:
            if not v or any(not row for row in v):
                raise ValueError("shipment_cost.inbound_fixed must be a non-empty 2D list")
            if len({len(row) for row in v}) != 1:
                raise ValueError("shipment_cost.inbound_fixed must be rectangular (all rows same length)")
        return v
    
    @field_validator("inbound_variable")
    @classmethod
    def _check_inbound_variable_is_rectangular(cls, v: Optional[List[List[NonNegativeFloat]]]):
        if v is not None:
            if not v or any(not row for row in v):
                raise ValueError("shipment_cost.inbound_variable must be a non-empty 2D list")
            if len({len(row) for row in v}) != 1:
                raise ValueError("shipment_cost.inbound_variable must be rectangular (all rows same length)")
        return v
    
    @model_validator(mode="after")
    def _check_variable_matches_fixed_shape(self):
        # Check outbound costs
        if self.outbound_variable is not None:
            if self.outbound_fixed is None:
                raise ValueError(
                    "shipment_cost.outbound_variable cannot be specified without "
                    "shipment_cost.outbound_fixed"
                )
            if len(self.outbound_variable) != len(self.outbound_fixed):
                raise ValueError(
                    f"shipment_cost.outbound_variable must have same number of rows as outbound_fixed "
                    f"({len(self.outbound_fixed)}), got {len(self.outbound_variable)}"
                )
            if len(self.outbound_variable[0]) != len(self.outbound_fixed[0]):
                raise ValueError(
                    f"shipment_cost.outbound_variable must have same number of columns as outbound_fixed "
                    f"({len(self.outbound_fixed[0])}), got {len(self.outbound_variable[0])}"
                )
        
        # Check inbound costs
        if self.inbound_variable is not None:
            if self.inbound_fixed is None:
                raise ValueError(
                    "shipment_cost.inbound_variable cannot be specified without "
                    "shipment_cost.inbound_fixed"
                )
            if len(self.inbound_variable) != len(self.inbound_fixed):
                raise ValueError(
                    f"shipment_cost.inbound_variable must have same number of rows as inbound_fixed "
                    f"({len(self.inbound_fixed)}), got {len(self.inbound_variable)}"
                )
            if len(self.inbound_variable[0]) != len(self.inbound_fixed[0]):
                raise ValueError(
                    f"shipment_cost.inbound_variable must have same number of columns as inbound_fixed "
                    f"({len(self.inbound_fixed[0])}), got {len(self.inbound_variable[0])}"
                )
        return self

# Cost structures
class CostStructureConfig(BaseModel):
    """Configuration for cost structure."""

    holding_cost:  Union[PositiveFloat, List[PositiveFloat]]
    penalty_cost:  Union[NonNegativeFloat, List[NonNegativeFloat]]
    shipment_cost: ShipmentCostConfig
    sku_weights: Optional[List[PositiveFloat]] = None
    distances: Optional[List[List[NonNegativeFloat]]] = None
    model_config = ConfigDict(extra="forbid")

# ============================================================================
# Component Config Schemas
# ============================================================================

#  ---------- Demand sampler ----------
# Poisson sampler
class DemandSamplerPoisson(BaseModel):
    """Configuration for Poisson demand sampler."""

    type: Literal["poisson"]
    params: Dict[Literal["lambda_orders", "lambda_skus", "lambda_quantity"], PositiveFloat]
    model_config = ConfigDict(extra="forbid")

# Empirical sampler
class DemandSamplerEmpirical(BaseModel):
    """Configuration for empirical demand sampler."""

    type: Literal["empirical"]
    params: Optional[None] = None 
    model_config = ConfigDict(extra="forbid")

# Union of demand sampler configurations
DemandSamplerConfig = Union[DemandSamplerPoisson, DemandSamplerEmpirical]


#  ---------- Demand allocator ----------
# Greedy allocator
class DemandAllocatorGreedy(BaseModel):
    """Configuration for greedy demand allocator."""

    type: Literal["greedy"]
    params: Dict[Literal["max_splits"], Union[Literal["default"], NonNegativeInt]]
    model_config = ConfigDict(extra="forbid")

# TODO: Adjust when implementing LP allocator
# LP allocator (placeholder for future implementation)
class DemandAllocatorLP(BaseModel):
    """Configuration for LP-based demand allocator."""

    type: Literal["lp"]
    params: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="forbid")

# Union of demand allocator configurations
DemandAllocatorConfig = Union[DemandAllocatorGreedy, DemandAllocatorLP]


#  ---------- Lead time sampler ----------
# Uniform sampler
class LeadTimeSamplerUniform(BaseModel):
    """Configuration for uniform lead time sampler."""

    type: Literal["uniform"]
    params: Dict[Literal["min", "max"], NonNegativeInt]
    model_config = ConfigDict(extra="forbid")

    # Existence and shape of uniform parameters
    @model_validator(mode="after")
    def _check_uniform_keys(self):
        mn = self.params.get("min")
        mx = self.params.get("max")
        if mn is None or mx is None:
            raise ValueError("uniform params must contain min and max")
        if mn > mx:
            raise ValueError("uniform params must satisfy min <= max")
        return self

# Union of lead time sampler configurations
LeadTimeSamplerConfig = Union[LeadTimeSamplerUniform]


#  ---------- Lost sales handler ---------
# Cheapest assignment
class LostSalesClosest(BaseModel):
    """Configuration for closest lost sales assignment."""

    type: Literal["closest"]
    params: Optional[None] = None
    model_config = ConfigDict(extra="forbid")

# Shipment-based assignment
class LostSalesShipment(BaseModel):
    """Configuration for shipment-based lost sales assignment."""

    type: Literal["shipment"]
    params: Optional[None] = None
    model_config = ConfigDict(extra="forbid")

# Cost-based assignment
class LostSalesCost(BaseModel):
    """Configuration for cost-based lost sales assignment."""

    type: Literal["cost"]
    params: Dict[Literal["alpha"], NonNegativeFloat]
    model_config = ConfigDict(extra="forbid")

# Union of lost sales handler configurations
LostSalesHandlerConfig = Union[LostSalesClosest, LostSalesShipment, LostSalesCost]


#  ---------- Reward calculator----------
# Cost-based reward calculator
Scope = Literal["team", "agent"]
COST_TYPES = ["holding_cost", "penalty_cost", "outbound_shipment_cost", "inbound_shipment_cost"]

class RewardCostParams(BaseModel):
    scope: Scope
    scale_factor: PositiveFloat
    normalize: bool
    cost_weights: List[NonNegativeFloat] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("cost_weights")
    @classmethod
    def _validate_cost_weights(cls, v: List[float]):
        # Cost weights must be in [0.0, 1.0]
        if any(x < 0.0 or x > 1.0 for x in v):
            raise ValueError("cost_weights must be in [0.0, 1.0]")
        # Cost weights must sum to 1.0
        total = float(sum(v))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"cost_weights must sum to 1.0 (got {total})")

        return v

class RewardCalculatorCost(BaseModel):
    """Configuration for cost-based reward calculator."""
    type: Literal["cost"]
    params: RewardCostParams
    model_config = ConfigDict(extra="forbid")

    # Length of cost_weights must match number of cost types
    @model_validator(mode="after")
    def _check_cost_weights_length(self):
        expected = len(COST_TYPES)
        actual = len(self.params.cost_weights)
        if actual != expected:
            raise ValueError(
                f"reward_calculator.params.cost_weights must have length {expected} "
                f"(one weight per cost type: {COST_TYPES})"
            )
        return self

# Union of reward calculator configurations
RewardCalculatorConfig = Union[RewardCalculatorCost]

#  ---------- Top-level components ----------
# Conponent orchestrator
class ComponentsConfig(BaseModel):
    """Top-level configuration for components."""

    demand_sampler: DemandSamplerConfig = Field(..., discriminator="type")
    demand_allocator: DemandAllocatorConfig = Field(..., discriminator="type")
    lead_time_sampler: LeadTimeSamplerConfig = Field(..., discriminator="type")
    lost_sales_handler: LostSalesHandlerConfig = Field(..., discriminator="type")
    reward_calculator: RewardCalculatorConfig = Field(..., discriminator="type")
    model_config = ConfigDict(extra="forbid")


# ============================================================================
# Data Split Configuration Schemas
# ============================================================================

# Ratio-based data split
class DataSplitRatio(BaseModel):
    """Configuration for ratio-based data split."""

    type: Literal["ratio"]
    train_ratio: NonNegativeFloat
    model_config = ConfigDict(extra="forbid")

    @field_validator("train_ratio")
    @classmethod
    def _validate_train_ratio(cls, v: float):
        if v > 1.0:
            raise ValueError("train_ratio must be in [0.0, 1.0]")
        return v

# Explicit timestep-based data split
class DataSplitExplicit(BaseModel):
    """Configuration for explicit timestep-based data split."""

    type: Literal["explicit"]
    train_timesteps: List[int]
    val_timesteps: List[int]
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _check_no_overlap(self):
        train_set = set(self.train_timesteps)
        val_set = set(self.val_timesteps)
        if train_set & val_set:
            raise ValueError("train_timesteps and val_timesteps must not overlap")
        return self

# Union of data split configurations
DataSplitConfig = Annotated[
    Union[DataSplitRatio, DataSplitExplicit],
    Field(discriminator="type")
]

# ============================================================================
# Data Source Config Schemas
# ============================================================================

# Synthetic data source
class DataSourceSynthetic(BaseModel):
    """Configuration for synthetic data source."""

    type: Literal["synthetic"]
    path: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# Real-world data source
class DataSourceRealWorld(BaseModel):
    """Configuration for real-world data source."""

    type: Literal["real_world"]
    path: Path
    data_split: Optional[DataSplitConfig] = None
    model_config = ConfigDict(extra="forbid")

    # Path must exist (can be file or directory)
    @field_validator("path")
    @classmethod
    def _check_path_exists(cls, p: Path):
        if not p.exists():
            raise ValueError(f"real_world path does not exist: {p}")
        return p

# Union of data source configurations
DataSourceConfig = Union[DataSourceSynthetic, DataSourceRealWorld]

# ============================================================================
# Top-level Environment Config Schemas
# ============================================================================

class EnvironmentConfig(BaseModel):
    """Top-level configuration for environment."""

    n_warehouses: PositiveInt
    n_skus: PositiveInt
    n_regions: PositiveInt
    episode_length: PositiveInt
    max_order_quantity: PositiveInt

    initial_inventory: InitialInventoryConfig = Field(..., discriminator="type")
    cost_structure: CostStructureConfig
    components: ComponentsConfig
    data_source: DataSourceConfig = Field(..., discriminator="type")
    
    model_config = ConfigDict(extra="forbid")

    # Shape checks of interacting environment parameters
    @model_validator(mode="after")
    def _shape_checks(self):
        nw, ns, nr = self.n_warehouses, self.n_skus, self.n_regions

        # n_regions >= n_warehouses
        if nr < nw:
            raise ValueError(f"n_regions ({nr}) must be >= n_warehouses ({nw})")

        # Initial_inventory: custom list shape (n_warehouses, n_skus)
        if isinstance(self.initial_inventory, InitialInventoryCustom):
            value = self.initial_inventory.params["value"]
            if isinstance(value, list):
                if len(value) != nw:
                    raise ValueError(
                        f"initial_inventory.custom params.value must have {nw} rows (n_warehouses), "
                        f"got {len(value)}"
                    )
                for i, row in enumerate(value):
                    if not isinstance(row, list):
                        raise ValueError(f"initial_inventory.custom params.value[{i}] must be a list")
                    if len(row) != ns:
                        raise ValueError(
                            f"initial_inventory.custom params.value[{i}] must have length n_skus={ns}, "
                            f"got {len(row)}"
                        )

        # Holding_cost: scalar or (n_skus,)
        hc = self.cost_structure.holding_cost
        if isinstance(hc, list) and len(hc) != ns:
            raise ValueError(f"holding_cost list must have length n_skus={ns}, got {len(hc)}")

        # Penalty_cost: scalar or (n_skus,)
        pc = self.cost_structure.penalty_cost
        if isinstance(pc, list) and len(pc) != ns:
            raise ValueError(f"penalty_cost list must have length n_skus={ns}")

        # Outbound shipment costs: (n_warehouses, n_regions)
        sc_outbound_fixed = self.cost_structure.shipment_cost.outbound_fixed
        if sc_outbound_fixed is not None:
            if len(sc_outbound_fixed) != nw:
                raise ValueError(f"shipment_cost.outbound_fixed must have {nw} rows (n_warehouses), got {len(sc_outbound_fixed)}")
            if any(len(row) != nr for row in sc_outbound_fixed):
                raise ValueError(f"shipment_cost.outbound_fixed must have {nr} columns (n_regions) in every row")
        
        sc_outbound_variable = self.cost_structure.shipment_cost.outbound_variable
        if sc_outbound_variable is not None:
            if len(sc_outbound_variable) != nw:
                raise ValueError(f"shipment_cost.outbound_variable must have {nw} rows (n_warehouses), got {len(sc_outbound_variable)}")
            if any(len(row) != nr for row in sc_outbound_variable):
                raise ValueError(f"shipment_cost.outbound_variable must have {nr} columns (n_regions) in every row")
        
        # Inbound shipment costs: (n_suppliers, n_warehouses) where n_suppliers = n_skus
        sc_inbound_fixed = self.cost_structure.shipment_cost.inbound_fixed
        if sc_inbound_fixed is not None:
            if len(sc_inbound_fixed) != ns:
                raise ValueError(f"shipment_cost.inbound_fixed must have {ns} rows (n_suppliers=n_skus), got {len(sc_inbound_fixed)}")
            if any(len(row) != nw for row in sc_inbound_fixed):
                raise ValueError(f"shipment_cost.inbound_fixed must have {nw} columns (n_warehouses) in every row")
        
        sc_inbound_variable = self.cost_structure.shipment_cost.inbound_variable
        if sc_inbound_variable is not None:
            if len(sc_inbound_variable) != ns:
                raise ValueError(f"shipment_cost.inbound_variable must have {ns} rows (n_suppliers=n_skus), got {len(sc_inbound_variable)}")
            if any(len(row) != nw for row in sc_inbound_variable):
                raise ValueError(f"shipment_cost.inbound_variable must have {nw} columns (n_warehouses) in every row")

        # SKU_weights: optional (n_skus,) for synthetic mode
        sw = self.cost_structure.sku_weights
        if sw is not None and len(sw) != ns:
            raise ValueError(f"sku_weights list must have length n_skus={ns}, got {len(sw)}")

        # Distances: (n_warehouses, n_regions)
        dist = self.cost_structure.distances
        if dist is not None:
            if not dist or any(not row for row in dist):
                raise ValueError("distances must be a non-empty 2D list")
            if len({len(row) for row in dist}) != 1:
                raise ValueError("distances must be rectangular (all rows same length)")
            if len(dist) != nw:
                raise ValueError(f"distances must have {nw} rows (n_warehouses), got {len(dist)}")
            if any(len(row) != nr for row in dist):
                raise ValueError(f"distances must have {nr} columns (n_regions) in every row")

        return self

    # Post-checks of interacting environment parameters
    @model_validator(mode="after")
    def _post_checks(self):
        nw = self.n_warehouses

        # Demand_allocator: greedy max_splits handling
        alloc = self.components.demand_allocator
        if isinstance(alloc, DemandAllocatorGreedy):
            ms = alloc.params["max_splits"]
            if ms == "default":
                alloc.params["max_splits"] = nw - 1
            else:
                if ms >= nw:
                    raise ValueError(f"demand_allocator.greedy max_splits must be < n_warehouses={nw}")

        # Data source validation
        if isinstance(self.components.demand_sampler, DemandSamplerEmpirical):
            if self.data_source.type != "real_world":
                raise ValueError(
                    "demand_sampler.type='empirical' requires data_source.type='real_world', "
                    f"got data_source.type='{self.data_source.type}'"
                )
        
        # Shipment costs validation
        if self.data_source.type == "synthetic":
            if self.cost_structure.shipment_cost.outbound_fixed is None:
                raise ValueError(
                    "shipment_cost.fixed_per_order must be specified when using synthetic data source. "
                    "For real_world data source, fixed_per_order is extracted from the data."
                )
            if self.cost_structure.shipment_cost.outbound_variable is None:
                raise ValueError(
                    "shipment_cost.variable_per_weight must be specified when using synthetic data source. "
                    "For real_world data source, variable_per_weight is extracted from the data."
                )
            if self.cost_structure.shipment_cost.inbound_fixed is None:
                raise ValueError(
                    "shipment_cost.inbound_fixed must be specified when using synthetic data source. "
                    "For real_world data source, inbound_fixed is extracted from the data."
                )
            if self.cost_structure.shipment_cost.inbound_variable is None:
                raise ValueError(
                    "shipment_cost.inbound_variable must be specified when using synthetic data source. "
                    "For real_world data source, inbound_variable is extracted from the data."
                )
        
        # SKU weights validation
        if self.data_source.type == "synthetic":
            if self.cost_structure.sku_weights is None:
                raise ValueError(
                    "sku_weights must be specified when using synthetic data source. "
                    "For real_world data source, sku_weights are extracted from the data."
                )
        # Distances validation
        if self.data_source.type == "synthetic":
            if self.cost_structure.distances is None:
                raise ValueError(
                    "distances must be specified when using synthetic data source. "
                    "For real_world data source, distances are extracted from the data."
                )
        
        return self


# ============================================================================
# Network Configuration Schemas
# ============================================================================

# Allowed activation functions
ActivationName = Literal[
    "relu", "tanh", "sigmoid", "elu", "selu", "gelu",
    "swish", "mish", "hard_swish", "hard_sigmoid"
]

# MLP networks
class MLPConfig(BaseModel):
    """Configuration for MLP config."""

    hidden_sizes: List[PositiveInt] = Field(min_length=1)
    activation: ActivationName
    output_activation: Optional[ActivationName] = None
    output_dim: Optional[PositiveInt] = None
    model_config = ConfigDict(extra="forbid")

class NetworkMLP(BaseModel):
    """Configuration for MLP network."""

    type: Literal["mlp"]
    config: MLPConfig
    model_config = ConfigDict(extra="forbid")

# GRU networks
class GRUConfig(BaseModel):
    """Configuration for GRU config."""

    num_layers: PositiveInt
    hidden_size: PositiveInt
    bidirectional: bool = False
    dropout: NonNegativeFloat = 0.0
    max_seq_len: PositiveInt
    activation: Optional[ActivationName] = None
    output_activation: Optional[ActivationName] = None
    output_dim: Optional[PositiveInt] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, v: float):
        if v > 1.0:
            raise ValueError("dropout must be in [0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def _validate_dropout_vs_layers(self):
        # In PyTorch GRU, dropout is only applied when num_layers > 1.
        if self.dropout > 0.0 and self.num_layers == 1:
            raise ValueError("dropout > 0 requires num_layers > 1 for GRU")
        return self

class NetworkGRU(BaseModel):
    """Configuration for GRU network."""

    type: Literal["gru"]
    config: GRUConfig
    model_config = ConfigDict(extra="forbid")

# Union of network configurations
NetworkConfig = Annotated[Union[NetworkMLP, NetworkGRU], Field(discriminator="type")]

# Shared layer network
class OptionalSharedLayers(BaseModel):
    """Configuration for optional shared layers."""

    shared_layers: Optional[NetworkConfig] = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _require_output_dim_if_shared_present(self):
        if self.shared_layers is None:
            return self

        out_dim = self.shared_layers.config.output_dim
        if out_dim is None:
            raise ValueError("shared_layers.config.output_dim is required when shared_layers is provided")

        return self


# ============================================================================
# Algorithm Configuration Schemas
# ============================================================================

# Shared algorithm parameters
class SharedAlgorithmConfig(BaseModel):
    """Configuration for shared algorithm parameters."""

    num_iterations: PositiveInt
    checkpoint_freq: PositiveInt
    batch_size: PositiveInt
    num_epochs: PositiveInt
    num_minibatches: PositiveInt    
    learning_rate: PositiveFloat
    num_env_runners: NonNegativeInt = 0
    num_envs_per_env_runner: NonNegativeInt = 1
    num_cpus_per_env_runner: PositiveInt = 1
    eval_interval: PositiveInt = 1
    num_eval_episodes: PositiveInt = 1
    evaluation_parallel_to_training: bool = False
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _validate_shared_params(self):
        """Validate relationships between shared algorithm parameters."""
        # batch_size must be >= num_minibatches (each minibatch needs at least 1 sample)
        if self.batch_size < self.num_minibatches:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be >= num_minibatches ({self.num_minibatches}) "
                f"so each minibatch has at least 1 sample"
            )
        
        # batch_size should be divisible by num_minibatches for clean minibatch division
        if self.batch_size % self.num_minibatches != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by num_minibatches ({self.num_minibatches}) "
                f"for clean minibatch division. Current minibatch_size would be {self.batch_size // self.num_minibatches} "
                f"with {self.batch_size % self.num_minibatches} samples lost."
            )
        
        # checkpoint_freq should not exceed num_iterations
        if self.checkpoint_freq > self.num_iterations:
            raise ValueError(
                f"checkpoint_freq ({self.checkpoint_freq}) must be <= num_iterations ({self.num_iterations})"
            )
        
        return self

# Actor-critic network structure
class ActorCriticConfig(OptionalSharedLayers):
    """Configuration for actor-critic network structure."""
    
    actor: NetworkConfig
    critic: NetworkConfig
    model_config = ConfigDict(extra="forbid")

# PPO-specific parameters
class PPOConfig(BaseModel):
    """Configuration for PPO-specific parameters."""

    vf_loss_coeff: NonNegativeFloat = 0.5
    entropy_coeff: NonNegativeFloat = 0.01
    clip_param: PositiveFloat = 0.2
    use_gae: bool = True
    lam: NonNegativeFloat = 0.95
    gamma: NonNegativeFloat = 0.99
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _validate_ppo_params(self):
        """Validate PPO parameter relationships."""
        if self.clip_param > 1.0:
            raise ValueError("clip_param should typically be <= 1.0")
        if self.lam < 0.0 or self.lam > 1.0:
            raise ValueError("lam must be in [0.0, 1.0]")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in [0.0, 1.0]")
        return self

# IPPO-specific parameters
class IPPOSpecificConfig(PPOConfig):
    """Configuration for algorithm-specific parameters."""

    parameter_sharing: bool = False
    networks: ActorCriticConfig
    model_config = ConfigDict(extra="forbid")

# IPPO algorithm configuration
class IPPOConfig(BaseModel):
    """Configuration for IPPO algorithm."""
    
    name: Literal["ippo"]
    shared: SharedAlgorithmConfig
    algorithm_specific: IPPOSpecificConfig
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_gru_constraints(self):
        """Validate GRU-specific constraints."""
        # Extract all GRU configs from the entire algorithm_specific structure
        gru_configs = extract_gru_configs_from_model(self.algorithm_specific)
        validate_gru_constraints(self.shared.batch_size, self.shared.num_minibatches, gru_configs)
        return self

# MAPPO-specific parameters
class MAPPOSpecificConfig(PPOConfig):
    """Configuration for algorithm-specific parameters."""

    parameter_sharing: bool = False
    networks: ActorCriticConfig
    model_config = ConfigDict(extra="forbid")

# MAPPO algorithm configuration
class MAPPOConfig(BaseModel):
    """Configuration for IPPO algorithm."""
    
    name: Literal["mappo"]
    shared: SharedAlgorithmConfig
    algorithm_specific: MAPPOSpecificConfig
    model_config = ConfigDict(extra="forbid")

# Union of algorithm configurations
AlgorithmConfig = Annotated[Union[IPPOConfig, MAPPOConfig], Field(discriminator="name")]


# ============================================================================
# Algorithm Configuration Helper Functions
# ============================================================================

# Extract GRU configs from a single NetworkConfig
def extract_gru_configs(network: NetworkConfig) -> List[GRUConfig]:
    """Extract GRU config from a single NetworkConfig if it's a GRU."""
    if isinstance(network, NetworkGRU):
        return [network.config]
    return []

# Extract GRU configs from a Pydantic model containing NetworkConfig fields
def extract_gru_configs_from_model(model: BaseModel) -> List[GRUConfig]:
    """ Recursively extract all GRU configs from any Pydantic model containing NetworkConfig fields."""
    gru_configs = []
    
    # Iterate over all fields in the model
    for field_name, field_info in type(model).model_fields.items():
        field_value = getattr(model, field_name, None)
        
        if field_value is None:
            continue
        
        # Check if the field is a NetworkConfig (directly)
        if isinstance(field_value, (NetworkMLP, NetworkGRU)):
            gru_configs.extend(extract_gru_configs(field_value))
        
        # If it's another BaseModel, recurse into it
        elif isinstance(field_value, BaseModel):
            gru_configs.extend(extract_gru_configs_from_model(field_value))
        
        # Handle lists/tuples of models
        elif isinstance(field_value, (list, tuple)):
            for item in field_value:
                if isinstance(item, (NetworkMLP, NetworkGRU)):
                    gru_configs.extend(extract_gru_configs(item))
                elif isinstance(item, BaseModel):
                    gru_configs.extend(extract_gru_configs_from_model(item))
    
    return gru_configs

# Validate GRU-specific constraints
def validate_gru_constraints(batch_size: int, num_minibatches: int, gru_configs: List[GRUConfig]) -> None:
    """Validate GRU-specific constraints."""
    if not gru_configs:
        return  # No GRU networks, nothing to validate
    
    # Get the maximum max_seq_len value from all GRU configs
    max_seq_lens = [gru.max_seq_len for gru in gru_configs]
    max_seq_len = max(max_seq_lens)
    
    # Ensure batch size is sufficient for all minibatches
    if batch_size < max_seq_len * num_minibatches:
        raise ValueError(
            f"batch_size ({batch_size}) must be >= max_seq_len * num_minibatches "
            f"({max_seq_len * num_minibatches}) when using GRU networks. "
            f"Found GRU networks with max_seq_len values: {max_seq_lens}"
        )
    
    # Ensure minibatch size is sufficient for the maximum max_seq_len
    minibatch_size = batch_size // num_minibatches
    if minibatch_size < max_seq_len:
        raise ValueError(
            f"minibatch_size ({minibatch_size} = batch_size // num_minibatches) must be >= max_seq_len ({max_seq_len}) "
            f"when using GRU networks. Found GRU networks with max_seq_len values: {max_seq_lens}"
        )
    
    # Validate all max_seq_len values are equal if multiple GRUs exist
    if len(gru_configs) > 1:
        unique_max_seq_lens = set(max_seq_lens)
        if len(unique_max_seq_lens) > 1:
            raise ValueError(
                f"All GRU networks must have the same max_seq_len when multiple GRUs are present. "
                f"Found max_seq_len values: {sorted(unique_max_seq_lens)}"
            )


# ============================================================================
# Tune Configuration Schemas
# ============================================================================

# Uniform search space
class UniformSearch(BaseModel):
    """Configuration for uniform search space."""

    type: Literal["uniform"]
    low: float
    high: float
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _validate_range(self):
        if self.low >= self.high:
            raise ValueError(f"uniform search: low ({self.low}) must be < high ({self.high})")
        return self

# Log-uniform search space
class LogUniformSearch(BaseModel):
    """Configuration for log-uniform search space."""

    type: Literal["loguniform"]
    low: PositiveFloat
    high: PositiveFloat
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _validate_range(self):
        if self.low >= self.high:
            raise ValueError(f"loguniform search: low ({self.low}) must be < high ({self.high})")
        return self

# Choice search space
class ChoiceSearch(BaseModel):
    """Configuration for choice search space."""

    type: Literal["choice"]
    values: List[Union[int, float, str]] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")

# Random integer search space
class RandIntSearch(BaseModel):
    """Configuration for random integer search space."""

    type: Literal["randint"]
    low: int
    high: int
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _validate_range(self):
        if self.low >= self.high:
            raise ValueError(f"randint search: low ({self.low}) must be < high ({self.high})")
        return self

# Grid search space
class GridSearch(BaseModel):
    """Configuration for grid search space."""

    type: Literal["grid_search"]
    values: List[Union[int, float, str]] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")

# Union of search space types
SearchSpaceSpec = Annotated[
    Union[UniformSearch, LogUniformSearch, ChoiceSearch, RandIntSearch, GridSearch],
    Field(discriminator="type")
]

# Tune configuration
class TuneConfig(BaseModel):
    """Configuration for Ray Tune hyperparameter search."""
    
    shared: Optional[Dict[str, SearchSpaceSpec]] = None
    algorithm_specific: Optional[Dict[str, SearchSpaceSpec]] = None
    model_config = ConfigDict(extra="forbid")
    
    @model_validator(mode="after")
    def _validate_structure(self):
        """Validate that at least one section is provided."""
        if self.shared is None and self.algorithm_specific is None:
            raise ValueError(
                "At least one of 'shared' or 'algorithm_specific' must be provided"
            )
        return self

