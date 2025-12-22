# validation.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, field_validator, model_validator


# ============================================================================
# Initial Inventory Config Schemas
# ============================================================================

# Uniform method
class InitialInventoryUniform(BaseModel):
    """Configuration for uniform initial inventory setup."""

    type: Literal["uniform"]
    params: Dict[Literal["min", "max"], int]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of uniform parameters
    @model_validator(mode="after")
    def _check_uniform_params(self):
        low = self.params["min"]
        high = self.params["max"]
        if not (isinstance(low, int) and isinstance(high, int)):
            raise ValueError("uniform params min/max must be integers")
        if low < 0 or high < 0:
            raise ValueError("uniform params min/max must be non-negative")
        if low >= high:
            raise ValueError("uniform params must satisfy min < max")
    
        return self

# Custom method
class InitialInventoryCustom(BaseModel):
    """Configuration for custom initial inventory setup."""    

    type: Literal["custom"]
    params: Dict[Literal["value"], Union[int, List[List[int]]]]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of value parameter
    @field_validator("params")
    @classmethod
    def _check_custom_value_nonneg(cls, v: Dict[str, Any]):
        custom_value = v["value"]
        if isinstance(custom_value, int):
            if custom_value < 0:
                raise ValueError("custom params.value must be non-negative")
        elif isinstance(custom_value, list):
            # Must be 2D list
            if not all(isinstance(row, list) for row in custom_value):
                raise ValueError("custom params.value must be a 2D list (list of lists)")
            # Check all values are non-negative integers
            for row in custom_value:
                if any((not isinstance(x, int)) or x < 0 for x in row):
                    raise ValueError("custom params.value must contain only non-negative integers")
        else:
            raise ValueError("custom params.value must be an int or 2D list of ints")
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

# Cost structures
class CostStructureConfig(BaseModel):
    """Configuration for cost structure."""

    holding_cost:  Union[int, float, List[int], List[float]]
    penalty_cost:  Union[int, float, List[int], List[float]]
    shipment_cost: Union[List[List[int]], List[List[float]]]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of holding_cost parameter
    @field_validator("holding_cost")
    @classmethod
    def _check_holding_cost_nonneg(cls, v):
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("holding_cost must be non-negative")
        elif isinstance(v, list):
            if any((not isinstance(x, (int, float))) or x < 0 for x in v):
                raise ValueError("holding_cost list must contain only non-negative floats")
        else:
            raise ValueError("holding_cost must be a float or list of floats")
        return v

    # Type and shape of penalty_cost parameter
    @field_validator("penalty_cost")
    @classmethod
    def _check_penalty_cost_nonneg(cls, v):
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("penalty_cost must be non-negative")
        elif isinstance(v, list):
            if any((not isinstance(x, (int, float))) or x < 0 for x in v):
                raise ValueError("penalty_cost list must contain only non-negative floats")
        else:
            raise ValueError("penalty_cost must be a float or list of floats")
        return v

    # Type and shape of shipment_cost parameter
    @field_validator("shipment_cost")
    @classmethod
    def _check_shipment_cost_is_rectangular(cls, v: List[List[float]]):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("shipment_cost must be a non-empty 2D list")
        row_len = None
        for row in v:
            if not isinstance(row, list) or len(row) == 0:
                raise ValueError("shipment_cost must be a 2D list of floats")
            if row_len is None:
                row_len = len(row)
            elif len(row) != row_len:
                raise ValueError("shipment_cost must be rectangular (all rows same length)")
            for x in row:
                if not isinstance(x, (int, float)):
                    raise ValueError("shipment_cost entries must be floats")
                if x < 0:
                    raise ValueError("shipment_cost entries must be non-negative")
        return v


# ============================================================================
# Component Config Schemas
# ============================================================================

#  ---------- Demand sampler ----------
# Poisson sampler
class DemandSamplerPoisson(BaseModel):
    """Configuration for Poisson demand sampler."""

    type: Literal["poisson"]
    params: Dict[Literal["lambda_orders", "lambda_skus", "lambda_quantity"], float]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of lambda parameters
    @model_validator(mode="after")
    def _check_lambdas(self):
        lambda_orders = self.params["lambda_orders"]
        lambda_skus = self.params["lambda_skus"]
        lambda_quantity = self.params["lambda_quantity"]
        if not isinstance(lambda_orders, (int, float)) or not isinstance(lambda_skus, (int, float)) or not isinstance(lambda_quantity, (int, float)):
            raise ValueError("poisson demand_sampler lambda_orders, lambda_skus, and lambda_quantity must be floats")
        if lambda_orders <= 0 or lambda_skus <= 0 or lambda_quantity <= 0:
            raise ValueError("poisson demand_sampler lambda_orders, lambda_skus, and lambda_quantity must be positive")
        return self

# Empirical sampler
class DemandSamplerEmpirical(BaseModel):
    """Configuration for empirical demand sampler."""

    type: Literal["empirical"]
    params: Optional[None] = None  # No params needed - episode_length comes from environment config
    model_config = ConfigDict(extra="forbid")

# Union of demand sampler configurations
DemandSamplerConfig = Union[DemandSamplerPoisson, DemandSamplerEmpirical]


#  ---------- Demand allocator ----------
# Greedy allocator
class DemandAllocatorGreedy(BaseModel):
    """Configuration for greedy demand allocator."""

    type: Literal["greedy"]
    params: Dict[Literal["max_splits"], Union[Literal["default"], PositiveInt]]
    model_config = ConfigDict(extra="forbid")

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
    params: Dict[Literal["min", "max"], int]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of uniform parameters
    @model_validator(mode="after")
    def _check_uniform_params(self):
        mn = self.params["min"]
        mx = self.params["max"]
        if mn <= 0 or mx < 0:
            raise ValueError("uniform params min/max must be non-negative")
        if mn >= mx:
            raise ValueError("uniform params must satisfy min < max")
        return self

# Union of lead time sampler configurations
LeadTimeSamplerConfig = Union[LeadTimeSamplerUniform]


#  ---------- Lost sales handler ---------
# Cheapest assignment
class LostSalesCheapest(BaseModel):
    """Configuration for cheapest lost sales assignment."""

    type: Literal["cheapest"]
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
    params: Dict[Literal["alpha"], Union[int, float]]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of alpha parameter
    @field_validator("params")
    @classmethod
    def _check_alpha_nonneg(cls, v: Dict[str, Any]):
        alpha = v["alpha"]
        if not isinstance(alpha, (int, float)):
            raise ValueError("cost params.alpha must be a number")
        if alpha < 0:
            raise ValueError("cost params.alpha must be positive")
        return v

# Union of lost sales handler configurations
LostSalesHandlerConfig = Union[LostSalesCheapest, LostSalesShipment, LostSalesCost]


#  ---------- Reward calculator----------
# Cost-based reward calculator
class RewardCalculatorCost(BaseModel):
    """Configuration for cost-based reward calculator."""

    type: Literal["cost"]
    params: Dict[
        Literal["scope", "scale_factor", "normalize", "cost_weights"],
        Any
    ]
    model_config = ConfigDict(extra="forbid")

    # Type and shape of reward calculator parameters
    @field_validator("params")
    @classmethod
    def _validate_params_types_and_ranges(cls, v: Dict[str, Any]):
        # Scope
        scope = v.get("scope")
        if scope not in ("team", "agent"):
            raise ValueError("reward_calculator.params.scope must be 'team' or 'agent'")

        # Scale_factor
        sf = v.get("scale_factor")
        if not isinstance(sf, (int, float)):
            raise ValueError("reward_calculator.params.scale_factor must be an int or float")
        if sf <= 0:
            raise ValueError("reward_calculator.params.scale_factor must be > 0")

        # Normalize
        norm = v.get("normalize")
        if not isinstance(norm, bool):
            raise ValueError("reward_calculator.params.normalize must be a bool")

        # Cost_weights
        cw = v.get("cost_weights")
        if not isinstance(cw, list) or len(cw) == 0:
            raise ValueError("reward_calculator.params.cost_weights must be a non-empty list")
        total = 0
        for x in cw:
            if not isinstance(x, (int, float)):
                raise ValueError("reward_calculator.params.cost_weights entries must be int/float")
            if x < 0 or x > 1:
                raise ValueError("reward_calculator.params.cost_weights entries must be in [0, 1]")
            total += float(x)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reward_calculator.params.cost_weights must sum to 1.0 (got {total})")

        return v

    # Length of cost_weights must match number of cost types
    @model_validator(mode="after")
    def _check_cost_weights_length(self):
        expected = len(CostStructureConfig.model_fields)
        actual = len(self.params["cost_weights"])
        if actual != expected:
            raise ValueError(
                f"reward_calculator.params.cost_weights must have length {expected} "
                f"(one weight per cost type: {list(CostStructureConfig.model_fields.keys())})"
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

        # Holding_cost: scalar or (n_warehouses,)
        hc = self.cost_structure.holding_cost
        if isinstance(hc, list) and len(hc) != nw:
            raise ValueError(f"holding_cost list must have length n_warehouses={nw}")

        # Penalty_cost: scalar or (n_skus,)
        pc = self.cost_structure.penalty_cost
        if isinstance(pc, list) and len(pc) != ns:
            raise ValueError(f"penalty_cost list must have length n_skus={ns}")

        # Shipment_cost: (n_warehouses, n_regions)
        sc = self.cost_structure.shipment_cost
        if len(sc) != nw:
            raise ValueError(f"shipment_cost must have {nw} rows (n_warehouses)")
        if any(len(row) != nr for row in sc):
            raise ValueError(f"shipment_cost must have {nr} columns (n_regions) in every row")

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

        # Data source validation: empirical sampler requires real_world data_source
        if isinstance(self.components.demand_sampler, DemandSamplerEmpirical):
            if self.data_source.type != "real_world":
                raise ValueError(
                    "demand_sampler.type='empirical' requires data_source.type='real_world', "
                    f"got data_source.type='{self.data_source.type}'"
                )

        return self