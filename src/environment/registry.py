from typing import Dict, Type

from .components.demand_sampler import BaseDemandSampler, PoissonDemandSampler, EmpiricalDemandSampler
from .components.demand_allocator import BaseDemandAllocator, GreedyDemandAllocator
from .components.lead_time_sampler import BaseLeadTimeSampler, UniformLeadTimeSampler
from .components.lost_sales_handler import BaseLostSalesHandler, CheapestLostSalesHandler, ShipmentLostSalesHandler, CostLostSalesHandler
from .components.reward_calculator import BaseRewardCalculator, CostRewardCalculator
from src.environment.context import EnvironmentContext, create_environment_context
from src.config.schema import EnvironmentConfig


# Define separate registries for each component type to map component name to component class 
DEMAND_SAMPLER_REGISTRY: Dict[str, Type[BaseDemandSampler]] = {}
DEMAND_ALLOCATOR_REGISTRY: Dict[str, Type[BaseDemandAllocator]] = {}
LEAD_TIME_SAMPLER_REGISTRY: Dict[str, Type[BaseLeadTimeSampler]] = {}
LOST_SALES_HANDLER_REGISTRY: Dict[str, Type[BaseLostSalesHandler]] = {}
REWARD_CALCULATOR_REGISTRY: Dict[str, Type[BaseRewardCalculator]] = {}


# ============================================================================
# Demand Sampler Registry
# ============================================================================

def register_demand_sampler(name: str, sampler_class: Type[BaseDemandSampler]):
    """
    Registers a demand sampler implementation.
    
    Args:
        name (str): Unique identifier for the sampler (e.g., "poisson", "empirical").
        sampler_class (Type[BaseDemandSampler]): Class implementing the BaseDemandSampler interface.
    """

    DEMAND_SAMPLER_REGISTRY[name] = sampler_class

def get_demand_sampler(env_config: EnvironmentConfig) -> BaseDemandSampler:
    """
    Builds the demand sampler component by creating an EnvironmentContext, 
    extracting the demand sampler config, looking up the sampler class in the 
    registry, and instantiating it.

    Args:
        env_config (EnvironmentConfig): Full environment configuration.

    Returns:
        demand_sampler (BaseDemandSampler): Instantiated demand sampler component.
    """

    # Extract the demand sampler config and type from the environment configuration
    component_config = env_config.components.demand_sampler
    sampler_type = component_config.type
    
    # Check if the sampler type is registered
    if sampler_type not in DEMAND_SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown demand sampler: {sampler_type}. "
            f"Available: {list(DEMAND_SAMPLER_REGISTRY.keys())}"
        )
    
    # Create context and instantiate component
    context = create_environment_context(env_config)
    sampler_class = DEMAND_SAMPLER_REGISTRY[sampler_type]
    return sampler_class(context, component_config)


# ============================================================================
# Demand Allocator Registry
# ============================================================================

def register_demand_allocator(name: str, allocator_class: Type[BaseDemandAllocator]):
    """
    Registers a demand allocator implementation.
    
    Args:
        name (str): Unique identifier for the allocator (e.g., "greedy", "lp").
        allocator_class (Type[BaseDemandAllocator]): Class implementing the BaseDemandAllocator interface.
    """

    DEMAND_ALLOCATOR_REGISTRY[name] = allocator_class

def get_demand_allocator(env_config: EnvironmentConfig) -> BaseDemandAllocator:
    """
    Builds the demand allocator component by creating an EnvironmentContext, 
    extracting the demand allocator config, looking up the allocator class in the 
    registry, and instantiating it.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        
    Returns:
        demand_allocator (BaseDemandAllocator): Instantiated demand allocator component.
    """
    
    # Extract the demand allocator config and type from the environment configuration
    component_config = env_config.components.demand_allocator
    allocator_type = component_config.type
    
    # Check if the allocator type is registered
    if allocator_type not in DEMAND_ALLOCATOR_REGISTRY:
        raise ValueError(
            f"Unknown demand allocator: {allocator_type}. "
            f"Available: {list(DEMAND_ALLOCATOR_REGISTRY.keys())}"
        )
    
    # Create context and instantiate component
    context = create_environment_context(env_config)
    allocator_class = DEMAND_ALLOCATOR_REGISTRY[allocator_type]
    return allocator_class(context, component_config)


# ============================================================================
# Lead Time Sampler Registry
# ============================================================================

def register_lead_time_sampler(name: str, sampler_class: Type[BaseLeadTimeSampler]):
    """
    Registers a lead time sampler implementation.
    
    Args:
        name (str): Unique identifier for the sampler (e.g., "uniform").
        sampler_class (Type[BaseLeadTimeSampler]): Class implementing the BaseLeadTimeSampler interface.
    """

    LEAD_TIME_SAMPLER_REGISTRY[name] = sampler_class

def get_lead_time_sampler(env_config: EnvironmentConfig) -> BaseLeadTimeSampler:
    """
    Builds the lead time sampler component by creating an EnvironmentContext, 
    extracting the lead time sampler config, looking up the sampler class in the 
    registry, and instantiating it.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        
    Returns:
        lead_time_sampler (BaseLeadTimeSampler): Instantiated lead time sampler component.
    """

    # Extract the lead time sampler config and type from the environment configuration
    component_config = env_config.components.lead_time_sampler
    sampler_type = component_config.type
    
    # Check if the sampler type is registered
    if sampler_type not in LEAD_TIME_SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown lead time sampler: {sampler_type}. "
            f"Available: {list(LEAD_TIME_SAMPLER_REGISTRY.keys())}"
        )
    
    # Create context and instantiate component
    context = create_environment_context(env_config)
    sampler_class = LEAD_TIME_SAMPLER_REGISTRY[sampler_type]
    return sampler_class(context, component_config)


# ============================================================================
# Lost Sales Handler Registry
# ============================================================================

def register_lost_sales_handler(name: str, handler_class: Type[BaseLostSalesHandler]):
    """
    Registers a lost sales handler implementation.
    
    Args:
        name (str): Unique identifier for the handler (e.g., "cheapest", "shipment", "cost").
        handler_class (Type[BaseLostSalesHandler]): Class implementing the BaseLostSalesHandler interface.
    """

    LOST_SALES_HANDLER_REGISTRY[name] = handler_class

def get_lost_sales_handler(env_config: EnvironmentConfig) -> BaseLostSalesHandler:
    """
    Builds the lost sales handler component by creating an EnvironmentContext, 
    extracting the lost sales handler config, looking up the handler class in the 
    registry, and instantiating it.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        
    Returns:
        lost_sales_handler (BaseLostSalesHandler): Instantiated lost sales handler component.
    """

    # Extract the lost sales handler config and type from the environment configuration
    component_config = env_config.components.lost_sales_handler
    handler_type = component_config.type
    
    # Check if the handler type is registered
    if handler_type not in LOST_SALES_HANDLER_REGISTRY:
        raise ValueError(
            f"Unknown lost sales handler: {handler_type}. "
            f"Available: {list(LOST_SALES_HANDLER_REGISTRY.keys())}"
        )
    
    # Create context and instantiate component
    context = create_environment_context(env_config)
    handler_class = LOST_SALES_HANDLER_REGISTRY[handler_type]
    return handler_class(context, component_config)


# ============================================================================
# Reward Calculator Registry
# ============================================================================

def register_reward_calculator(name: str, calculator_class: Type[BaseRewardCalculator]):
    """
    Registers a reward calculator implementation.
    
    Args:
        name (str): Unique identifier for the calculator (e.g., "cost").
        calculator_class (Type[BaseRewardCalculator]): Class implementing the BaseRewardCalculator interface.
    """

    REWARD_CALCULATOR_REGISTRY[name] = calculator_class

def get_reward_calculator(env_config: EnvironmentConfig) -> BaseRewardCalculator:
    """
    Builds the reward calculator component by creating an EnvironmentContext, 
    extracting the reward calculator config, looking up the calculator class in the 
    registry, and instantiating it.
    
    Args:
        env_config (EnvironmentConfig): Full environment configuration.
        
    Returns:
        reward_calculator (BaseRewardCalculator): Instantiated reward calculator component.
    """

    # Extract the reward calculator config and type from the environment configuration
    component_config = env_config.components.reward_calculator
    calculator_type = component_config.type
    
    # Check if the calculator type is registered
    if calculator_type not in REWARD_CALCULATOR_REGISTRY:
        raise ValueError(
            f"Unknown reward calculator: {calculator_type}. "
            f"Available: {list(REWARD_CALCULATOR_REGISTRY.keys())}"
        )
    
    # Create context and instantiate component
    context = create_environment_context(env_config)
    calculator_class = REWARD_CALCULATOR_REGISTRY[calculator_type]
    return calculator_class(context, component_config)


# Register implementations
register_demand_sampler("poisson", PoissonDemandSampler)
register_demand_sampler("empirical", EmpiricalDemandSampler)
register_demand_allocator("greedy", GreedyDemandAllocator)
register_lead_time_sampler("uniform", UniformLeadTimeSampler)
register_lost_sales_handler("cheapest", CheapestLostSalesHandler)
register_lost_sales_handler("shipment", ShipmentLostSalesHandler)
register_lost_sales_handler("cost", CostLostSalesHandler)
register_reward_calculator("cost", CostRewardCalculator)

