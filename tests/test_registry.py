import pytest
import numpy as np

from src.environment.components.registry import (
    register_demand_sampler,
    get_demand_sampler,
    register_demand_allocator,
    get_demand_allocator,
    register_lead_time_sampler,
    get_lead_time_sampler,
    register_lost_sales_handler,
    get_lost_sales_handler,
    register_reward_calculator,
    get_reward_calculator,
    DEMAND_SAMPLER_REGISTRY,
    DEMAND_ALLOCATOR_REGISTRY,
    LEAD_TIME_SAMPLER_REGISTRY,
    LOST_SALES_HANDLER_REGISTRY,
    REWARD_CALCULATOR_REGISTRY
)
from src.environment.components.demand_sampler import DemandSampler
from src.environment.components.demand_allocator import DemandAllocator
from src.environment.components.lead_time_sampler import LeadTimeSampler
from src.environment.components.lost_sales_handler import LostSalesHandler
from src.environment.components.reward_calculator import RewardCalculator


class TestDemandSamplerRegistry:
    """Tests for demand sampler registry."""
    
    def test_register_and_get_sampler(self):
        """Test registering and retrieving a sampler."""
        class TestSampler(DemandSampler):
            def __init__(self, param1=5.0):
                self.param1 = param1
            
            def sample(self, warehouse_id, sku_id, timestep):
                return self.param1
            
            def reset(self, seed=None):
                pass
        
        register_demand_sampler("test_sampler", TestSampler)
        assert "test_sampler" in DEMAND_SAMPLER_REGISTRY
        
        config = {"params": {"param1": 10.0}}
        sampler = get_demand_sampler("test_sampler", config)
        assert isinstance(sampler, TestSampler)
        assert sampler.param1 == 10.0
        assert sampler.sample(0, 0, 0) == 10.0
    
    def test_get_unknown_sampler_raises_error(self):
        """Test that getting unknown sampler raises error."""
        with pytest.raises(ValueError, match="Unknown demand sampler"):
            get_demand_sampler("nonexistent", {})


class TestDemandAllocatorRegistry:
    """Tests for demand allocator registry."""
    
    def test_register_and_get_allocator(self):
        """Test registering and retrieving an allocator."""
        class TestAllocator(DemandAllocator):
            def __init__(self, param1=True):
                self.param1 = param1
            
            def allocate(self, unfulfilled_demand, available_inventories):
                n_warehouses, n_skus = unfulfilled_demand.shape
                return np.zeros((n_warehouses, n_warehouses, n_skus))
        
        register_demand_allocator("test_allocator", TestAllocator)
        assert "test_allocator" in DEMAND_ALLOCATOR_REGISTRY
        
        config = {"params": {"param1": False}}
        allocator = get_demand_allocator("test_allocator", config)
        assert isinstance(allocator, TestAllocator)
        assert allocator.param1 == False
    
    def test_get_unknown_allocator_raises_error(self):
        """Test that getting unknown allocator raises error."""
        with pytest.raises(ValueError, match="Unknown demand allocator"):
            get_demand_allocator("nonexistent", {})


class TestLeadTimeSamplerRegistry:
    """Tests for lead time sampler registry."""
    
    def test_register_and_get_sampler(self):
        """Test registering and retrieving a lead time sampler."""
        class TestSampler(LeadTimeSampler):
            def __init__(self, min_time=1):
                self.min_time = min_time
            
            def sample(self, sku_id):
                return self.min_time
            
            def reset(self, seed=None):
                pass
        
        register_lead_time_sampler("test_lead_sampler", TestSampler)
        assert "test_lead_sampler" in LEAD_TIME_SAMPLER_REGISTRY
        
        config = {"params": {"min_time": 5}}
        sampler = get_lead_time_sampler("test_lead_sampler", config)
        assert isinstance(sampler, TestSampler)
        assert sampler.min_time == 5
    
    def test_get_unknown_sampler_raises_error(self):
        """Test that getting unknown sampler raises error."""
        with pytest.raises(ValueError, match="Unknown lead time sampler"):
            get_lead_time_sampler("nonexistent", {})


class TestLostSalesHandlerRegistry:
    """Tests for lost sales handler registry."""
    
    def test_register_and_get_handler(self):
        """Test registering and retrieving a lost sales handler."""
        class TestHandler(LostSalesHandler):
            def __init__(self, scale=1.0):
                self.scale = scale
            
            def calculate_lost_sales(self, unfulfilled_demand):
                return unfulfilled_demand * self.scale
        
        register_lost_sales_handler("test_handler", TestHandler)
        assert "test_handler" in LOST_SALES_HANDLER_REGISTRY
        
        config = {"params": {"scale": 2.0}}
        handler = get_lost_sales_handler("test_handler", config)
        assert isinstance(handler, TestHandler)
        assert handler.scale == 2.0
        
        demand = np.array([[1.0, 2.0]])
        result = handler.calculate_lost_sales(demand)
        np.testing.assert_array_equal(result, np.array([[2.0, 4.0]]))
    
    def test_get_unknown_handler_raises_error(self):
        """Test that getting unknown handler raises error."""
        with pytest.raises(ValueError, match="Unknown lost sales handler"):
            get_lost_sales_handler("nonexistent", {})


class TestRewardCalculatorRegistry:
    """Tests for reward calculator registry."""
    
    def test_register_and_get_calculator(self):
        """Test registering and retrieving a reward calculator."""
        class TestCalculator(RewardCalculator):
            def __init__(self, scale=1.0):
                self.scale = scale
            
            def calculate(self, holding_costs, penalty_costs, shipment_costs):
                total = holding_costs.sum(axis=1) + penalty_costs.sum(axis=1)
                return -total * self.scale
        
        register_reward_calculator("test_calculator", TestCalculator)
        assert "test_calculator" in REWARD_CALCULATOR_REGISTRY
        
        config = {"params": {"scale": 0.5}}
        calculator = get_reward_calculator("test_calculator", config)
        assert isinstance(calculator, TestCalculator)
        assert calculator.scale == 0.5
    
    def test_get_unknown_calculator_raises_error(self):
        """Test that getting unknown calculator raises error."""
        with pytest.raises(ValueError, match="Unknown reward calculator"):
            get_reward_calculator("nonexistent", {})

