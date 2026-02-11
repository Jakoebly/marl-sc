import pytest
import numpy as np

from src.environment.components.demand_sampler import DemandSampler
from src.environment.components.demand_allocator import DemandAllocator
from src.environment.components.lead_time_sampler import LeadTimeSampler
from src.environment.components.lost_sales_handler import LostSalesHandler
from src.environment.components.reward_calculator import RewardCalculator


class TestDemandSampler:
    """Tests for DemandSampler base class."""
    
    def test_cannot_instantiate_base_class(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            DemandSampler()
    
    def test_concrete_implementation_must_implement_methods(self):
        """Test that concrete implementations must implement all methods."""
        class IncompleteDemandSampler(DemandSampler):
            def sample(self, warehouse_id, sku_id, timestep):
                return 5.0
            # Missing reset method
        
        with pytest.raises(TypeError):
            IncompleteDemandSampler()
    
    def test_complete_implementation_can_be_instantiated(self):
        """Test that complete implementation can be instantiated."""
        class CompleteDemandSampler(DemandSampler):
            def sample(self, warehouse_id, sku_id, timestep):
                return 5.0
            
            def reset(self, seed=None):
                pass
        
        sampler = CompleteDemandSampler()
        assert sampler.sample(0, 0, 0) == 5.0


class TestDemandAllocator:
    """Tests for DemandAllocator base class."""
    
    def test_cannot_instantiate_base_class(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            DemandAllocator()
    
    def test_complete_implementation(self):
        """Test that complete implementation works."""
        class CompleteDemandAllocator(DemandAllocator):
            def allocate(self, unfulfilled_demand, available_inventories):
                # Simple implementation: no allocation
                n_warehouses, n_skus = unfulfilled_demand.shape
                return np.zeros((n_warehouses, n_warehouses, n_skus))
        
        allocator = CompleteDemandAllocator()
        demand = np.array([[1.0, 2.0], [3.0, 4.0]])
        inventory = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = allocator.allocate(demand, inventory)
        assert result.shape == (2, 2, 2)


class TestLeadTimeSampler:
    """Tests for LeadTimeSampler base class."""
    
    def test_cannot_instantiate_base_class(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            LeadTimeSampler()
    
    def test_complete_implementation(self):
        """Test that complete implementation works."""
        class CompleteLeadTimeSampler(LeadTimeSampler):
            def sample(self, sku_id):
                return 3
            
            def reset(self, seed=None):
                pass
        
        sampler = CompleteLeadTimeSampler()
        assert sampler.sample(0) == 3


class TestLostSalesHandler:
    """Tests for LostSalesHandler base class."""
    
    def test_cannot_instantiate_base_class(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            LostSalesHandler()
    
    def test_complete_implementation(self):
        """Test that complete implementation works."""
        class CompleteLostSalesHandler(LostSalesHandler):
            def calculate_lost_sales(self, unfulfilled_demand):
                return unfulfilled_demand
        
        handler = CompleteLostSalesHandler()
        demand = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = handler.calculate_lost_sales(demand)
        np.testing.assert_array_equal(result, demand)


class TestRewardCalculator:
    """Tests for RewardCalculator base class."""
    
    def test_cannot_instantiate_base_class(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            RewardCalculator()
    
    def test_complete_implementation(self):
        """Test that complete implementation works."""
        class CompleteRewardCalculator(RewardCalculator):
            def calculate(self, holding_costs, penalty_costs, shipment_costs):
                # Simple: sum all costs per warehouse
                holding_total = holding_costs.sum(axis=1)
                penalty_total = penalty_costs.sum(axis=1)
                shipment_total = shipment_costs.sum(axis=(1, 2))
                return -(holding_total + penalty_total + shipment_total)
        
        calculator = CompleteRewardCalculator()
        holding = np.array([[1.0, 2.0], [3.0, 4.0]])
        penalty = np.array([[0.5, 1.0], [1.5, 2.0]])
        shipment = np.zeros((2, 2, 2))
        result = calculator.calculate(holding, penalty, shipment)
        assert result.shape == (2,)
        assert result[0] == -(3.0 + 1.5)  # -(1+2) - (0.5+1.0)

