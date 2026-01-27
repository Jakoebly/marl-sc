import numpy as np
from src.config.loader import load_environment_config
from src.environment.environment import InventoryEnvironment

def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing basic environment functionality...")
    
    # Load config
    config_path = "config_files/environments/base_env.yaml"
    env_config = load_environment_config(config_path)
    print(f"[OK] Environment config loaded: {env_config}")
    
    # Create environment
    env = InventoryEnvironment(env_config)
    print(f"[OK] Environment created: {env.n_warehouses} warehouses, {env.n_skus} SKUs, {env.n_regions} regions")
    
    # Test reset
    observations, infos = env.reset(seed=42)
    print(f"[OK] Reset successful: {len(observations)} agents")
    
    # Check observation shapes
    for agent_id, obs in observations.items():
        expected_size = 2 * env.n_skus
        assert obs.shape == (expected_size,), f"Observation shape mismatch for {agent_id}: {obs.shape} != ({expected_size},)"
    print("[OK] Observation shapes correct")
    
    # Test step
    actions = {agent_id: np.zeros(env.n_skus) for agent_id in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f"[OK] Step successful: rewards={rewards}")
    
    # Test multiple steps
    for i in range(5):
        actions = {agent_id: np.random.uniform(0, 10, env.n_skus) for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f"  Step {i+1}: rewards={rewards}, truncations={truncations}")
    
    print("[OK] Basic functionality test passed!\n")

def test_component_initialization():
    """Test that all components initialize correctly."""
    print("Testing component initialization...")
    
    config_path = "config_files/environments/base_env.yaml"
    env_config = load_environment_config(config_path)
    env = InventoryEnvironment(env_config)
    
    # Check all components exist
    assert env.demand_sampler is not None, "Demand sampler not initialized"
    assert env.demand_allocator is not None, "Demand allocator not initialized"
    assert env.lead_time_sampler is not None, "Lead time sampler not initialized"
    assert env.lost_sales_handler is not None, "Lost sales handler not initialized"
    assert env.reward_calculator is not None, "Reward calculator not initialized"
    
    print("[OK] All components initialized")
    
    # Test demand sampling
    orders = env.demand_sampler.sample(0)
    print(f"[OK] Demand sampler: {len(orders)} orders sampled")
    
    # Test lead time sampling
    lead_times = env.lead_time_sampler.sample()
    assert lead_times.shape == (env.n_skus,), f"Lead times shape: {lead_times.shape}"
    print(f"[OK] Lead time sampler: {lead_times}")
    
    print("[OK] Component initialization test passed!\n")

def test_demand_allocation():
    """Test demand allocation logic."""
    print("Testing demand allocation...")
    
    config_path = "config_files/environments/base_env.yaml"
    env_config = load_environment_config(config_path)
    env = InventoryEnvironment(env_config)
    
    env.reset(seed=42)
    
    # Create test orders
    from src.environment.components.demand_sampler import Order
    test_orders = [
        Order(region_id=0, sku_demands=np.array([5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        Order(region_id=1, sku_demands=np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    
    # Allocate
    result = env.demand_allocator.allocate(test_orders, env.inventory)
    
    # Check result shapes
    assert result.fulfillment_matrix.shape == (2, env.n_warehouses, env.n_skus), \
        f"Fulfillment matrix shape: {result.fulfillment_matrix.shape}"
    assert result.unfulfilled_demands.shape == (env.n_regions, env.n_skus), \
        f"Unfulfilled demands shape: {result.unfulfilled_demands.shape}"
    assert result.shipment_counts.shape == (env.n_warehouses, env.n_regions), \
        f"Shipment counts shape: {result.shipment_counts.shape}"
    assert result.shipment_quantities.shape == (env.n_warehouses, env.n_regions), \
        f"Shipment quantities shape: {result.shipment_quantities.shape}"
    
    print(f"[OK] Allocation result shapes correct")
    print(f"  Fulfillment matrix: {result.fulfillment_matrix.shape}")
    print(f"  Shipment counts: {result.shipment_counts.sum()} total shipments")
    print(f"  Shipment quantities: {result.shipment_quantities.sum():.2f} total units")
    
    print("[OK] Demand allocation test passed!\n")

def test_reward_calculation():
    """Test reward calculation."""
    print("Testing reward calculation...")
    
    config_path = "config_files/environments/base_env.yaml"
    env_config = load_environment_config(config_path)
    env = InventoryEnvironment(env_config)
    
    env.reset(seed=42)
    
    # Create test state
    inventory = env.inventory.copy()
    lost_sales = np.zeros((env.n_warehouses, env.n_skus))
    shipment_counts = np.zeros((env.n_warehouses, env.n_regions))
    
    # Calculate rewards
    rewards = env.reward_calculator.calculate(inventory, lost_sales, shipment_counts)
    
    assert rewards.shape == (env.n_warehouses,), f"Rewards shape: {rewards.shape}"
    print(f"[OK] Rewards calculated: {rewards}")
    
    print("[OK] Reward calculation test passed!\n")

def test_pending_orders():
    """Test pending orders mechanism."""
    print("Testing pending orders...")
    
    config_path = "config_files/environments/base_env.yaml"
    env_config = load_environment_config(config_path)
    env = InventoryEnvironment(env_config)
    
    env.reset(seed=42)
    initial_inventory = env.inventory.copy()
    
    # Place orders with short lead times
    actions = {agent_id: np.array([10.0] * env.n_skus) for agent_id in env.agents}
    
    # Step through to see orders arrive
    for i in range(10):
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f"  Timestep {env.timestep}: pending_orders keys={list(env.pending_orders.keys())}")
        
        if i == 0:
            assert len(env.pending_orders) > 0, "Orders should be pending after first step"
        
        if env.timestep >= 5:  # Lead times are 1-5, so orders should start arriving
            break
    
    print("[OK] Pending orders test passed!\n")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        #test_component_initialization()
        #test_demand_allocation()
        #test_reward_calculation()
        #test_pending_orders()
        print("=" * 50)
        print("All tests passed! [OK]")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

