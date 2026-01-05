from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from src.config.schema import EnvironmentConfig, InitialInventoryUniform, InitialInventoryCustom, InitialInventoryZero
from src.environment.registry import (
    get_demand_sampler,
    get_demand_allocator,
    get_lead_time_sampler,
    get_lost_sales_handler,
    get_reward_calculator,
)
from src.environment.components.base import StochasticComponent
from src.environment.components.demand_sampler import Order
from src.environment.components.demand_allocator import AllocationResult
from numpy.random import SeedSequence
from src.utils.seed_manager import SeedManager, ENVIRONMENT_SEED_REGISTRY


class InventoryEnvironment(ParallelEnv):
    """
    Implements a multi-agent inventory management environment compatible with PettingZoo. The environment 
    orchestrates demand sampling, order allocation, lead times, lost sales handling, and reward calculation 
    through pluggable components. Each warehouse is an agent that makes replenishment decisions.
    
    Attributes:
        n_warehouses (int): Number of warehouses.
        n_skus (int): Number of stock-keeping units.
        n_regions (int): Number of demand regions.
        episode_length (int): Maximum number of timesteps per episode.
        inventory (np.ndarray): Current inventory levels. Shape: (n_warehouses, n_skus).
        pending_orders (Dict[int, np.ndarray]): Orders scheduled to arrive at a given timestep. 
            Shape: {arrival_timestep: (n_warehouses, n_skus)}.
        timestep (int): Current timestep in the episode.
    """
    
    metadata = {"render_modes": ["human"], "name": "inventory_env_v0"}
    
    def __init__(self, env_config: EnvironmentConfig, seed: Optional[int] = None):
        """
        Initializes the environment from configuration.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration.
            seed (Optional[int]): Root seed for reproducibility. If provided, used for preprocessing and stored
                for reset(). Defaults to None.
        """

        # Store general environment parameters
        self.n_warehouses = env_config.n_warehouses 
        self.n_skus = env_config.n_skus
        self.n_regions = env_config.n_regions
        self.episode_length = env_config.episode_length
        self.max_order_quantity = env_config.max_order_quantity
        
        # Initialize seed manager
        self.seed_manager = SeedManager(root_seed=seed, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        
        # Create environment context
        from src.environment.context import create_environment_context
        context = create_environment_context(env_config, seed_manager=self.seed_manager)
        
        # Initialize components via registry
        self.demand_sampler = get_demand_sampler(env_config, context=context)
        self.demand_allocator = get_demand_allocator(env_config, context=context)
        self.lead_time_sampler = get_lead_time_sampler(env_config, context=context)
        self.lost_sales_handler = get_lost_sales_handler(env_config, context=context)
        self.reward_calculator = get_reward_calculator(env_config, context=context)
        
        # Store cost structure and initial inventory config
        self.cost_structure = env_config.cost_structure
        self.initial_inventory_config = env_config.initial_inventory

        # Pre-compute closest warehouses for each region
        self.closest_warehouses = np.argmin(context.shipment_cost, axis=0)  # Shape: (n_regions,)
        
        # Set agent IDs
        self.agents = [f"warehouse_{i}" for i in range(self.n_warehouses)]
        self.possible_agents = self.agents.copy()

        # Initialize state
        self.inventory = None  # Shape: (n_warehouses, n_skus)
        self.pending_orders = {}  # Shape: {arrival_timestep: (n_warehouses, n_skus)}
        self.timestep = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Resets the environment to its initial state by resetting all stochastic components with independent seeds, 
        initializing inventory, clearing pending orders, and returning initial observations.
        
        Args:
            seed (Optional[int]): Random seed. If None, stochastic components are reset without explicit seeds.
            options (Optional[Dict]): Optional dictionary of reset options (unused).
            
        Returns:
            Tuple containing:
                - observations (Dict[str, np.ndarray]): Dictionary mapping agent_id to observation array.
                    Shape: {warehouse_id: (2 * n_skus,)}.
                - infos (Dict[str, Dict]): Dictionary mapping agent_id to info dict (empty for now).
        """

        # Update root seed if provided, otherwise use stored seeds
        if seed is not None:
            self.seed_manager.update_root_seed(seed)
        
        # Get seeds for stochastic components and initial inventory
        from src.environment.components.base import STOCHASTIC_COMPONENT_REGISTRY
        stochastic_seeds = self.seed_manager.get_seeds_int_for_components(STOCHASTIC_COMPONENT_REGISTRY)
        inventory_seed = self.seed_manager.get_seed_int('inventory')
        
        # Reset stochastic components
        self._reset_stochastic_components(seeds=stochastic_seeds)      

        # Reset state
        self.inventory = self._initialize_inventory(seed=inventory_seed)
        self.pending_orders = {}
        self.timestep = 0
        
        # Get initial observations
        observations = self._get_observations() # Shape: {warehouse_id: (2 * n_skus,)}

        # Initialize infos
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Executes one environment step by applying the following sequence of events:

            1. Place replenishment orders and update pending orders
            2. Receive replenishment orders that are scheduled to arrive in this timestep
            3. Sample new customer demand
            4. Allocate and ship demand across warehouses
            5. Update inventory levels
            6. Assign lost sales from regions to warehouses
            7. Calculate rewards
            8. Update observations
            9. Increment timestep and check terminations/truncations
        
        Args:
            actions (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to action array.
                Shape: {agent_id: (n_skus,)}.
                
        Returns:
            Tuple containing:
                - observations (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to observation array.
                    Shape: {warehouse_id: (2 * n_skus,)}.
                - rewards (Dict[str, float]): Dictionary mapping warehouse_id to reward array.
                - terminations (Dict[str, bool]): Dictionary mapping warehouse_id to termination flag.
                - truncations (Dict[str, bool]): Dictionary mapping warehouse_id to truncation flag.
                - infos (Dict[str, Dict]): Dictionary mapping warehouse_id to info dictionary (empty for now).
        """

        # 1. Rescale normalized actions to order quantities and place replenishment orders
        order_quantities = self._rescale_actions_to_quantities(actions)
        self._apply_orders(actions=order_quantities)

        # 2. Receive replenishment orders arriving in this timestep
        self._apply_arrivals()
        
        # 3. Sample demand 
        orders = self.demand_sampler.sample(self.timestep)
    
        # 4. Allocate and ship orders
        allocation_result = self.demand_allocator.allocate(orders, self.inventory)
        fulfillment_matrix = allocation_result.fulfillment_matrix  # Shape: (n_orders, n_warehouses, n_skus)
        shipment_counts = allocation_result.shipment_counts  # Shape: (n_warehouses, n_regions)
        shipment_quantities = allocation_result.shipment_quantities  # Shape: (n_warehouses, n_regions)
        unfulfilled_demands = allocation_result.unfulfilled_demands  # Shape: (n_regions, n_skus)
                
        # 5. Update inventories
        self.inventory = np.maximum(self.inventory - fulfillment_matrix.sum(axis=0), 0.0)
        
        # 6. Assign lost sales
        lost_sales = self.lost_sales_handler.calculate_lost_sales(unfulfilled_demands, shipment_quantities) # Shape: (n_warehouses, n_skus)
        
        # 7. Calculate rewards 
        rewards_array = self.reward_calculator.calculate(self.inventory, lost_sales, shipment_counts)  # Shape: (n_warehouses,)
        
        # Convert rewards array to dictionary keyed by warehouse IDs
        rewards = {agent_id: float(rewards_array[i]) for i, agent_id in enumerate(self.agents)}
        
        # 8. Update observations
        observations = self._get_observations()
        
        # 9. Increment timestep and check terminations and truncations
        self.timestep += 1
        terminations = {agent: False for agent in self.agents}  # No early termination
        truncations = {agent: (self.timestep >= self.episode_length) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def observation_space(self, agent: str) -> Box:
        """
        Returns the observation space for a warehouse consisting of the following features:        
        - Current inventory level
        - Sum of all pending orders
        
        Args:
            agent (str): Warehouse ID (unused, all warehouses have same observation space).
            
        Returns:
            observation_space (Box): Box space that contains the observation space for a warehouse. Shape: (2 * n_skus,).
        """

        # Set the observation dimension
        obs_size = 2 * self.n_skus

        # Create the observation space
        observation_space = Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)

        return observation_space
    
    def state_space(self) -> Box:
        """
        Returns the observation space for the global state.
        
        Returns:
            global_state_space (Box): Box space for global state.
                Shape: (n_warehouses * 2 * n_skus,).
        """

        # Set the observation dimension and global state size
        obs_dim = 2 * self.n_skus
        global_state_size = self.n_warehouses * obs_dim

        # Create the global state space
        global_state_space = Box(low=0.0, high=np.inf, shape=(global_state_size,), dtype=np.float32)

        return global_state_space

    def action_space(self, agent: str) -> Box:
        """
        Returns the action space (normalized replenishment order quantities) for a warehouse. Actions are in the
        range [-1, 1] and will be rescaled to [0, max_order_quantity] internally.

        Args:
            agent (str): Warehouse ID (unused, all warehouses have the same action space).
            
        Returns:
            action_space (Box): Box space that contains the action space for a warehouse. Shape: (n_skus,).
        """

        # Calculate the action space size
        act_size = self.n_skus

        # Create the normalized action space [-1, 1]
        action_space = Box(low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32)

        return action_space

    def render(self):
        """
        Renders the environment by [...]. Not implemented yet.
        """
        pass


    # ============================================================================
    # Environment Helper Functions
    # ============================================================================

    def _initialize_inventory(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Initializes inventory levels for each warehouse and SKU. 
        
        Args:
            seed (Optional[int]): Random seed for initialization (used for uniform method).
            
        Returns:
            inventory (np.ndarray): Initial inventory array. Shape: (n_warehouses, n_skus).
        """
       
        # Get the initial inventory method from config
        inv_config = self.initial_inventory_config
        
        # Use a temporary RNG for initialization (used for uniform method)
        rng = np.random.default_rng(seed)
        
        # Initialize inventory based on the initial inventory method
        # Uniform random initialization
        if isinstance(inv_config, InitialInventoryUniform):
            min_val = inv_config.params["min"]
            max_val = inv_config.params["max"]
            inventory = rng.integers(min_val, max_val + 1, size=(self.n_warehouses, self.n_skus))

        # Custom initialization
        elif isinstance(inv_config, InitialInventoryCustom):
            value = inv_config.params["value"]
            if isinstance(value, int): # Scalar
                inventory = np.full((self.n_warehouses, self.n_skus), value, dtype=int)
            else: # 2D list
                inventory = np.array(value, dtype=int)

        # Zero initialization
        elif isinstance(inv_config, InitialInventoryZero):
            inventory = np.zeros((self.n_warehouses, self.n_skus), dtype=int)
        
        return inventory.astype(float)
    
    def _reset_stochastic_components(self, seeds: Optional[List[SeedSequence]] = None):
        """
        Resets all stochastic components using pre-spawned seeds.
        
        Args:
            seeds (Optional[List[SeedSequence]]): List of seeds for stochastic components. If None, components are reset without explicit seeds.
        """

        # Get stochastic components from the environment
        from src.environment.components.base import STOCHASTIC_COMPONENT_REGISTRY
        stochastic_components = [
            getattr(self, attr_name) 
            for attr_name in STOCHASTIC_COMPONENT_REGISTRY
            if isinstance(getattr(self, attr_name), StochasticComponent)
        ]
        
        # If seeds are provided, ensure that they match the number of stochastic components and use them to reset the components
        if seeds is not None:
            if len(seeds) != len(stochastic_components):
                raise ValueError(
                    f"Number of seeds ({len(seeds)}) must match "
                    f"number of stochastic components ({len(stochastic_components)})"
                )
            
            for comp, seed_obj in zip(stochastic_components, seeds):
                if isinstance(seed_obj, SeedSequence):
                    seed_int = SeedManager.seed_to_int(seed_obj)
                else:
                    seed_int = seed_obj  # Already an int
                comp.reset(seed_int)

        # If no seeds provided, reset all stochastic components without explicit seeds
        else:
            for comp in stochastic_components:
                comp.reset(None)

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Builds per-warehouse observations consisting of the following features:        
        - Current inventory level
        - Sum of all pending orders
        
        Returns:
            observations (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to observation array.
                Shape: {warehouse_id: (2 * n_skus,)}.
        """

        # Initialize dictionary to store observations
        observations = {}
        
        # Build observations for each warehouse
        for warehouse_idx, warehouse_id in enumerate(self.agents):
            # Fetch all features for the current warehouse
            obs_inventory = self.inventory[warehouse_idx, :].copy()
            pending_total = np.zeros(self.n_skus, dtype=float)
            for orders_array in self.pending_orders.values():
                pending_total += orders_array[warehouse_idx, :]
            
            # Concatenate features and store them in the observations dictionary
            observation = np.concatenate([obs_inventory, pending_total])
            observations[warehouse_id] = observation
        
        return observations
    
    def get_global_state(self) -> np.ndarray:
        """
        Returns the global state by concatenating all agent observations.
        
        Returns:
            global_state (np.ndarray): Concatenated observations from all agents. Shape: (n_warehouses * obs_dim)
        """

        # Get observations for all agents
        observations = self._get_observations()
        
        # Concatenate observations in agent order
        global_state = np.concatenate([observations[agent_id] for agent_id in self.agents])
        
        return global_state

    def _rescale_actions_to_quantities(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Rescales normalized actions from a [-1, 1] range to integer order quantities in the range [0, max_order_quantity].
        
        Args:
            actions (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to normalized action array. 
                Shape: {agent_id: (n_skus,)}
                
        Returns:
            quantities (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to rescaled action array.
                Shape: {agent_id: (n_skus,)}
        """
        
        # Initialize dictionary to store quantities
        quantities = {}
        
        # Rescale actions to quantities for each warehouse
        for agent_id, action in actions.items():
            scaled = (action + 1.0) / 2.0 * self.max_order_quantity 
            integer_action = np.round(scaled).astype(int)
            integer_action = np.clip(integer_action, 0, self.max_order_quantity) # Clip to ensure [0, max_order_quantity] range
            quantities[agent_id] = integer_action.astype(float) # Convert to float for compatibility with _apply_orders
        
        return quantities

    def _apply_orders(self, actions: Dict[str, np.ndarray]):
        """
        Applies replenishment orders to pending orders by sampling lead times and scheduling orders for future arrival. 
        Filters out orders that would arrive after episode ends.
        
        Args:
            actions: Dict mapping agent_id to action array.
                Shape: {agent_id: (n_skus,)} - replenishment quantities.
        """

        # Sample lead times for all SKUs
        lead_times = self.lead_time_sampler.sample() # Shape: (n_skus,)
        
        # Stack all actions into a matrix
        actions_matrix = np.array([actions[agent_id] for agent_id in self.agents]) # Shape: (n_warehouses, n_skus)
        
        # Calculate arrival timesteps for all (warehouse, sku) pairs      
        arrival_timesteps = self.timestep + np.broadcast_to(lead_times, (self.n_warehouses, self.n_skus)).astype(int) # Shape: (n_warehouses, n_skus)
        
        # Define masks to filter only orders with quantity > 0 and that arrive before the episode ends
        ordered_skus = actions_matrix > 0
        valid_skus = ordered_skus & (arrival_timesteps < self.episode_length)

        # Return early if there are no valid orders
        if not np.any(valid_skus):
            return 
        
        # Get unique arrival timesteps that have valid orders
        unique_arrival_timesteps = np.unique(arrival_timesteps[valid_skus])

        # Add valid orders to pending orders
        for arrival_timestep in unique_arrival_timesteps:
            # Define a mask for orders arriving at this timestep
            mask = (arrival_timesteps == arrival_timestep) & valid_skus

            # Initialize entry for this arrival timestep if it doesn't exist
            if arrival_timestep not in self.pending_orders:
                self.pending_orders[arrival_timestep] = np.zeros(
                    (self.n_warehouses, self.n_skus), dtype=float
                )

            # Add orders arriving in the current arrival timestep to pending_orders
            self.pending_orders[arrival_timestep] += np.where(mask, actions_matrix, 0.0)

    def _apply_arrivals(self):
        """
        Adds arriving orders to inventory and removes them from pending_orders.
        """
        
        # Add arriving orders to inventory if there are any
        if self.timestep in self.pending_orders:
            self.inventory += self.pending_orders[self.timestep]
            del self.pending_orders[self.timestep]





