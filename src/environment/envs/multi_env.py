from collections import defaultdict, deque
from typing import Dict, List, NamedTuple, Optional, Tuple, Any

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
from src.utils.seed_manager import SeedManager, ENVIRONMENT_SEEDS, STOCHASTIC_SEEDS


class PendingOrder(NamedTuple):
    """
    Defines a single in-transit order for one (warehouse, SKU) pair.
    
    Attributes:
        quantity (float): Quantity of the order.
        actual_arrival (int): Actual arrival timestep.
        expected_arrival (int): Expected arrival timestep.
    """

    # Order parameters
    quantity: float
    actual_arrival: int
    expected_arrival: int


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
        pending_orders (Dict[tuple, List[PendingOrder]]): In-transit orders keyed by
            (warehouse_idx, sku_idx). Each entry stores quantity, actual arrival, and
            expected arrival timesteps.
        timestep (int): Current timestep in the episode.
    """
    
    metadata = {"render_modes": ["human"], "name": "multi_env"}
    
    def __init__(
        self, 
        env_config: EnvironmentConfig, 
        seed: Optional[int] = None,
        env_meta: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the environment from configuration.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration.
            seed (Optional[int]): Seed for reproducibility. If provided, used as root seed for SeedManager
                which spawns component seeds (preprocessing, inventory, demand_sampler, lead_time_sampler).
                When used with RLlib, this comes from RLlib's `env_meta["seed"]` (train_seed or eval_seed).
                Defaults to None.
            env_meta (Optional[Dict[str, Any]]): RLlib's environment metadata dict. May contain 'data_mode'
                parameter that determines which dataset to use ("train" or "val"), and 'seed' parameter
                containing the seed. Defaults to None.
        """

        # Store environment config for factory to use
        self.env_config = env_config
        
        # Store general environment parameters
        self.n_warehouses = env_config.n_warehouses
        self.n_skus = env_config.n_skus
        self.n_regions = env_config.n_regions
        self.episode_length = env_config.episode_length
        self.max_wh_capacities = env_config.max_wh_capacities

        # Store action space formulation parameters
        self.action_space_type = env_config.action_space.type
        if self.action_space_type == "direct":
            self.max_order_quantities = np.asarray(
                env_config.action_space.params.max_order_quantities, dtype=float
            )
        elif self.action_space_type == "demand_centered":
            self.max_quantity_adjustment = np.asarray(
                env_config.action_space.params.max_quantity_adjustment, dtype=float
            )
        elif self.action_space_type == "base_stock":
            self.max_stock_level = np.asarray(
                env_config.action_space.params.max_stock_level, dtype=float
            )
        
        # Initialize seed manager
        self.seed_manager = SeedManager(root_seed=seed, seed_registry=ENVIRONMENT_SEEDS)
        
        # Extract data_mode from RLlib's config dict (defaults to "train")
        data_mode = "train"
        if env_meta is not None:
            data_mode = env_meta.get("data_mode", "train")
        
        # Create environment context with data_mode
        from src.environment.context import create_environment_context
        context = create_environment_context(
            env_config, 
            seed_manager=self.seed_manager,
            data_mode=data_mode
        )
        
        # Initialize components via registry
        self.demand_sampler = get_demand_sampler(env_config, context=context)
        self.demand_allocator = get_demand_allocator(env_config, context=context)
        self.lead_time_sampler = get_lead_time_sampler(env_config, context=context)
        self.lost_sales_handler = get_lost_sales_handler(env_config, context=context)
        self.reward_calculator = get_reward_calculator(env_config, context=context)
        
        # Store expected lead times and max pipeline horizon from lead time sampler
        self.expected_lead_times = self.lead_time_sampler.get_expected()  # Shape: (n_warehouses, n_skus)
        self.max_expected_lead_time = self.lead_time_sampler.get_max_expected()

        # Store cost structure and initial inventory config
        self.cost_structure = env_config.cost_structure
        self.initial_inventory_config = env_config.initial_inventory
        
        # Compute home region mapping as each warehouse's closest region (by distance)
        self.distances = context.distances  # Shape: (n_warehouses, n_regions)
        self.home_regions = np.argmin(self.distances, axis=1)  # Shape: (n_warehouses,)

        # Set number of timesteps for rolling demand mean
        self.rolling_window = 5
        
        # Set exponential smoothing parameter for demand forecast
        self.ema_alpha = 0.3

        # Set observation normalization mode and optional precomputed statistics
        self.obs_normalization = "off"
        self.obs_stats = None
        if env_meta is not None:
            self.obs_normalization = env_meta.get("obs_normalization", "off")
            self.obs_stats = env_meta.get("obs_stats", None)

        # Set include warehouse ID flag for parameter sharing
        self.include_warehouse_id = False
        if env_meta is not None:
            self.include_warehouse_id = env_meta.get("include_warehouse_id", False)

        # Eval episode cycling: when set, the episode counter resets to 0
        # every N episodes so that each eval round replays the same episodes.
        self._num_eval_episodes = None
        if env_meta is not None:
            self._num_eval_episodes = env_meta.get("num_eval_episodes", None)

        # Set agent IDs
        self.agents = [f"warehouse_{i}" for i in range(self.n_warehouses)]
        self.possible_agents = self.agents.copy()

        # Initialize state
        self.inventory = None  # Shape: (n_warehouses, n_skus)
        self.pending_orders: Dict[tuple, List[PendingOrder]] = defaultdict(list)
        self.timestep = 0

        # Observation feature buffers (initialized in reset)
        self._incoming_demand_home = None   # Shape: (n_warehouses, n_skus)
        self._units_shipped_home = None     # Shape: (n_warehouses, n_skus)
        self._units_shipped_away = None     # Shape: (n_warehouses, n_skus)
        self._stockout = None               # Shape: (n_warehouses, n_skus)
        self._rolling_demand_mean_home = None  # Shape: (n_warehouses, n_skus)
        self._demand_forecast_home = None   # Shape: (n_warehouses, n_skus)
        self._demand_history_home = None    # deque of (n_warehouses, n_skus) arrays

        # Visualization flag: when True, step() populates infos with detailed per-step data.
        # Default False to avoid overhead during training. Set to True for manual rollout.
        self.collect_step_info = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Resets the environment to its initial state by resetting all stochastic components with independent seeds, 
        initializing inventory, clearing pending orders, and returning initial observations.
        
        Args:
            seed (Optional[int]): Random seed. If None, stochastic components are reset without explicit seeds.
            options (Optional[Dict]): Optional dictionary of reset options (unused).
            
        Returns:
            Tuple containing:
                - observations (Dict[str, Dict[str, np.ndarray]]): Dictionary mapping agent_id to dictionary containing local and global observations.
                    Shape: {warehouse_id: {"local": (2 * n_skus,), "global": (n_warehouses * 2 * n_skus,)}}.
                - infos (Dict[str, Dict]): Dictionary containing any additional information (unused for now).
        """

        # Update root seed if provided, otherwise advance to next episode seeds.
        # Eval envs (num_eval_episodes set) ignore explicit seeds from RLlib's
        # init calls so that original_root_seed is never overwritten; the counter
        # is reset to 0 on explicit-seed calls and at cycle boundaries.
        if self._num_eval_episodes is not None:
            if seed is not None:
                self.seed_manager._episode_counter = 0
            elif self.seed_manager._episode_counter >= self._num_eval_episodes:
                self.seed_manager._episode_counter = 0
            self.seed_manager.advance_episode()
        elif seed is not None:
            self.seed_manager.update_root_seed(seed)
        else:
            self.seed_manager.advance_episode()

        if self._num_eval_episodes is not None:
            print(f"[EVAL ENV] Episode counter: {self.seed_manager._episode_counter}, "
                  f"root_seed: {self.seed_manager.root_seed}, "
                  f"original_root_seed: {self.seed_manager._original_root_seed}, "
                  f"seed_arg: {seed}")

        # Track cumulative reward for per-episode debug output
        self._episode_cumulative_reward = 0.0

        # Reset stochastic components with RNGs from SeedManager
        self._reset_stochastic_components()

        # Reset state and observation feature buffers
        self.inventory = self._initialize_inventory(rng=self.seed_manager.get_rng('inventory'))
        self.pending_orders = defaultdict(list)
        self._incoming_demand_home = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)
        self._units_shipped_home = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)
        self._units_shipped_away = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)
        self._stockout = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)
        self._rolling_demand_mean_home = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)
        self._demand_forecast_home = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)
        self._demand_history_home = deque(maxlen=self.rolling_window)
        self.timestep = 0
        
        # Get initial observations
        observations = self._get_observations()
        
        # Add any additional information to infos
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
            6. Compute observation features (demand, shipment, stockout, forecast)
            7. Assign lost sales from regions to warehouses
            8. Calculate rewards
            9. Update observations
            10. Increment timestep and check terminations/truncations
        
        Args:
            actions (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to action array.
                Shape: {agent_id: (n_skus,)}.
                
        Returns:
            Tuple containing:
                - observations (Dict[str, Dict[str, np.ndarray]]): Dictionary mapping warehouse_id to dictionary containing local and global observations.
                    Shape: {warehouse_id: (2 * n_skus,)}.
                - rewards (Dict[str, float]): Dictionary mapping warehouse_id to reward value.
                - terminations (Dict[str, bool]): Dictionary mapping warehouse_id to termination flag.
                - truncations (Dict[str, bool]): Dictionary mapping warehouse_id to truncation flag.
                - infos (Dict[str, Dict]): Dictionary containing any additional information (unused for now).
        """

        # Capture pre-step state for visualization (before any modifications)
        if self.collect_step_info:
            inventory_before = self.inventory.copy()
            pending_total = self._compute_pending_matrix()

        # 1. Rescale normalized actions to order quantities and place replenishment orders
        order_quantities = self._rescale_actions_to_quantities(actions)
        ordered_skus = self._apply_orders(actions=order_quantities) # Shape: (n_warehouses, n_skus)

        # 2. Receive replenishment orders arriving in this timestep
        self._apply_arrivals()
        
        # 3. Sample demand 
        orders = self.demand_sampler.sample(self.timestep)
    
        # 4. Allocate and ship orders
        allocation_result = self.demand_allocator.allocate(orders, self.inventory)
        fulfillment_matrix = allocation_result.fulfillment_matrix  # Shape: (n_orders, n_warehouses, n_skus)
        shipment_counts = allocation_result.shipment_counts  # Shape: (n_warehouses, n_regions)
        shipment_quantities = allocation_result.shipment_quantities  # Shape: (n_warehouses, n_regions)
        shipment_quantities_by_sku = allocation_result.shipment_quantities_by_sku  # Shape: (n_warehouses, n_regions, n_skus)
        unfulfilled_demands = allocation_result.unfulfilled_demands  # Shape: (n_regions, n_skus)
        lost_order_counts = allocation_result.lost_order_counts  # Shape: (n_regions,)
                
        # 5. Update inventories
        self.inventory = np.maximum(self.inventory - fulfillment_matrix.sum(axis=0), 0.0)
        
        # 6. Compute observation features from allocation result
        self._update_observations(orders, shipment_quantities_by_sku)

        # 7. Assign lost sales
        lost_sales = self.lost_sales_handler.calculate_lost_sales(lost_order_counts, unfulfilled_demands, shipment_quantities) # Shape: (n_warehouses, n_skus)
        
        # 8. Calculate rewards 
        rewards_array = self.reward_calculator.calculate(self.inventory, ordered_skus, lost_sales, shipment_counts, shipment_quantities_by_sku)  # Shape: (n_warehouses,)
        
        # Convert rewards array to dictionary keyed by warehouse IDs
        rewards = {agent_id: float(rewards_array[i]) for i, agent_id in enumerate(self.agents)}
        
        # 9. Update observations
        observations = self._get_observations()
        
        # 10. Increment timestep and check terminations and truncations
        self.timestep += 1
        terminations = {agent: False for agent in self.agents}  # No early termination
        truncations = {agent: (self.timestep >= self.episode_length) for agent in self.agents}

        # Accumulate reward and print at episode end
        self._episode_cumulative_reward += sum(rewards.values())
        if self.timestep >= self.episode_length:
            print(f"[EPISODE DONE] total_reward: {self._episode_cumulative_reward:.4f}, "
                  f"root_seed: {self.seed_manager.root_seed}, "
                  f"eval: {self._num_eval_episodes is not None}")
        
        # Populate detailed step info for visualization (if enabled)
        if self.collect_step_info:
            # Aggregate demand from orders into a (n_regions, n_skus) matrix
            demand_per_region = np.zeros((self.n_regions, self.n_skus))
            unique_skus_per_order = []
            for order in orders:
                demand_per_region[order.region_id] += order.sku_demands
                unique_skus_per_order.append(np.sum(order.sku_demands > 0))

            n_orders = len(orders)
            mean_unique_skus_per_order = (
                np.mean(unique_skus_per_order) if unique_skus_per_order else 0.0
            )

            # Read cost breakdown stored by reward calculator
            cost_breakdown = getattr(self.reward_calculator, '_cost_breakdown', {})

            step_info = {
                "inventory": inventory_before,
                "pending_total": pending_total,
                "order_quantities": ordered_skus.copy(),
                "demand_per_region": demand_per_region,
                "fulfilled_per_warehouse": fulfillment_matrix.sum(axis=0).copy(),
                "unfulfilled_demands": unfulfilled_demands.copy(),
                "shipment_counts": shipment_counts.copy(),
                "shipment_quantities": shipment_quantities.copy(),
                "shipment_quantities_by_sku": shipment_quantities_by_sku.copy(),
                "lost_order_counts": lost_order_counts.copy(),
                "lost_sales": lost_sales.copy(),
                "n_orders": n_orders,
                "mean_unique_skus_per_order": mean_unique_skus_per_order,
                **cost_breakdown,
            }
            infos = {agent: step_info for agent in self.agents}
        else:
            infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def observation_space(self, agent: str) -> Box:
        """
        Returns the observation space for a warehouse consisting of 8 feature groups per 
        warehouse w (S = n_skus, L = max_expected_lead_time):

            - inventory              (S per-SKU + 1 aggregate            = S+1)
            - pipeline               (L*S per-slot-per-SKU + 1 aggregate = L*S+1)
            - incoming demand (home) (S per-SKU + 1 aggregate            = S+1)
            - units shipped (home)   (S per-SKU                          = S)
            - units shipped (away)   (S per-SKU + 1 aggregate            = S+1)
            - stockout               (S per-SKU                          = S)
            - rolling demand mean    (S per-SKU + 1 aggregate            = S+1)
            - demand forecast        (S per-SKU + 1 aggregate            = S+1)

        The observation is a flat vector that concatenates the local observation (this warehouse's
        features) followed by the global observation (all warehouses' features concatenated).
        The RLModule splits this flat vector using local_obs_dim passed via model_config.
        
        Args:
            agent (str): Warehouse ID (unused, all warehouses have same observation space).
            
        Returns:
            observation_space (Box): Flat observation space for a warehouse.
                Shape: (local_obs_dim + global_obs_dim,) where
                local_obs_dim = (7 + max_expected_lead_time) * n_skus + 6
                and global_obs_dim = n_warehouses * local_obs_dim.
        """

        # Compute dimensions of local and global observation spaces
        local_obs_dim = (7 + self.max_expected_lead_time) * self.n_skus + 6
        if self.include_warehouse_id:
            local_obs_dim += self.n_warehouses  
        global_obs_dim = self.n_warehouses * local_obs_dim

        # Create the flat observation space (local + global concatenated)
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(local_obs_dim + global_obs_dim,), dtype=np.float32
        )

        return observation_space
    
    def global_observation_space(self) -> Box:
        """
        Returns the global observation space (concatenation of all local observations).
        
        Returns:
            global_observation_space (Box): Box space for global observation space. 
                Shape: (n_warehouses * (8 * n_skus + 6),).
        """

        # Compute dimension of global observation space
        local_obs_dim = (7 + self.max_expected_lead_time) * self.n_skus + 6
        if self.include_warehouse_id:
            local_obs_dim += self.n_warehouses 
        global_obs_dim = self.n_warehouses * local_obs_dim

        # Create the global observation space
        global_observation_space = Box(low=-np.inf, high=np.inf, shape=(global_obs_dim,), dtype=np.float32)

        return global_observation_space

    def action_space(self, agent: str) -> Box:
        """
        Returns the action space for a warehouse. Actions are in the range [-1, 1] and
        will be rescaled internally according to the configured action_space_type
        ("direct", "demand_centered", or "base_stock").

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

    def _initialize_inventory(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Initializes inventory levels for each warehouse and SKU. 
        
        Args:
            rng (Optional[np.random.Generator]): Generator from SeedManager (used for uniform method).
            
        Returns:
            inventory (np.ndarray): Initial inventory array. Shape: (n_warehouses, n_skus).
        """
       
        # Get the initial inventory method from config
        inv_config = self.initial_inventory_config
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize inventory based on the initial inventory method
        # Uniform random initialization
        if isinstance(inv_config, InitialInventoryUniform):
            min_val = inv_config.params["min"]
            max_val = inv_config.params["max"]
            inventory = rng.integers(min_val, max_val + 1, size=(self.n_warehouses, self.n_skus))

        # Custom initialization
        elif isinstance(inv_config, InitialInventoryCustom):
            value = inv_config.params["values"]
            if isinstance(value, int): # Scalar
                inventory = np.full((self.n_warehouses, self.n_skus), value, dtype=int)
            else: # 2D list
                inventory = np.array(value, dtype=int)

        # Zero initialization
        elif isinstance(inv_config, InitialInventoryZero):
            inventory = np.zeros((self.n_warehouses, self.n_skus), dtype=int)
        
        return inventory.astype(float)
    
    def _reset_stochastic_components(self):
        """Resets all stochastic components with RNGs from the SeedManager."""
        for name in STOCHASTIC_SEEDS:
            comp = getattr(self, name, None)
            if isinstance(comp, StochasticComponent):
                comp.reset(rng=self.seed_manager.get_rng(name))

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Builds per-warehouse local and global observations consisting of 8 feature groups.
        For obs_normalization == "ratio", per-SKU features are ratio-normalized (fractions)
        and aggregate features are log1p-scaled totals or ratios. For "meanstd_custom",
        raw features are assembled and then the entire local vector is normalized using
        precomputed mean/std from a random-policy rollout. For "off" and "meanstd",
        per-SKU features are raw values with the same aggregate features appended ("meanstd" 
        lets RLlib handle mean/std normalization).
        
        Feature groups per warehouse w (S = n_skus, L = max_expected_lead_time):
            1. Inventory:           S per-SKU + 1 aggregate            = S+1)
            2. Pipeline:            L*S per-slot-per-SKU + 1 aggregate  (L*S+1)
            3. Demand home:         S per-SKU + 1 aggregate             (S+1)
            4. Shipped home:        S per-SKU                           (S)
            5. Shipped away:        S per-SKU + 1 aggregate     	    (S+1)
            6. Stockout:            S per-SKU                           (S)
            7. Rolling demand mean: S per-SKU + 1 aggregate             (S+1)
            8. Demand forecast:     S per-SKU + 1 aggregate             (S+1)
        
        Returns:
            observations (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to flat 
                observation vector (local obs concatenated with global obs).
                Shape: {warehouse_id: (local_obs_dim + global_obs_dim,)}.
        """

        # Build local observation for each warehouse
        local_obs = {
            warehouse_id: self._build_local_obs(warehouse_idx)
            for warehouse_idx, warehouse_id in enumerate(self.agents)
        }

        # Build global observation (concatenation of all local observations)
        global_obs = np.concatenate([local_obs[agent_id] for agent_id in self.agents])

        # Build flat observations dictionary (local + global concatenated per agent)
        obs = {
            agent_id: np.concatenate([local_obs[agent_id], global_obs])
            for agent_id in self.agents
        }

        return obs

    def _build_local_obs(self, warehouse_idx: int) -> np.ndarray:
        """
        Builds the local observation vector for a single warehouse by extracting raw
        per-SKU features, computing the expected pipeline breakdown, optionally applying
        ratio normalization, and concatenating everything into a flat vector.

        Args:
            warehouse_idx (int): Index of the warehouse in the agents list.

        Returns:
            local (np.ndarray): Flat local observation vector. Shape: (local_obs_dim,).
        """

        # Set ratio normalization flag and epsilon
        use_ratio_norm = self.obs_normalization == "ratio"
        eps = 1e-8

        # Extract raw per-SKU features
        sku_inventory = self.inventory[warehouse_idx, :].copy()
        demand_home = self._incoming_demand_home[warehouse_idx, :]
        shipped_home = self._units_shipped_home[warehouse_idx, :]
        shipped_away = self._units_shipped_away[warehouse_idx, :]
        stockout = self._stockout[warehouse_idx, :]
        rolling_mean = self._rolling_demand_mean_home[warehouse_idx, :]
        demand_forecast = self._demand_forecast_home[warehouse_idx, :]

        # Extract pipeline feature and apply ratio normalization if enabled
        pipeline = self._compute_pipeline(warehouse_idx)  # Shape: (max_expected_lead_time, n_skus)
        pipeline_flat = pipeline.ravel()                   # Shape: (max_expected_lead_time * n_skus,)
        pending_total = float(pipeline_flat.sum())
        if use_ratio_norm:
            pipeline_normed = pipeline_flat / (pending_total + eps)
        else:
            pipeline_normed = pipeline_flat

        # Compute aggregate totals
        inventory_total = sku_inventory.sum()
        demand_home_total = demand_home.sum()
        shipped_total = (shipped_home + shipped_away).sum()
        rolling_mean_total = rolling_mean.sum()
        demand_forecast_total = demand_forecast.sum()

        # Concatenate feature blocks into a single local observation vector
        local = np.concatenate([
            # 1. Inventory: per-SKU + total aggregate
            self._feature_block(sku_inventory, inventory_total, inventory_total,
                                use_ratio_norm, eps),
            # 2. Pipeline: L*S per-slot-per-SKU + total aggregate
            np.append(pipeline_normed, pending_total),
            # 3. Incoming demand (home): per-SKU + total aggregate
            self._feature_block(demand_home, demand_home_total, demand_home_total,
                                use_ratio_norm, eps),
            # 4. Units shipped (home): per-SKU only
            self._feature_block(shipped_home, demand_home_total, None,
                                use_ratio_norm, eps),
            # 5. Units shipped (away): per-SKU + ratio aggregate
            self._feature_block(shipped_away, shipped_total, shipped_away.sum() / (shipped_total + eps),
                                use_ratio_norm, eps),
            # 6. Stockout: per-SKU only
            self._feature_block(stockout, demand_home_total, None,
                                use_ratio_norm, eps),
            # 7. Rolling demand mean: per-SKU + total aggregate
            self._feature_block(rolling_mean, rolling_mean_total, rolling_mean_total,
                                use_ratio_norm, eps),
            # 8. Demand forecast: per-SKU + total aggregate
            self._feature_block(demand_forecast, demand_forecast_total, demand_forecast_total,
                                use_ratio_norm, eps),
        ]).astype(np.float32)

        # Apply meanstd_custom or meanstd_grouped normalization if enabled
        if self.obs_normalization in ("meanstd_custom", "meanstd_grouped") and self.obs_stats is not None:
            obs_mean, obs_std = self.obs_stats
            local = (local - obs_mean) / obs_std

        # Prepend one-hot warehouse identifier when parameter sharing is enabled (include_warehouse_id is True)
        if self.include_warehouse_id:
            warehouse_id_onehot = np.zeros(self.n_warehouses, dtype=np.float32)
            warehouse_id_onehot[warehouse_idx] = 1.0
            local = np.concatenate([warehouse_id_onehot, local])

        return local

    def _feature_block(
        self,
        sku_features: np.ndarray,
        ratio_denom: float,
        agg_value: Optional[float],
        use_ratio_norm: bool,
        eps: float,
    ) -> np.ndarray:
        """
        Builds one observation feature block consisting of per-SKU features (optionally
        ratio-normalized) followed by an optional scalar aggregate.

        Args:
            sku_features (np.ndarray): Raw per-SKU feature values. Shape: (n_skus,).
            ratio_denom (float): Denominator for ratio normalization (total across SKUs).
            agg_value (Optional[float]): Scalar aggregate to append, or None to omit.
            use_ratio_norm (bool): If True, divide sku_features by (ratio_denom + eps).
            eps (float): Small constant to avoid division by zero.

        Returns:
            block (np.ndarray): Per-SKU features with optional appended aggregate.
        """

        # Apply ratio normalization if enabled
        if use_ratio_norm:
            normed = sku_features / (ratio_denom + eps)
        else:
            normed = sku_features

        # Append aggregate scalar if provided
        if agg_value is not None:
            normed = np.concatenate([normed, [np.float32(agg_value)]])

        return normed

    def _update_observations(
        self, 
        orders: List[Order], 
        shipment_quantities_by_sku: np.ndarray,
    ) -> None:
        """
        Computes and stores observation features derived from the current timestep's 
        demand and allocation results.
        
        Args:
            orders (List[Order]): Customer orders sampled this timestep.
            shipment_quantities_by_sku (np.ndarray): Units shipped per warehouse-region-SKU.
                Shape: (n_warehouses, n_regions, n_skus).
        """

        # Aggregate demand from orders into per-region totals
        demand_per_region = np.zeros((self.n_regions, self.n_skus), dtype=np.float32)
        for order in orders:
            demand_per_region[order.region_id] += order.sku_demands

        # Compute incoming demand from each warehouse's home region
        self._incoming_demand_home = demand_per_region[self.home_regions, :]  # Shape: (n_warehouses, n_skus)

        # Compute units shipped to each warehouse's home region
        self._units_shipped_home = shipment_quantities_by_sku[
            np.arange(self.n_warehouses), self.home_regions, :
        ]  # Shape: (n_warehouses, n_skus)

        # Compute units shipped to non-home regions
        total_shipped = shipment_quantities_by_sku.sum(axis=1)  # Shape: (n_warehouses, n_skus)
        self._units_shipped_away = total_shipped - self._units_shipped_home  # Shape: (n_warehouses, n_skus)

        # Compute stockout for each warehouse
        self._stockout = np.maximum(
            self._incoming_demand_home - self._units_shipped_home, 0.0
        ).astype(np.float32)  # Shape: (n_warehouses, n_skus)

        # Update rolling demand history and compute rolling mean
        self._demand_history_home.append(self._incoming_demand_home.copy())
        history_stack = np.array(self._demand_history_home)  # Shape: (min(t+1, window), n_warehouses, n_skus)
        self._rolling_demand_mean_home = history_stack.mean(axis=0).astype(np.float32)

        # Update demand forecast via exponential smoothing:
        alpha = self.ema_alpha
        self._demand_forecast_home = (
            alpha * self._incoming_demand_home + (1 - alpha) * self._demand_forecast_home
        ).astype(np.float32)

    def _rescale_actions_to_quantities(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Rescales normalized actions from a [-1, 1] range to integer order quantities
        according to the configured action_space_type:

            - "direct": maps [-1, 1] → [0, max_order_quantities] via linear scaling.
            - "demand_centered": maps [-1, 1] → [-max_quantity_adjustment, max_quantity_adjustment],
              then adds incoming home demand and clips to non-negative.
            - "base_stock": maps [-1, 1] → [0, max_stock_level] as target stock level,
              then subtracts incoming home demand and pending orders and clips to non-negative.
        
        Args:
            actions (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to normalized action array. 
                Shape: {agent_id: (n_skus,)}
                
        Returns:
            quantities (Dict[str, np.ndarray]): Dictionary mapping warehouse_id to rescaled action array.
                Shape: {agent_id: (n_skus,)}
        """
        
        # Initialize dictionary to store quantities
        quantities = {}
        
        # Rescale actions to quantities for each warehouse (element-wise per SKU)
        for agent_id, action in actions.items():
            # Get the warehouse index
            warehouse_idx = self.agents.index(agent_id)

            # Rescale actions to quantities for direct action space
            if self.action_space_type == "direct":
                scaled = (action + 1.0) / 2.0 * self.max_order_quantities
                integer_action = np.round(scaled).astype(int)
                integer_action = np.clip(integer_action, 0, self.max_order_quantities)
                quantities[agent_id] = integer_action.astype(float)

            # Rescale actions to quantities for demand-centered action space
            elif self.action_space_type == "demand_centered":
                adjustment = np.round(self.max_quantity_adjustment * action).astype(int)
                demand = self._incoming_demand_home[warehouse_idx].astype(int)
                order_qty = np.maximum(0, adjustment + demand)
                quantities[agent_id] = order_qty.astype(float)

            # Rescale actions to quantities for base-stock action space
            elif self.action_space_type == "base_stock":
                target = (action + 1.0) / 2.0 * self.max_stock_level
                demand = self._incoming_demand_home[warehouse_idx]
                sku_pending = np.zeros(self.n_skus, dtype=np.float32)
                for sku in range(self.n_skus):
                    for entry in self.pending_orders[(warehouse_idx, sku)]:
                        sku_pending[sku] += entry.quantity
                order_qty = np.maximum(0, np.round(target - demand - sku_pending)).astype(int)
                quantities[agent_id] = order_qty.astype(float)

        return quantities

    def _apply_orders(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Applies replenishment orders by sampling lead times and adding PendingOrder
        entries to the per-(warehouse, SKU) queues. Filters out orders that would
        arrive after the episode ends.

        Args:
            actions: Dict mapping agent_id to order quantity array.
                Shape: {agent_id: (n_skus,)}.

        Returns:
            ordered_skus (np.ndarray): Quantities actually ordered (valid orders only).
                Shape: (n_warehouses, n_skus).
        """

        # Sample actual lead times
        actual_lead_times = self.lead_time_sampler.sample()  # Shape: (n_warehouses, n_skus)

        # Get expected lead times
        expected_lead_times = self.expected_lead_times # Shape: (n_warehouses, n_skus)

        # Convert actions into a matrix
        actions_matrix = np.array([actions[agent_id] for agent_id in self.agents])  # Shape: (n_warehouses, n_skus)

        # Compute actual and expected arrival timesteps
        actual_arrivals = self.timestep + actual_lead_times.astype(int)    # Shape: (n_warehouses, n_skus)
        expected_arrivals = self.timestep + expected_lead_times.astype(int) # Shape: (n_warehouses, n_skus)

        # Create a mask for orders that contain quantities
        valid_mask = actions_matrix > 0

        # If no valid orders, return empty array
        if not np.any(valid_mask):
            ordered_skus = np.zeros((self.n_warehouses, self.n_skus), dtype=float)
            return ordered_skus

        # Add valid orders to pending orders
        for wh in range(self.n_warehouses):
            for sku in range(self.n_skus):
                if valid_mask[wh, sku]:
                    self.pending_orders[(wh, sku)].append(
                        PendingOrder(
                            quantity=actions_matrix[wh, sku],
                            actual_arrival=int(actual_arrivals[wh, sku]),
                            expected_arrival=int(expected_arrivals[wh, sku]),
                        )
                    )

        # Get the actual ordered quantities (valid orders only)
        ordered_skus = np.where(valid_mask, actions_matrix, 0.0)
        
        return ordered_skus

    def _apply_arrivals(self):
        """
        Processes arriving order by adding arriving quantities to inventory and removing
        the corresponding PendingOrder entries from the per-(warehouse, SKU) queues.
        """

        # Add arriving orders to inventory and remove corresponding PendingOrder entries
        for wh in range(self.n_warehouses):
            for sku in range(self.n_skus):
                queue = self.pending_orders[(wh, sku)]
                surviving = []
                for entry in queue:
                    if entry.actual_arrival == self.timestep:
                        self.inventory[wh, sku] += entry.quantity
                    else:
                        surviving.append(entry)
                self.pending_orders[(wh, sku)] = surviving

    def _compute_pending_matrix(self) -> np.ndarray:
        """
        Computes total pending quantities per (warehouse, SKU) by summing
        over all in-transit orders.

        Returns:
            pending (np.ndarray): Total pending per (warehouse, SKU).
                Shape: (n_warehouses, n_skus).
        """

        # Initialize pending matrix
        pending = np.zeros((self.n_warehouses, self.n_skus), dtype=np.float32)

        # Sum over all in-transit orders
        for (wh, sku), queue in self.pending_orders.items():
            for entry in queue:
                pending[wh, sku] += entry.quantity

        return pending

    def _compute_pipeline(self, warehouse_idx: int) -> np.ndarray:
        """
        Computes the expected pipeline breakdown for one warehouse: for each
        of the next max_expected_lead_time arrival slots, how much quantity
        is expected to arrive per SKU. Late deliveries (expected arrival already passed but order hasn't
        actually arrived) are attributed to slot 0 (imminent).

        Args:
            warehouse_idx (int): Index of the warehouse.

        Returns:
            pipeline (np.ndarray): Expected arrivals per slot and SKU.
                Shape: (max_expected_lead_time, n_skus).
        """

        # Initialize pipeline array
        pipeline = np.zeros((self.max_expected_lead_time, self.n_skus), dtype=np.float32)

        # Compute expected pipeline breakdown for each SKU
        for sku in range(self.n_skus):
            for entry in self.pending_orders[(warehouse_idx, sku)]:
                slot = entry.expected_arrival - self.timestep
                if 1 <= slot <= self.max_expected_lead_time:
                    pipeline[slot - 1, sku] += entry.quantity
                elif slot <= 0:
                    pipeline[0, sku] += entry.quantity

        return pipeline
