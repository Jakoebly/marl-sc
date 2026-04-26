"""
Runs baseline policies on the InventoryEnvironment as multi-seed evaluations,
in the same statistical/output protocol as the RL seed-evaluation pipeline
(see :mod:`src.experiments.utils.seed_evaluation`).

Available baselines:

  1. Random: Uniform random orders in [-1, 1].
  2. Constant Order (calibrated): Per-(warehouse, SKU) constant quantities
     derived from observed demand in pilot episodes, swept over a shared
     multiplier ``alpha``.
  3. BS-Newsvendor: Analytical newsvendor base-stock using *true* demand
     parameters, swept over safety factor ``z``.
  4. BS-Adaptive: Rolling-mean base-stock using *observed* demand history,
     swept over safety factor ``z`` and window width ``H``.
  5. BS-Optimized: Simulation-optimized base-stock via Bayesian optimization.
  6. BS-Independent: Independently optimized base-stock via iterated best
     response (per-warehouse BO).
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.loader import load_environment_config
from src.config.schema import EnvironmentConfig
from src.environment.envs.multi_env import InventoryEnvironment
from src.experiments.utils.args import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_SEED,
)
from src.experiments.utils.experiment_utils import (
    generate_baseline_experiment_name,
    save_env_config,
)
from src.experiments.utils.seed_evaluation import aggregate_seed_evaluation
from src.utils.seed_manager import EXPERIMENT_SEEDS, SeedManager


# ============================================================================
# Baseline registry
# ============================================================================

BASELINE_NAMES: Tuple[str, ...] = (
    "random",
    "constant",
    "bs_newsvendor",
    "bs_adaptive",
    "bs_optimized",
    "bs_independent",
)

BASELINE_DISPLAY_NAMES: Dict[str, str] = {
    "random": "Random",
    "constant": "ConstantOrder",
    "bs_newsvendor": "BSNewsvendor",
    "bs_adaptive": "BSAdaptive",
    "bs_optimized": "BSOptimized",
    "bs_independent": "BSIndependent",
}


# ============================================================================
# Action function factories
# ============================================================================

def make_random_action_fn(rng: np.random.Generator) -> Callable:
    """
    Returns an action function that samples uniform random orders in
    ``[-1, 1]`` for every warehouse-SKU pair.

    Args:
        rng (np.random.Generator): Random number generator.

    Returns:
        action_fn (Callable): Action function that produces actions in
            ``[-1, 1]`` for each agent.
    """

    # Define the action function
    def action_fn(env, obs):
        actions = {
            agent_id: rng.uniform(-1.0, 1.0, size=(env.n_skus,)).astype(np.float32)
            for agent_id in env.agents
        }
        return actions

    return action_fn

def make_constant_action_fn(
    quantities: np.ndarray,
    max_order_quantities: np.ndarray,
) -> Callable:
    """
    Returns an action function that orders fixed per-(warehouse, SKU)
    quantities at each timestep, regardless of the current state.

    Requires a ``"direct"`` action space so the mapping from order quantity
    to normalized ``[-1, 1]`` action is a simple linear rescaling.

    Args:
        quantities (np.ndarray): Fixed order quantities per (warehouse, SKU).
            Shape: ``(n_warehouses, n_skus)``.
        max_order_quantities (np.ndarray): Maximum order quantities per SKU.

    Returns:
        action_fn (Callable): Action function in ``[-1, 1]`` per agent.
    """

    # Precompute constant actions for each warehouse-SKU pair
    clipped = np.clip(quantities, 0.0, max_order_quantities)
    constant_actions = {}
    for wh_idx in range(quantities.shape[0]):
        action = (2.0 * clipped[wh_idx] / max_order_quantities - 1.0).astype(np.float32)
        constant_actions[wh_idx] = action

    # Define the action function    
    def action_fn(env, obs):
        actions = {
            agent_id: constant_actions[wh_idx]
            for wh_idx, agent_id in enumerate(env.agents)
        }
        return actions

    return action_fn

def make_bs_newsvendor_action_fn(env_config: EnvironmentConfig, z: float) -> Callable:
    """
    Returns an action function implementing an oracle base-stock heuristic
    (BS-Newsvendor) that uses true (not observed) demand statistics.

    The base-stock levels are calculated as follows:

        1. Calculate expected demand per home-region, per SKU:
            ``E[D] = lambda_orders * probability_skus * lambda_quantity``
        2. Calculate expected demand over lead time:
            ``E[D_L] = L * E[D]``
        3. Calculate Poisson approximation (Var(D) = E[D]):
            ``sigma_L ≈ sqrt(L * E[D])``
        4. Calculate base-stock level:
            ``S = E[D_L] + z * sigma_L``
        5. Calculate order quantity:
            ``order  = max(0, S - on_hand - pipeline)``

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        z (float): Safety factor for the base-stock level.

    Returns:
        action_fn (Callable): Action function in ``[-1, 1]`` per agent.
    """

    # Extract number of warehouses and SKUs
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus

    # Extract demand parameters 
    demand_params = env_config.components.demand_sampler.params
    lambda_orders = np.array(demand_params["lambda_orders"])
    probability_skus = np.array(demand_params["probability_skus"])
    lambda_quantity = np.array(demand_params["lambda_quantity"])

    # Extract lead times
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times
    )

    # Calculate home regions for each warehouse
    distances = np.array(env_config.cost_structure.distances)
    home_regions = np.argmin(distances, axis=1)

    # Extract max order quantities (baselines assume "direct" action space)
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    # Compute base-stock heuristic
    base_stock = np.zeros((n_warehouses, n_skus))
    for wh in range(n_warehouses):
        home = home_regions[wh]
        for sku in range(n_skus):
            L = lead_times[wh, sku]
            E_D = lambda_orders[home] * probability_skus[home] * lambda_quantity[home, sku]
            E_D_L = L * E_D
            sigma_L = np.sqrt(L * E_D)
            base_stock[wh, sku] = E_D_L + z * sigma_L

    # Define the action function
    def action_fn(env, obs):
        inventory = env.inventory
        pipeline = env._compute_pending_matrix().astype(float)
        actions = {}
        for wh_idx, agent_id in enumerate(env.agents):
            qty = np.clip(
                base_stock[wh_idx] - inventory[wh_idx] - pipeline[wh_idx],
                0.0,
                max_qty,
            )
            action = 2.0 * qty / max_qty - 1.0
            actions[agent_id] = action.astype(np.float32)
        return actions

    return action_fn

def make_adaptive_bs_action_fn(
    env_config: EnvironmentConfig,
    z: float,
    H: int,
) -> Callable:
    """
    Returns an action function implementing a rolling-mean base-stock 
    heuristic (BS-Adaptive) that uses observed (not true)
    demand statistics over the last ``H`` steps.

    The base-stock levels are calculated in each timestep ``t`` as follows:

        1. Record the demand realised in the *previous* step
           (``env._incoming_demand_home``).
        2. Compute a rolling mean ``D_mean[w,k]`` and rolling variance
           ``D_var[w,k]`` over the last ``H`` observations (or fewer at
           the start of the episode).
        3. Derive a base-stock level per (warehouse, SKU):
           ``S[w,k] = L[w,k] * D_mean + z * sqrt(L[w,k] * D_var)``
        4. Order ``max(0, S - on_hand - pipeline)``, clipped to
           ``max_order_quantities``.

    At ``t = 0`` (no demand observed yet) the policy orders zero.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        z (float): Safety factor for the base-stock level.
        H (int): Rolling-window width.

    Returns:
        action_fn (Callable): Action function in ``[-1, 1]`` per agent.
    """

    # Extract number of SKUs, lead times, and max order quantities
    n_skus = env_config.n_skus
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times,
        dtype=float,
    ) # Shape: (n_warehouses, n_skus)
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

     # Define mutable states to be shared across calls within an episode
    demand_buffer: List[np.ndarray] = []
    prev_timestep = [-1]


    def action_fn(env, obs):
        # Extract the current timestep and detect episode reset
        t = env.timestep
        if t <= prev_timestep[0] or t == 0:
            demand_buffer.clear()
        prev_timestep[0] = t

        # Record demand from the step that just completed
        demand_now = env._incoming_demand_home.copy()
        if t > 0:
            demand_buffer.append(demand_now)

        # If not enough history yet, order zero
        if len(demand_buffer) == 0:
            return {agent_id: -np.ones(n_skus, dtype=np.float32) for agent_id in env.agents}

        # Compute rolling statistics over last H observations
        window = demand_buffer[-H:]
        stack = np.array(window)
        D_mean = stack.mean(axis=0)
        D_var = stack.var(axis=0, ddof=0) if len(window) > 1 else D_mean.copy()

        # Compute base-stock level: S = L * D_mean + z * sqrt(L * D_var)
        base_stock = lead_times * D_mean + z * np.sqrt(lead_times * D_var)

        # Order = max(0, S - inventory - pipeline), clipped
        inventory = env.inventory
        pipeline = env._compute_pending_matrix().astype(float)
        actions = {}
        for wh_idx, agent_id in enumerate(env.agents):
            qty = np.clip(
                base_stock[wh_idx] - inventory[wh_idx] - pipeline[wh_idx],
                0.0,
                max_qty,
            )
            action = 2.0 * qty / max_qty - 1.0
            actions[agent_id] = action.astype(np.float32)
        return actions

    return action_fn

def make_bs_optimized_action_fn(
    base_stock_levels: np.ndarray,
    max_order_quantities: np.ndarray,
) -> Callable:
    """
    Returns an action function implementing a fixed base-stock policy with
    *given* base-stock levels ``S``. At each step the policy orders:
    ``order[w,k] = max(0, S[w,k] - inventory[w,k] - pipeline[w,k])``

    This is the same decision rule used by :func:`make_bs_newsvendor_action_fn`,
    but the levels ``S`` are provided directly (e.g. found by Bayesian
    optimization) rather than computed analytically.

    Args:
        base_stock_levels (np.ndarray): Target base-stock levels per
            (warehouse, SKU). Shape: ``(n_warehouses, n_skus)``.
        max_order_quantities (np.ndarray): Maximum order quantities per SKU.

    Returns:
        action_fn (Callable): Action function in ``[-1, 1]`` per agent.
    """

    # Copy the base-stock levels and max order quantities
    S = base_stock_levels.copy()
    max_qty = max_order_quantities.copy()

    # Define the action function
    def action_fn(env, obs):
        inventory = env.inventory
        pipeline = env._compute_pending_matrix().astype(float)
        actions = {}
        for wh_idx, agent_id in enumerate(env.agents):
            qty = np.clip(
                S[wh_idx] - inventory[wh_idx] - pipeline[wh_idx],
                0.0,
                max_qty,
            )
            action = 2.0 * qty / max_qty - 1.0
            actions[agent_id] = action.astype(np.float32)
        return actions

    return action_fn


# ============================================================================
# Environment construction helpers
# ============================================================================

def make_eval_env(
    env_config: EnvironmentConfig,
    eval_seed: int,
    eval_episodes: int,
) -> InventoryEnvironment:
    """
    Builds an evaluation env that mirrors RLlib's eval env exactly.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        eval_seed (int): Fixed root seed for the evaluation distribution.
        eval_episodes (int): Number of rollout episodes; sets the env-side
            cycle so the counter does not wrap mid-loop.

    Returns:
        env (InventoryEnvironment): Eval-mode environment instance.
    """

    return InventoryEnvironment(
        env_config,
        seed=eval_seed,
        env_meta={"data_mode": "val", "num_eval_episodes": eval_episodes},
    )

def make_train_env(
    env_config: EnvironmentConfig,
    seed: int,
) -> InventoryEnvironment:
    """
    Builds a training env that mirrors RLlib's train env exactly.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        seed (int): Root seed for the training-side episodes.

    Returns:
        env (InventoryEnvironment): Train-mode environment instance.
    """

    return InventoryEnvironment(
        env_config,
        seed=seed,
        env_meta={"data_mode": "train"},
    )


# ============================================================================
# Rollout and aggregation
# ============================================================================

def baseline_rollout(
    env: InventoryEnvironment,
    action_fn: Callable,
    num_episodes: int,
    collect_step_info: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Runs ``num_episodes`` deterministic rollout episodes using a fixed
    ``action_fn`` (no learning, no exploration noise).

    Args:
        env (InventoryEnvironment): Environment instance to evaluate on.
        action_fn (Callable): ``action_fn(env, obs)`` returning agent_id ->
            action arrays in ``[-1, 1]``.
        num_episodes (int): Number of episodes to roll out.
        collect_step_info (bool): If ``True``, collect per-step info to
            enable cost decomposition. Set to ``False`` inside hot loops
            (e.g., BO objective) to skip the bookkeeping cost.

    Returns:
        all_episodes (List[Dict[str, np.ndarray]]): Per-episode data dicts.
    """

    # Enable per-step info collection
    env.collect_step_info = collect_step_info

    # Run manual rollout
    all_episodes: List[Dict[str, np.ndarray]] = []
    for _ in range(num_episodes):
        # Initialize episode data and reset environment
        episode_data: Dict[str, List[Any]] = {}
        obs, _ = env.reset()

        # Run episode
        done = False
        while not done:
            # Query the action function for actions and step environment
            actions = action_fn(env, obs)
            obs, rewards, terms, truncs, infos = env.step(actions)

    	    # Extract step info (shared across all agents)
            if collect_step_info:
                step_info = infos[env.agents[0]]
                for key, value in step_info.items():
                    episode_data.setdefault(key, []).append(
                        value.copy() if isinstance(value, np.ndarray) else value
                    )

            # Record per-warehouse rewards
            rewards_array = np.array([rewards[a] for a in env.agents])
            episode_data.setdefault("rewards", []).append(rewards_array)

    	    # Check if episode is done
            done = all(truncs.values()) or all(terms.values())

        episode_dict = {k: np.array(v) for k, v in episode_data.items()}
        all_episodes.append(episode_dict)

    # Disable per-step info collection
    env.collect_step_info = False

    return all_episodes


def episode_costs(episode: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Computes per-component total costs and reward for a single episode.

    Args:
        episode (Dict[str, np.ndarray]): Episode data dict (must have
            been produced with ``collect_step_info=True``).

    Returns:
        costs (Dict[str, float]): Reward and cost components for the episode.
    """

    return {
        "reward": float(episode["rewards"].sum()),
        "holding": float(episode["holding_cost"].sum()),
        "penalty": float(episode["penalty_cost"].sum()),
        "outbound": float(episode["outbound_shipment_cost"].sum()),
        "inbound": float(episode["inbound_shipment_cost"].sum()),
    }


def aggregate_costs(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Aggregates per-episode costs into mean/std summaries.

    Args:
        episodes (List[Dict[str, np.ndarray]]): Episodes from
            :func:`baseline_rollout` (with ``collect_step_info=True``).

    Returns:
        agg (Dict[str, float]): Mean and std summaries.
    """

    costs = [episode_costs(ep) for ep in episodes]
    rewards = [c["reward"] for c in costs]
    total_costs = [c["holding"] + c["penalty"] + c["outbound"] + c["inbound"] for c in costs]

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "holding_mean": float(np.mean([c["holding"] for c in costs])),
        "penalty_mean": float(np.mean([c["penalty"] for c in costs])),
        "outbound_mean": float(np.mean([c["outbound"] for c in costs])),
        "inbound_mean": float(np.mean([c["inbound"] for c in costs])),
        "total_cost_mean": float(np.mean(total_costs)),
        "total_cost_std": float(np.std(total_costs, ddof=1) if len(total_costs) > 1 else 0.0),
    }


# ============================================================================
# Demand calibration and sweep helpers
# ============================================================================

def calibrate_demand(
    env_config: EnvironmentConfig,
    calibration_seed: int,
    num_calibration_episodes: int = 50,
) -> np.ndarray:
    """
    Estimates empirical mean demand per (warehouse, SKU) by running
    pilot episodes with a zero-order policy on ``data_mode="train"``.

    The zero-order policy ensures observations are not confounded by
    inventory-dependent allocation effects. Demand is read from
    ``env._incoming_demand_home``, which records raw home-region demand
    independent of the agent's actions.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        calibration_seed (int): Seed for reproducible calibration rollouts,
            independent of the eval seed.
        num_calibration_episodes (int): Number of pilot episodes.

    Returns:
        mean_demand (np.ndarray): Mean demand per timestep, shape
            ``(n_warehouses, n_skus)``.
    """

    env = make_train_env(env_config, calibration_seed)
    demand_observations: List[np.ndarray] = []

    for _ in range(num_calibration_episodes):
        env.reset()
        done = False
        while not done:
            actions = {
                agent_id: np.full(env.n_skus, -1.0, dtype=np.float32)
                for agent_id in env.agents
            }
            _, _, terms, truncs, _ = env.step(actions)
            demand_observations.append(env._incoming_demand_home.copy())
            done = all(truncs.values()) or all(terms.values())

    return np.array(demand_observations).mean(axis=0)


def run_validation_sweep(
    env_config: EnvironmentConfig,
    validation_seed: int,
    num_episodes: int,
    sweep_values: List[Any],
    make_action_fn: Callable[[Any], Callable],
    label_fn: Callable[[Any], str],
) -> Tuple[List[float], int]:
    """
    Evaluates a sequence of candidate hyperparameters on a *validation*
    rollout (``data_mode="train"``, seeded by ``validation_seed``) and
    returns the per-value mean reward and the argmax index.

    Important: the validation rollout is independent of the final eval.
    Hyperparameter selection on the eval distribution would constitute
    selecting on the test set.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        validation_seed (int): Seed for the validation rollouts; must be
            distinct from ``eval_seed`` and any other run's eval seed.
        num_episodes (int): Episodes per sweep value.
        sweep_values (List[Any]): Candidate hyperparameter values.
        make_action_fn (Callable[[Any], Callable]): Maps a candidate value
            to an action function.
        label_fn (Callable[[Any], str]): Maps a candidate value to a
            human-readable label.

    Returns:
        validation_rewards (List[float]): Mean reward per candidate.
        best_idx (int): Index of the argmax candidate.
    """

    rewards: List[float] = []
    best_idx = 0
    best_reward = -np.inf

    for idx, value in enumerate(sweep_values):
        env = make_train_env(env_config, validation_seed)
        episodes = baseline_rollout(
            env,
            make_action_fn(value),
            num_episodes,
            collect_step_info=False,
        )
        ep_rewards = [float(ep["rewards"].sum()) for ep in episodes]
        mean_reward = float(np.mean(ep_rewards))
        rewards.append(mean_reward)
        print(f"    [val] {label_fn(value)}  ->  reward={mean_reward:>10.1f}")
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_idx = idx

    return rewards, best_idx


# ============================================================================
# Bayesian optimization
# ============================================================================

def run_bs_optimization(
    env_config: EnvironmentConfig,
    optimization_seed: int,
    n_calls: int = 300,
    n_random_starts: int = 50,
    n_obj_episodes: int = 50,
    upper_bound: float = 200.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Finds optimal base-stock levels ``S*`` via Gaussian-process Bayesian
    optimization. Each candidate is evaluated by averaging ``n_obj_episodes``
    rollouts on ``data_mode="train"`` with deterministic per-episode seeds
    derived from ``optimization_seed``.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        optimization_seed (int): Base seed for BO objective rollouts and
            for ``gp_minimize``'s ``random_state``.
        n_calls (int): Total BO evaluations.
        n_random_starts (int): Initial random evaluations before the GP
            surrogate takes over.
        n_obj_episodes (int): Episodes per BO objective evaluation.
        upper_bound (float): Upper bound on each base-stock level.

    Returns:
        S_star (np.ndarray): Best base-stock levels found, shape
            ``(n_warehouses, n_skus)``.
        convergence (List[float]): Best (negative) objective per call.
    """

    from skopt import gp_minimize

    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)
    n_params = n_warehouses * n_skus

    call_count = [0]

    def objective(S_flat: List[float]) -> float:
        S = np.array(S_flat).reshape(n_warehouses, n_skus)
        action_fn = make_bs_optimized_action_fn(S, max_qty)

        total_rewards = []
        for ep_idx in range(n_obj_episodes):
            env = make_train_env(env_config, optimization_seed + ep_idx)
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                actions = action_fn(env, obs)
                obs, rewards, terms, truncs, _ = env.step(actions)
                ep_reward += sum(rewards.values())
                done = all(truncs.values()) or all(terms.values())
            total_rewards.append(ep_reward)

        mean_reward = float(np.mean(total_rewards))
        call_count[0] += 1
        if call_count[0] % 25 == 0 or call_count[0] <= 5:
            print(
                f"    BO call {call_count[0]:4d}/{n_calls} -> "
                f"reward={mean_reward:>10.1f}  "
                f"S=[{min(S_flat):.1f}, {max(S_flat):.1f}]"
            )
        return -mean_reward

    dimensions = [(0.0, upper_bound)] * n_params
    print(
        f"  Starting BO ({n_params} params, {n_calls} calls, "
        f"{n_obj_episodes} episodes/call)..."
    )
    result = gp_minimize(
        objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=optimization_seed,
        verbose=False,
    )

    S_star = np.array(result.x).reshape(n_warehouses, n_skus)
    convergence = np.minimum.accumulate(result.func_vals).tolist()
    print(f"  BO complete. Best train reward = {-result.fun:.1f}")

    return S_star, convergence


def run_bs_independent_optimization(
    env_config: EnvironmentConfig,
    optimization_seed: int,
    S_init: np.ndarray,
    n_rounds: int = 2,
    n_calls_per_wh: int = 100,
    n_random_starts_per_wh: int = 50,
    n_obj_episodes: int = 50,
    upper_bound: float = 200.0,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Iterated-best-response BO: for each round, each warehouse's ``S`` is
    optimized while the others are held fixed. Mirrors the incentive
    structure of independent learning agents.

    The per-(round, warehouse) BO uses an independent child seed spawned
    from ``optimization_seed`` so the search trajectories are independent
    yet deterministic.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        optimization_seed (int): Parent seed for spawning per-(round, wh)
            child seeds and for objective rollouts.
        S_init (np.ndarray): Initial base-stock levels.
        n_rounds (int): Iterated-best-response rounds.
        n_calls_per_wh (int): BO evaluations per (round, wh) call.
        n_random_starts_per_wh (int): Initial random evaluations per call.
        n_obj_episodes (int): Episodes per BO objective evaluation.
        upper_bound (float): Upper bound on each base-stock level.

    Returns:
        S_star (np.ndarray): Independently-optimized base-stock levels.
        convergence_log (Dict[str, List[float]]): Per-(round, wh) convergence.
    """

    from skopt import gp_minimize

    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    # Independent child seeds for each (round, wh) BO call so the GP
    # search and rollout sampling are independent across calls. Spawning
    # from a fresh SeedSequence keeps results deterministic and order-
    # independent of any other code that consumes the SeedManager.
    parent_ss = np.random.SeedSequence(optimization_seed)
    child_ss = parent_ss.spawn(n_rounds * n_warehouses)

    S_current = S_init.copy()
    convergence_log: Dict[str, List[float]] = {}

    for rnd in range(n_rounds):
        print(f"\n  === Independent BO -- Round {rnd + 1}/{n_rounds} ===")

        for wh_target in range(n_warehouses):
            child_idx = rnd * n_warehouses + wh_target
            wh_seed = int(child_ss[child_idx].generate_state(1, dtype=np.uint32)[0])
            print(
                f"    Optimising WH {wh_target} ({n_skus} params, "
                f"{n_calls_per_wh} calls, child_seed={wh_seed})..."
            )

            S_snapshot = S_current.copy()
            call_count = [0]

            def objective(s_wh_flat: List[float], _target=wh_target,
                          _snapshot=S_snapshot, _seed=wh_seed) -> float:
                S_candidate = _snapshot.copy()
                S_candidate[_target] = np.array(s_wh_flat)
                action_fn = make_bs_optimized_action_fn(S_candidate, max_qty)

                wh_rewards = []
                for ep_idx in range(n_obj_episodes):
                    env = make_train_env(env_config, _seed + ep_idx)
                    obs, _ = env.reset()
                    ep_wh_reward = 0.0
                    done = False
                    while not done:
                        actions = action_fn(env, obs)
                        obs, rewards, terms, truncs, _ = env.step(actions)
                        ep_wh_reward += sum(rewards.values())
                        done = all(truncs.values()) or all(terms.values())
                    wh_rewards.append(ep_wh_reward)

                mean_reward = float(np.mean(wh_rewards))
                call_count[0] += 1
                if call_count[0] % 25 == 0 or call_count[0] <= 3:
                    print(
                        f"      BO WH{_target} call {call_count[0]:4d}/"
                        f"{n_calls_per_wh} -> reward={mean_reward:>10.1f}  "
                        f"S=[{min(s_wh_flat):.1f}, {max(s_wh_flat):.1f}]"
                    )
                return -mean_reward

            result = gp_minimize(
                objective,
                dimensions=[(0.0, upper_bound)] * n_skus,
                n_calls=n_calls_per_wh,
                n_random_starts=n_random_starts_per_wh,
                random_state=wh_seed,
                verbose=False,
            )

            S_current[wh_target] = np.array(result.x)
            key = f"round{rnd + 1}_wh{wh_target}"
            convergence_log[key] = np.minimum.accumulate(result.func_vals).tolist()
            print(
                f"      WH {wh_target} done. "
                f"Best wh_reward = {-result.fun:.1f}, "
                f"S = {S_current[wh_target].round(1)}"
            )

    print("\n  Independent BO complete. Final S (per WH x SKU):")
    for wh in range(n_warehouses):
        print(f"    WH {wh}: {S_current[wh].round(1)}")

    return S_current, convergence_log


# ============================================================================
# Per-baseline runners
# ============================================================================

def _eval_action_fn_on_eval_seed(
    env_config: EnvironmentConfig,
    action_fn: Callable,
    eval_seed: int,
    eval_episodes: int,
) -> Dict[str, Any]:
    """
    Runs the final evaluation of a fixed policy on the held-out eval seed
    and returns a dict matching the schema produced by RL's
    ``EvaluationRunner.run()``.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        action_fn (Callable): Pre-built action function for the policy.
        eval_seed (int): Held-out root seed for the final evaluation
            (default 123 across all RL and baseline runs).
        eval_episodes (int): Number of evaluation episodes.

    Returns:
        eval_results (Dict[str, Any]): Top-level dict including
            ``episode_return_mean`` and per-component cost statistics.
    """

    env = make_eval_env(env_config, eval_seed, eval_episodes)
    episodes = baseline_rollout(env, action_fn, eval_episodes, collect_step_info=True)
    agg = aggregate_costs(episodes)

    return {
        "episode_return_mean": agg["reward_mean"],
        "episode_return_std": agg["reward_std"],
        "episode_return_min": agg["reward_min"],
        "episode_return_max": agg["reward_max"],
        "num_episodes": int(eval_episodes),
        "cost_components": {
            "holding_mean": agg["holding_mean"],
            "penalty_mean": agg["penalty_mean"],
            "outbound_mean": agg["outbound_mean"],
            "inbound_mean": agg["inbound_mean"],
            "total_cost_mean": agg["total_cost_mean"],
            "total_cost_std": agg["total_cost_std"],
        },
    }


def run_random_baseline(
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
) -> Dict[str, Any]:
    """
    Random baseline: uniform random actions in ``[-1, 1]``.

    Has no calibration step. The action RNG is seeded directly from
    ``root_seed`` so different seed runs produce independent random
    policies (the across-seed std then captures the effect of policy
    stochasticity, analogous to RL across-seed training stochasticity).

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed (e.g., ``100, 200, ...``).
        eval_seed (int): Fixed eval seed.
        eval_episodes (int): Number of evaluation episodes.

    Returns:
        result (Dict[str, Any]): Eval results dict to be written to
            ``eval_results_best.yaml``.
    """

    rng = np.random.default_rng(root_seed)
    action_fn = make_random_action_fn(rng)
    eval_results = _eval_action_fn_on_eval_seed(
        env_config, action_fn, eval_seed, eval_episodes,
    )
    return {
        "baseline": "random",
        "selected_hparams": {},
        **eval_results,
    }


def run_constant_baseline(
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
    alpha_values: Optional[List[float]] = None,
    num_calibration_episodes: int = 50,
    num_validation_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Constant-order baseline: per-(WH, SKU) constant order quantities
    ``round(alpha * mean_demand)``, with ``alpha`` selected by a validation
    sweep on ``data_mode="train"``.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed.
        eval_seed (int): Fixed eval seed.
        eval_episodes (int): Number of evaluation episodes.
        alpha_values (Optional[List[float]]): Sweep grid for ``alpha``.
        num_calibration_episodes (int): Pilot episodes for demand
            calibration.
        num_validation_episodes (Optional[int]): Episodes per sweep value
            on validation (defaults to ``eval_episodes``).

    Returns:
        result (Dict[str, Any]): Eval results dict.
    """

    sm = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)
    calibration_seed, validation_seed = sm.spawn_child_seeds("train", 2)

    if alpha_values is None:
        alpha_values = [round(0.05 * i, 2) for i in range(1, 41)]
    val_episodes = num_validation_episodes or eval_episodes

    print("  Calibrating mean demand...")
    mean_demand = calibrate_demand(
        env_config, calibration_seed, num_calibration_episodes,
    )
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    print(f"  Validating alpha grid ({len(alpha_values)} values, "
          f"{val_episodes} eps each, val_seed={validation_seed})...")
    val_rewards, best_idx = run_validation_sweep(
        env_config=env_config,
        validation_seed=validation_seed,
        num_episodes=val_episodes,
        sweep_values=alpha_values,
        make_action_fn=lambda alpha: make_constant_action_fn(
            np.round(alpha * mean_demand), max_qty,
        ),
        label_fn=lambda alpha: f"alpha={alpha:.2f}",
    )

    best_alpha = alpha_values[best_idx]
    best_quantities = np.round(best_alpha * mean_demand)
    action_fn = make_constant_action_fn(best_quantities, max_qty)
    eval_results = _eval_action_fn_on_eval_seed(
        env_config, action_fn, eval_seed, eval_episodes,
    )

    return {
        "baseline": "constant",
        "selected_hparams": {
            "best_alpha": float(best_alpha),
            "calibrated_mean_demand": mean_demand.tolist(),
            "best_quantities": best_quantities.tolist(),
        },
        "validation": {
            "validation_seed": int(validation_seed),
            "validation_episodes": int(val_episodes),
            "alpha_values": [float(a) for a in alpha_values],
            "validation_rewards": [float(r) for r in val_rewards],
            "best_idx": int(best_idx),
        },
        "calibration": {
            "calibration_seed": int(calibration_seed),
            "num_calibration_episodes": int(num_calibration_episodes),
        },
        **eval_results,
    }


def run_bs_newsvendor_baseline(
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
    z_values: Optional[List[float]] = None,
    num_validation_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    BS-Newsvendor baseline: oracle base-stock with true demand parameters,
    swept over safety factor ``z`` on a validation rollout.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed.
        eval_seed (int): Fixed eval seed.
        eval_episodes (int): Number of evaluation episodes.
        z_values (Optional[List[float]]): Sweep grid for ``z``.
        num_validation_episodes (Optional[int]): Episodes per sweep value.

    Returns:
        result (Dict[str, Any]): Eval results dict.
    """

    sm = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)
    (validation_seed,) = sm.spawn_child_seeds("train", 1)

    if z_values is None:
        z_values = [round(0.25 * i, 2) for i in range(25)]
    val_episodes = num_validation_episodes or eval_episodes

    print(f"  Validating z grid ({len(z_values)} values, "
          f"{val_episodes} eps each, val_seed={validation_seed})...")
    val_rewards, best_idx = run_validation_sweep(
        env_config=env_config,
        validation_seed=validation_seed,
        num_episodes=val_episodes,
        sweep_values=z_values,
        make_action_fn=lambda z: make_bs_newsvendor_action_fn(env_config, z=z),
        label_fn=lambda z: f"z={z:5.2f}",
    )

    best_z = z_values[best_idx]
    action_fn = make_bs_newsvendor_action_fn(env_config, z=best_z)
    eval_results = _eval_action_fn_on_eval_seed(
        env_config, action_fn, eval_seed, eval_episodes,
    )

    return {
        "baseline": "bs_newsvendor",
        "selected_hparams": {"best_z": float(best_z)},
        "validation": {
            "validation_seed": int(validation_seed),
            "validation_episodes": int(val_episodes),
            "z_values": [float(z) for z in z_values],
            "validation_rewards": [float(r) for r in val_rewards],
            "best_idx": int(best_idx),
        },
        **eval_results,
    }


def run_bs_adaptive_baseline(
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
    z_values: Optional[List[float]] = None,
    H_values: Optional[List[int]] = None,
    num_validation_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    BS-Adaptive baseline: rolling-mean base-stock, swept over safety
    factor ``z`` and rolling-window width ``H``.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed.
        eval_seed (int): Fixed eval seed.
        eval_episodes (int): Number of evaluation episodes.
        z_values (Optional[List[float]]): Sweep grid for ``z``.
        H_values (Optional[List[int]]): Sweep grid for ``H``.
        num_validation_episodes (Optional[int]): Episodes per sweep value.

    Returns:
        result (Dict[str, Any]): Eval results dict.
    """

    sm = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)
    (validation_seed,) = sm.spawn_child_seeds("train", 1)

    if z_values is None:
        z_values = [round(0.25 * i, 2) for i in range(25)]
    if H_values is None:
        H_values = [5, 10, 20, 30, 40, 50, 100]
    val_episodes = num_validation_episodes or eval_episodes

    sweep_params = [(z, H) for H in H_values for z in z_values]

    print(f"  Validating (z, H) grid ({len(sweep_params)} pairs, "
          f"{val_episodes} eps each, val_seed={validation_seed})...")
    val_rewards, best_idx = run_validation_sweep(
        env_config=env_config,
        validation_seed=validation_seed,
        num_episodes=val_episodes,
        sweep_values=sweep_params,
        make_action_fn=lambda zh: make_adaptive_bs_action_fn(
            env_config, z=zh[0], H=zh[1],
        ),
        label_fn=lambda zh: f"z={zh[0]:5.2f}, H={zh[1]:3d}",
    )

    best_z, best_H = sweep_params[best_idx]
    action_fn = make_adaptive_bs_action_fn(env_config, z=best_z, H=best_H)
    eval_results = _eval_action_fn_on_eval_seed(
        env_config, action_fn, eval_seed, eval_episodes,
    )

    return {
        "baseline": "bs_adaptive",
        "selected_hparams": {
            "best_z": float(best_z),
            "best_H": int(best_H),
        },
        "validation": {
            "validation_seed": int(validation_seed),
            "validation_episodes": int(val_episodes),
            "z_values": [float(z) for z in z_values],
            "H_values": [int(h) for h in H_values],
            "sweep_params": [[float(z), int(h)] for (z, h) in sweep_params],
            "validation_rewards": [float(r) for r in val_rewards],
            "best_idx": int(best_idx),
        },
        **eval_results,
    }


def run_bs_optimized_baseline(
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
    n_calls: int = 300,
    n_random_starts: int = 50,
    n_obj_episodes: int = 50,
    upper_bound: float = 200.0,
) -> Dict[str, Any]:
    """
    BS-Optimized baseline: simulation-optimized base-stock via Bayesian
    optimization. The BO objective is averaged over training-mode
    rollouts; the optimum is then evaluated on the held-out eval seed.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed.
        eval_seed (int): Fixed eval seed.
        eval_episodes (int): Number of evaluation episodes.
        n_calls (int): Total BO evaluations.
        n_random_starts (int): Initial random BO evaluations.
        n_obj_episodes (int): Episodes per BO objective evaluation.
        upper_bound (float): Upper bound on each base-stock parameter.

    Returns:
        result (Dict[str, Any]): Eval results dict.
    """

    sm = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)
    (optimization_seed,) = sm.spawn_child_seeds("train", 1)

    print(f"  BO optimization_seed={optimization_seed}")
    S_star, bo_convergence = run_bs_optimization(
        env_config=env_config,
        optimization_seed=optimization_seed,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        n_obj_episodes=n_obj_episodes,
        upper_bound=upper_bound,
    )

    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)
    action_fn = make_bs_optimized_action_fn(S_star, max_qty)
    eval_results = _eval_action_fn_on_eval_seed(
        env_config, action_fn, eval_seed, eval_episodes,
    )

    return {
        "baseline": "bs_optimized",
        "selected_hparams": {
            "best_base_stock_levels": S_star.tolist(),
        },
        "optimization": {
            "optimization_seed": int(optimization_seed),
            "n_calls": int(n_calls),
            "n_random_starts": int(n_random_starts),
            "n_obj_episodes": int(n_obj_episodes),
            "upper_bound": float(upper_bound),
            "bo_convergence": [float(c) for c in bo_convergence],
        },
        **eval_results,
    }


def run_bs_independent_baseline(
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
    n_rounds: int = 3,
    n_calls_per_wh: int = 100,
    n_random_starts_per_wh: int = 50,
    n_obj_episodes: int = 50,
    upper_bound: float = 200.0,
    num_calibration_episodes: int = 50,
) -> Dict[str, Any]:
    """
    BS-Independent baseline: per-warehouse independent BO with iterated
    best response. Initial ``S`` derived from a newsvendor formula on
    calibrated mean demand.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed.
        eval_seed (int): Fixed eval seed.
        eval_episodes (int): Number of evaluation episodes.
        n_rounds (int): Iterated best-response rounds.
        n_calls_per_wh (int): BO evaluations per (round, warehouse).
        n_random_starts_per_wh (int): Initial random evaluations per call.
        n_obj_episodes (int): Episodes per BO objective evaluation.
        upper_bound (float): Upper bound on each base-stock parameter.
        num_calibration_episodes (int): Pilot episodes for the
            initial-S calibration.

    Returns:
        result (Dict[str, Any]): Eval results dict.
    """

    sm = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)
    calibration_seed, optimization_seed = sm.spawn_child_seeds("train", 2)

    print(f"  Calibrating initial S (cal_seed={calibration_seed})...")
    mean_demand = calibrate_demand(
        env_config, calibration_seed, num_calibration_episodes,
    )
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times,
        dtype=float,
    )
    z_init = 1.0
    S_init = lead_times * mean_demand + z_init * np.sqrt(lead_times * mean_demand)

    print(f"  Iterated-BR BO (opt_seed={optimization_seed})...")
    S_indep, indep_convergence = run_bs_independent_optimization(
        env_config=env_config,
        optimization_seed=optimization_seed,
        S_init=S_init,
        n_rounds=n_rounds,
        n_calls_per_wh=n_calls_per_wh,
        n_random_starts_per_wh=n_random_starts_per_wh,
        n_obj_episodes=n_obj_episodes,
        upper_bound=upper_bound,
    )

    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)
    action_fn = make_bs_optimized_action_fn(S_indep, max_qty)
    eval_results = _eval_action_fn_on_eval_seed(
        env_config, action_fn, eval_seed, eval_episodes,
    )

    return {
        "baseline": "bs_independent",
        "selected_hparams": {
            "best_base_stock_levels": S_indep.tolist(),
            "initial_base_stock_levels": S_init.tolist(),
            "z_init": float(z_init),
        },
        "optimization": {
            "calibration_seed": int(calibration_seed),
            "optimization_seed": int(optimization_seed),
            "n_rounds": int(n_rounds),
            "n_calls_per_wh": int(n_calls_per_wh),
            "n_random_starts_per_wh": int(n_random_starts_per_wh),
            "n_obj_episodes": int(n_obj_episodes),
            "upper_bound": float(upper_bound),
            "convergence_log": {
                k: [float(c) for c in v] for k, v in indep_convergence.items()
            },
        },
        **eval_results,
    }


BASELINE_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "random": run_random_baseline,
    "constant": run_constant_baseline,
    "bs_newsvendor": run_bs_newsvendor_baseline,
    "bs_adaptive": run_bs_adaptive_baseline,
    "bs_optimized": run_bs_optimized_baseline,
    "bs_independent": run_bs_independent_baseline,
}


# ============================================================================
# Per-(baseline, seed) worker and multi-seed orchestrator
# ============================================================================

def run_single_baseline_seed_eval(
    baseline: str,
    env_config: EnvironmentConfig,
    root_seed: int,
    eval_seed: int,
    eval_episodes: int,
    output_dir: Path,
    skip_if_done: bool = True,
) -> Dict[str, Any]:
    """
    Runs one ``(baseline, root_seed)`` pair and writes
    ``eval_results_best.yaml`` into ``output_dir``.

    Args:
        baseline (str): Baseline key in :data:`BASELINE_REGISTRY`.
        env_config (EnvironmentConfig): Environment configuration.
        root_seed (int): Per-run root seed (e.g. ``100, 200, ...``).
        eval_seed (int): Fixed eval seed shared with RL runs.
        eval_episodes (int): Number of evaluation episodes.
        output_dir (Path): Directory to write into. Should follow the
            convention ``<seed_eval_dir>/<DisplayName>_Seed<root_seed>/``.
        skip_if_done (bool): If ``True`` and a valid
            ``eval_results_best.yaml`` already exists, skip and return it.

    Returns:
        eval_results (Dict[str, Any]): The result dict written to disk.
    """

    if baseline not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline {baseline!r}. "
            f"Choose from: {sorted(BASELINE_REGISTRY.keys())}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_file = output_dir / "eval_results_best.yaml"

    if skip_if_done and eval_file.exists():
        try:
            with open(eval_file, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
            if isinstance(existing.get("episode_return_mean"), (int, float)):
                print(
                    f"[SKIP] {output_dir.name}: eval_results_best.yaml already complete "
                    f"(episode_return_mean={existing['episode_return_mean']:.4f})"
                )
                return existing
        except (OSError, yaml.YAMLError):
            pass

    if env_config.action_space.type != "direct":
        raise ValueError(
            f"Baselines require action_space.type='direct', got "
            f"'{env_config.action_space.type}'."
        )

    print(f"\n{'=' * 75}")
    print(
        f"  BASELINE: {BASELINE_DISPLAY_NAMES[baseline]} | "
        f"root_seed={root_seed} | eval_seed={eval_seed} | "
        f"eval_episodes={eval_episodes}"
    )
    print(f"{'=' * 75}")

    runner = BASELINE_REGISTRY[baseline]
    eval_results = runner(
        env_config=env_config,
        root_seed=root_seed,
        eval_seed=eval_seed,
        eval_episodes=eval_episodes,
    )

    eval_results["root_seed"] = int(root_seed)
    eval_results["eval_seed"] = int(eval_seed)
    eval_results["display_name"] = BASELINE_DISPLAY_NAMES[baseline]

    with open(eval_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_results, f, default_flow_style=False, sort_keys=False)
    print(
        f"\n[INFO] {BASELINE_DISPLAY_NAMES[baseline]} "
        f"(root_seed={root_seed}): episode_return_mean = "
        f"{eval_results['episode_return_mean']:.4f}"
    )
    print(f"[INFO] Saved: {eval_file}")
    return eval_results


def run_baselines_across_seeds(
    env_config: EnvironmentConfig,
    n_seeds: int,
    eval_seed: int,
    eval_episodes: int,
    seed_eval_dir: Path,
    baselines: Optional[List[str]] = None,
    skip_if_done: bool = True,
    aggregate: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Runs every ``(baseline, root_seed)`` pair sequentially, then optionally
    calls :func:`aggregate_seed_evaluation` to produce
    ``seed_evaluation_summary.yaml``.

    Output layout::

        <seed_eval_dir>/<DisplayName>_Seed<root_seed>/eval_results_best.yaml

    This matches the directory layout used by the RL seed-evaluation
    pipeline so dropping baseline outputs into the same ``seed_eval_dir``
    as RL runs produces a combined ranked table for free.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        n_seeds (int): Number of seeds. Root seeds are ``100, 200, ...``.
        eval_seed (int): Fixed eval seed shared with RL runs.
        eval_episodes (int): Number of evaluation episodes.
        seed_eval_dir (Path): Parent directory for ``<DisplayName>_Seed<N>/``.
        baselines (Optional[List[str]]): Subset of baselines to run.
        skip_if_done (bool): Skip pairs that already have a complete
            ``eval_results_best.yaml``.
        aggregate (bool): If True, run :func:`aggregate_seed_evaluation`
            after the loop.

    Returns:
        summary (Optional[Dict[str, Any]]): Aggregator summary or None.
    """

    seed_eval_dir = Path(seed_eval_dir)
    seed_eval_dir.mkdir(parents=True, exist_ok=True)
    if baselines is None:
        baselines = list(BASELINE_NAMES)

    for baseline in baselines:
        if baseline not in BASELINE_REGISTRY:
            raise ValueError(
                f"Unknown baseline {baseline!r}. "
                f"Choose from: {sorted(BASELINE_REGISTRY.keys())}"
            )
        display = BASELINE_DISPLAY_NAMES[baseline]
        for seed_idx in range(1, n_seeds + 1):
            root_seed = seed_idx * 100
            run_single_baseline_seed_eval(
                baseline=baseline,
                env_config=env_config,
                root_seed=root_seed,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                output_dir=seed_eval_dir / f"{display}_Seed{root_seed}",
                skip_if_done=skip_if_done,
            )

    if aggregate:
        return aggregate_seed_evaluation(seed_eval_dir)
    return None


# ============================================================================
# SLURM helpers (task-id mapping and self-heal scan)
# ============================================================================

def baseline_task_layout(
    n_seeds: int,
    baselines: Optional[List[str]] = None,
) -> List[Tuple[int, str, int, int]]:
    """
    Returns the canonical ``(task_id, baseline, seed_idx, root_seed)``
    layout the SLURM array launcher expects:

        task_id    = baseline_idx * n_seeds + (seed_idx - 1)
        seed_idx   = task_id % n_seeds + 1
        root_seed  = seed_idx * 100

    Args:
        n_seeds (int): Number of seeds per baseline.
        baselines (Optional[List[str]]): Ordered baseline list. Defaults to
            :data:`BASELINE_NAMES`.

    Returns:
        layout (List[Tuple[int, str, int, int]]): ``(task_id, baseline,
            seed_idx, root_seed)`` tuples.
    """

    if baselines is None:
        baselines = list(BASELINE_NAMES)
    layout: List[Tuple[int, str, int, int]] = []
    for baseline_idx, baseline in enumerate(baselines):
        for seed_idx in range(1, n_seeds + 1):
            task_id = baseline_idx * n_seeds + (seed_idx - 1)
            layout.append((task_id, baseline, seed_idx, seed_idx * 100))
    return layout


def find_missing_baseline_tasks(
    seed_eval_dir: Path,
    n_seeds: int,
    baselines: Optional[List[str]] = None,
) -> List[int]:
    """
    Scans ``seed_eval_dir`` and returns SLURM array task IDs whose
    ``eval_results_best.yaml`` is missing or unreadable. Used by the
    self-healing aggregate phase in ``scripts/run_baselines.sh``.

    Args:
        seed_eval_dir (Path): Parent directory of ``<DisplayName>_Seed<N>/``.
        n_seeds (int): Number of seeds per baseline.
        baselines (Optional[List[str]]): Ordered baseline list. Defaults
            to :data:`BASELINE_NAMES`.

    Returns:
        missing (List[int]): Sorted task IDs still pending.
    """

    seed_eval_dir = Path(seed_eval_dir)
    missing: List[int] = []
    for task_id, baseline, _, root_seed in baseline_task_layout(n_seeds, baselines):
        display = BASELINE_DISPLAY_NAMES[baseline]
        seed_dir = seed_eval_dir / f"{display}_Seed{root_seed}"
        eval_file = seed_dir / "eval_results_best.yaml"
        if not eval_file.exists():
            missing.append(task_id)
            continue
        try:
            with open(eval_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            missing.append(task_id)
            continue
        if not isinstance(data.get("episode_return_mean"), (int, float)):
            missing.append(task_id)
    return sorted(missing)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parses and validates CLI arguments.

    Available modes (mirror the RL pipeline -- ``run_experiment.py``):

        - ``--mode single``: runs one ``(baseline, root_seed)`` pair and
            writes ``eval_results_best.yaml``. This is the atomic unit of
            work, called per worker by ``scripts/run_baselines.sh`` and
            usable directly for local one-off runs.
        - ``--mode all-seeds``: runs every ``(baseline, root_seed)`` pair
            sequentially in one process and then calls the aggregator.
            Used for local non-SLURM execution; the SLURM launcher fans
            this out into parallel ``--mode single`` workers instead.

    SLURM helpers (:func:`find_missing_baseline_tasks`, :data:`BASELINE_NAMES`,
    and :func:`aggregate_seed_evaluation`) are imported directly by
    ``scripts/run_baselines.sh`` via ``python -c`` snippets, exactly like
    ``run_seed_evaluation.sh`` does for the RL pipeline.
    """

    # Create parser
    parser = argparse.ArgumentParser(
        description="Run baseline policies as seed-comparable evaluations."
    )

    # Add arguments
    parser.add_argument(
        "--mode",
        choices=["single", "all-seeds"],
        default="all-seeds",
        help=(
            "single: run one (baseline, root_seed) pair (atomic unit, "
            "used by SLURM workers and local one-offs). "
            "all-seeds: run every (baseline, root_seed) sequentially in "
            "one process and aggregate (local default)."
        ),
    )
    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="Path to the environment config YAML.",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="./experiment_outputs/Runs",
        help="Parent directory for the experiment folder "
            "(default: ./experiment_outputs/Runs).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment folder name. If not provided, auto-generated as "
            "'BASELINE_<n_warehouses>WH<n_skus>SKU_<env_class>'.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds per baseline (--mode all-seeds only).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Number of held-out evaluation episodes "
            f"(default: {DEFAULT_EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=f"Fixed eval root seed shared by all baseline and RL "
            f"(config, root_seed) pairs so paired comparisons see "
            f"identical eval episodes (default: {DEFAULT_EVAL_SEED}).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        choices=list(BASELINE_NAMES),
        default=None,
        help="Subset of baselines to run (default: all). "
            "Only valid for --mode all-seeds.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINE_NAMES),
        default=None,
        help="Baseline name (--mode single only).",
    )
    parser.add_argument(
        "--root-seed",
        type=int,
        default=None,
        help="Per-run root seed, e.g. 100, 200, ... (--mode single only).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "single":
        if args.baseline is None or args.root_seed is None:
            parser.error("--mode single requires --baseline and --root-seed")
        if args.baselines is not None:
            parser.error(
                "--baselines (plural) is incompatible with --mode single; "
                "use --baseline (singular) instead."
            )
    elif args.mode == "all-seeds":
        if args.baseline is not None or args.root_seed is not None:
            parser.error(
                "--baseline / --root-seed are only valid for --mode single; "
                "use --baselines (plural) and --n-seeds for --mode all-seeds."
            )

    return args


def resolve_seed_eval_dir(
    args: argparse.Namespace,
    env_config: EnvironmentConfig,
) -> Path:
    """
    Resolves ``<storage-dir>/<experiment-name>/seed_evaluation/`` from CLI
    args. When ``--experiment-name`` is omitted, the name is auto-generated
    from the env config (e.g. ``BASELINE_3WH5SKU_SymmetricEnv``).

    Args:
        args (argparse.Namespace): Parsed CLI args.
        env_config (EnvironmentConfig): Loaded env config (used for
            auto-naming when no explicit name is given).

    Returns:
        seed_eval_dir (Path): Resolved seed-evaluation directory.
    """

    experiment_name = args.experiment_name or generate_baseline_experiment_name(env_config)
    return Path(args.storage_dir) / experiment_name / "seed_evaluation"


def main():
    """Main CLI entry point."""

    args = parse_args()
    env_config = load_environment_config(args.env_config)
    seed_eval_dir = resolve_seed_eval_dir(args, env_config=env_config)
    seed_eval_dir.mkdir(parents=True, exist_ok=True)

    # Persist the env config alongside the seed_evaluation directory so
    # downstream notebooks / aggregators can reproduce the experiment
    # without relying on the original config path.
    save_env_config(env_config, seed_eval_dir.parent)

    if args.mode == "single":
        display = BASELINE_DISPLAY_NAMES[args.baseline]
        output_dir = seed_eval_dir / f"{display}_Seed{args.root_seed}"
        run_single_baseline_seed_eval(
            baseline=args.baseline,
            env_config=env_config,
            root_seed=args.root_seed,
            eval_seed=args.eval_seed,
            eval_episodes=args.eval_episodes,
            output_dir=output_dir,
        )
        return

    # all-seeds: sequential local run + aggregate.
    run_baselines_across_seeds(
        env_config=env_config,
        n_seeds=args.n_seeds,
        eval_seed=args.eval_seed,
        eval_episodes=args.eval_episodes,
        seed_eval_dir=seed_eval_dir,
        baselines=args.baselines,
        aggregate=True,
    )
    print(f"\n[INFO] All baseline outputs in: {seed_eval_dir.resolve()}")


if __name__ == "__main__":
    main()
