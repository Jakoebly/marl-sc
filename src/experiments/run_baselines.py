"""
Runs baseline policies on the InventoryEnvironment:
  1. Random: Uniform random orders in [-1, 1].
  2. Constant Order (calibrated): Per-(warehouse, SKU) constant quantities
     derived from observed demand in pilot episodes, swept over a shared
     multiplier ``alpha``.
  3. BS-Newsvendor: Analytical newsvendor base-stock using *true* demand
     parameters, swept over safety factor ``z``.
  4. BS-Adaptive: Rolling-mean base-stock using *observed* demand history,
     swept over safety factor ``z`` and window width ``H``.
  5. BS-Optimized: Simulation-optimized base-stock via Bayesian optimization.
     Finds the best time-invariant base-stock levels ``S*`` by GP-based
     sequential optimisation over rollout episodes.
  6. BS-Independent: Independently optimised base-stock via iterated best
     response.  Each warehouse's ``S`` is optimised via BO while other
     warehouses are held fixed, mirroring independent learning.

Each baseline is evaluated over multiple rollout episodes with visualization.
Seeds are derived from a SeedManager: ``eval_seed`` for evaluation rollouts,
``calibration_seed`` (from the 'train' slot) for demand calibration and BO.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_environment_config
from src.config.schema import EnvironmentConfig
from src.environment.envs.multi_env import InventoryEnvironment
from src.experiments.utils.args import DEFAULT_ROOT_SEED
from src.experiments.utils.experiment_utils import (
    generate_baseline_experiment_name,
    save_env_config,
)
from src.experiments.utils.visualization import generate_visualizations
from src.utils.seed_manager import SeedManager, EXPERIMENT_SEEDS


# ============================================================================
# Baseline Action Functions
# ============================================================================

def make_random_action_fn(rng: np.random.Generator) -> Callable:
    """
    Returns an action function that samples uniform random orders in
    ``[0, max_order_quantity]`` for every warehouse-SKU pair.

    Args:
        rng (np.random.Generator): Random number generator.

    Returns:
        action_fn (Callable): Action function that produces actions in ``[-1, 1]`` 
            for each agent.
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
    Returns an action function that orders fixed per-(warehouse, SKU) quantities
    at each timestep, regardless of the current state.
    The constant order quantities are calculated for each warehouse-SKU pair as 
    `constant_order[w,k] = round(α * mean_demand[w,k])`

    Requires a ``"direct"`` action space so that the mapping from order
    quantity to normalized ``[-1, 1]`` action is a simple linear rescaling.

    Args:
        quantities (np.ndarray): Fixed order quantities per (warehouse, SKU).
            Shape: ``(n_warehouses, n_skus)``.
        max_order_quantities (np.ndarray): Maximum order quantities per SKU
            (from ``env_config.action_space.params.max_order_quantities``).

    Returns:
        action_fn (Callable): Action function that produces actions in ``[-1, 1]`` 
            for each agent.
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
    Returns an action function implementing an oracle base-stock 
    heuristic (BS-Newsvendor) that uses true (not observed)
    demand statistics.

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
        action_fn (Callable): Action function that produces actions in ``[-1, 1]`` 
            for each agent.
    """

    # Extract number of warehouses and SKUs
    n_warehouses = env_config.n_warehouses # Shape: (1,)
    n_skus = env_config.n_skus # Shape: (1,)

    # Extract demand parameters 
    demand_params = env_config.components.demand_sampler.params
    lambda_orders = np.array(demand_params["lambda_orders"]) # Shape:(n_regions,)
    probability_skus = np.array(demand_params["probability_skus"]) # Shape:(n_regions,)
    lambda_quantity = np.array(demand_params["lambda_quantity"]) # Shape:(n_regions, n_skus)

    # Extract lead times
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times # Shape:(n_warehouses, n_skus)
    )
    
    # Calculate home regions for each warehouse
    distances = np.array(env_config.cost_structure.distances) # Shape:(n_warehouses, n_regions)
    home_regions = np.argmin(distances, axis=1) # Shape:(n_warehouses,)

    # Extract max order quantities (baselines assume "direct" action space)
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    # Initialize base-stock levels for each warehouse-SKU pair
    base_stock = np.zeros((n_warehouses, n_skus))

    # Compute base-stock heuristic
    for wh in range(n_warehouses):
        home = home_regions[wh]
        for sku in range(n_skus):
            L = lead_times[wh, sku]
            E_D = lambda_orders[home] * probability_skus[home] * lambda_quantity[home, sku]
            E_D_L = L * E_D
            sigma_L = np.sqrt(L * E_D)
            base_stock[wh, sku] = E_D_L + z * sigma_L

    # Print base-stock levels for each warehouse-SKU pair
    print(f"  [Heuristic z={z:.2f}] base-stock levels (per WH x SKU):")
    for wh in range(n_warehouses):
        print(f"    WH {wh}: {base_stock[wh].round(1)}")

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
    demand statistics.

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
        H (int): Rolling-window width (number of past demand observations).

    Returns:
        action_fn (Callable): Action function producing ``[-1, 1]`` actions.
    """

    # Extract number of warehouses and SKUs, lead times, and max order quantities
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times,
        dtype=float,
    )  # Shape: (n_warehouses, n_skus)
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    # Define mutable states to be shared across calls within an episode
    demand_buffer: List[np.ndarray] = []
    prev_timestep = [-1]  # track episode resets

    # Define the action function
    def action_fn(env, obs):
        # Extract the current timestep and detect episode reset
        t = env.timestep  
        if t <= prev_timestep[0] or t == 0:
            demand_buffer.clear()
        prev_timestep[0] = t

        # Record demand from the step that just completed
        demand_now = env._incoming_demand_home.copy()  # Shape:(n_warehouses, n_skus)
        if t > 0:
            demand_buffer.append(demand_now)

        # If not enough history yet, order zero
        if len(demand_buffer) == 0:
            return {agent_id: -np.ones(n_skus, dtype=np.float32) for agent_id in env.agents}

        # Compute rolling statistics over last H observations
        window = demand_buffer[-H:]
        stack = np.array(window)  # Shape: (min(t, H), n_warehouses, n_skus)
        D_mean = stack.mean(axis=0)  # Shape:(n_warehouses, n_skus)
        D_var = stack.var(axis=0, ddof=0) if len(window) > 1 else D_mean.copy()

        # Compute base-stock level: S = L * D_mean + z * sqrt(L * D_var)
        base_stock = lead_times * D_mean + z * np.sqrt(lead_times * D_var)

        # Order = max(0, S - inventory - pipeline), clipped
        inventory = env.inventory  # (n_warehouses, n_skus)
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
    *given* base-stock levels ``S``.

    At each step the policy orders:
        ``order[w,k] = max(0, S[w,k] - inventory[w,k] - pipeline[w,k])``

    This is the same decision rule used by :func:`make_bs_newsvendor_action_fn`,
    but the levels ``S`` are provided directly (e.g. found by Bayesian
    optimization) rather than computed analytically.

    Args:
        base_stock_levels (np.ndarray): Target base-stock levels per
            (warehouse, SKU). Shape: ``(n_warehouses, n_skus)``.
        max_order_quantities (np.ndarray): Maximum order quantities per SKU.

    Returns:
        action_fn (Callable): Action function producing ``[-1, 1]`` actions.
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
# Baseline Orchestration Function
# ============================================================================

def run_all_baselines(
    env_config: EnvironmentConfig,
    eval_seed: int,
    calibration_seed: int,
    num_episodes: int,
    rng: np.random.Generator,
    viz_dir: Path,
) -> Dict[str, Any]:
    """
    Runs all baseline policies and returns a combined results dict.

    Executes the random baseline, a calibrated constant-order sweep,
    a BS-Newsvendor (analytical newsvendor) sweep, a BS-Adaptive
    (rolling-mean) sweep, a BS-Optimized (Bayesian-optimisation)
    baseline, and a BS-Independent (iterated best response) baseline.
    Visualisations and sweep curves are saved under ``viz_dir``.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        eval_seed (int): Seed used to initialise each evaluation environment.
        calibration_seed (int): Separate seed for demand calibration
            (derived from the ``'train'`` slot, independent of ``eval_seed``).
        num_episodes (int): Rollout episodes per configuration.
        rng (np.random.Generator): RNG for the random baseline.
        viz_dir (Path): Directory for visualisation output.

    Returns:
        results (Dict[str, Any]): Dictionary containing the results of the 
            baseline experiments.
    """

    # Check that the action space is direct; required for constant baseline
    if env_config.action_space.type != "direct":
        raise ValueError(
            f"Baselines require action_space.type='direct', "
            f"got '{env_config.action_space.type}'. "
            f"Update the environment config to use a 'direct' action space."
        )

    # Initialize results dictionary
    results: Dict[str, Any] = {"baselines": {}, "comparison": {}}


    # ------------------------------------------------------------------
    # 1. Random Baseline Policy
    # ------------------------------------------------------------------
    print("\n[1/6] Running Random baseline...")

    # Create environment and run rollout
    env = InventoryEnvironment(env_config, seed=eval_seed)
    episodes = baseline_rollout(env, make_random_action_fn(rng), num_episodes)

    # Store results and print summary
    random_agg = aggregate_costs(episodes)
    results["baselines"]["random"] = random_agg
    print_summary_block("Random Baseline", random_agg)
    generate_visualizations(episodes, str(viz_dir / "random"))
    

    # ------------------------------------------------------------------
    # 2. Constant Order Baseline (calibrated α-sweep)
    # ------------------------------------------------------------------
    print("\n[2/6] Running Constant Order baseline (calibrated)...")

    # Extract max order quantities from environment config
    max_qty = np.array(
        env_config.action_space.params.max_order_quantities, dtype=float,
    )

    # Calibrate per-(warehouse, SKU) mean demand from pilot episodes
    print("  Calibrating demand from pilot episodes...")
    mean_demand = calibrate_demand(env_config, calibration_seed)
    print(f"  Calibrated mean demand per timestep (per WH × SKU):")
    for wh in range(env_config.n_warehouses):
        print(f"    WH {wh}: {mean_demand[wh].round(2)}")

    # Define the range for α to sweep over
    alpha_values = [round(0.05 * i, 2) for i in range(1, 41)]  

    # Run rollout for each alpha value
    const_sweep_stats, best_const_idx, best_const_episodes = run_sweep(
        env_config=env_config,
        eval_seed=eval_seed,
        num_episodes=num_episodes,
        sweep_values=alpha_values,
        make_action_fn=lambda alpha: make_constant_action_fn(
            np.round(alpha * mean_demand),
            max_qty,
        ),
        label_fn=lambda alpha: f"α={alpha:.2f}",
    )

    # Store results and print summary
    best_alpha = alpha_values[best_const_idx]
    best_quantities = np.round(best_alpha * mean_demand)
    results["baselines"]["constant"] = {
        "sweep": {str(a): s for a, s in zip(alpha_values, const_sweep_stats)},
        "best_alpha": best_alpha,
        "calibrated_mean_demand": mean_demand.tolist(),
        "best_quantities": best_quantities.tolist(),
        "best": const_sweep_stats[best_const_idx],
    }
    print(f"  Best α={best_alpha:.2f} → per-(WH, SKU) quantities:")
    for wh in range(env_config.n_warehouses):
        print(f"    WH {wh}: {best_quantities[wh].astype(int)}")
    print_summary_block(
        f"Best Constant Baseline (α={best_alpha})",
        const_sweep_stats[best_const_idx],
    )

    # Generate visualizations for the best alpha value
    generate_visualizations(
        best_const_episodes, str(viz_dir / f"constant_best_a{best_alpha}"),
    )

    # Plot the sweep results on a curve
    plot_sweep_curve(
        x_values=alpha_values,
        sweep_stats=const_sweep_stats,
        x_label="Demand Multiplier α",
        title="Constant Order Baseline — Sweep over α (calibrated from observed demand)",
        output_path=viz_dir / "sweep_constant.png",
        best_idx=best_const_idx,
    )


    # ------------------------------------------------------------------
    # 3. BS-Newsvendor: Analytical Base-Stock with True Parameters (sweep over z)
    # ------------------------------------------------------------------
    print("\n[3/6] Running BS-Newsvendor baseline sweep...")

    # Define sweep parameters and values
    z_values = [round(0.25 * i, 2) for i in range(25)]

    # Run rollout for each z-value
    heur_sweep_stats, best_heur_idx, best_heur_episodes = run_sweep(
        env_config=env_config,
        eval_seed=eval_seed,
        num_episodes=num_episodes,
        sweep_values=z_values,
        make_action_fn=lambda z: make_bs_newsvendor_action_fn(env_config, z=z),
        label_fn=lambda z: f"z={z:5.2f}",
    )

    # Store results and print summary
    best_z = z_values[best_heur_idx]
    results["baselines"]["bs_newsvendor"] = {
        "sweep": {str(z): s for z, s in zip(z_values, heur_sweep_stats)},
        "best_z": best_z,
        "best": heur_sweep_stats[best_heur_idx],
    }
    print_summary_block(
        f"Best BS-Newsvendor Baseline (z={best_z})",
        heur_sweep_stats[best_heur_idx],
    )

    # Generate visualizations for the best z-value
    generate_visualizations(
        best_heur_episodes, str(viz_dir / f"bs_newsvendor_best_z{best_z}"),
    )

    # Plot the sweep results on a curve
    plot_sweep_curve(
        x_values=z_values,
        sweep_stats=heur_sweep_stats,
        x_label="Safety Factor z",
        title="BS-Newsvendor — Sweep over Safety Factor z",
        output_path=viz_dir / "sweep_bs_newsvendor.png",
        best_idx=best_heur_idx,
    )


    # ------------------------------------------------------------------
    # 4. BS-Adaptive: Analytical Base-Stock with Observed Rolling-Mean Demand
    # ------------------------------------------------------------------
    print("\n[4/6] Running BS-Adaptive baseline sweep (z × H)...")

    # Define sweep grids
    adaptive_z_values = [round(0.25 * i, 2) for i in range(25)]  
    adaptive_H_values = [5, 10, 20, 30, 40, 50, 100]

    # Flatten into (z, H) tuples
    adaptive_sweep_params = [
        (z_val, H_val)
        for H_val in adaptive_H_values
        for z_val in adaptive_z_values
    ]

    # Run 2D sweep
    adaptive_sweep_stats, best_adapt_idx, best_adapt_episodes = run_sweep(
        env_config=env_config,
        eval_seed=eval_seed,
        num_episodes=num_episodes,
        sweep_values=adaptive_sweep_params,
        make_action_fn=lambda params: make_adaptive_bs_action_fn(
            env_config, z=params[0], H=params[1],
        ),
        label_fn=lambda params: f"z={params[0]:5.2f}, H={params[1]:3d}",
    )

    # Identify best (z, H) pair
    best_adapt_z, best_adapt_H = adaptive_sweep_params[best_adapt_idx]

    # Store results
    results["baselines"]["adaptive_bs"] = {
        "sweep": {
            f"z={z_val}_H={H_val}": s
            for (z_val, H_val), s in zip(adaptive_sweep_params, adaptive_sweep_stats)
        },
        "best_z": best_adapt_z,
        "best_H": best_adapt_H,
        "best": adaptive_sweep_stats[best_adapt_idx],
    }
    print_summary_block(
        f"Best BS-Adaptive (z={best_adapt_z}, H={best_adapt_H})",
        adaptive_sweep_stats[best_adapt_idx],
    )

    # Generate visualizations for the best (z, H) configuration
    generate_visualizations(
        best_adapt_episodes,
        str(viz_dir / f"adaptive_bs_best_z{best_adapt_z}_H{best_adapt_H}"),
    )

    # Plot multi-line sweep: one line per H, reward vs z
    plot_adaptive_bs_sweep(
        z_values=adaptive_z_values,
        H_values=adaptive_H_values,
        sweep_stats=adaptive_sweep_stats,
        best_z=best_adapt_z,
        best_H=best_adapt_H,
        output_path=viz_dir / "sweep_adaptive_bs.png",
    )

    # Also plot a single-line sweep curve for the best H (reward vs z)
    best_H_slice_stats = [
        s for (z_val, H_val), s in zip(adaptive_sweep_params, adaptive_sweep_stats)
        if H_val == best_adapt_H
    ]
    best_in_slice = adaptive_z_values.index(best_adapt_z)
    plot_sweep_curve(
        x_values=adaptive_z_values,
        sweep_stats=best_H_slice_stats,
        x_label="Safety Factor z",
        title=f"BS-Adaptive (H={best_adapt_H}) — Sweep over Safety Factor z",
        output_path=viz_dir / f"sweep_adaptive_bs_H{best_adapt_H}.png",
        best_idx=best_in_slice,
    )


    # ------------------------------------------------------------------
    # 5. BS-Optimized: Simulation-Optimized Base-Stock via Bayesian Optimization
    # ------------------------------------------------------------------
    print("\n[5/6] Running BS-Optimized (Bayesian Optimization)...")

    # Extract max order quantities from environment config
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    # Run Bayesian optimization
    S_star, bo_convergence = run_bs_optimization(
        env_config=env_config,
        optimization_seed=calibration_seed,
        n_calls=300,
        n_random_starts=50,
        n_obj_episodes=50,
        upper_bound=200.0,
    )

    # Final evaluation on held-out eval seed
    print("  Evaluating optimised S* on held-out eval episodes...")
    env_bo_eval = InventoryEnvironment(env_config, seed=eval_seed)
    bo_action_fn = make_bs_optimized_action_fn(S_star, max_qty)
    bo_episodes = baseline_rollout(env_bo_eval, bo_action_fn, num_episodes)
    bo_agg = aggregate_costs(bo_episodes)

    # Store results
    results["baselines"]["bs_optimized"] = {
        "best_base_stock_levels": S_star.tolist(),
        "bo_convergence": bo_convergence,
        "bo_n_calls": 300,
        "bo_n_random_starts": 50,
        "bo_n_obj_episodes": 50,
        "best": bo_agg,
    }
    print_summary_block("BS-Optimized (Bayesian Opt.)", bo_agg)

    # Generate episode visualizations for S*
    generate_visualizations(
        bo_episodes, str(viz_dir / "bs_optimized_best"),
    )

    # Plot BO convergence curve
    plot_bo_convergence(
        convergence=bo_convergence,
        output_path=viz_dir / "sweep_bs_optimized_convergence.png",
    )


    # ------------------------------------------------------------------
    # 6. BS-Independent: Independently Optimized Base-Stock with Iterated Best Response
    # ------------------------------------------------------------------
    print("\n[6/6] Running BS-Independent (Iterated Best Response BO)...")

    # Compute initial S from calibrated demand + lead times.
    # Uses a newsvendor-style formula with z=1.0 as a reasonable starting
    # point (roughly the 84th-percentile service level).
    mean_demand = calibrate_demand(env_config, calibration_seed)
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times,
        dtype=float,
    )
    z_init = 1.0
    S_init = lead_times * mean_demand + z_init * np.sqrt(lead_times * mean_demand)
    print(f"  Initial S (newsvendor z={z_init} on calibrated demand):")
    for wh in range(env_config.n_warehouses):
        print(f"    WH {wh}: {S_init[wh].round(1)}")

    # Run iterated best response BO
    n_rounds = 3
    S_indep, indep_convergence = run_bs_independent_optimization(
        env_config=env_config,
        optimization_seed=calibration_seed,
        S_init=S_init,
        n_rounds=n_rounds,
        n_calls_per_wh=100,
        n_random_starts_per_wh=50,
        n_obj_episodes=50,
        upper_bound=200.0,
    )

    # Final evaluation on held-out eval seed
    print("  Evaluating independently-optimised S on held-out eval episodes...")
    env_indep_eval = InventoryEnvironment(env_config, seed=eval_seed)
    indep_action_fn = make_bs_optimized_action_fn(S_indep, max_qty)
    indep_episodes = baseline_rollout(env_indep_eval, indep_action_fn, num_episodes)
    indep_agg = aggregate_costs(indep_episodes)

    # Store results
    results["baselines"]["bs_independent"] = {
        "best_base_stock_levels": S_indep.tolist(),
        "initial_base_stock_levels": S_init.tolist(),
        "convergence_log": indep_convergence,
        "n_rounds": n_rounds,
        "n_calls_per_wh": 300,
        "n_random_starts_per_wh": 50,
        "n_obj_episodes": 50,
        "best": indep_agg,
    }
    print_summary_block("BS-Independent (Iterated Best Response)", indep_agg)

    # Generate episode visualizations
    generate_visualizations(
        indep_episodes, str(viz_dir / "bs_independent_best"),
    )

    # Plot per-warehouse convergence curves
    plot_bo_independent_convergence(
        convergence_log=indep_convergence,
        n_warehouses=env_config.n_warehouses,
        n_rounds=n_rounds,
        output_path=viz_dir / "sweep_bs_independent_convergence.png",
    )


    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------

    # Print comparison summary
    print(f"\n{'=' * 75}")
    print("  BASELINE COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Baseline':<40s}  {'Mean Reward':>12s}")
    print(f"  {'-' * 40}  {'-' * 12}")
    print(f"  {'Random':<40s}  {random_agg['reward']:>12.1f}")
    print(f"  {f'Constant (α={best_alpha})':<40s}  {const_sweep_stats[best_const_idx]['reward']:>12.1f}")
    print(f"  {f'BS-Newsvendor (z={best_z})':<40s}  {heur_sweep_stats[best_heur_idx]['reward']:>12.1f}")
    adapt_label = f'BS-Adaptive (z={best_adapt_z}, H={best_adapt_H})'
    print(f"  {adapt_label:<40s}  {adaptive_sweep_stats[best_adapt_idx]['reward']:>12.1f}")
    print(f"  {'BS-Optimized (BO)':<40s}  {bo_agg['reward']:>12.1f}")
    print(f"  {'BS-Independent (IBR)':<40s}  {indep_agg['reward']:>12.1f}")
    print(f"{'=' * 75}")

    # Store comparison results
    results["comparison"] = {
        "random": random_agg["reward"],
        f"constant_a{best_alpha}": const_sweep_stats[best_const_idx]["reward"],
        f"bs_newsvendor_z{best_z}": heur_sweep_stats[best_heur_idx]["reward"],
        f"bs_adaptive_z{best_adapt_z}_H{best_adapt_H}": adaptive_sweep_stats[best_adapt_idx]["reward"],
        "bs_optimized": bo_agg["reward"],
        "bs_independent": indep_agg["reward"],
    }

    return results


# ============================================================================
# Baseline Rollout Functions
# ============================================================================

def baseline_rollout(
    env: InventoryEnvironment,
    action_fn: Callable,
    num_episodes: int = 10,
) -> List[Dict[str, np.ndarray]]:
    """
    Runs rollout episodes using a baseline action function (no RL policy).
    Collects the same per-step data format as ``BaseAlgorithmWrapper.rollout()``.

    Args:
        env (InventoryEnvironment): Environment instance to evaluate on.
        action_fn (Callable): ``action_fn(env, obs)`` returning a dict that
            maps agent IDs to action arrays in ``[-1, 1]``.
        num_episodes (int): Number of episodes to roll out.

    Returns:
        all_episodes (List[Dict[str, np.ndarray]]): Per-episode data dicts (same schema as
            ``rollout()``).
    """

    # Enable per-step info collection
    env.collect_step_info = True

    # Initialize a list to store all rollout episodes
    all_episodes = []

    # Run manual rollout
    for _ in range(num_episodes):
        # Initialize episode data and reset environment
        episode_data = defaultdict(list)
        obs, _ = env.reset()

        # Run manual rollout loop
        done = False
        while not done:
            # Query action function for actions
            actions = action_fn(env, obs)

            # Step environment and record step info
            obs, rewards, terms, truncs, infos = env.step(actions)

            # Extract step info (shared across all agents)
            step_info = infos[env.agents[0]]
            for key, value in step_info.items():
                episode_data[key].append(
                    value.copy() if isinstance(value, np.ndarray) else value
                )

            # Record per-warehouse rewards
            rewards_array = np.array([rewards[a] for a in env.agents])
            episode_data["rewards"].append(rewards_array)

            # Check if episode is done
            done = all(truncs.values()) or all(terms.values())

        # Convert all lists to numpy arrays and add env metadata for visualization
        episode_data = {k: np.array(v) for k, v in episode_data.items()}
        episode_data["n_skus"] = env.n_skus
        episode_data["max_expected_lead_time"] = env.max_expected_lead_time
        episode_data["feature_config"] = env.feature_config.model_dump()
        episode_data["include_warehouse_id"] = env.include_warehouse_id
        episode_data["rolling_window"] = env.rolling_window
        all_episodes.append(episode_data)

    # Disable step info collection after rollout
    env.collect_step_info = False

    # Return all episodes
    return all_episodes

def run_bs_optimization(
    env_config: EnvironmentConfig,
    optimization_seed: int,
    n_calls: int = 300,
    n_random_starts: int = 50,
    n_obj_episodes: int = 50,
    upper_bound: float = 200.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Finds optimal base-stock levels ``S*`` via Bayesian optimization
    (Gaussian-process-based sequential model optimization).

    Each candidate solution is a vector of ``n_warehouses × n_skus``
    continuous values.  The objective function evaluates a candidate by
    running ``n_obj_episodes`` rollout episodes using the *optimization
    seed* (NOT the eval seed) and returning the *negative* mean episode
    reward (since ``gp_minimize`` minimises).

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        optimization_seed (int): Base seed for optimization rollouts,
            independent of the evaluation seed.
        n_calls (int): Total number of BO evaluations (including
            ``n_random_starts`` initial random points).
        n_random_starts (int): Number of initial random evaluations
            before the GP surrogate takes over.
        n_obj_episodes (int): Number of episodes per objective evaluation
            (averaged to reduce stochastic noise).
        upper_bound (float): Upper bound for each base-stock level
            parameter.

    Returns:
        S_star (np.ndarray): Optimised base-stock levels, shape
            ``(n_warehouses, n_skus)``.
        convergence (List[float]): Best *negative* objective found after
            each evaluation (length ``n_calls``).  Negate to get reward.
    """

    from skopt import gp_minimize

    # Extract number of warehouses, SKUs, and max order quantities
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)
    n_params = n_warehouses * n_skus

    def make_objective(n_calls_total: int):
        """Factory that creates the BO objective with its own call counter."""
        call_count = [0]

        # Define the objective function for Bayesian optimization
        def objective(S_flat: List[float]) -> float:
            # Reshape base-stock levels and create action function
            S = np.array(S_flat).reshape(n_warehouses, n_skus)
            action_fn = make_bs_optimized_action_fn(S, max_qty)

            # Run the objective function rollout
            total_rewards = []
            for ep_idx in range(n_obj_episodes):
                env = InventoryEnvironment(env_config, seed=optimization_seed + ep_idx)
                env.collect_step_info = False
                obs, _ = env.reset()
                ep_reward = 0.0
                done = False
                while not done:
                    actions = action_fn(env, obs)
                    obs, rewards, terms, truncs, _ = env.step(actions)
                    ep_reward += sum(rewards.values())
                    done = all(truncs.values()) or all(terms.values())
                total_rewards.append(ep_reward)

            # Calculate the mean reward and update the call counter
            mean_reward = float(np.mean(total_rewards))
            call_count[0] += 1

            # Print progress every 25 calls or the first 5 calls
            if call_count[0] % 25 == 0 or call_count[0] <= 5:
                print(f"    BO call {call_count[0]:4d}/{n_calls} — "
                    f"reward = {mean_reward:10.1f}  "
                    f"S range = [{min(S_flat):.1f}, {max(S_flat):.1f}]")

            # Return the negative mean reward (gp_minimize minimizes)
            return -mean_reward  

        return objective

    # Build objective and search space
    objective_fn = make_objective(n_calls)
    dimensions = [(0.0, upper_bound)] * n_params

    # Run the Bayesian optimization
    print(f"  Starting Bayesian optimization ({n_params} params, "
          f"{n_calls} calls, {n_obj_episodes} episodes/call)...")
    result = gp_minimize(
        objective_fn,
        dimensions=dimensions,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=optimization_seed,
        verbose=False,
    )

    # Reshape the best solution found into a 2D array of base-stock levels
    S_star = np.array(result.x).reshape(n_warehouses, n_skus)

    # Build convergence curve
    best_so_far = np.minimum.accumulate(result.func_vals)
    convergence = best_so_far.tolist()

    # Print the final best reward and the optimized base-stock levels
    print(f"  BO complete. Best reward = {-result.fun:.1f}")
    print(f"  Optimised base-stock levels (per WH × SKU):")
    for wh in range(n_warehouses):
        print(f"    WH {wh}: {S_star[wh].round(1)}")

    return S_star, convergence

def run_sweep(
    env_config: EnvironmentConfig,
    eval_seed: int,
    num_episodes: int,
    sweep_values: List[Any],
    make_action_fn: Callable[[Any], Callable],
    label_fn: Callable[[Any], str],
) -> Tuple[List[Dict[str, float]], int, List[Dict[str, np.ndarray]]]:
    """
    Runs a baseline sweep over a list of parameter values.

    For each value the environment is freshly created, rollouts are executed,
    costs are aggregated, and the result with the highest mean reward is
    tracked.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        eval_seed (int): Seed used to initialise each environment.
        num_episodes (int): Number of rollout episodes per sweep point.
        sweep_values (List[Any]): Parameter values to iterate over.
        make_action_fn (Callable[[Any], Callable]): Factory that maps a sweep
            value to a baseline action function.
        label_fn (Callable[[Any], str]): Maps a sweep value to a printable
            label for ``print_sweep_row``.

    Returns:
        sweep_stats (List[Dict[str, float]]): Aggregate cost dicts per sweep value.
        best_idx (int): Index of the best sweep value.
        best_episodes (List[Dict[str, np.ndarray]]): Episodes corresponding to the best sweep value.
            ``(sweep_stats, best_idx, best_episodes)`` — aggregate cost dicts,
            index of the best sweep value, and the corresponding episodes.
    """

    # Initialize lists to store sweep stats, best index, and best episodes
    sweep_stats: List[Dict[str, float]] = []
    best_idx = 0
    best_reward = -np.inf
    best_episodes = None

    # Run a baseline rollout for each sweep value
    for idx, value in enumerate(sweep_values):
        env = InventoryEnvironment(env_config, seed=eval_seed)
        eps = baseline_rollout(env, make_action_fn(value), num_episodes)
        agg = aggregate_costs(eps)
        sweep_stats.append(agg)
        print_sweep_row(label_fn(value), agg)
        if agg["reward"] > best_reward:
            best_reward = agg["reward"]
            best_idx = idx
            best_episodes = eps

    return sweep_stats, best_idx, best_episodes

def run_bs_independent_optimization(
    env_config: EnvironmentConfig,
    optimization_seed: int,
    S_init: np.ndarray,
    n_rounds: int = 2,
    n_calls_per_wh: int = 300,
    n_random_starts_per_wh: int = 50,
    n_obj_episodes: int = 50,
    upper_bound: float = 200.0,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Finds base-stock levels ``S*`` via **iterated best response**, where
    each warehouse's parameters are optimised independently while the other
    warehouses' policies are held fixed.

    Since the environment uses ``scope: "agent"``, each warehouse's BO
    objective minimises *that warehouse's own* negative reward, matching
    the incentive structure faced by an independent RL agent.

    Procedure (per round):
        1. For warehouse 0: fix all others at their current ``S`` values,
           run BO over ``S[0, :]`` (``n_skus`` parameters).
        2. For warehouse 1: fix WH 0 at its newly optimised values and
           WH 2+ at their current values, optimise ``S[1, :]``.
        3. Continue for all warehouses.
        4. Repeat for ``n_rounds`` rounds.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        optimization_seed (int): Base seed for optimization rollouts.
        S_init (np.ndarray): Initial base-stock levels, shape
            ``(n_warehouses, n_skus)``.  Typically derived from the best
            BS-Adaptive or BS-Newsvendor policy.
        n_rounds (int): Number of iterated-best-response rounds.
        n_calls_per_wh (int): BO evaluations per warehouse per round.
        n_random_starts_per_wh (int): Initial random evaluations per
            warehouse per round.
        n_obj_episodes (int): Episodes per objective evaluation.
        upper_bound (float): Upper bound for each base-stock parameter.

    Returns:
        S_star (np.ndarray): Independently-optimised base-stock levels,
            shape ``(n_warehouses, n_skus)``.
        convergence_log (Dict[str, List[float]]): Convergence curves keyed
            by ``"round{r}_wh{w}"`` — best negative objective per BO step.
    """

    from skopt import gp_minimize

    # Extract the number of warehouses, SKUs, and max order quantities
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    max_qty = np.array(env_config.action_space.params.max_order_quantities, dtype=float)

    # Initialize the current base-stock levels and convergence log
    S_current = S_init.copy()
    convergence_log: Dict[str, List[float]] = {}

    # Run the iterated best response optimization
    for rnd in range(n_rounds):
        print(f"\n  === Independent BO — Round {rnd + 1}/{n_rounds} ===")

        # Optimize each warehouse independently
        for wh_target in range(n_warehouses):
            print(f"    Optimising WH {wh_target} ({n_skus} params, "
                  f"{n_calls_per_wh} calls)...")

            def make_objective(target_wh: int, S_fixed: np.ndarray,
                               n_calls_total: int):
                """Factory to avoid late-binding closure issues."""
                # Create a snapshot of the base-stock levels and initialize  call count
                S_snapshot = S_fixed.copy()
                call_count = [0]

                # Define the objective function for Bayesian Optimization
                def objective(s_wh_flat: List[float]) -> float:
                    # Replace the target warehouse's base-stock level with the candidate
                    S_candidate = S_snapshot.copy()
                    S_candidate[target_wh] = np.array(s_wh_flat)

                    # Create the action function
                    action_fn = make_bs_optimized_action_fn(S_candidate, max_qty)

                    # Run the objective function rollout
                    wh_rewards = []
                    for ep_idx in range(n_obj_episodes):
                        env = InventoryEnvironment(
                            env_config, seed=optimization_seed + ep_idx,
                        )
                        env.collect_step_info = False
                        obs, _ = env.reset()
                        ep_wh_reward = 0.0
                        done = False
                        while not done:
                            actions = action_fn(env, obs)
                            obs, rewards, terms, truncs, _ = env.step(actions)
                            ep_wh_reward += sum(rewards.values())
                            done = all(truncs.values()) or all(terms.values())
                        wh_rewards.append(ep_wh_reward)

                    # Calculate the mean reward and update the call counter
                    mean_reward = float(np.mean(wh_rewards))
                    call_count[0] += 1

                    # Print progress every 25 calls or the first 5 calls
                    if call_count[0] % 25 == 0 or call_count[0] <= 3:
                        print(
                            f"      BO WH{target_wh} call "
                            f"{call_count[0]:4d}/{n_calls_total} — "
                            f"wh_reward = {mean_reward:10.1f}  "
                            f"S = [{min(s_wh_flat):.1f}, {max(s_wh_flat):.1f}]"
                        )
                    
                    # Return the negative mean reward (gp_minimize minimizes)
                    return -mean_reward
                return objective

            objective_fn = make_objective(wh_target, S_current, n_calls_per_wh)
            dimensions = [(0.0, upper_bound)] * n_skus

            # Use a deterministic but distinct random_state per (round, wh)
            rs = optimization_seed + rnd * 1000 + wh_target * 100

            # Run the Bayesian optimization
            result = gp_minimize(
                objective_fn,
                dimensions=dimensions,
                n_calls=n_calls_per_wh,
                n_random_starts=n_random_starts_per_wh,
                random_state=rs,
                verbose=False,
            )

            # Update the current base-stock levels with the best solution found
            S_current[wh_target] = np.array(result.x)

            # Build convergence curve
            key = f"round{rnd + 1}_wh{wh_target}"
            convergence_log[key] = np.minimum.accumulate(
                result.func_vals
            ).tolist()

            # Print progress
            print(f"      WH {wh_target} done. "
                  f"Best wh_reward = {-result.fun:.1f}, "
                  f"S = {S_current[wh_target].round(1)}")

    # Print final base-stock levels
    print(f"\n  Independent BO complete. Final S (per WH × SKU):")
    for wh in range(n_warehouses):
        print(f"    WH {wh}: {S_current[wh].round(1)}")

    return S_current, convergence_log


# ============================================================================
# General Helpers
# ============================================================================

def calibrate_demand(
    env_config: EnvironmentConfig,
    calibration_seed: int,
    num_calibration_episodes: int = 50,
) -> np.ndarray:
    """
    Estimates empirical mean demand per (warehouse, SKU) pair by running
    calibration episodes with a zero-order policy.

    The zero-order policy ensures demand observations are not confounded by
    inventory-dependent allocation effects. Demand is read from the
    environment's ``_incoming_demand_home`` attribute, which records raw
    home-region demand independently of the agent's actions.

    Uses a separate ``calibration_seed`` (derived from the ``'train'`` slot
    of the experiment-level :class:`SeedManager`) so that calibration
    episodes do not consume or overlap with the evaluation seed sequence.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        calibration_seed (int): Seed for reproducible calibration rollouts,
            independent of the evaluation seed.
        num_calibration_episodes (int): Number of episodes to run.

    Returns:
        mean_demand (np.ndarray): Mean demand per timestep for each
            (warehouse, SKU) pair.  Shape: ``(n_warehouses, n_skus)``.
    """

    # Create environment for pilot rollouts
    env = InventoryEnvironment(env_config, seed=calibration_seed)

    # Initialize list to store demand observations
    demand_observations: List[np.ndarray] = []

    # Run pilot rollouts
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

    demand_matrix = np.array(demand_observations)
    mean_demand = demand_matrix.mean(axis=0)

    return mean_demand


# ============================================================================
# Output and Reporting Helpers
# ============================================================================

def episode_costs(episode: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Computes total costs for a single episode.

    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.

    Returns:
        costs (Dict[str, float]): Total costs broken down by component.
    """

    # Compute costs components for the episode
    costs = {
        "reward": float(episode["rewards"].sum()),
        "holding": float(episode["holding_cost"].sum()),
        "penalty": float(episode["penalty_cost"].sum()),
        "outbound": float(episode["outbound_shipment_cost"].sum()),
        "inbound": float(episode["inbound_shipment_cost"].sum()),
    }

    return costs

def aggregate_costs(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Computes mean costs across episodes.

    Args:
        episodes (List[Dict[str, np.ndarray]]): List of episode data dicts.

    Returns:
        agg_costs (Dict[str, float]): Mean and std of costs across episodes.
    """

    # Compute aggregate costs over episodes
    costs = [episode_costs(ep) for ep in episodes]
    agg_costs = {
        "reward": float(np.mean([c["reward"] for c in costs])),
        "reward_std": float(np.std([c["reward"] for c in costs])),
        "holding": float(np.mean([c["holding"] for c in costs])),
        "penalty": float(np.mean([c["penalty"] for c in costs])),
        "outbound": float(np.mean([c["outbound"] for c in costs])),
        "inbound": float(np.mean([c["inbound"] for c in costs])),
        "total_cost": float(np.mean([
            c["holding"] + c["penalty"] + c["outbound"] + c["inbound"]
            for c in costs
        ])),
        "total_cost_std": float(np.std([
            c["holding"] + c["penalty"] + c["outbound"] + c["inbound"]
            for c in costs
        ])),
    }

    return agg_costs

def print_sweep_row(label: str, agg: Dict[str, float]):
    """
    Prints a single sweep row with cost breakdown.

    Args:
        label (str): Label for the sweep row.
        agg (Dict[str, float]): Dictionary containing aggregate costs.
    """

    print(
        f"  {label}  →  reward={agg['reward']:>10.1f}   cost={agg['total_cost']:>10.1f}"
        f"   [H={agg['holding']:>.0f}  P={agg['penalty']:>.0f}"
        f"  O={agg['outbound']:>.0f}  I={agg['inbound']:>.0f}]"
    )

def print_summary_block(label: str, agg: Dict[str, float]):
    """
    Prints a detailed summary block for a baseline.

    Args:
        label (str): Label for the summary block.
        agg (Dict[str, float]): Dictionary containing aggregate costs.
    """

    print(f"\n{'=' * 75}")
    print(f"  {label}")
    print(f"{'=' * 75}")
    print(f"  Reward:     {agg['reward']:>10.1f}  +/- {agg['reward_std']:.1f}")
    print(f"  Total Cost: {agg['total_cost']:>10.1f}  +/- {agg['total_cost_std']:.1f}")
    print(f"  Holding:    {agg['holding']:>10.1f}")
    print(f"  Penalty:    {agg['penalty']:>10.1f}")
    print(f"  Outbound:   {agg['outbound']:>10.1f}")
    print(f"  Inbound:    {agg['inbound']:>10.1f}")
    print(f"{'=' * 75}")

def plot_sweep_curve(
    x_values: List[float],
    sweep_stats: List[Dict[str, float]],
    x_label: str,
    title: str,
    output_path: Path,
    best_idx: Optional[int] = None,
):
    """
    Plots a two-panel sweep curve:
      - Top:    mean episode reward vs sweep parameter (with best marked)
      - Bottom: stacked cost components vs sweep parameter

    Args:
        x_values (List[float]): Sweep parameter values (e.g. quantities or z values).
        sweep_stats (List[Dict[str, float]]): List of aggregate cost dicts per sweep value 
        x_label (str): Label for x-axis.
        title (str): Overall figure title.
        output_path (Path): Path to save the plot.
        best_idx (Optional[int]): Index of the best sweep value (marked on the plot).
    """

    # Convert x values to numpy array and extract metrics
    x = np.array(x_values)
    rewards = np.array([s["reward"] for s in sweep_stats])
    reward_std = np.array([s["reward_std"] for s in sweep_stats])
    holding = np.array([s["holding"] for s in sweep_stats])
    penalty = np.array([s["penalty"] for s in sweep_stats])
    outbound = np.array([s["outbound"] for s in sweep_stats])
    inbound = np.array([s["inbound"] for s in sweep_stats])

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: reward curve
    ax1.plot(x, rewards, "o-", color="#4c72b0", linewidth=1.5, markersize=5)
    ax1.fill_between(x, rewards - reward_std, rewards + reward_std,
                      alpha=0.15, color="#4c72b0")
    if best_idx is not None:
        ax1.axvline(x[best_idx], color="#dd8452", linestyle="--", alpha=0.7,
                     label=f"Best ({x_label}={x[best_idx]:g})")
        ax1.plot(x[best_idx], rewards[best_idx], "D", color="#dd8452",
                 markersize=8, zorder=5)
        ax1.legend(fontsize=9)
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Bottom: stacked cost components
    ax2.stackplot(
        x, holding, penalty, outbound, inbound,
        labels=["Holding", "Penalty", "Outbound", "Inbound"],
        colors=["#4c72b0", "#dd8452", "#55a868", "#c44e52"],
        alpha=0.8,
    )
    if best_idx is not None:
        ax2.axvline(x[best_idx], color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Mean Episode Cost")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Set layout and save figure
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_adaptive_bs_sweep(
    z_values: List[float],
    H_values: List[int],
    sweep_stats: List[Dict[str, float]],
    best_z: float,
    best_H: int,
    output_path: Path,
):
    """
    Plots a multi-line chart for the BS-Adaptive 2-D sweep.

    Each line represents a different rolling-window width *H* and shows
    mean episode reward versus safety factor *z*.  The overall best
    (z, H) combination is highlighted with a marker.

    Args:
        z_values (List[float]): Safety factor grid (inner loop).
        H_values (List[int]): Rolling-window widths (outer loop).
        sweep_stats (List[Dict[str, float]]): Flat list of aggregate stats
            ordered as ``H_values`` outer × ``z_values`` inner.
        best_z (float): Best safety factor.
        best_H (int): Best rolling-window width.
        output_path (Path): Path to save the plot.
    """

    # Convert z and H values to numpy arrays
    n_z = len(z_values)
    z_arr = np.array(z_values)

    # Define the color map and colors for the lines
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(H_values) - 1, 1)) for i in range(len(H_values))]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the reward curve for each rolling-window width
    for h_idx, H_val in enumerate(H_values):
        start = h_idx * n_z
        rewards = np.array([s["reward"] for s in sweep_stats[start:start + n_z]])
        label = f"H={H_val}"
        ax.plot(z_arr, rewards, "o-", color=colors[h_idx], linewidth=1.3,
                markersize=3, label=label)

    # Mark the overall best
    ax.plot(best_z, sweep_stats[
        H_values.index(best_H) * n_z + z_values.index(best_z)
    ]["reward"], "D", color="#dd8452", markersize=5, zorder=5,
            label=f"Best (z={best_z}, H={best_H})")

    # Set labels and title
    ax.set_xlabel("Safety Factor z")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("BS-Adaptive — Sweep over Safety Factor z and Window H")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    # Set layout and save figure
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_bo_convergence(
    convergence: List[float],
    output_path: Path,
):
    """
    Plots the Bayesian-optimization convergence curve.

    Shows the best objective (reward, not negated) found as a function
    of the number of BO evaluations.

    Args:
        convergence (List[float]): Best *negative* objective at each
            evaluation (from :func:`run_bs_optimization`).
        output_path (Path): Path to save the plot.
    """

    # Convert convergence list to numpy array and negate back to reward
    iters = np.arange(1, len(convergence) + 1)
    best_reward = -np.array(convergence) 

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the best reward curve
    ax.plot(iters, best_reward, "-", color="#4c72b0", linewidth=1.5)
    ax.set_xlabel("BO Evaluation")
    ax.set_ylabel("Best Mean Episode Reward")
    ax.set_title("BS-Optimized — Bayesian Optimization Convergence")
    ax.grid(True, alpha=0.3)

    # Mark the final best
    ax.axhline(best_reward[-1], color="#dd8452", linestyle="--", alpha=0.7,
               label=f"Final best = {best_reward[-1]:.1f}")
    ax.legend(fontsize=9)

    # Set layout and save figure
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bo_independent_convergence(
    convergence_log: Dict[str, List[float]],
    n_warehouses: int,
    n_rounds: int,
    output_path: Path,
):
    """
    Plots per-warehouse BO convergence curves for BS-Independent.

    Creates a subplot per round, with one line per warehouse showing
    the best own-reward found as a function of BO evaluations.

    Args:
        convergence_log (Dict[str, List[float]]): Keyed by
            ``"round{r}_wh{w}"``, values are best negative objective
            per step.
        n_warehouses (int): Number of warehouses.
        n_rounds (int): Number of iterated-best-response rounds.
        output_path (Path): Path to save the plot.
    """

    fig, axes = plt.subplots(
        1, n_rounds, figsize=(6 * n_rounds, 5), squeeze=False,
    )

    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(n_warehouses)]

    for rnd in range(n_rounds):
        ax = axes[0, rnd]
        for wh in range(n_warehouses):
            key = f"round{rnd + 1}_wh{wh}"
            if key not in convergence_log:
                continue
            curve = -np.array(convergence_log[key])
            iters = np.arange(1, len(curve) + 1)
            ax.plot(iters, curve, "-", color=colors[wh], linewidth=1.3,
                    label=f"WH {wh}")
        ax.set_xlabel("BO Evaluation")
        ax.set_ylabel("Best WH Reward")
        ax.set_title(f"Round {rnd + 1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("BS-Independent — Per-Warehouse BO Convergence",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results_json(results: Dict[str, Any], experiment_dir: Path):
    """
    Saves baseline results to ``baseline_results.json``.

    Args:
        results (Dict[str, Any]): Baseline result data.
        experiment_dir (Path): Experiment output directory.
    """

    # Save baseline results to a JSON file
    results_path = experiment_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] Saved baseline results to: {results_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main CLI entry point."""

    # Create simple CLI parser for baseline experiments
    parser = argparse.ArgumentParser(description="Run baseline sanity checks")
    parser.add_argument("--env-config", type=str, default="config_files/environments/env_simplified_symmetric.yaml")
    parser.add_argument("--storage-dir", type=str, default="./experiment_outputs")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment folder name. If not provided, auto-generated.")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    args = parser.parse_args()

    # Load environment config
    env_config = load_environment_config(args.env_config)

    # Derive seeds via SeedManager (same derivation as EvaluationRunner)
    seed_manager = SeedManager(root_seed=args.root_seed, seed_registry=EXPERIMENT_SEEDS)
    eval_seed = seed_manager.get_seed_int('eval')
    calibration_seed = seed_manager.get_seed_int('train')

    # Separate RNG for the random baseline policy (independent of env seeds)
    rng = np.random.default_rng(args.root_seed)

    # Create experiment directory
    experiment_name = args.experiment_name or generate_baseline_experiment_name(env_config)
    experiment_dir = Path(args.storage_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = experiment_dir / "visualizations"

    # Save env config
    save_env_config(env_config, experiment_dir)

    # Run all baselines
    baseline_results = run_all_baselines(
        env_config=env_config,
        eval_seed=eval_seed,
        calibration_seed=calibration_seed,
        num_episodes=args.num_episodes,
        rng=rng,
        viz_dir=viz_dir,
    )

    # Save combined results
    all_results: Dict[str, Any] = {
        "root_seed": args.root_seed,
        "eval_seed": eval_seed,
        "calibration_seed": calibration_seed,
        "num_episodes": args.num_episodes,
        "env_config_path": str(args.env_config),
        **baseline_results,
    }
    save_results_json(all_results, experiment_dir)
    print(f"\n[INFO] All outputs saved to: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
