"""
Runs baseline sanity-check policies on the InventoryEnvironment:
  1. Random: Uniform random orders in [0, max_order_quantity]
  2. Constant Order: Every warehouse-SKU pair orders the same fixed quantity each timestep
  3. Heuristic: Newsvendor base-stock policy, sweep over safety factor z

Each baseline is evaluated over multiple rollout episodes with visualization.
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
from src.experiments.utils.experiment_utils import (
    generate_baseline_experiment_name,
    save_env_config,
)
from src.experiments.visualization import generate_visualizations
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

    # Define action function for random orders
    def action_fn(env, obs):
        actions = {
            agent_id: rng.uniform(-1.0, 1.0, size=(env.n_skus,)).astype(np.float32)
            for agent_id in env.agents
        }
        return actions

    return action_fn

def make_constant_action_fn(quantity: float, max_order_quantities: np.ndarray) -> Callable:
    """
    Returns an action function that orders a fixed ``quantity`` for every
    warehouse-SKU pair at each timestep.

    Requires a ``"direct"`` action space so that the mapping from order
    quantity to normalized ``[-1, 1]`` action is a simple linear rescaling.

    Args:
        quantity (float): Fixed order quantity.
        max_order_quantities (np.ndarray): Maximum order quantities per SKU
            (from ``env_config.action_space.params.max_order_quantities``).

    Returns:
        action_fn (Callable): Action function that produces actions in ``[-1, 1]`` 
            for each agent.
    """
    
    # Define action function for constant orders
    def action_fn(env, obs):
        static_action = (2.0 * quantity / max_order_quantities - 1.0).astype(np.float32)
        actions = {agent_id: static_action for agent_id in env.agents}
        return actions

    return action_fn

def make_heuristic_action_fn(env_config: EnvironmentConfig, z: float) -> Callable:
    """
    Returns an action function implementing the Newsvendor base-stock
    heuristic:

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

    # Define action function for heuristic base-stock orders
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


# ============================================================================
# Generic Baseline Rollout
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


# ============================================================================
# Swwep Orchestration Functions
# ============================================================================

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

def run_all_baselines(
    env_config: EnvironmentConfig,
    eval_seed: int,
    num_episodes: int,
    rng: np.random.Generator,
    viz_dir: Path,
) -> Dict[str, Any]:
    """
    Runs all baseline policies and returns a combined results dict.

    Executes the random baseline, a constant-order sweep, and a
    Newsvendor heuristic sweep.  Visualisations and sweep curves are
    saved under ``viz_dir``.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        eval_seed (int): Seed used to initialise each environment.
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
    print("\n[1/3] Running Random baseline...")

    # Create environment and run rollout
    env = InventoryEnvironment(env_config, seed=eval_seed)
    episodes = baseline_rollout(env, make_random_action_fn(rng), num_episodes)

    # Store results and print summary
    random_agg = aggregate_costs(episodes)
    results["baselines"]["random"] = random_agg
    print_summary_block("Random Baseline", random_agg)
    generate_visualizations(episodes, str(viz_dir / "random"))
    


    # ------------------------------------------------------------------
    # 2. Constant Order Baseline (sweep)
    # ------------------------------------------------------------------
    print("\n[2/3] Running Constant Order baseline sweep...")

    # Define sweep parameters and values
    step_size = 1
    sweep_max = 30
    constant_amounts = list(range(1, sweep_max + 1, step_size))

    # Run rollout for each constant order quantity
    const_sweep_stats, best_const_idx, best_const_episodes = run_sweep(
        env_config=env_config,
        eval_seed=eval_seed,
        num_episodes=num_episodes,
        sweep_values=constant_amounts,
        make_action_fn=lambda amount: make_constant_action_fn(
            float(amount),
            np.array(env_config.action_space.params.max_order_quantities, dtype=float),
        ),
        label_fn=lambda amount: f"qty={amount:3g}",
    )

    # Store results and print summary
    best_amount = constant_amounts[best_const_idx]
    results["baselines"]["constant"] = {
        "sweep": {str(q): s for q, s in zip(constant_amounts, const_sweep_stats)},
        "best_quantity": best_amount,
        "best": const_sweep_stats[best_const_idx],
    }
    print_summary_block(
        f"Best Constant Baseline (qty={best_amount})",
        const_sweep_stats[best_const_idx],
    )

    # Generate visualizations for the best constant order quantity
    generate_visualizations(
        best_const_episodes, str(viz_dir / f"constant_best_{best_amount}"),
    )

    # Plot the sweep results on a curve
    plot_sweep_curve(
        x_values=constant_amounts,
        sweep_stats=const_sweep_stats,
        x_label="Order Quantity",
        title="Constant Order Baseline — Sweep over Order Quantity",
        output_path=viz_dir / "sweep_constant.png",
        best_idx=best_const_idx,
    )


    # ------------------------------------------------------------------
    # 3. Heuristic (Newsvendor) Baseline (sweep over z)
    # ------------------------------------------------------------------
    print("\n[3/3] Running Heuristic (Newsvendor) baseline sweep...")

    # Define sweep parameters and values
    z_values = [0.0, 0.5, 0.97, 1.0, 2.0, 4.0, 5.0, 6.0]

    # Run rollout for each z-value
    heur_sweep_stats, best_heur_idx, best_heur_episodes = run_sweep(
        env_config=env_config,
        eval_seed=eval_seed,
        num_episodes=num_episodes,
        sweep_values=z_values,
        make_action_fn=lambda z: make_heuristic_action_fn(env_config, z=z),
        label_fn=lambda z: f"z={z:5.2f}",
    )

    # Store results and print summary
    best_z = z_values[best_heur_idx]
    results["baselines"]["heuristic"] = {
        "sweep": {str(z): s for z, s in zip(z_values, heur_sweep_stats)},
        "best_z": best_z,
        "best": heur_sweep_stats[best_heur_idx],
    }
    print_summary_block(
        f"Best Heuristic Baseline (z={best_z})",
        heur_sweep_stats[best_heur_idx],
    )

    # Generate visualizations for the best z-value
    generate_visualizations(
        best_heur_episodes, str(viz_dir / f"heuristic_best_z{best_z}"),
    )

    # Plot the sweep results on a curve
    plot_sweep_curve(
        x_values=z_values,
        sweep_stats=heur_sweep_stats,
        x_label="Safety Factor z",
        title="Heuristic (Newsvendor) Baseline — Sweep over Safety Factor z",
        output_path=viz_dir / "sweep_heuristic.png",
        best_idx=best_heur_idx,
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
    print(f"  {f'Constant (qty={best_amount})':<40s}  {const_sweep_stats[best_const_idx]['reward']:>12.1f}")
    print(f"  {f'Heuristic (z={best_z})':<40s}  {heur_sweep_stats[best_heur_idx]['reward']:>12.1f}")
    print(f"{'=' * 75}")

    # Store comparison results
    results["comparison"] = {
        "random": random_agg["reward"],
        f"constant_qty{best_amount}": const_sweep_stats[best_const_idx]["reward"],
        f"heuristic_z{best_z}": heur_sweep_stats[best_heur_idx]["reward"],
    }

    return results


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
    parser.add_argument("--root-seed", type=int, default=42)
    args = parser.parse_args()

    # Load environment config
    env_config = load_environment_config(args.env_config)

    # Derive eval_seed via SeedManager (same derivation as EvaluationRunner)
    seed_manager = SeedManager(root_seed=args.root_seed, seed_registry=EXPERIMENT_SEEDS)
    eval_seed = seed_manager.get_seed_int('eval')

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
        num_episodes=args.num_episodes,
        rng=rng,
        viz_dir=viz_dir,
    )

    # Save combined results
    all_results: Dict[str, Any] = {
        "root_seed": args.root_seed,
        "eval_seed": eval_seed,
        "num_episodes": args.num_episodes,
        "env_config_path": str(args.env_config),
        **baseline_results,
    }
    save_results_json(all_results, experiment_dir)
    print(f"\n[INFO] All outputs saved to: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
