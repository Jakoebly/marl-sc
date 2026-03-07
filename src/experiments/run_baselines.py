"""
Runs baseline sanity-check policies on the InventoryEnvironment:
  1. Random: Uniform random orders in [0, max_order_quantity]
  2. Constant Order: Every warehouse-SKU pair orders the same fixed quantity each timestep
  3. Heuristic: Newsvendor base-stock policy, sweep over safety factor z

Each baseline is evaluated over multiple rollout episodes with visualization.
"""

import argparse
import json
import shutil
import sys
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_environment_config
from src.config.schema import EnvironmentConfig
from src.environment.envs.multi_env import InventoryEnvironment
from src.experiments.visualization import generate_visualizations


# ---------------------------------------------------------------------------
# Generic baseline rollout
# ---------------------------------------------------------------------------

def baseline_rollout(
    env: InventoryEnvironment,
    action_fn: Callable,
    num_episodes: int = 10,
) -> List[Dict[str, np.ndarray]]:
    """
    Runs rollout episodes using a baseline action function (no RL policy).
    Collects the same per-step data format as BaseAlgorithmWrapper.rollout().

    Args:
        env (InventoryEnvironment): InventoryEnvironment instance.
        action_fn (Callable): Callable(env, obs) -> Dict[agent_id, np.ndarray] returning
            actions in [-1, 1] for each agent.
        num_episodes (int): Number of episodes to roll out.

    Returns:
        List of episode data dicts (same schema as rollout()).
    """

    # Enable step info collection
    env.collect_step_info = True

    # Initialize list to store all episodes
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

        # Convert all lists to numpy arrays
        episode_data = {k: np.array(v) for k, v in episode_data.items()}
        all_episodes.append(episode_data)

    # Disable step info collection after rollout
    env.collect_step_info = False

    # Return all episodes
    return all_episodes


# ---------------------------------------------------------------------------
# Baseline action functions
# ---------------------------------------------------------------------------

def make_random_action_fn(rng: np.random.Generator):
    """
    Samples uniform random orders in [0, max_order_quantity] for every warehouse-SKU pair.

    Args:
        rng (np.random.Generator): Random number generator.

    Returns:
        action_fn (Callable): Callable(env, obs) -> Dict[agent_id, np.ndarray] returning
            actions in [-1, 1] for each agent.
    """

    # Define action function for random orders
    def action_fn(env, obs):
        actions = {
            agent_id: rng.uniform(-1.0, 1.0, size=(env.n_skus,)).astype(np.float32)
            for agent_id in env.agents
        }
        return actions

    return action_fn


def make_constant_action_fn(quantity: float):
    """
    Makes every warehouse-SKU pair order the same fixed quantity each timestep.

    Args:
        quantity (float): Fixed quantity to order for each warehouse-SKU pair.

    Returns:
        action_fn (Callable): Callable(env, obs) -> Dict[agent_id, np.ndarray] returning
            actions in [-1, 1] for each agent.
    """

    # Define action function for constant orders
    def action_fn(env, obs):
        action = 2.0 * quantity / env.max_order_quantities - 1.0
        actions = {
            agent_id: action.astype(np.float32)
            for agent_id in env.agents
        }
        return actions

    return action_fn


def make_heuristic_action_fn(env_config: EnvironmentConfig, z: float):
    """
    Makes each warehouse-SKU pair order according to the Newsvendor base-stock heuristic:

        1. Calculate expected demand per home-region, per SKU:
            E[D] = lambda_orders * probability_skus * lambda_quantity
        2. Calculate expected demand over lead time:
            E[D_L] = L * E[D]
        3. Calculate Poisson approximation (Var(D) = E[D]):
            sigma_L ≈ sqrt(L * E[D])
        4. Calculate base-stock level:
            S = E[D_L] + z * sigma_L
        5. Calculate order quantity:
            order  = max(0, S - on_hand - pipeline)

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        z (float): Safety factor for the Newsvendor base-stock heuristic.

    Returns:
        action_fn (Callable): Callable(env, obs) -> Dict[agent_id, np.ndarray] returning
            actions in [-1, 1] for each agent.
    """

    # Extract number of warehouses and SKUs
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus

    # Extract demand parameters
    demand_params = env_config.components.demand_sampler.params
    lambda_orders = np.array(demand_params["lambda_orders"])        # Shape:(n_regions,)
    probability_skus = np.array(demand_params["probability_skus"])  # Shape:(n_regions,)
    lambda_quantity = np.array(demand_params["lambda_quantity"])     # Shape:(n_regions, n_skus)

    # Extract lead time parameters
    lead_times = np.array(
        env_config.components.lead_time_sampler.params["values"]    # Shape:(n_warehouses, n_skus)
    )

    # Extract cost structure parameters
    distances = np.array(env_config.cost_structure.distances)       # Shape:(n_warehouses, n_regions)

    # Calculate home regions for each warehouse
    home_regions = np.argmin(distances, axis=1)                     # Shape:(n_warehouses,)

    # Extract max order quantities
    max_qty = np.array(env_config.max_order_quantities, dtype=float)

    # Initialize base-stock levels for each warehouse-SKU pair
    base_stock = np.zeros((n_warehouses, n_skus))

    # Compute base-stock levels
    for wh in range(n_warehouses):
        # Extract home region for current warehouse
        home = home_regions[wh]
        # Compute base-stock level for each SKU for the current warehouse
        for sku in range(n_skus):
            # Extract lead time for current warehouse-SKU pair
            L = lead_times[wh, sku]
            # Compute expected demand for current warehouse-SKU pair
            E_D = lambda_orders[home] * probability_skus[home] * lambda_quantity[home, sku]
            # Compute expected demand over lead time for current warehouse-SKU pair
            E_D_L = L * E_D
            # Compute Poisson approximation for current warehouse-SKU pair
            sigma_L = np.sqrt(L * E_D)
            # Compute base-stock level for current warehouse-SKU pair
            base_stock[wh, sku] = E_D_L + z * sigma_L

    # Print base-stock levels for each warehouse-SKU pair
    print(f"  [Heuristic z={z:.2f}] base-stock levels (per WH x SKU):")
    for wh in range(n_warehouses):
        print(f"    WH {wh}: {base_stock[wh].round(1)}")

    # Define action function for heuristic base-stock orders
    def action_fn(env, obs):
        # Extract inventory and pipeline for each warehouse
        inventory = env.inventory
        pipeline = np.zeros((n_warehouses, n_skus), dtype=float)
        for arr in env.pending_orders.values():
            pipeline += arr

        # Compute actions for each warehouse based on base-stock levels
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


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def episode_costs(ep: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Computes total costs for a single episode.

    Args:
        ep (Dict[str, np.ndarray]): Episode data dict.

    Returns:
        costs (Dict[str, float]): Dictionary containing total costs for the episode.
    """
    return {
        "reward": float(ep["rewards"].sum()),
        "holding": float(ep["holding_cost"].sum()),
        "penalty": float(ep["penalty_cost"].sum()),
        "outbound": float(ep["outbound_shipment_cost"].sum()),
        "inbound": float(ep["inbound_shipment_cost"].sum()),
    }


def aggregate_costs(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Computes mean costs across episodes.

    Args:
        episodes (List[Dict[str, np.ndarray]]): List of episode data dicts.

    Returns:
        costs (Dict[str, float]): Dictionary containing mean costs across episodes.
    """
    costs = [episode_costs(ep) for ep in episodes]
    return {
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


# ---------------------------------------------------------------------------
# Sweep curve visualizations
# ---------------------------------------------------------------------------

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
        sweep_stats (List[Dict[str, float]]): List of aggregate cost dicts (one per sweep value).
        x_label (str): Label for x-axis (e.g. "Order Quantity" or "Safety Factor z").
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def generate_experiment_name(env_config: EnvironmentConfig) -> str:
    """
    Generates a timestamped experiment folder name matching the project convention.

    Args:
        env_config (EnvironmentConfig): Environment configuration.

    Returns:
        experiment_name (str): Timestamped experiment folder name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"BASELINE_{env_config.n_warehouses}WH_{env_config.n_skus}SKU_{timestamp}"


def save_env_config(env_config: EnvironmentConfig, experiment_dir: Path):
    """
    Saves environment config to the experiment directory (same format as training).

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        experiment_dir (Path): Path to the experiment directory.
    """
    env_config_path = experiment_dir / "env_config.yaml"
    if not env_config_path.exists():
        with open(env_config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"environment": env_config.model_dump()},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"[INFO] Saved environment config to: {env_config_path}")


def save_results_json(results: Dict[str, Any], experiment_dir: Path):
    """
    Saves baseline results to a JSON file.

    Args:
        results (Dict[str, Any]): Dictionary containing baseline results.
        experiment_dir (Path): Path to the experiment directory.
    """
    results_path = experiment_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] Saved baseline results to: {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description="Run baseline sanity checks")
    parser.add_argument("--env-config", type=str, default="config_files/environments/env_simplified_symmetric.yaml")
    parser.add_argument("--output-dir", type=str, default="./experiment_outputs")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment folder name. If not provided, auto-generated.")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args() # Parse command line arguments

    # Load environment config
    env_config = load_environment_config(args.env_config)
    max_qty = np.array(env_config.max_order_quantities, dtype=float)
    rng = np.random.default_rng(args.seed)

    # Create experiment directory (matching project convention)
    experiment_name = args.experiment_name or generate_experiment_name(env_config)
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = experiment_dir / "visualizations"

    # Save env config alongside results
    save_env_config(env_config, experiment_dir)

    # Copy original env config file for reference
    src_config = Path(args.env_config)
    if src_config.exists():
        shutil.copy2(src_config, experiment_dir / src_config.name)

    # Collect all results for the JSON summary
    all_results: Dict[str, Any] = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "env_config_path": str(args.env_config),
        "baselines": {},
    }

    # ==================================================================
    # 1. Random Baseline
    # ==================================================================

    print("\n[1/3] Running Random baseline...")
    env = InventoryEnvironment(env_config, seed=args.seed)
    episodes = baseline_rollout(env, make_random_action_fn(rng), args.num_episodes)
    random_agg = aggregate_costs(episodes)
    print_summary_block("Random Baseline", random_agg)
    generate_visualizations(episodes, str(viz_dir / "random"))
    all_results["baselines"]["random"] = random_agg


    # ==================================================================
    # 2. Constant Order Baseline (sweep)
    # ==================================================================

    print("\n[2/3] Running Constant Order baseline sweep...")
    step_size = 5
    sweep_max = int(max_qty.min())
    constant_amounts = list(range(step_size, sweep_max + 1, step_size))

    const_sweep_stats: List[Dict[str, float]] = []
    best_const_idx = 0
    best_const_reward = -np.inf
    best_const_episodes = None

    for idx, amount in enumerate(constant_amounts):
        env = InventoryEnvironment(env_config, seed=args.seed)
        eps = baseline_rollout(env, make_constant_action_fn(float(amount)), args.num_episodes)
        agg = aggregate_costs(eps)
        const_sweep_stats.append(agg)
        print_sweep_row(f"qty={amount:3d}", agg)
        if agg["reward"] > best_const_reward:
            best_const_reward = agg["reward"]
            best_const_idx = idx
            best_const_episodes = eps

    best_amount = constant_amounts[best_const_idx]
    print_summary_block(
        f"Best Constant Baseline (qty={best_amount})",
        const_sweep_stats[best_const_idx],
    )
    generate_visualizations(best_const_episodes, str(viz_dir / f"constant_best_{best_amount}"))
    plot_sweep_curve(
        x_values=constant_amounts,
        sweep_stats=const_sweep_stats,
        x_label="Order Quantity",
        title="Constant Order Baseline — Sweep over Order Quantity",
        output_path=viz_dir / "sweep_constant.png",
        best_idx=best_const_idx,
    )
    all_results["baselines"]["constant"] = {
        "sweep": {str(q): s for q, s in zip(constant_amounts, const_sweep_stats)},
        "best_quantity": best_amount,
        "best": const_sweep_stats[best_const_idx],
    }


    # ==================================================================
    # 3. Heuristic (Newsvendor) Baseline (sweep over z)
    # ==================================================================

    print("\n[3/3] Running Heuristic (Newsvendor) baseline sweep...")
    z_values = [0.0, 0.5, 0.97, 1.0, 2.0]

    heur_sweep_stats: List[Dict[str, float]] = []
    best_heur_idx = 0
    best_heur_reward = -np.inf
    best_heur_episodes = None

    for idx, z in enumerate(z_values):
        env = InventoryEnvironment(env_config, seed=args.seed)
        action_fn = make_heuristic_action_fn(env_config, z=z)
        eps = baseline_rollout(env, action_fn, args.num_episodes)
        agg = aggregate_costs(eps)
        heur_sweep_stats.append(agg)
        print_sweep_row(f"z={z:5.2f}", agg)
        if agg["reward"] > best_heur_reward:
            best_heur_reward = agg["reward"]
            best_heur_idx = idx
            best_heur_episodes = eps

    best_z = z_values[best_heur_idx]
    print_summary_block(
        f"Best Heuristic Baseline (z={best_z})",
        heur_sweep_stats[best_heur_idx],
    )
    generate_visualizations(best_heur_episodes, str(viz_dir / f"heuristic_best_z{best_z}"))
    plot_sweep_curve(
        x_values=z_values,
        sweep_stats=heur_sweep_stats,
        x_label="Safety Factor z",
        title="Heuristic (Newsvendor) Baseline — Sweep over Safety Factor z",
        output_path=viz_dir / "sweep_heuristic.png",
        best_idx=best_heur_idx,
    )
    all_results["baselines"]["heuristic"] = {
        "sweep": {str(z): s for z, s in zip(z_values, heur_sweep_stats)},
        "best_z": best_z,
        "best": heur_sweep_stats[best_heur_idx],
    }


    # ==================================================================
    # Final comparison + save
    # ==================================================================
    
    print(f"\n{'=' * 75}")
    print("  BASELINE COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Baseline':<40s}  {'Mean Reward':>12s}")
    print(f"  {'-' * 40}  {'-' * 12}")
    print(f"  {'Random':<40s}  {random_agg['reward']:>12.1f}")
    print(f"  {f'Constant (qty={best_amount})':<40s}  {const_sweep_stats[best_const_idx]['reward']:>12.1f}")
    print(f"  {f'Heuristic (z={best_z})':<40s}  {heur_sweep_stats[best_heur_idx]['reward']:>12.1f}")
    print(f"{'=' * 75}")

    all_results["comparison"] = {
        "random": random_agg["reward"],
        f"constant_qty{best_amount}": const_sweep_stats[best_const_idx]["reward"],
        f"heuristic_z{best_z}": heur_sweep_stats[best_heur_idx]["reward"],
    }

    save_results_json(all_results, experiment_dir)
    print(f"\n[INFO] All outputs saved to: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
