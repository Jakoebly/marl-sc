from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def generate_visualizations(
    episodes_data: List[Dict[str, np.ndarray]],
    output_dir: str,
    episode_idx: int = 0,
) -> None:
    """
    Generates and saves all visualization plots from rollout data.
    
    Plots the first episode in detail and generates an episode summary across all episodes.
    
    Args:
        episodes_data (List[Dict[str, np.ndarray]]): List of episode data dicts from rollout().
        output_dir (str): Directory to save plot files.
        episode_idx (int): Which episode to plot in detail. Defaults to 0 (first episode).
    """

    # Create output directory
    viz_dir = Path(output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Get the episode to plot in detail
    episode = episodes_data[episode_idx]

    # Generate individual plots
    plot_inventory(episode, viz_dir)
    plot_orders(episode, viz_dir)
    plot_cost_breakdown(episode, viz_dir)
    plot_demand_fulfillment(episode, viz_dir)
    plot_shipment_heatmap(episode, viz_dir)

    # Generate episode summary across all episodes
    if len(episodes_data) > 1:
        plot_episode_summary(episodes_data, viz_dir)

    print(f"[INFO] Visualizations saved to {viz_dir}")


def plot_inventory(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots inventory levels and pending orders over time for each warehouse.
    
    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """

    # Get inventory and pending orders
    inventory = episode["inventory"]               # Shape: (T, num_warehouses, num_skus)
    pending = episode["pending_total"]              # Shape: (T, n_warehouses, n_skus)

    # Get number of timesteps, warehouses, and SKUs
    T, n_warehouses, n_skus = inventory.shape
    timesteps = np.arange(T)

    # Create figure and axes
    fig, axes = plt.subplots(n_warehouses, 1, figsize=(14, 4 * n_warehouses), sharex=True)
    if n_warehouses == 1:
        axes = [axes]

    # Plot inventory and pending orders for each warehouse
    for wh in range(n_warehouses):
        ax = axes[wh]
        for sku in range(n_skus):
            ax.plot(timesteps, inventory[:, wh, sku], label=f"SKU {sku}", linewidth=1.5)
            ax.plot(timesteps, pending[:, wh, sku], label=f"SKU {sku} (pending)",
                    linestyle="--", alpha=0.5, linewidth=1.0)

        ax.set_ylabel("Quantity")
        ax.set_title(f"Warehouse {wh} — Inventory & Pending Orders")
        ax.legend(loc="upper right", fontsize=7, ncol=min(n_skus, 5))
        ax.grid(True, alpha=0.3)

    # Set x label and save figure
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "01_inventory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_orders(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots replenishment order quantities over time for each warehouse.
    
    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """

    # Get order quantities
    orders = episode["order_quantities"]            # Shape: (T, n_warehouses, n_skus)

    # Get number of timesteps, warehouses, and SKUs
    T, n_warehouses, n_skus = orders.shape
    timesteps = np.arange(T)

    # Create figure and axes
    fig, axes = plt.subplots(n_warehouses, 1, figsize=(14, 4 * n_warehouses), sharex=True)
    if n_warehouses == 1:
        axes = [axes]

    # Plot order quantities for each warehouse
    for wh in range(n_warehouses):
        ax = axes[wh]
        for sku in range(n_skus):
            ax.step(timesteps, orders[:, wh, sku], label=f"SKU {sku}", where="mid", linewidth=1.2)

        ax.set_ylabel("Order Quantity")
        ax.set_title(f"Warehouse {wh} — Replenishment Orders")
        ax.legend(loc="upper right", fontsize=7, ncol=min(n_skus, 5))
        ax.grid(True, alpha=0.3)

    # Set x label and save figure
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "02_orders.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots stacked cost breakdown over time for each warehouse.
    
    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """

    # Get cost breakdown
    holding = episode["holding_cost"]               # Shape: (T, n_warehouses)
    penalty = episode["penalty_cost"]               # Shape: (T, n_warehouses)
    outbound = episode["outbound_shipment_cost"]    # Shape: (T, n_warehouses)
    inbound = episode["inbound_shipment_cost"]      # Shape: (T, n_warehouses)

    # Get number of timesteps and warehouses
    T, n_warehouses = holding.shape
    timesteps = np.arange(T)

    # Create cost labels and colors
    cost_labels = ["Holding", "Penalty", "Outbound Shipping", "Inbound Shipping"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # Create figure and axes
    fig, axes = plt.subplots(n_warehouses, 1, figsize=(14, 4 * n_warehouses), sharex=True)
    if n_warehouses == 1:
        axes = [axes]

    # Plot cost breakdown for each warehouse
    for wh in range(n_warehouses):
        ax = axes[wh]
        costs = np.stack([holding[:, wh], penalty[:, wh], outbound[:, wh], inbound[:, wh]])
        ax.stackplot(timesteps, costs, labels=cost_labels, colors=colors, alpha=0.8)
        ax.set_ylabel("Cost")
        ax.set_title(f"Warehouse {wh} — Cost Breakdown")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Set x label and save figure
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "03_cost_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_demand_fulfillment(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots system-wide demand vs. fulfillment and fill rate over time.
    
    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """

    demand = episode["demand_per_region"]               # Shape: (T, n_regions, n_skus)
    fulfilled = episode["fulfilled_per_warehouse"]      # Shape: (T, n_warehouses, n_skus)
    unfulfilled = episode["unfulfilled_demands"]        # Shape: (T, n_regions, n_skus)

    # Get number of timesteps
    T = demand.shape[0]
    timesteps = np.arange(T)

    # Aggregate to system-wide totals per timestep
    total_demand = demand.sum(axis=(1, 2))              # (T,)
    total_fulfilled = fulfilled.sum(axis=(1, 2))        # (T,)
    total_unfulfilled = unfulfilled.sum(axis=(1, 2))    # (T,)

    # Compute fill rate (avoid division by zero)
    fill_rate = np.where(total_demand > 0, total_fulfilled / total_demand, 1.0)

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: demand vs fulfilled vs unfulfilled
    ax1.bar(timesteps, total_demand, alpha=0.3, color="gray", label="Demand", width=0.9)
    ax1.bar(timesteps, total_fulfilled, alpha=0.7, color="#55a868", label="Fulfilled", width=0.9)
    ax1.bar(timesteps, total_unfulfilled, bottom=total_fulfilled, alpha=0.7, 
            color="#c44e52", label="Unfulfilled", width=0.9)
    ax1.set_ylabel("Quantity")
    ax1.set_title("System-Wide Demand vs. Fulfillment")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: fill rate over time
    ax2.plot(timesteps, fill_rate, color="#4c72b0", linewidth=1.5, label="Fill Rate")
    ax2.axhline(y=fill_rate.mean(), color="#dd8452", linestyle="--", alpha=0.7, 
                label=f"Mean = {fill_rate.mean():.2%}")
    ax2.set_ylabel("Fill Rate")
    ax2.set_xlabel("Timestep")
    ax2.set_title("System-Wide Fill Rate")
    ax2.set_ylim(-0.05, 1.1)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Set x label and save figure
    plt.tight_layout()
    plt.savefig(output_dir / "04_demand_fulfillment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shipment_heatmap(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots heatmap of average shipment quantities from warehouses to regions.
    Shows geographic specialization of agents.
    
    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """

    # Get shipment quantities
    shipments = episode["shipment_quantities"]          # Shape: (T, n_warehouses, n_regions)
    avg_shipments = shipments.mean(axis=0)              # Shape: (n_warehouses, n_regions)

    # Get number of warehouses and regions
    n_warehouses, n_regions = avg_shipments.shape

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(max(6, n_regions * 2), max(4, n_warehouses * 1.5)))

    # Plot heatmap  
    im = ax.imshow(avg_shipments, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg. Shipment Quantity per Timestep")

    # Set x and y ticks and labels
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels([f"Region {r}" for r in range(n_regions)])
    ax.set_yticks(range(n_warehouses))
    ax.set_yticklabels([f"Warehouse {w}" for w in range(n_warehouses)])
    ax.set_title("Warehouse → Region Shipment Flow (Episode Average)")

    # Annotate cells with values
    for w in range(n_warehouses):
        for r in range(n_regions):
            ax.text(r, w, f"{avg_shipments[w, r]:.1f}", ha="center", va="center",
                    color="white" if avg_shipments[w, r] > avg_shipments.max() * 0.6 else "black",
                    fontsize=10, fontweight="bold")

    # Set x label and save figure
    plt.tight_layout()
    plt.savefig(output_dir / "05_shipment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_summary(episodes_data: List[Dict[str, np.ndarray]], output_dir: Path) -> None:
    """
    Plots summary metrics across all episodes (total reward, cost breakdown).
    
    Args:
        episodes_data (List[Dict[str, np.ndarray]]): List of episode data dicts.
        output_dir (Path): Directory to save the plot.
    """

    # Get number of episodes
    n_episodes = len(episodes_data)
    episode_indices = np.arange(n_episodes)

    # Compute per-episode totals
    total_rewards = []
    total_holding = []
    total_penalty = []
    total_outbound = []
    total_inbound = []

    # Compute per-episode totals
    for ep_data in episodes_data:
        total_rewards.append(ep_data["rewards"].sum())
        total_holding.append(ep_data["holding_cost"].sum())
        total_penalty.append(ep_data["penalty_cost"].sum())
        total_outbound.append(ep_data["outbound_shipment_cost"].sum())
        total_inbound.append(ep_data["inbound_shipment_cost"].sum())

    # Convert to numpy arrays
    total_rewards = np.array(total_rewards)
    total_holding = np.array(total_holding)
    total_penalty = np.array(total_penalty)
    total_outbound = np.array(total_outbound)
    total_inbound = np.array(total_inbound)

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total reward per episode
    bars = ax1.bar(episode_indices, total_rewards, color="#4c72b0", alpha=0.8)
    ax1.axhline(y=total_rewards.mean(), color="#dd8452", linestyle="--", 
                label=f"Mean = {total_rewards.mean():.1f}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Total Reward per Episode")
    ax1.set_xticks(episode_indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: stacked cost breakdown per episode
    bar_width = 0.6
    ax2.bar(episode_indices, total_holding, bar_width, label="Holding", color="#4c72b0", alpha=0.8)
    ax2.bar(episode_indices, total_penalty, bar_width, bottom=total_holding, 
            label="Penalty", color="#dd8452", alpha=0.8)
    ax2.bar(episode_indices, total_outbound, bar_width, 
            bottom=total_holding + total_penalty, label="Outbound", color="#55a868", alpha=0.8)
    ax2.bar(episode_indices, total_inbound, bar_width, 
            bottom=total_holding + total_penalty + total_outbound, 
            label="Inbound", color="#c44e52", alpha=0.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Cost")
    ax2.set_title("Cost Breakdown per Episode")
    ax2.set_xticks(episode_indices)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Set x label and save figure
    plt.tight_layout()
    plt.savefig(output_dir / "06_episode_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
