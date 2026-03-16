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
    plot_orders_summary(episode, viz_dir)
    plot_cost_breakdown(episode, viz_dir)
    plot_demand_fulfillment(episode, viz_dir)
    plot_shipment_heatmap(episode, viz_dir)
    plot_observations(episode, viz_dir)
    plot_obs_normalization(episode, viz_dir)

    # Generate episode summary across all episodes
    if len(episodes_data) > 1:
        plot_episode_summary(episodes_data, viz_dir)

    print(f"[INFO] Visualizations saved to {viz_dir}")


def plot_inventory(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots inventory levels, pending orders, and actual order quantities over time for
    each warehouse-SKU pair (one subplot per pair).
    
    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """

    # Get inventory and pending orders
    inventory = episode["inventory"]               # Shape: (T, n_warehouses, n_skus)
    pending = episode["pending_total"]              # Shape: (T, n_warehouses, n_skus)
    order_qty = episode["order_quantities"]         # Shape: (T, n_warehouses, n_skus)

    # Get number of timesteps, warehouses, and SKUs
    T, n_warehouses, n_skus = inventory.shape
    timesteps = np.arange(T)
    n_plots = n_warehouses * n_skus

    # Create figure and axes
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Plot inventory and pending orders for each warehouse
    plot_idx = 0
    for wh in range(n_warehouses):
        for sku in range(n_skus):
            ax = axes[plot_idx]
            ax.plot(timesteps, inventory[:, wh, sku], label="Inventory",
                    linewidth=1.5, color="#4c72b0")
            ax.plot(timesteps, pending[:, wh, sku], label="Pending Orders",
                    linestyle="--", alpha=0.6, linewidth=1.0, color="#55a868")
            ax.bar(timesteps, order_qty[:, wh, sku], alpha=0.3, color="#dd8452",
                   label="Order Quantity", width=0.9)
            ax.set_ylabel("Quantity")
            ax.set_title(f"Warehouse {wh}, SKU {sku} — Inventory & Pending Orders")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    # Set x label and save figure
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "01_inventory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_orders(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots all types of replenishment order quantities over time for each warehouse-SKU
    pair (one subplot per pair). Shows actual order quantities (rescaled), raw sampled
    actions, and optionally the actor's mu/sigma distribution parameters.
    
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
    actions_raw = episode.get("actions_raw")        # Shape: (T, n_warehouses, n_skus) or None
    actor_mu = episode.get("actor_mu")              # Shape: (T, n_warehouses, n_skus) or None
    actor_sigma = episode.get("actor_sigma")        # Shape: (T, n_warehouses, n_skus) or None

    n_plots = n_warehouses * n_skus
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Plot order quantities for each warehouse
    has_twin = actions_raw is not None or (actor_mu is not None and actor_sigma is not None)

    plot_idx = 0
    for wh in range(n_warehouses):
        for sku in range(n_skus):
            ax = axes[plot_idx]

            # Actual order quantities (after rescaling and rounding)
            ax.step(timesteps, orders[:, wh, sku], label="Actual Order Qty",
                    where="mid", linewidth=1.5, color="#4c72b0")

            if has_twin:
                ax2 = ax.twinx()
                ax2.set_ylabel("Raw Action [-1, 1]", color="#c44e52", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="#c44e52")
                ax2.set_ylim(-1.5, 1.5)

                # Raw sampled actions (normalized [-1, 1])
                if actions_raw is not None:
                    ax2.step(timesteps, actions_raw[:, wh, sku], label="Raw Action",
                             where="mid", linewidth=1.0, linestyle="--",
                             color="#c44e52", alpha=0.7)

                # Actor mu and sigma (on the raw action scale)
                if actor_mu is not None and actor_sigma is not None:
                    mu = actor_mu[:, wh, sku]
                    sigma = actor_sigma[:, wh, sku]
                    ax2.plot(timesteps, mu, label="Actor mu", linewidth=1.0,
                             color="#55a868", alpha=0.8)
                    ax2.fill_between(timesteps, mu - sigma, mu + sigma,
                                     alpha=0.15, color="#55a868", label="Actor +/- sigma")

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          loc="upper right", fontsize=7)
            else:
                ax.legend(loc="upper right", fontsize=7)

            ax.set_ylabel("Order Quantity")
            ax.set_title(f"Warehouse {wh}, SKU {sku} — Replenishment Orders")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "02_orders.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_orders_summary(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots orders overview: per-warehouse replenishment totals, customer order statistics,
    and average demand per region-SKU pair.

    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """
    order_quantities = episode["order_quantities"]      # (T, n_warehouses, n_skus)
    demand_per_region = episode["demand_per_region"]    # (T, n_regions, n_skus)

    T, n_warehouses, n_skus = order_quantities.shape
    n_regions = demand_per_region.shape[1]
    timesteps = np.arange(T)

    n_orders = episode.get("n_orders", np.zeros(T, dtype=int))
    mean_unique_skus = episode.get("mean_unique_skus_per_order", np.zeros(T, dtype=float))
    if isinstance(n_orders, (int, float)):
        n_orders = np.full(T, n_orders)
    if isinstance(mean_unique_skus, (int, float)):
        mean_unique_skus = np.full(T, mean_unique_skus)

    wh_colors = plt.cm.Set2.colors
    sku_colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Replenishment orders per warehouse (stacked area)
    ax1 = axes[0, 0]
    wh_totals = [order_quantities[:, wh, :].sum(axis=1) for wh in range(n_warehouses)]
    ax1.stackplot(timesteps, wh_totals,
                  labels=[f"WH {w}" for w in range(n_warehouses)],
                  colors=[wh_colors[w % len(wh_colors)] for w in range(n_warehouses)],
                  alpha=0.75)
    ax1.set_ylabel("Total Order Qty")
    ax1.set_xlabel("Timestep")
    ax1.set_title("Replenishment Orders by Warehouse")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.2)

    # 2. Customer orders and mean unique SKUs
    ax2 = axes[0, 1]
    ax2.fill_between(timesteps, n_orders, alpha=0.3, color="#55a868")
    ln1 = ax2.plot(timesteps, n_orders, linewidth=1.2, color="#55a868", label="Num. Orders")
    ax2.set_ylabel("Customer Orders", color="#55a868")
    ax2.tick_params(axis="y", labelcolor="#55a868")
    ax2_twin = ax2.twinx()
    ln2 = ax2_twin.plot(timesteps, mean_unique_skus, linewidth=1.2, color="#c44e52",
                        linestyle="--", label="Mean Unique SKUs")
    ax2_twin.set_ylabel("Mean Unique SKUs / Order", color="#c44e52")
    ax2_twin.tick_params(axis="y", labelcolor="#c44e52")
    ax2.set_xlabel("Timestep")
    ax2.set_title("Customer Order Statistics")
    ax2.legend(handles=ln1 + ln2, fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.2)

    # 3. Total demand per SKU over time
    ax3 = axes[1, 0]
    total_demand_per_sku = demand_per_region.sum(axis=1)  # (T, n_skus)
    for sku in range(n_skus):
        ax3.plot(timesteps, total_demand_per_sku[:, sku], linewidth=1.2,
                 color=sku_colors[sku % len(sku_colors)], label=f"SKU {sku}")
    ax3.set_ylabel("Total Demand Qty")
    ax3.set_xlabel("Timestep")
    ax3.set_title("Total Customer Demand per SKU")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.2)

    # 4. Average demand per region-SKU pair (heatmap)
    ax4 = axes[1, 1]
    avg_demand = demand_per_region.mean(axis=0)  # (n_regions, n_skus)
    im = ax4.imshow(avg_demand, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax4, shrink=0.85)
    cbar.set_label("Avg. Demand / Timestep", fontsize=8)
    ax4.set_xticks(range(n_skus))
    ax4.set_xticklabels([f"SKU {s}" for s in range(n_skus)], fontsize=8)
    ax4.set_yticks(range(n_regions))
    ax4.set_yticklabels([f"Region {r}" for r in range(n_regions)], fontsize=8)
    ax4.set_xlabel("SKU")
    ax4.set_ylabel("Region")
    ax4.set_title("Avg. Demand per Region × SKU")
    vmax = avg_demand.max() if avg_demand.max() > 0 else 1
    for r in range(n_regions):
        for s in range(n_skus):
            val = avg_demand[r, s]
            ax4.text(s, r, f"{val:.0f}", ha="center", va="center",
                     color="white" if val > vmax * 0.55 else "black",
                     fontsize=9, fontweight="bold")

    fig.suptitle("Orders Summary", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "03_orders_summary.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(output_dir / "04_cost_breakdown.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(output_dir / "05_demand_fulfillment.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(output_dir / "06_shipment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_observations(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plots the normalized local observation features over time for each warehouse.
    Creates one figure per warehouse with a subplot for each feature group,
    each showing one line per SKU plus aggregate features where present.

    Args:
        episode (Dict[str, np.ndarray]): Episode data dict (must include
            ``obs_normalized``, ``n_skus``, and ``max_expected_lead_time``).
        output_dir (Path): Directory to save the plot.
    """
    obs_norm = episode.get("obs_normalized")  # (T, n_warehouses, obs_dim)
    if obs_norm is None:
        return

    n_skus = int(episode.get("n_skus", 0))
    max_lt = int(episode.get("max_expected_lead_time", 0))
    if n_skus <= 0 or max_lt <= 0:
        return

    T, n_warehouses, obs_dim = obs_norm.shape
    timesteps = np.arange(T)

    # Feature groups: (name, n_sku_columns, has_aggregate)
    # Pipeline uses L*S columns instead of S
    feature_groups = [
        ("Inventory",              n_skus,          True),
        ("Pipeline",               max_lt * n_skus, True),
        ("Incoming Demand (Home)", n_skus,          True),
        ("Shipped (Home)",         n_skus,          False),
        ("Shipped (Away)",         n_skus,          True),
        ("Stockout",               n_skus,          False),
        ("Rolling Demand Mean",    n_skus,          True),
        ("Demand Forecast",        n_skus,          True),
    ]

    # local obs = [warehouse_onehot? | features]
    # features = (7 + L) * n_skus + 6
    base_local = (7 + max_lt) * n_skus + 6
    local_dim_full = obs_dim // (1 + n_warehouses)
    warehouse_id_offset = local_dim_full - base_local

    sku_colors = plt.cm.tab10.colors
    n_subplots = len(feature_groups)

    for wh in range(n_warehouses):
        fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 2.5 * n_subplots), sharex=True)

        local_obs = obs_norm[:, wh, warehouse_id_offset:warehouse_id_offset + base_local]

        offset = 0
        for feat_idx, (feat_name, n_cols, has_agg) in enumerate(feature_groups):
            ax = axes[feat_idx]

            if feat_name == "Pipeline":
                for slot in range(max_lt):
                    slot_total = np.zeros(T)
                    for sku in range(n_skus):
                        col = offset + slot * n_skus + sku
                        slot_total += local_obs[:, col]
                    ax.plot(timesteps, slot_total,
                            label=f"Slot t+{slot+1}", linewidth=1.0,
                            color=sku_colors[slot % len(sku_colors)], alpha=0.85)
            else:
                for sku in range(n_cols):
                    ax.plot(timesteps, local_obs[:, offset + sku],
                            label=f"SKU {sku}", linewidth=1.0,
                            color=sku_colors[sku % len(sku_colors)], alpha=0.85)
            offset += n_cols

            if has_agg:
                ax.plot(timesteps, local_obs[:, offset],
                        label="Aggregate", linewidth=1.2, linestyle="--",
                        color="black", alpha=0.7)
                offset += 1

            ax.set_ylabel("Value", fontsize=7)
            ax.set_title(f"{feat_name}", fontsize=9, loc="left")
            ax.legend(fontsize=6, ncol=min(max(n_skus, max_lt) + 1, 8), loc="upper right")
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Timestep")
        fig.suptitle(f"Warehouse {wh} — Local Observations", fontsize=12, y=1.0)
        plt.tight_layout()
        plt.savefig(output_dir / f"10_observations_wh{wh}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_obs_normalization(episode: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Diagnostic plot comparing raw vs. normalized observations to verify that
    the MeanStdFilter is correctly applied during evaluation rollouts.

    Generates two visualizations:
    1. Heatmaps of raw vs. normalized obs over time for warehouse 0.
    2. Per-dimension mean/std summary across the episode for all warehouses,
       showing whether normalized values are centered around 0 with unit variance.

    Args:
        episode (Dict[str, np.ndarray]): Episode data dict.
        output_dir (Path): Directory to save the plot.
    """
    obs_raw = episode.get("obs_raw")              # (T, n_warehouses, obs_dim)
    obs_norm = episode.get("obs_normalized")      # (T, n_warehouses, obs_dim)

    if obs_raw is None or obs_norm is None:
        return

    T, n_warehouses, obs_dim = obs_raw.shape

    # --- Plot 1: Heatmap for warehouse 0 (raw vs normalized) ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    for ax, data, title in [
        (axes[0], obs_raw[:, 0, :], "Warehouse 0 — Raw Observations"),
        (axes[1], obs_norm[:, 0, :], "Warehouse 0 — Normalized Observations"),
    ]:
        im = ax.imshow(data.T, aspect="auto", interpolation="nearest", cmap="RdBu_r")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_ylabel("Obs Dimension")
        ax.set_title(title)

    axes[1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_dir / "08_obs_normalization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Per-dimension mean/std summary for each warehouse ---
    fig, axes = plt.subplots(n_warehouses, 1, figsize=(16, 4 * n_warehouses), squeeze=False)
    dims = np.arange(obs_dim)

    for wh in range(n_warehouses):
        ax = axes[wh, 0]
        raw_mean = obs_raw[:, wh, :].mean(axis=0)
        raw_std = obs_raw[:, wh, :].std(axis=0)
        norm_mean = obs_norm[:, wh, :].mean(axis=0)
        norm_std = obs_norm[:, wh, :].std(axis=0)

        ax.bar(dims - 0.2, raw_mean, width=0.4, color="#c44e52", alpha=0.7, label="Raw mean")
        ax.errorbar(dims - 0.2, raw_mean, yerr=raw_std, fmt="none", ecolor="#c44e52", alpha=0.4, capsize=2)
        ax.bar(dims + 0.2, norm_mean, width=0.4, color="#4c72b0", alpha=0.7, label="Norm mean")
        ax.errorbar(dims + 0.2, norm_mean, yerr=norm_std, fmt="none", ecolor="#4c72b0", alpha=0.4, capsize=2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Value")
        ax.set_title(f"Warehouse {wh} — Per-Dimension Obs Statistics (episode avg ± std)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Observation Dimension Index")
    plt.tight_layout()
    plt.savefig(output_dir / "09_obs_normalization_stats.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(output_dir / "07_episode_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
