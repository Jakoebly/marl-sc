"""
Pure plotting helpers for seed-evaluation training curves and baseline bands.

These functions take already-loaded metrics or per-config stats (no file I/O
beyond the final ``savefig``) and are shared by both the in-run aggregators
in :mod:`src.experiments.utils.seed_evaluation` and the thesis-plotting
notebook in :mod:`src.visualization`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.experiments.utils.experiment_utils import SEED_EVAL_PLOTS_SUBDIR


def plot_seed_evaluation_curves(
    all_metrics: Dict[str, List[List[dict]]],
    output_dir: Path,
) -> None:
    """
    Plots mean +/- std eval-return and train-return curves across seeds for
    each config.

    For every config, at each training iteration the return is averaged over
    seeds and drawn as a solid line with a shaded +/- 1 std band. One curve
    per config per figure.

    Args:
        all_metrics (Dict[str, List[List[dict]]]): ``{config_name:
            [seed_1_metrics, seed_2_metrics, ...]}`` where each
            ``seed_metrics`` is a list of dicts with keys
            ``iteration``, ``train_return``, and ``eval_return``.
        output_dir (Path): Directory to save the plots.
    """

    # Convert to Path and create directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot eval return during training
    _plot_metric_curves(
        all_metrics,
        metric_key="eval_return",
        ylabel="Eval Return (mean +/- std)",
        title="Seed Evaluation — Eval Return During Training",
        output_path=output_dir / "seed_eval_curves.png",
    )

    # Plot train return during training
    _plot_metric_curves(
        all_metrics,
        metric_key="train_return",
        ylabel="Train Return (mean +/- std)",
        title="Seed Evaluation — Train Return During Training",
        output_path=output_dir / "seed_train_curves.png",
    )

    print(f"[INFO] Seed evaluation plots saved to: {output_dir}")

def _plot_metric_curves(
    all_metrics: Dict[str, List[List[dict]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plots mean +/- std curves for a single metric across configs.

    Args:
        all_metrics (Dict[str, List[List[dict]]]): Mapping of config name to
            per-seed metrics lists.
        metric_key (str): Key inside each metrics dict to plot
            (e.g. ``"eval_return"``).
        ylabel (str): Y-axis label.
        title (str): Figure title.
        output_path (Path): Path to save the plot.
    """

    # Lazy import to avoid a circular dependency with seed_evaluation
    from src.experiments.utils.seed_evaluation import compute_curve_stats_across_seeds

    # Set the colors for the plots
    colors = plt.cm.tab10.colors

    # Plot mean +/- std curve for each config
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (config_name, seed_runs) in enumerate(sorted(all_metrics.items())):
        iterations, means, stds = compute_curve_stats_across_seeds(seed_runs, key=metric_key)
        if len(iterations) == 0:
            continue
        color = colors[idx % len(colors)]
        ax.plot(iterations, means, label=config_name, linewidth=1.5, color=color)
        ax.fill_between(
            iterations, means - stds, means + stds, alpha=0.2, color=color,
        )
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_baseline_curves(
    config_stats: List[dict],
    output_path: Path,
) -> None:
    """
    Plots each baseline as a horizontal mean line with a +/- 1 std band,
    color-cycled to match :func:`_plot_metric_curves`.

    Args:
        config_stats (List[dict]): Per-baseline stats dicts as returned by
            :func:`compute_seed_statistics`.
        output_path (Path): Path to save the plot.
    """

    # Set the colors for the plots
    colors = plt.cm.tab10.colors

    # Plot horizontal mean line + std band for each baseline
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, cfg in enumerate(sorted(config_stats, key=lambda c: c["name"])):
        color = colors[idx % len(colors)]
        mean = cfg["mean"]
        std = cfg["std"]
        ax.axhline(mean, label=cfg["name"], linewidth=1.5, color=color)
        ax.axhspan(mean - std, mean + std, alpha=0.2, color=color)
    ax.set_ylabel("Eval Return (mean +/- std)")
    ax.set_title("Seed Evaluation — Baseline Eval Return")
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_combined_training_curves(seed_dir: Path) -> None:
    """
    Loads ``training_metrics.yaml`` from every per-config subfolder under
    ``seed_dir`` and plots combined training curves across all configs.

    Used in tune mode after :func:`save_combined_seed_evaluation_summary`.

    Args:
        seed_dir (Path): Path to the ``seed_evaluation/`` directory that
            contains the per-config subfolders.
    """

    # Lazy import to avoid a circular dependency with seed_evaluation
    from src.experiments.utils.seed_evaluation import load_training_metrics_by_config

    # Load training metrics from all per-config subfolders
    seed_dir = Path(seed_dir)
    all_curve_metrics: Dict[str, List[List[dict]]] = {}
    for sub in sorted(seed_dir.iterdir()):
        if sub.is_dir() and sub.name != SEED_EVAL_PLOTS_SUBDIR:
            sub_metrics = load_training_metrics_by_config(sub)
            all_curve_metrics.update(sub_metrics)

    # Plot combined curves if any metrics were loaded
    if all_curve_metrics:
        seed_plots_dir = seed_dir / SEED_EVAL_PLOTS_SUBDIR
        seed_plots_dir.mkdir(parents=True, exist_ok=True)
        plot_seed_evaluation_curves(all_curve_metrics, seed_plots_dir)

def plot_combined_baseline_curves(all_metrics: List[dict], seed_dir: Path) -> None:
    """
    Plots combined baseline curves across all configs using the per-config stats.

    Args:
        all_metrics (List[dict]): List of per-baseline statistics dicts.
        seed_dir (Path): Path to the ``seed_evaluation/`` directory that
            contains the per-config subfolders.
    """

    # Create the seed evaluation plots directory and plot the baseline curves
    if all_metrics:
        seed_plots_dir = seed_dir / SEED_EVAL_PLOTS_SUBDIR
        seed_plots_dir.mkdir(parents=True, exist_ok=True)
        plot_baseline_curves(
            all_metrics,
            output_path=seed_plots_dir / "seed_eval_curves.png",
        )
        print(f"[INFO] Seed evaluation plots saved to: {seed_plots_dir}")
