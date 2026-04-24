"""
Seed-evaluation helpers: orchestration helpers used by ``run_seed_evaluation``,
training-curve plotting, disk-based aggregation, and the
``aggregate_and_plot_seed_evaluation`` entry point used by the parallel SLURM path.

The sequential orchestrator (``run_seed_evaluation`` in ``run_experiment.py``)
calls ``build_seed_evaluation_configs``, ``evaluate_config_across_seeds``, and
``save_combined_seed_evaluation_summary`` directly. The parallel path invokes
``aggregate_and_plot_seed_evaluation`` from the aggregate SLURM phase after all
worker tasks have finished.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import yaml

from src.config.loader import load_algorithm_config, load_environment_config
from src.experiments.runner import ExperimentRunner
from src.experiments.utils.experiment_utils import (
    find_checkpoint_dir,
    save_run_metadata,
)
from src.experiments.utils.wandb import setup_wandb
from src.utils.seed_manager import EXPERIMENT_SEEDS, SeedManager


# ============================================================================
# Seed evaluation orchestration (sequential path)
# ============================================================================

def build_seed_evaluation_configs(
    env_config_path: Optional[str] = None,
    algorithm_config_path: Optional[str] = None,
    storage_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tune_name: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Builds the list of configs to evaluate across seeds.

    In tune mode (``tune_name`` given), reads the top-K trial configs from the
    tune experiment's ``best_trial_results.yaml``. In single mode, wraps the
    explicit env/algorithm configs into a one-entry list.

    Args:
        env_config_path (Optional[str]): Path to environment config (single mode).
        algorithm_config_path (Optional[str]): Path to algorithm config (single mode).
        storage_dir (Optional[str]): Root directory for outputs (single mode).
        experiment_name (Optional[str]): Experiment name (single mode).
        tune_name (Optional[str]): Name of a completed tune experiment (tune mode).
        top_k (int): Number of top trials to evaluate (tune mode).

    Returns:
        configs_to_eval (List[Dict[str, Any]]): Per-config entries with keys
            ``config_name``, ``env_config_path``, ``algorithm_config_path``,
            and ``storage_dir``.
    """

    # If in tune mode, read the top-K trial configs from the tune's best trials
    if tune_name is not None:
        return _build_tune_mode_configs(tune_name, top_k)

    # If in single mode, use the provided env and algorithm configs
    if env_config_path is not None and algorithm_config_path is not None:
        return _build_single_mode_config(
            env_config_path=env_config_path,
            algorithm_config_path=algorithm_config_path,
            storage_dir=storage_dir,
            experiment_name=experiment_name,
        )

    # If neither mode is provided, raise an error
    raise ValueError(
        "Provide either --tune-name (tune mode) or "
        "--env-config + --algorithm-config (single mode)"
    )

def _build_tune_mode_configs(
    tune_name: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Reads the top-K trial configs from a completed tune experiment and produces
    one config entry per trial for seed evaluation.

    Args:
        tune_name (str): Name of a completed tune experiment.
        top_k (int): Number of top trials to evaluate.

    Returns:
        configs_to_eval (List[Dict[str, Any]]): Per-trial config entries.
    """

    # Get the tune directory
    tuning_base = "experiment_outputs/Tuning"
    tune_dir = Path(tuning_base) / tune_name
    if not tune_dir.exists():
        raise FileNotFoundError(f"Tune directory not found: {tune_dir}")

    # Get the top-K trial configs
    results_path = tune_dir / "best_trial_results.yaml"
    if not results_path.exists():
        raise FileNotFoundError(
            f"best_trial_results.yaml not found in {tune_dir}"
        )
    with open(results_path, encoding="utf-8") as f:
        results_data = yaml.safe_load(f)
    available_trials = results_data.get("top_k_trials", [])
    if top_k > len(available_trials):
        print(
            f"[WARN] Requested top_k={top_k} but best_trial_results.yaml "
            f"only contains {len(available_trials)} trials. "
            f"Using all {len(available_trials)} available trials."
        )
    trials = available_trials[:top_k]

    # Create the directory for seed evaluation runs
    seed_eval_base = tune_dir / "seed_evaluation"
    seed_eval_base.mkdir(parents=True, exist_ok=True)

    # Loop over the top-K trials and add their configs to the list of configs
    configs_to_eval: List[Dict[str, Any]] = []
    for trial in trials:
        rank = trial["rank"]
        short_id = trial["short_id"]
        trial_path = trial["trial_path"]
        config_name = f"{rank:02d}_{short_id}"
        config_storage = str(seed_eval_base / config_name)
        env_cfg = str(Path(trial_path) / "env_config.yaml")
        algo_cfg = str(Path(trial_path) / "algorithm_config.yaml")
        configs_to_eval.append({
            "config_name": config_name,
            "env_config_path": env_cfg,
            "algorithm_config_path": algo_cfg,
            "storage_dir": config_storage,
        })

    return configs_to_eval

def _build_single_mode_config(
    env_config_path: str,
    algorithm_config_path: str,
    storage_dir: Optional[str],
    experiment_name: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Wraps the explicit env and algorithm configs into a one-entry config list.

    Args:
        env_config_path (str): Path to environment config.
        algorithm_config_path (str): Path to algorithm config.
        storage_dir (Optional[str]): Root directory for outputs.
        experiment_name (Optional[str]): Experiment name.

    Returns:
        configs_to_eval (List[Dict[str, Any]]): One-entry config list.
    """

    # Create the directory for seed evaluation runs
    if storage_dir is None:
        storage_dir = "./experiment_outputs/Runs"
    if experiment_name is None:
        raise ValueError(
            "--experiment-name is required for single-mode seed evaluation"
        )
    base_dir = str(Path(storage_dir) / experiment_name)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Add the explizit configs to the list of configs to evaluate
    configs_to_eval = [{
        "config_name": experiment_name,
        "env_config_path": env_config_path,
        "algorithm_config_path": algorithm_config_path,
        "storage_dir": base_dir,
    }]

    return configs_to_eval

def evaluate_config_across_seeds(
    cfg: Dict[str, Any],
    n_seeds: int,
    eval_episodes: int,
    eval_seed: int,
    num_iterations: Optional[int] = None,
    wandb_project: Optional[str] = None,
) -> dict:
    """
    Trains ``n_seeds`` seeds for one config, evaluates each with a fixed eval
    seed, computes per-config statistics, and writes the per-config summary +
    plot.

    Args:
        cfg (Dict[str, Any]): Config entry from
            :func:`build_seed_evaluation_configs`, containing ``config_name``,
            ``env_config_path``, ``algorithm_config_path``, and ``storage_dir``.
        n_seeds (int): Number of seeds to train (root seeds will be
            100, 200, ..., N*100).
        eval_episodes (int): Episodes for the final deterministic evaluation.
        eval_seed (int): Fixed root seed used for all final evaluations.
        num_iterations (Optional[int]): Override training iterations
            (``None`` keeps config default).
        wandb_project (Optional[str]): Optional WandB project name.

    Returns:
        stats (dict): Aggregated statistics as returned by
            :func:`compute_seed_statistics`.
    """

    # Lazy import of run_evaluation to avoid a circular import with run_experiment
    from src.experiments.run_experiment import run_evaluation

    # Extract names
    config_name = cfg["config_name"]
    cfg_storage = cfg["storage_dir"]
    cfg_env = cfg["env_config_path"]
    cfg_algo = cfg["algorithm_config_path"]

    # Initialize list to store seed evaluation results
    seed_values: List[tuple] = []

    print("\n" + "=" * 80)
    print(f"SEED EVALUATION: {config_name}")
    print("=" * 80)

    # Loop over each seed
    for seed_idx in range(1, n_seeds + 1):
        # Calculate a spaced-out root seed and set the run name
        root_seed = seed_idx * 100
        seed_name = f"{config_name}_Seed{root_seed}"

        print(f"\n--- Seed {seed_idx}/{n_seeds} (root_seed={root_seed}) ---")

        # Train the model for this seed
        _run_single_seed_eval(
            env_config_path=cfg_env,
            algorithm_config_path=cfg_algo,
            storage_dir=cfg_storage,
            experiment_name=seed_name,
            root_seed=root_seed,
            wandb_project=wandb_project,
            num_iterations=num_iterations,
        )

        # Evaluate with fixed eval seed
        seed_experiment_dir = str(Path(cfg_storage) / seed_name)
        checkpoint_dir = str(
            find_checkpoint_dir(Path(seed_experiment_dir))
        )
        eval_result = run_evaluation(
            checkpoint_dir=checkpoint_dir,
            experiment_dir=seed_experiment_dir,
            eval_episodes=eval_episodes,
            root_seed=eval_seed,
        )
        mean_return = eval_result.get("evaluation", {}).get(
            "episode_return_mean", 0.0,
        )
        seed_values.append((root_seed, mean_return))

    # Compute statistics for this config
    stats = compute_seed_statistics(config_name, seed_values)

    # Save per-config results
    results_dir = Path(cfg_storage) / "seed_evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"configs": [stats]}
    summary_path = results_dir / "seed_evaluation_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, default_flow_style=False, sort_keys=False)

    # Load training metrics from disk and plot curves for this config
    per_config_metrics = load_training_metrics_from_disk(Path(cfg_storage))
    plot_seed_evaluation_curves(per_config_metrics, results_dir)

    return stats

def _run_single_seed_eval(
    env_config_path: str,
    algorithm_config_path: str,
    storage_dir: str,
    experiment_name: str,
    root_seed: int,
    wandb_project: Optional[str] = None,
    num_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs a single training run for seed evaluation.

    Thin wrapper around the standard training path that builds a dedicated
    ``SeedManager``, loads configs, sets up WandB, and invokes
    ``ExperimentRunner.run()``. Training metrics are automatically saved to
    ``training_metrics.yaml`` by the runner and loaded from disk during
    aggregation.

    Args:
        env_config_path (str): Path to environment config.
        algorithm_config_path (str): Path to algorithm config.
        storage_dir (str): Root directory for experiment outputs.
        experiment_name (str): Name for this seed's experiment run.
        root_seed (int): Root seed for reproducibility.
        wandb_project (Optional[str]): WandB project name.
        num_iterations (Optional[int]): Override training iterations
            (``None`` keeps config default).

    Returns:
        Dict[str, Any]: Training result dictionary.
    """

    # Create the single top-level SeedManager for this seed's run
    seed_manager = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)

    # Load configs
    env_config = load_environment_config(env_config_path, seed_manager=seed_manager)
    algorithm_config = load_algorithm_config(algorithm_config_path)

    # Override training iterations if set
    if num_iterations is not None:
        algorithm_config.shared.num_iterations = num_iterations

    # Setup WandB to log metrics
    wandb_config, _ = setup_wandb(
        wandb_project=wandb_project,
        mode="single",
        wandb_name=experiment_name,
    )

    # Create the experiment directory
    experiment_dir = Path(storage_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(experiment_dir)

    # Create the experiment runner
    runner = ExperimentRunner(
        env_config=env_config,
        algorithm_config=algorithm_config,
        seed_manager=seed_manager,
        checkpoint_dir=checkpoint_dir,
        wandb_config=wandb_config,
    )

    # Save run metadata
    save_run_metadata(
        output_dir=checkpoint_dir,
        runner=runner,
        root_seed=root_seed,
    )

    # Run the experiment
    result = runner.run()

    return result

def save_combined_seed_evaluation_summary(
    tune_name: str,
    all_config_stats: List[dict],
) -> None:
    """
    Writes the combined ``seed_evaluation_summary.yaml`` across all tune configs
    and plots the combined training curves.

    Used only in tune mode when multiple configs have been evaluated.

    Args:
        tune_name (str): Name of the completed tune experiment.
        all_config_stats (List[dict]): Per-config stats dicts, sorted best-first.
    """

    # Create the overall directory for the seed evaluation results
    overall_dir = Path("experiment_outputs/Tuning") / tune_name / "seed_evaluation" / "seed_evaluation_results"
    overall_dir.mkdir(parents=True, exist_ok=True)

    # Create and save the overall summary
    overall_summary = {
        "configs": all_config_stats,
        "best_config": {
            "name": all_config_stats[0]["name"],
            "mean": all_config_stats[0]["mean"],
        },
    }
    overall_path = overall_dir / "seed_evaluation_summary.yaml"
    with open(overall_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(overall_summary, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Overall summary saved to: {overall_path}")

    # Load training metrics from all seed eval subfolders and plot combined curves
    seed_eval_base = Path("experiment_outputs/Tuning") / tune_name / "seed_evaluation"
    all_curve_metrics: Dict[str, List[List[dict]]] = {}
    for sub in sorted(seed_eval_base.iterdir()):
        if sub.is_dir() and sub.name != "seed_evaluation_results":
            sub_metrics = load_training_metrics_from_disk(sub)
            all_curve_metrics.update(sub_metrics)
    plot_seed_evaluation_curves(all_curve_metrics, overall_dir)


# ============================================================================
# Aggregation orchestration (parallel SLURM path)
# ============================================================================

def aggregate_and_plot_seed_evaluation(
    mode: str,
    name: str,
    tuning_base: str = "experiment_outputs/Tuning",
    runs_base: str = "experiment_outputs/Runs",
) -> None:
    """
    Locates the seed-evaluation directory, runs the disk-based aggregation
    (summary table + YAML), and generates the training-curve plots.

    Functions as the entry point for the aggregate phase of the parallel SLURM
    seed evaluation.

    Args:
        mode (str): ``"tune"`` or ``"single"``.
        name (str): Tune experiment name or single-run experiment name.
        tuning_base (str): Base directory for tuning experiments.
        runs_base (str): Base directory for single-run experiments.
    """

    from src.experiments.utils.experiment_utils import find_experiment_dir

    # If in tune mode, locate the tune directory and dispatch
    if mode == "tune":
        tune_dir = Path(tuning_base) / name
        if not tune_dir.exists():
            raise FileNotFoundError(f"Tune directory not found: {tune_dir}")
        _aggregate_and_plot_tune(tune_dir)

    # If in single mode, locate the experiment directory and dispatch
    elif mode == "single":
        experiment_dir = find_experiment_dir(runs_base, name)
        _aggregate_and_plot_single(experiment_dir)

    else:
        raise ValueError(f"Unknown seed-eval mode: {mode!r} (expected 'tune' or 'single')")

def _aggregate_and_plot_tune(tune_dir: Path) -> None:
    """
    Aggregates a tune-mode seed evaluation directory by completing per-config 
    aggregation and plotting as well as top-level combined plotting  across all 
    configs.

    Args:
        tune_dir (Path): Path to the tune experiment directory.
    """

    # Get directories
    seed_eval_dir = tune_dir / "seed_evaluation"

    # Loop over each config subfolder and aggregate seed evaluation results
    for config_dir in sorted(seed_eval_dir.iterdir()):
        # Skip if not a directory
        if not config_dir.is_dir():
            continue

        # Aggregate the seed evaluation results
        results_dir = config_dir / "seed_evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        try:
            aggregate_seed_evaluation(config_dir)
        except ValueError as e:
            print(f"[WARN] Skipping {config_dir.name}: {e}")
            continue

        # Load metrics and plot curves
        metrics = load_training_metrics_from_disk(config_dir)
        if metrics:
            plot_seed_evaluation_curves(metrics, results_dir)

    # Aggregate across all configs at the top level
    all_metrics: Dict[str, List[List[dict]]] = {}
    for config_dir in sorted(seed_eval_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_metrics = load_training_metrics_from_disk(config_dir)
        all_metrics.update(config_metrics)
    if all_metrics:
        top_results_dir = seed_eval_dir / "seed_evaluation_results"
        top_results_dir.mkdir(parents=True, exist_ok=True)
        plot_seed_evaluation_curves(all_metrics, top_results_dir)

def _aggregate_and_plot_single(experiment_dir: Path) -> None:
    """
    Aggregates a single-mode seed evaluation directory.

    Args:
        experiment_dir (Path): Path to the single-run experiment directory.
    """

    # Get directories
    seed_eval_dir = experiment_dir / "seed_evaluation"
    if not seed_eval_dir.exists():
        raise FileNotFoundError(
            f"No seed_evaluation/ directory in {experiment_dir}"
        )
    results_dir = seed_eval_dir / "seed_evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate the seed evaluation results
    aggregate_seed_evaluation(seed_eval_dir)

    # Load metrics and plot curves
    metrics = load_training_metrics_from_disk(seed_eval_dir)
    if metrics:
        plot_seed_evaluation_curves(metrics, results_dir)

def aggregate_seed_evaluation(seed_eval_dir: str | Path) -> dict:
    """
    Scans a ``seed_evaluation/`` directory, groups runs by config name,
    and computes per-config statistics (mean, std, 95% CI).

    Expects subdirectory names matching ``{name}_Seed{N}`` (or
    ``{name}_seed{N}``).  Each subdirectory must contain an
    ``eval_results_*.yaml`` with an ``episode_return_mean`` key.

    Writes ``seed_evaluation_summary.yaml`` into ``seed_eval_dir`` and
    returns the summary as a dict.

    Args:
        seed_eval_dir (str | Path): Path to the ``seed_evaluation/`` directory.

    Returns:
        summary (dict): Aggregated results with per-config statistics.
    """

    # Get the seed evaluation directory and check if it exists
    seed_eval_dir = Path(seed_eval_dir)
    if not seed_eval_dir.exists():
        raise FileNotFoundError(
            f"Seed evaluation directory not found: {seed_eval_dir}"
        )

    # Scan the subdirectories and collect (seed, reward) pairs per config
    groups = _collect_seed_evaluation_groups(seed_eval_dir)

    # Check if any valid seed evaluation results were found
    if not groups:
        raise ValueError(
            f"No valid seed evaluation results found in {seed_eval_dir}"
        )

    # Compute per-config statistics and sort best-first
    configs = [
        compute_seed_statistics(name, seeds)
        for name, seeds in sorted(groups.items())
    ]
    configs.sort(key=lambda c: c["mean"], reverse=True)

    # Create and save the summary
    summary = {
        "configs": configs,
        "best_config": {
            "name": configs[0]["name"],
            "mean": float(configs[0]["mean"]),
        },
    }
    summary_path = seed_eval_dir / "seed_evaluation_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, default_flow_style=False, sort_keys=False)

    # Print the summary table
    print_seed_evaluation_table(configs, summary_path)

    return summary

def _collect_seed_evaluation_groups(
    seed_eval_dir: Path,
) -> dict[str, list[tuple[int, float]]]:
    """
    Scans ``{name}_Seed{N}`` subdirectories for ``eval_results_best.yaml`` and
    groups ``(seed_number, reward)`` pairs by config name.

    Args:
        seed_eval_dir (Path): Path to the ``seed_evaluation/`` directory.

    Returns:
        groups (dict[str, list[tuple[int, float]]]): Mapping
            ``{config_name: [(seed_number, reward), ...]}``.
    """

    # Compile regex pattern and initialize groups
    pattern = re.compile(r"^(.+)_[Ss]eed(\d+)$")
    groups: dict[str, list[tuple[int, float]]] = {}

    # Iterate over the subdirectories in the seed evaluation directory
    for subdir in sorted(seed_eval_dir.iterdir()):
        # Check if the subdirectory is a directory
        if not subdir.is_dir():
            continue
        # Check if the subdirectory name matches the pattern and skip if not
        match = pattern.match(subdir.name)
        if not match:
            continue

        # Extract the config name and seed number from the subdirectory name
        config_name = match.group(1)
        seed_number = int(match.group(2))

        # Get the evaluation results file and check if it exists
        eval_file = subdir / "eval_results_best.yaml"
        if not eval_file.exists():
            print(f"[WARN] No eval_results_best.yaml in {subdir.name}, skipping")
            continue
        with open(eval_file, encoding="utf-8") as f:
            eval_data = yaml.safe_load(f)

        reward = eval_data.get("episode_return_mean")
        if reward is None:
            print(
                f"[WARN] No episode_return_mean in {eval_file.name} "
                f"({subdir.name}), skipping"
            )
            continue

        # Add the reward to its corresponding config group
        groups.setdefault(config_name, []).append((seed_number, float(reward)))

    return groups

def find_missing_seed_evaluation_tasks(
    mode: str,
    name: str,
    n_seeds: int,
    top_k: int = 10,
    tuning_base: str = "experiment_outputs/Tuning",
    runs_base: str = "experiment_outputs/Runs",
) -> List[int]:
    """
    Returns the SLURM array task IDs whose ``eval_results_best.yaml`` is
    missing or unreadable, so the parallel aggregate phase can self-heal by
    re-submitting only the failed/unfinished (config, seed) pairs.

    The array mapping matches :file:`scripts/run_seed_evaluation.sh`::

        TASK_ID    = config_idx * n_seeds + (seed_idx - 1)
        CONFIG_IDX = TASK_ID // n_seeds
        SEED_IDX   = TASK_ID % n_seeds + 1
        ROOT_SEED  = SEED_IDX * 100

    A task is considered complete when its expected run directory contains
    an ``eval_results_best.yaml`` with a numeric ``episode_return_mean``.

    Args:
        mode (str): ``"tune"`` or ``"single"``.
        name (str): Tune experiment name (tune mode) or single-run name.
        n_seeds (int): Number of seeds per config.
        top_k (int): Number of top trials expected (tune mode only).
        tuning_base (str): Base directory for tuning experiments.
        runs_base (str): Base directory for single-run experiments.

    Returns:
        missing_task_ids (List[int]): Sorted SLURM array indices still pending.
    """
    from src.experiments.utils.experiment_utils import find_experiment_dir

    # Build the list of (config_idx, config_name, config_storage_dir) triples
    # in the same order that the shell launcher walks them.
    configs: List[Tuple[int, str, Path]] = []

    # If in tune mode, build the config list by using the top-k 
    # trials from the best_trial_results.yaml file
    if mode == "tune":
        tune_dir = Path(tuning_base) / name
        if not tune_dir.exists():
            raise FileNotFoundError(f"Tune directory not found: {tune_dir}")
        results_path = tune_dir / "best_trial_results.yaml"
        if not results_path.exists():
            raise FileNotFoundError(
                f"best_trial_results.yaml not found in {tune_dir}"
            )
        with open(results_path, encoding="utf-8") as f:
            results_data = yaml.safe_load(f)
        trials = results_data.get("top_k_trials", [])[:top_k]
        seed_eval_base = tune_dir / "seed_evaluation"
        for config_idx, trial in enumerate(trials):
            rank = trial["rank"]
            short_id = trial["short_id"]
            config_name = f"{rank:02d}_{short_id}"
            config_storage = seed_eval_base / config_name
            configs.append((config_idx, config_name, config_storage))

    # If in single mode, build the config list by using the experiment directory
    elif mode == "single":
        experiment_dir = find_experiment_dir(runs_base, name)
        config_storage = experiment_dir / "seed_evaluation"
        configs.append((0, name, config_storage))

    # If mode is not tune or single, raise an error
    else:
        raise ValueError(
            f"Unknown seed-eval mode: {mode!r} (expected 'tune' or 'single')"
        )

    # Walk every expected (config_idx, seed_idx) and check for completed output
    missing: List[int] = []
    for config_idx, config_name, config_storage in configs:
        for seed_idx in range(1, n_seeds + 1):
            root_seed = seed_idx * 100
            seed_dir = config_storage / f"{config_name}_Seed{root_seed}"
            eval_file = seed_dir / "eval_results_best.yaml"
            task_id = config_idx * n_seeds + (seed_idx - 1)
            if not eval_file.exists():
                missing.append(task_id)
                continue
            try:
                with open(eval_file, encoding="utf-8") as f:
                    eval_data = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                missing.append(task_id)
                continue
            if not isinstance(eval_data.get("episode_return_mean"), (int, float)):
                missing.append(task_id)

    return sorted(missing)


# ============================================================================
# Disk-based metrics loading
# ============================================================================

def load_training_metrics_from_disk(
    seed_eval_dir: Path,
) -> Dict[str, List[List[dict]]]:
    """
    Reads ``training_metrics.yaml`` from every ``{name}_Seed{N}`` subfolder
    under *seed_eval_dir* and groups them by config name.

    Args:
        seed_eval_dir (Path): Directory containing per-seed run folders.

    Returns:
        groups (Dict[str, List[List[dict]]]): Mapping ``{config_name: 
            [seed_1_metrics, seed_2_metrics, ...]}``.
    """
    
    # Compile regex pattern to match the config name and seed number and initialize groups
    pattern = re.compile(r"^(.+)_[Ss]eed(\d+)$")
    groups: Dict[str, List[List[dict]]] = {}

    # Loop over all subdirectories in seed_eval_dir
    for subdir in sorted(seed_eval_dir.iterdir()):
        # Skip if not a directory
        if not subdir.is_dir():
            continue

        # Match the subdirectory name to the config name and seed number
        match = pattern.match(subdir.name)
        if not match:
            continue
        config_name = match.group(1)

        # Load the metrics from the training_metrics.yaml file and store them
        metrics_file = subdir / "training_metrics.yaml"
        if not metrics_file.exists():
            continue
        with open(metrics_file, encoding="utf-8") as f:
            metrics = yaml.safe_load(f)
        if isinstance(metrics, list):
            groups.setdefault(config_name, []).append(metrics)

    return groups


# ============================================================================
# Statistics and reporting helpers
# ============================================================================

def _compute_curve_stats_across_seeds(
    seed_runs: List[List[dict]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregates one metric across seeds into mean/std curves over training iterations.
    Missing values (``None`` or absent key) become NaN and are ignored in the mean/std.

    Args:
        seed_runs (List[List[dict]]): Per-seed metrics lists.
        key (str): Key to compute the mean/std for.

    Returns:
        iterations (np.ndarray): Array containing the training iterations.
        means (np.ndarray): Array containing the mean values for each iteration.
        stds (np.ndarray): Array containing the standard deviations for each iteration.
    """

    # If there are no seed runs, return empty arrays
    if not seed_runs:
        return np.array([]), np.array([]), np.array([])

    # Get the maximum length of the per-seed metrics lists
    max_len = max(len(run) for run in seed_runs)
    if max_len == 0:
        return np.array([]), np.array([]), np.array([])

    # Get iterations from the first seed run (equal for all seed runs)
    iterations = np.array([m["iteration"] for m in seed_runs[0]])

    # Extract the values for the key for each seed run into a list
    values_per_seed = []
    for run in seed_runs:
        vals = np.full(max_len, np.nan)
        for i, m in enumerate(run):
            v = m.get(key)
            if v is not None:
                vals[i] = v
        values_per_seed.append(vals)

    # Stack the values per seed into a matrix 
    matrix = np.stack(values_per_seed, axis=0)  # Shape: (n_seeds, n_iterations)

    # Drop iterations where every seed is missing a value for the key
    valid_mask = np.any(~np.isnan(matrix), axis=0)
    iterations = iterations[valid_mask]
    matrix = matrix[:, valid_mask]

    # Compute the mean and std for each iteration
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(means)

    return iterations, means, stds

def compute_seed_statistics(
    name: str,
    seed_values: list[tuple[int, float]],
    confidence: float = 0.95,
) -> dict:
    """
    Computes mean, std, and confidence interval for a set of seed evaluation results.

    Args:
        name (str): Config / group name.
        seed_values (list[tuple[int, float]]): ``(seed_number, metric_value)`` pairs.
        confidence (float): Confidence level for the CI (default 0.95).

    Returns:
        stats (dict): Dictionary with keys ``name``, ``mean``, ``std``,
            ``ci_95``, ``n_seeds``, and ``per_seed``.
    """

    # Sort the seed values by seed number and get the mean and std values
    seed_values_sorted = sorted(seed_values, key=lambda x: x[0])
    values = np.array([v for _, v in seed_values_sorted])
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0

    # Compute the confidence interval if there are multiple seeds
    if n > 1:
        se = std / np.sqrt(n)
        alpha = 1.0 - confidence
        t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2, df=n - 1))
        ci_low = float(mean - t_crit * se)
        ci_high = float(mean + t_crit * se)
    else:
        ci_low = ci_high = float(mean)

    # Create the stats dictionary (plain float/int so yaml.safe_dump stays readable)
    stats = {
        "name": name,
        "mean": float(round(mean, 4)),
        "std": float(round(std, 4)),
        "ci_95": [float(round(ci_low, 4)), float(round(ci_high, 4))],
        "n_seeds": n,
        "per_seed": [
            {"seed": s, "value": float(round(float(v), 4))}
            for s, v in seed_values_sorted
        ],
    }

    return stats

def print_seed_evaluation_table(
    configs: list[dict],
    summary_path: Optional[Path] = None,
) -> None:
    """
    Prints a ranked summary table of seed evaluation results.

    Args:
        configs (list[dict]): Config statistics dicts as returned by
            :func:`compute_seed_statistics`, sorted best-first.
        summary_path (Optional[Path]): If given, prints the save location.
    """

    # Print results
    print(f"\n{'=' * 70}")
    print(f"Seed Evaluation Summary ({len(configs)} config(s))")
    print(f"{'=' * 70}")
    print(
        f"{'Rank':<6}{'Name':<35}{'Mean':>10}{'Std':>10}"
        f"{'95% CI':>20}{'N':>5}"
    )
    print(f"{'-' * 86}")
    for i, c in enumerate(configs):
        ci_str = f"[{c['ci_95'][0]:.2f}, {c['ci_95'][1]:.2f}]"
        print(
            f"{i + 1:<6}{c['name']:<35}{c['mean']:>10.4f}"
            f"{c['std']:>10.4f}{ci_str:>20}{c['n_seeds']:>5}"
        )
    print(f"{'=' * 86}")
    print(f"Best config: {configs[0]['name']} (mean={configs[0]['mean']:.4f})")
    if summary_path:
        print(f"Summary saved to: {summary_path}")


# ============================================================================
# Plotting helpers
# ============================================================================

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

    # Set the colors for the plots
    colors = plt.cm.tab10.colors

    # Plot mean +/- std curve for each config
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (config_name, seed_runs) in enumerate(sorted(all_metrics.items())):
        iterations, means, stds = _compute_curve_stats_across_seeds(seed_runs, key=metric_key)
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

