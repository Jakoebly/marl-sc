"""
Seed-evaluation helpers: orchestration helpers used by ``run_seed_evaluation``,
training-curve plotting, disk-based aggregation, and the
``aggregate_and_plot_seed_evaluation`` entry point used by the parallel SLURM path.

The sequential orchestrator (``run_seed_evaluation`` in ``run_experiment.py``)
calls ``build_seed_evaluation_configs``, ``evaluate_config_across_seeds``,
``save_combined_seed_evaluation_summary``, and ``plot_combined_training_curves`` directly.
The parallel path invokes aggregate_and_plot_seed_evaluation`` from the aggregate SLURM
phase after all worker tasks have finished.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
import yaml

from src.config.loader import load_algorithm_config, load_environment_config
from src.experiments.runner import ExperimentRunner
from src.experiments.utils.experiment_utils import (
    DEFAULT_RUN_STORAGE_DIR,
    DEFAULT_TUNE_STORAGE_DIR,
    SEED_EVAL_PLOTS_SUBDIR,
    SEED_EVAL_SUBDIR,
    SEED_EVAL_SUMMARY_FILENAME,
    TUNE_BEST_TRIALS_FILENAME,
    find_checkpoint_dir,
    find_experiment_dir,
    find_latest_periodic_checkpoint,
    find_missing_seed_eval_files,
    save_run_metadata,
)
from src.experiments.utils.wandb import setup_wandb
from src.utils.seed_manager import EXPERIMENT_SEEDS, SeedManager
from src.visualization.utils.seed_eval_plots import (
    plot_baseline_curves,
    plot_combined_baseline_curves,
    plot_combined_training_curves,
    plot_seed_evaluation_curves,
)


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
    tune experiment's ``best_trial_results.yaml``. In single mode, wraps a
    single (env, algorithm) config into a one-entry list.

    Single-mode config resolution priority:
        1. Explicit ``env_config_path`` + ``algorithm_config_path`` (fresh
           seed eval; no prior training run required).
        2. Saved ``env_config.yaml`` + ``algorithm_config.yaml`` under
           ``storage_dir/experiment_name/`` (existing-run seed eval).

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

    # If in single mode, use the env and algorithm configs (either provided or resolved)
    if experiment_name is not None:
        return _build_single_mode_config(
            env_config_path=env_config_path,
            algorithm_config_path=algorithm_config_path,
            storage_dir=storage_dir,
            experiment_name=experiment_name,
        )

    # If neither mode is provided, raise an error
    raise ValueError(
        "Provide either --tune-name (tune mode) or "
        "--experiment-name with optional --env-config + --algorithm-config (single mode)"
    )

def _build_tune_mode_configs(
    name: str,
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
    storage_dir = DEFAULT_TUNE_STORAGE_DIR
    experiment_dir = find_experiment_dir(storage_dir, name)

    # Check if the tune results file exists
    tune_results_file = experiment_dir / TUNE_BEST_TRIALS_FILENAME
    if not tune_results_file.exists():
        raise FileNotFoundError(
            f"{TUNE_BEST_TRIALS_FILENAME} not found in {experiment_dir}"
        )
    
    # Open the tune results file to get the top-k trials
    with open(tune_results_file, encoding="utf-8") as f:
        results_data = yaml.safe_load(f)
    trials = results_data.get("top_k_trials", [])

    # If the requested top-k is greater than the number of trials, print a warning
    if top_k > len(trials):
        print(
            f"[WARN] Requested top_k={top_k} but {TUNE_BEST_TRIALS_FILENAME} "
            f"only contains {len(trials)} trials. "
            f"Using all {len(trials)} available trials."
        )
    
    # Get the top-k trials
    trials = trials[:top_k]

    # Create the directory for seed evaluation runs
    seed_dir = experiment_dir / SEED_EVAL_SUBDIR
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the top-K trials and add their configs to the list of configs
    configs_to_eval: List[Dict[str, Any]] = []
    for trial in trials:
        rank = trial["rank"]
        short_id = trial["short_id"]
        trial_path = trial["trial_path"]
        config_name = f"{rank:02d}_{short_id}"
        config_storage = str(seed_dir / config_name)
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
    env_config_path: Optional[str],
    algorithm_config_path: Optional[str],
    storage_dir: Optional[str],
    experiment_name: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Wraps a single (env, algorithm) config pair into a one-entry config list.

    Two sub-cases:
        - New-config sub-case: explicit ``env_config_path`` and
          ``algorithm_config_path`` provided; the run directory is the
          literal ``storage_dir/experiment_name/`` (created on demand).
        - Existing-run sub-case: configs omitted; the run directory is
          resolved via ``find_experiment_dir`` (supports prefix match), and
          env/algorithm configs are loaded from the saved YAMLs inside that
          resolved directory.

    Args:
        env_config_path (Optional[str]): Path to environment config; if
            ``None``, resolved from the existing run directory.
        algorithm_config_path (Optional[str]): Path to algorithm config; if
            ``None``, resolved from the existing run directory.
        storage_dir (Optional[str]): Root directory for outputs (defaults to
            :data:`DEFAULT_RUN_STORAGE_DIR`).
        experiment_name (Optional[str]): Experiment name.

    Returns:
        configs_to_eval (List[Dict[str, Any]]): One-entry config list.
    """

    # Safeguard against missing storage_dir or experiment_name
    if storage_dir is None:
        storage_dir = DEFAULT_RUN_STORAGE_DIR
    if experiment_name is None:
        raise ValueError(
            "--experiment-name is required for single-mode seed evaluation"
        )

    # Resolve the experiment directory
    # If explicit env and algorithm configs are provided, create a new run directory
    if env_config_path is not None and algorithm_config_path is not None:
        experiment_dir = Path(storage_dir) / experiment_name
    
    # If no explicit env and algorithm configs are provided, find 
    # the experiment directory and check for existing env and algorithm configs
    else:
        experiment_dir = find_experiment_dir(storage_dir, experiment_name)
        if env_config_path is None:
            candidate = experiment_dir / "env_config.yaml"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"No env_config.yaml found at {candidate}. Either pass "
                    f"--env-config explicitly or point --experiment-name at "
                    f"an existing run that has a saved env_config.yaml."
                )
            env_config_path = str(candidate)
        if algorithm_config_path is None:
            candidate = experiment_dir / "algorithm_config.yaml"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"No algorithm_config.yaml found at {candidate}. Either "
                    f"pass --algorithm-config explicitly or point "
                    f"--experiment-name at an existing run that has a saved "
                    f"algorithm_config.yaml."
                )
            algorithm_config_path = str(candidate)

    # Create the directory for seed evaluation runs
    seed_dir = str(experiment_dir / SEED_EVAL_SUBDIR)
    Path(seed_dir).mkdir(parents=True, exist_ok=True)

    # Wrap into the standard one-entry config list
    configs_to_eval = [{
        "config_name": experiment_name,
        "env_config_path": env_config_path,
        "algorithm_config_path": algorithm_config_path,
        "storage_dir": seed_dir,
    }]

    return configs_to_eval

def evaluate_config_across_seeds(
    cfg: Dict[str, Any],
    n_seeds: int,
    eval_episodes: int,
    eval_seed: int,
    num_iterations: Optional[int] = None,
    eval_interval: Optional[int] = None,
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
        eval_episodes (int): Episodes for both intra-training evaluation
            (overrides ``algorithm.shared.num_eval_episodes``) and the final
            deterministic evaluation.
        eval_seed (int): Fixed root seed used for all final evaluations.
        num_iterations (Optional[int]): Override training iterations
            (``None`` keeps config default).
        eval_interval (Optional[int]): Override intra-training eval interval
            via ``algorithm.shared.eval_interval`` (``None`` keeps config default).
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
        seed_experiment_name = f"{config_name}_Seed{root_seed}"
        seed_experiment_dir = Path(cfg_storage) / seed_experiment_name

        # Skip the seed if a valid eval_results_final.yaml is already on disk 
        # or if the file is empty
        eval_results_file = seed_experiment_dir / "eval_results_final.yaml"
        if eval_results_file.exists():
            try:
                with open(eval_results_file, encoding="utf-8") as f:
                    prev_eval = yaml.safe_load(f) or {}
                prev_mean = prev_eval.get("episode_return_mean")
            except (OSError, yaml.YAMLError):
                prev_mean = None
            if isinstance(prev_mean, (int, float)):
                print(
                    f"\n--- Seed {seed_idx}/{n_seeds} (root_seed={root_seed}): "
                    f"already complete, skipping ---"
                )
                seed_values.append((root_seed, float(prev_mean)))
                continue

        print(f"\n--- Seed {seed_idx}/{n_seeds} (root_seed={root_seed}) ---")

        # Train the model for the given seed
        _run_single_seed_eval(
            env_config_path=cfg_env,
            algorithm_config_path=cfg_algo,
            storage_dir=cfg_storage,
            experiment_name=seed_experiment_name,
            root_seed=root_seed,
            eval_seed=eval_seed,
            wandb_project=wandb_project,
            num_iterations=num_iterations,
            eval_interval=eval_interval,
            num_eval_episodes=eval_episodes,
        )

        # Evaluate with fixed eval seed
        checkpoint_dir = str(
            find_checkpoint_dir(seed_experiment_dir)
        )
        eval_result = run_evaluation(
            checkpoint_dir=checkpoint_dir,
            experiment_dir=str(seed_experiment_dir),
            eval_episodes=eval_episodes,
            root_seed=eval_seed,
        )
        mean_return = eval_result.get("evaluation", {}).get(
            "episode_return_mean", 0.0,
        )
        seed_values.append((root_seed, mean_return))

    # Compute statistics for this config
    stats = compute_seed_statistics(config_name, seed_values)

    # Save per-config summary
    summary = {
        "configs": [stats],
        "best_config": {
            "name": stats["name"],
            "mean": float(stats["mean"]),
        },
    }
    summary_path = Path(cfg_storage) / SEED_EVAL_SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, default_flow_style=False, sort_keys=False)

    # Load training metrics from disk and plot curves for this config
    plots_dir = Path(cfg_storage) / SEED_EVAL_PLOTS_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_config_metrics = load_training_metrics_by_config(Path(cfg_storage))
    plot_seed_evaluation_curves(per_config_metrics, plots_dir)

    return stats

def _run_single_seed_eval(
    env_config_path: str,
    algorithm_config_path: str,
    storage_dir: str,
    experiment_name: str,
    root_seed: int,
    eval_seed: Optional[int] = None,
    wandb_project: Optional[str] = None,
    num_iterations: Optional[int] = None,
    eval_interval: Optional[int] = None,
    num_eval_episodes: Optional[int] = None,
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
        eval_seed (Optional[int]): Fixed eval root seed forwarded to
            ``ExperimentRunner`` as ``eval_seed_override``. When set, the
            PPO intra-training eval uses the same eval distribution across
            all seeds and matches the post-training benchmark.
        wandb_project (Optional[str]): WandB project name.
        num_iterations (Optional[int]): Override training iterations
            (``None`` keeps config default).
        eval_interval (Optional[int]): Override intra-training eval interval
            via ``algorithm.shared.eval_interval`` (``None`` keeps config default).
        num_eval_episodes (Optional[int]): Override intra-training eval episode
            count via ``algorithm.shared.num_eval_episodes`` (``None`` keeps
            config default). Used to keep intra-training and final-eval
            episode counts in sync for low-noise plot points.

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

    # Override intra-training eval interval / episodes if set
    if eval_interval is not None:
        algorithm_config.shared.eval_interval = eval_interval
    if num_eval_episodes is not None:
        algorithm_config.shared.num_eval_episodes = num_eval_episodes

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

    # Check if there is a periodic checkpoint to resume from 
    # because the job was killed before.
    resume_from = find_latest_periodic_checkpoint(experiment_dir)
    if resume_from:
        print(f"[RESUME] Found existing periodic checkpoint: {resume_from}")

    # Create the experiment runner
    runner = ExperimentRunner(
        env_config=env_config,
        algorithm_config=algorithm_config,
        seed_manager=seed_manager,
        checkpoint_dir=checkpoint_dir,
        wandb_config=wandb_config,
        eval_seed_override=eval_seed,
    )

    # Save run metadata
    save_run_metadata(
        output_dir=checkpoint_dir,
        runner=runner,
        root_seed=root_seed,
    )

    # Run the experiment
    result = runner.run(resume_from=resume_from)

    return result

# ============================================================================
# Aggregation orchestration (parallel SLURM path)
# ============================================================================

def aggregate_and_plot_seed_evaluation(
    mode: str,
    name: str,
    storage_dir: Optional[str] = None,
) -> None:
    """
    Locates the seed evaluation directory, runs the disk-based aggregation
    (summary table + YAML), and generates the seed-evaluation plots.

    Functions as the entry point for the aggregate phase of the parallel SLURM
    seed evaluation.

    Args:
        mode (str): ``"tune"``, ``"single"``, or ``"baseline"``.
        name (str): Tune / single-run / baseline experiment name.
        storage_dir (Optional[str]): Base directory where the folder with ``name`` lives.
    """

    # If in tune mode, locate the experiment directory and dispatch
    if mode == "tune":
        if storage_dir is None:
            storage_dir = DEFAULT_TUNE_STORAGE_DIR
        experiment_dir = find_experiment_dir(storage_dir, name)
        _aggregate_and_plot_tune(experiment_dir)

    # If in single mode, locate the experiment directory and dispatch
    elif mode == "single":
        if storage_dir is None:
            storage_dir = DEFAULT_RUN_STORAGE_DIR
        experiment_dir = find_experiment_dir(storage_dir, name)
        _aggregate_and_plot_single(experiment_dir)

    # If in baseline mode, locate the experiment directory and dispatch
    elif mode == "baseline":
        if storage_dir is None:
            storage_dir = DEFAULT_RUN_STORAGE_DIR
        experiment_dir = find_experiment_dir(storage_dir, name)
        _aggregate_and_plot_baseline(experiment_dir)

    # If mode is not single, tune, or baseline, raise an error
    else:
        raise ValueError(
            f"Unknown seed-eval mode: {mode!r} "
            f"(expected 'tune', 'single', or 'baseline')"
        )

def _aggregate_and_plot_tune(experiment_dir: Path) -> None:
    """
    Aggregates a tune-mode seed evaluation directory by completing per-config 
    aggregation/plotting as well as top-level combined plotting across all 
    configs.

    Args:
        experiment_dir (Path): Path to the tune experiment directory.
    """

    # Get the seed evaluation directory
    seed_dir = experiment_dir / SEED_EVAL_SUBDIR
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"No {SEED_EVAL_SUBDIR}/ directory in {experiment_dir}"
        )

    # Loop over each trainable config subfolder and aggregate seed evaluation results
    all_config_stats: List[dict] = []
    for config_dir in sorted(seed_dir.iterdir()):
        # Skip if not a directory or if the plots directory
        if not config_dir.is_dir() or config_dir.name == SEED_EVAL_PLOTS_SUBDIR:
            continue

        # Aggregate the seed evaluation results for the current trainable config
        try:
            cfg_summary = aggregate_seed_evaluation(config_dir)
        except ValueError as e:
            print(f"[WARN] Skipping {config_dir.name}: {e}")
            continue
        all_config_stats.extend(cfg_summary.get("configs", []))

        # Load metrics and plot curves
        seed_plots_dir = config_dir / SEED_EVAL_PLOTS_SUBDIR
        seed_plots_dir.mkdir(parents=True, exist_ok=True)
        training_metrics = load_training_metrics_by_config(config_dir)
        if training_metrics:
            plot_seed_evaluation_curves(training_metrics, seed_plots_dir)

    # Aggregate across all trainable configs at the top level
    all_config_stats.sort(key=lambda c: c["mean"], reverse=True)
    save_combined_seed_evaluation_summary(seed_dir, all_config_stats)
    plot_combined_training_curves(seed_dir)

def _aggregate_and_plot_baseline(experiment_dir: Path) -> None:
    """
    Aggregates a baseline seed evaluation directory by computing per-baseline
    statistics, writing the combined summary, and plotting each baseline as a
    horizontal mean line with a +/- 1 std band (no training curves since
    baselines do not train).

    Args:
        experiment_dir (Path): Path to the baseline experiment directory.
    """

    # Get the seed evaluation directory
    seed_dir = experiment_dir / SEED_EVAL_SUBDIR
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"No {SEED_EVAL_SUBDIR}/ directory in {experiment_dir}"
        )

    # Loop over each baseline subfolder and aggregate seed evaluation results
    all_config_stats: List[dict] = []
    for config_dir in sorted(seed_dir.iterdir()):
        # Skip if not a directory or if the plots directory
        if not config_dir.is_dir() or config_dir.name == SEED_EVAL_PLOTS_SUBDIR:
            continue

        # Aggregate the seed evaluation results for the current baseline
        try:
            cfg_summary = aggregate_seed_evaluation(config_dir)
        except ValueError as e:
            print(f"[WARN] Skipping {config_dir.name}: {e}")
            continue
        all_config_stats.extend(cfg_summary.get("configs", []))

        # Plot the baseline as a horizontal mean line + std band
        seed_plots_dir = config_dir / SEED_EVAL_PLOTS_SUBDIR
        seed_plots_dir.mkdir(parents=True, exist_ok=True)
        config_stats = cfg_summary.get("configs", [])
        if config_stats:
            plot_baseline_curves(
                config_stats, output_path=seed_plots_dir / "seed_eval_curves.png",
            )

    # Aggregate across all baselines at the top level
    all_config_stats.sort(key=lambda c: c["mean"], reverse=True)
    save_combined_seed_evaluation_summary(seed_dir, all_config_stats)
    plot_combined_baseline_curves(all_config_stats, seed_dir)

def _aggregate_and_plot_single(experiment_dir: Path) -> None:
    """
    Aggregates and plots a single-mode seed evaluation directory.

    Args:
        experiment_dir (Path): Path to the single-run experiment directory.
    """

    # Get directories
    seed_dir = experiment_dir / SEED_EVAL_SUBDIR
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"No {SEED_EVAL_SUBDIR}/ directory in {experiment_dir}"
        )
    seed_plots_dir = seed_dir / SEED_EVAL_PLOTS_SUBDIR
    seed_plots_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate the seed evaluation results
    aggregate_seed_evaluation(seed_dir)

    # Load metrics and plot curves
    metrics = load_training_metrics_by_config(seed_dir)
    if metrics:
        plot_seed_evaluation_curves(metrics, seed_plots_dir)

def aggregate_seed_evaluation(seed_dir: str | Path) -> dict:
    """
    Groups seed evaluation runs by config name and computes per-config 
    statistics (mean, std, 95% CI).

    Expects subdirectory names matching ``{name}_Seed{N}`` (or
    ``{name}_seed{N}``).  Each subdirectory must contain an
    ``eval_results_final.yaml`` with an ``episode_return_mean`` key.

    Args:
        seed_dir (str | Path): Path to the directory containing 
            the seed evaluation experiment runs.

    Returns:
        summary (dict): Aggregated results with per-config statistics.
    """

    # Get the seed evaluation directory and check if it exists
    seed_dir = Path(seed_dir)
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Seed evaluation directory not found: {seed_dir}"
        )

    # Scan the subdirectories and collect (seed, reward) pairs per config
    seed_groups = _load_eval_returns_by_config(seed_dir)

    # Check if any valid seed evaluation results were found
    if not seed_groups:
        raise ValueError(
            f"No valid seed evaluation results found in {seed_dir}"
        )

    if not seed_groups:
        raise ValueError(
            f"No valid seed evaluation results found in {seed_dir}"
        )
    # Compute per-config statistics and sort best-first
    configs = [
        compute_seed_statistics(config_name, seed_values)
        for config_name, seed_values in sorted(seed_groups.items())
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
    summary_file = seed_dir / SEED_EVAL_SUMMARY_FILENAME
    with open(summary_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, default_flow_style=False, sort_keys=False)

    # Print the summary table
    print_seed_evaluation_table(configs, summary_file)

    return summary

def _load_eval_returns_by_config(
    seed_dir: Path,
) -> dict[str, list[tuple[int, float]]]:
    """
    Scans ``seed_experiment_dir/`` for subdirectories matching ``{name}_Seed{N}`` and
    containing an ``eval_results_final.yaml`` file. Groups ``(seed_number, reward)`` 
    pairs by config name.

    Args:
        seed_dir (Path): Path to the directory containing per-seed run folders.

    Returns:
        seed_groups (dict[str, list[tuple[int, float]]]): Mapping
            ``{config_name: [(seed_number, reward), ...]}``.
    """

    # Iterate over the subdirectories in the seed evaluation directory
    seed_groups: dict[str, list[tuple[int, float]]] = {}
    for subdir in sorted(seed_dir.iterdir()):
        # Skip if not a directory
        if not subdir.is_dir():
            continue

        # Match the subdirectory name and skip if it does not match the convention
        matched = match_per_seed_run_dirname(subdir.name)
        if matched is None:
            continue
        config_name, seed_number = matched
        
        # Load the evaluation results file and check if it exists
        eval_results_file = subdir / "eval_results_final.yaml"
        if not eval_results_file.exists():
            print(f"[WARN] No eval_results_final.yaml in {subdir.name}, skipping")
            continue
        with open(eval_results_file, encoding="utf-8") as f:
            eval_results = yaml.safe_load(f)

        # Extract the episode return mean from the 
        # evaluation results and check if it exists
        episode_return_mean = eval_results.get("episode_return_mean")
        if episode_return_mean is None:
            print(
                f"[WARN] No episode_return_mean in {eval_results_file.name} "
                f"({subdir.name}), skipping"
            )
            continue

        # Add the reward to its corresponding config group
        seed_groups.setdefault(config_name, []).append((seed_number, float(episode_return_mean)))

    return seed_groups

def find_missing_seed_evaluation_tasks(
    mode: str,
    name: str,
    n_seeds: int,
    top_k: int = 10,
    storage_dir: Optional[str] = None,
) -> List[int]:
    """
    Returns the SLURM array task IDs whose ``eval_results_final.yaml`` is
    missing or unreadable.

    Used for RL experiments only (tune and single modes). The baseline counterpart
    is ``find_missing_baseline_tasks`` in ``run_baselines.py``.

    Args:
        mode (str): ``"tune"`` or ``"single"``.
        name (str): Tune experiment name (tune mode) or single-run name.
        n_seeds (int): Number of seeds per config.
        top_k (int): Number of top trials expected (tune mode only).
        storage_dir (Optional[str]): Base directory for experiments.

    Returns:
        missing (List[int]): Sorted SLURM array indices still pending.
    """

    # Initialize a list to store (config_idx, config_name, config_storage) triples
    configs: List[Tuple[int, str, Path]] = []

    # If in tune mode, build the config list by using the top-k
    # trials from the best-trial-results file
    if mode == "tune":
        # Resolve directories
        if storage_dir is None: 
            storage_dir = DEFAULT_TUNE_STORAGE_DIR
        experiment_dir = find_experiment_dir(storage_dir, name)
        seed_dir = experiment_dir / SEED_EVAL_SUBDIR

        # Check if the tune results file exists
        tune_results_file = experiment_dir / TUNE_BEST_TRIALS_FILENAME
        if not tune_results_file.exists():
            raise FileNotFoundError(
                f"{TUNE_BEST_TRIALS_FILENAME} not found in {experiment_dir}"
            )

        # Open the tune results file to get the top-k trials
        with open(tune_results_file, encoding="utf-8") as f:
            results_data = yaml.safe_load(f)
        trials = results_data.get("top_k_trials", [])[:top_k]

        # Collect all existing configs
        for config_idx, trial in enumerate(trials):
            rank = trial["rank"]
            short_id = trial["short_id"]
            config_name = f"{rank:02d}_{short_id}"
            config_storage = seed_dir / config_name
            configs.append((config_idx, config_name, config_storage))

    # If in single mode, use the config based on the given name
    elif mode == "single":
        # Resolve directories
        if storage_dir is None: 
            storage_dir = DEFAULT_RUN_STORAGE_DIR
        experiment_dir = find_experiment_dir(storage_dir, name)
        seed_dir = experiment_dir / SEED_EVAL_SUBDIR

        # Collect the single config
        configs.append((0, name, seed_dir))

    # If mode is not tune or single, raise an error
    else:
        raise ValueError(
            f"Unknown seed-eval mode: {mode!r} (expected 'tune' or 'single')"
        )

    # Check file-existence for each (config_idx, config_name, config_storage) tuple
    missing = find_missing_seed_eval_files(configs, n_seeds)
    
    return missing


# ============================================================================
# Loading and naming helpers
# ============================================================================

def load_training_metrics_by_config(
    seed_dir: Path,
) -> Dict[str, List[List[dict]]]:
    """
    Reads ``training_metrics.yaml`` from every ``{name}_Seed{N}`` subfolder
    under ``seed_dir`` and groups them by config name.

    Args:
        seed_dir (Path): Directory containing per-seed run folders.

    Returns:
        seed_groups (Dict[str, List[List[dict]]]): Mapping ``{config_name: 
            [seed_1_metrics, seed_2_metrics, ...]}``.
    """

    # Loop over all subdirectories in seed_eval_dir
    seed_groups: Dict[str, List[List[dict]]] = {}
    for subdir in sorted(seed_dir.iterdir()):
        # Skip if not a directory
        if not subdir.is_dir():
            continue

        # Match the subdirectory name and skip if it does not match the convention
        matched = match_per_seed_run_dirname (subdir.name)
        if matched is None:
            continue
        config_name, _ = matched

        # Load the metrics from the training_metrics.yaml file and store them
        training_metrics_file = subdir / "training_metrics.yaml"
        if not training_metrics_file.exists():
            continue
        with open(training_metrics_file, encoding="utf-8") as f:
            metrics = yaml.safe_load(f)
        if isinstance(metrics, list):
            seed_groups.setdefault(config_name, []).append(metrics)

    return seed_groups

def match_per_seed_run_dirname (name: str) -> Optional[Tuple[str, int]]:
    """
    Matches a per-seed run directory name of the form ``{config_name}_Seed{N}``
    (or ``{config_name}_seed{N}``) and returns ``(config_name, seed_number)``,
    or ``None`` if the name does not match.

    Args:
        name (str): Directory name to parse.

    Returns:
        matched (Optional[Tuple[str, int]]): ``(config_name, seed_number)`` on a
            successful match, otherwise ``None``.
    """

    # Match the per-seed run directory naming convention
    match = re.match(r"^(.+)_[Ss]eed(\d+)$", name)
    if not match:
        return None

    # Get the matched config name and seed number
    matched = match.group(1), int(match.group(2))

    return matched


# ============================================================================
# Statistics and reporting helpers
# ============================================================================

def compute_curve_stats_across_seeds(
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

    # Get the maximum length of the per-seed metrics lists and check if it is zero
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
    config_name: str,
    seed_values: list[tuple[int, float]],
    confidence: float = 0.95,
) -> dict:
    """
    Computes mean, std, and confidence interval for a set of seed evaluation results.

    Args:
        config_name (str): Config name.
        seed_values (list[tuple[int, float]]): ``(seed_number, metric_value)`` pairs.
        confidence (float): Confidence level for the CI (default 0.95).

    Returns:
        stats (dict): Dictionary with keys ``name``, ``mean``, ``std``,
            ``ci_95``, ``n_seeds``, and ``per_seed``.
    """

    # Sort the seed values by seed number and get the mean and std values
    seed_values_sorted = sorted(seed_values, key=lambda x: x[0])
    reward_values = np.array([v for _, v in seed_values_sorted])
    n = len(reward_values)
    mean = float(reward_values.mean())
    std = float(reward_values.std(ddof=1)) if n > 1 else 0.0

    # Compute the confidence interval if there are multiple seeds
    if n > 1:
        se = std / np.sqrt(n)
        alpha = 1.0 - confidence
        t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2, df=n - 1))
        ci_low = float(mean - t_crit * se)
        ci_high = float(mean + t_crit * se)
    else:
        ci_low = ci_high = float(mean)

    # Create the stats dictionary
    stats = {
        "name": config_name,
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
    summary_file: Optional[Path] = None,
) -> None:
    """
    Prints a ranked summary table of seed evaluation results to the console.

    Args:
        configs (list[dict]): Config statistics dicts as returned by
            :func:`compute_seed_statistics`, sorted best-first.
        summary_file (Optional[Path]): If given, prints the save location.
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
    if summary_file:
        print(f"Summary saved to: {summary_file}")

def save_combined_seed_evaluation_summary(
    seed_dir: Path,
    all_config_stats: List[dict],
) -> None:
    """
    Writes the combined ``seed_evaluation_summary.yaml`` across all configs.

    Args:
        seed_dir (Path): Path to the ``seed_evaluation/`` directory that
            contains the per-config subfolders.
        all_config_stats (List[dict]): Per-config stats dicts, sorted best-first.
    """

    # Skip if no configs were aggregated
    if not all_config_stats:
        return

    # Create the overall directory for the seed evaluation results
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Create and save the overall summary
    overall_summary = {
        "configs": all_config_stats,
        "best_config": {
            "name": all_config_stats[0]["name"],
            "mean": float(all_config_stats[0]["mean"]),
        },
    }
    eval_summary_file = seed_dir / SEED_EVAL_SUMMARY_FILENAME
    with open(eval_summary_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(overall_summary, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Overall summary saved to: {eval_summary_file}")

