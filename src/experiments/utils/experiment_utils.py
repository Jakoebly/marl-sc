"""
Shared helpers for experiment file I/O and directory resolution.

Provides utilities for generating experiment names, saving/loading run
metadata, locating experiment and checkpoint directories, and resolving
saved config files.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import stats as scipy_stats
import yaml

from src.config.loader import (
    load_algorithm_config,
    load_environment_config,
)

if TYPE_CHECKING:
    from src.config.schema import EnvironmentConfig
    from src.experiments.runner import ExperimentRunner


# ============================================================================
# Directory and Loading Helpers
# ============================================================================

def load_root_seed_from_metadata(experiment_dir: Path) -> Optional[int]:
    """
    Reads the ``root_seed`` from a previously saved ``metadata.json``.

    Args:
        experiment_dir (Path): Experiment directory containing ``metadata.json``.

    Returns:
        root_seed (Optional[int]): The stored root seed, or ``None`` if not found.
    """

    # Get the metadata file path
    meta_path = experiment_dir / "metadata.json"

    # If the metadata file does not exist, return None
    if not meta_path.exists():
        return None

    # Load the metadata from the file
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Get the root seed
    root_seed = meta.get("root_seed")

    return root_seed

def resolve_saved_config(
    experiment_dir: Path,
    filename: str,
    explicit_path: Optional[str] = None,
) -> str:
    """
    Resolves a saved config file path. 
    
    Returns ``explicit_path`` if provided, otherwise resolves
    ``experiment_dir / filename`` and raises if the file is missing.

    Args:
        experiment_dir (Path): Directory containing saved config files.
        filename (str): Config filename (e.g. ``"env_config.yaml"``).
        explicit_path (Optional[str]): Caller-supplied path that takes priority.

    Returns:
        resolved_config_path (str): Resolved config file path.
    """

    # Return early if an explicit path is provided
    if explicit_path is not None:
        return explicit_path

    # Resolve the config file path
    resolved_config_path = str(experiment_dir / filename)
    if not Path(resolved_config_path).exists():
        raise FileNotFoundError(
            f"No saved config found at: {resolved_config_path}\n"
            f"Provide the config path explicitly."
        )
    print(f"[INFO] Loading saved config from: {resolved_config_path}")

    return resolved_config_path

def find_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Searches recursively for a directory matching ``experiment_name``
    under ``base_dir``.  Supports exact matches as well as prefix matching
    so that tune trial directories can be found with a short prefix
    (e.g. ``trainable_da92fc08``) instead of the full long name.

    Args:
        base_dir (str): Root directory to search under.
        experiment_name (str): Exact directory name or a unique prefix.

    Returns:
        experiment_dir (Path): Resolved experiment directory.
    """

    # Get the base directory and check if it exists
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Base directory '{base_dir}' does not exist."
        )

    # Search for directories with exact matching first
    matches = [
        p for p in base.rglob(experiment_name)
        if p.is_dir() and p.name == experiment_name
    ]

    # If no exact matches are found, fall back to prefix match
    if len(matches) == 0:
        matches = [
            p for p in base.rglob(f"{experiment_name}*")
            if p.is_dir() and p.name.startswith(experiment_name)
        ]

    # If no exact or prefix matches are found, raise an error
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Experiment '{experiment_name}' not found under '{base_dir}'."
        )
    
    # If multiple matches are found, raise an error
    if len(matches) > 1:
        paths_str = "\n  ".join(str(m) for m in matches)
        raise ValueError(
            f"Multiple directories matching '{experiment_name}' found under '{base_dir}':\n  {paths_str}\n"
            f"Please provide a more specific prefix or the full name."
        )

    return matches[0]

def find_checkpoint_dir(
    experiment_dir: Path,
    checkpoint_number: Optional[int] = None,
) -> Path:
    """
    Resolves the checkpoint directory under an experiment run folder.

    If ``checkpoint_number`` is given, uses ``checkpoint_<n>`` or zero-padded Ray Tune
    naming when the unpadded folder is missing. Otherwise prefers ``checkpoint_best``,
    then ``checkpoint_final``, then the last sorted ``checkpoint_*`` directory, or
    ``checkpoint_final`` as a fallback name (may not exist yet).

    Args:
        experiment_dir (Path): Path to the experiment directory.
        checkpoint_number (Optional[int]): Specific checkpoint number to resolve.

    Returns:
        checkpoint_dir (Path): Resolved checkpoint directory.
    """

    # If checkpoint number is provided, use it to find the checkpoint directory
    if checkpoint_number is not None:
        checkpoint_folder = f"checkpoint_{checkpoint_number}"
        if not (experiment_dir / checkpoint_folder).is_dir():
            padded = f"checkpoint_{int(checkpoint_number):06d}"
            if (experiment_dir / padded).is_dir():
                checkpoint_folder = padded
    
    # If no checkpoint number is provided, use the best checkpoint directory first
    elif (experiment_dir / "checkpoint_best").is_dir():
        checkpoint_folder = "checkpoint_best"
    
    # If no best checkpoint directory is found, use the final checkpoint directory
    elif (experiment_dir / "checkpoint_final").is_dir():
        checkpoint_folder = "checkpoint_final"
    
    # If no best or final checkpoint directory is found, use the last sorted checkpoint
    # directory
    else:
        tune_chkpts = sorted(
            p for p in experiment_dir.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint_")
        )
        if tune_chkpts:
            checkpoint_folder = tune_chkpts[-1].name
        else:
            checkpoint_folder = "checkpoint_final"

    # Assemble the checkpoint directory 
    checkpoint_dir = experiment_dir / checkpoint_folder

    return checkpoint_dir

def find_experiment_dir_from_checkpoint(checkpoint_dir: str) -> Path:
    """
    Walks up from a checkpoint directory to find the experiment directory,
    identified by the presence of ``env_config.yaml`` (saved during training).
    Falls back to the immediate parent if no config file is found.

    Args:
        checkpoint_dir (str): Path to a checkpoint directory.

    Returns:
        experiment_dir (Path): Resolved experiment directory.
    """
    
    # Get the given checkpoint directory
    current = Path(checkpoint_dir).resolve()

    # Walk up the directory tree until the experiment directory is found
    for parent in current.parents:
        if (parent / "env_config.yaml").exists():
            return parent
        if parent == parent.parent:
            break

    # Assemble the experiment dir
    experiment_dir = Path(checkpoint_dir).parent

    return experiment_dir

def checkpoint_suffix(checkpoint_dir: str) -> str:
    """
    Derives a short label from a checkpoint directory name. 
    Examples: 
        - ``checkpoint_best`` -> ``best``
        - ``checkpoint_final`` -> ``final``
        - ``checkpoint_42`` -> ``chkpt42``
        - ``checkpoint_42`` -> ``chkpt42``

    Args:
        checkpoint_dir (str): Path to a checkpoint directory.

    Returns:
        suffix (str): Short suffix string.
    """
    
    # Get the checkpoint name
    checkpoint_name = Path(checkpoint_dir).name

    # Extract the suffix from the checkpoint name
    if checkpoint_name in ("checkpoint_final", "checkpoint_best"):
        suffix = checkpoint_name.replace("checkpoint_", "")
        return suffix
    if checkpoint_name.startswith("checkpoint_"):
        suffix = f"chkpt{checkpoint_name.replace('checkpoint_', '')}"
        return suffix
    return checkpoint_name


# ============================================================================
# Experiment Name Helpers
# ============================================================================

def generate_experiment_name(
    env_config_path: str,
    algorithm_config_path: str,
    mode: str = "single",
    search_type: Optional[str] = None,
    scheduler_type: Optional[str] = None,
) -> str:
    """
    Generates a default experiment name from config metadata and a timestamp.

    Args:
        env_config_path (str): Path to environment config.
        algorithm_config_path (str): Path to algorithm config.
        mode (str): Experiment mode (``"single"`` or ``"tune"``).
        search_type (Optional[str]): Search algorithm type (e.g. ``"optuna"``).
        scheduler_type (Optional[str]): Scheduler type (e.g. ``"asha"``).

    Returns:
        experiment_name (str): Generated experiment name.
    """

    # Get algorithm name from config
    try:
        algorithm_config = load_algorithm_config(algorithm_config_path)
        algo_name = algorithm_config.name
    except Exception:
        # Fallback to config filename if loading fails
        algo_name = Path(algorithm_config_path).stem

    env_config = load_environment_config(env_config_path)
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Build name components
    name_parts = [algo_name.upper(), mode, f"{n_warehouses}WH", f"{n_skus}SKU"]
    if mode == "tune":
        if search_type and search_type != "random":
            name_parts.append(search_type)
        if scheduler_type and scheduler_type != "fifo":
            name_parts.append(scheduler_type)
    name_parts.append(timestamp)
    
    # Join with underscores
    experiment_name = "_".join(name_parts)
    
    return experiment_name

def generate_baseline_experiment_name(env_config: "EnvironmentConfig") -> str:
    """
    Generates a experiment name for baseline runs.

    Args:
        env_config (EnvironmentConfig): Environment configuration.

    Returns:
        str: Folder name in the form ``BASELINE_<W>WH_<S>SKU_<timestamp>``.
    """

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Build the experiment name
    experiment_name = [
        f"BASELINE_{env_config.n_warehouses}WH_{env_config.n_skus}SKU_{timestamp}"
        ]

    return experiment_name


# ============================================================================
# Output Helpers
# ============================================================================

def save_run_metadata(
    output_dir: str,
    runner: ExperimentRunner,
    ray_trial_id: Optional[str] = None,
    root_seed: Optional[int] = None,
) -> None:
    """
    Writes a one-time ``metadata.json`` at the start of a run or trial.

    Args:
        output_dir (str): Directory to write the metadata file into.
        runner (ExperimentRunner): Experiment runner (provides RLlib config).
        ray_trial_id (Optional[str]): Ray Tune trial ID, or ``None`` for single runs.
        root_seed (Optional[int]): Root seed used for reproducibility.
    """

    # Set the filename for the metadata file
    METADATA_FILENAME = "metadata.json"

    # Create the path to the metadata file and check if it already exists
    meta_path = Path(output_dir) / METADATA_FILENAME
    if meta_path.exists():
        return

    # Get the trainer from the runner instance
    trainer = runner.algorithm.trainer

    # Create the metadata dictionary
    meta = {
        "ray_trial_id": ray_trial_id or "single",
        "root_seed": root_seed,
        "config": trainer.config.to_dict(),
    }

    # Create the directory if it does not exist
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the metadata to the file
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[INFO] Saved run metadata to: {meta_path}")

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
        se = std / math.sqrt(n)
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


def aggregate_seed_evaluation(seed_eval_dir: str | Path) -> dict:
    """
    Scans a ``seed_evaluation/`` directory, groups runs by config name,
    and computes per-config statistics (mean, std, 95% CI).

    Expects subdirectory names matching ``{name}_seed{N}``.  Each
    subdirectory must contain an ``eval_results_*.yaml`` with an
    ``episode_return_mean`` key.

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

    # Compile regex pattern and initialize groups
    pattern = re.compile(r"^(.+)_seed(\d+)$")
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

    # Check if any valid seed evaluation results were found
    if not groups:
        raise ValueError(
            f"No valid seed evaluation results found in {seed_eval_dir}"
        )

    configs = [
        compute_seed_statistics(name, seeds)
        for name, seeds in sorted(groups.items())
    ]
    configs.sort(key=lambda c: c["mean"], reverse=True)

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

    print_seed_evaluation_table(configs, summary_path)

    return summary


def save_env_config(env_config: EnvironmentConfig, experiment_dir: Path) -> None:
    """
    Saves an ``EnvironmentConfig`` to ``env_config.yaml`` in
    ``experiment_dir``. Skips writing if the file already exists.

    Args:
        env_config (EnvironmentConfig): Environment configuration.
        experiment_dir (Path): Experiment output directory.
    """

    # Set the filename for the environment config file
    env_config_path = experiment_dir / "env_config.yaml"

    # Save the environment config if the file does not exist
    if not env_config_path.exists():
        with open(env_config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"environment": env_config.model_dump()},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"[INFO] Saved environment config to: {env_config_path}")