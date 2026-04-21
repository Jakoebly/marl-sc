"""
Shared helpers for experiment file I/O and directory resolution.

Provides utilities for generating experiment names, saving/loading run
metadata, locating experiment and checkpoint directories, and resolving
saved config files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from src.config.loader import (
    load_algorithm_config,
    load_environment_config,
)

if TYPE_CHECKING:
    from src.config.schema import AlgorithmConfig, EnvironmentConfig
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

def save_algorithm_config(
    algorithm_config: "AlgorithmConfig", experiment_dir: Path
) -> None:
    """
    Saves an ``AlgorithmConfig`` to ``algorithm_config.yaml`` in
    ``experiment_dir``. Skips writing if the file already exists.

    Args:
        algorithm_config (AlgorithmConfig): Algorithm configuration.
        experiment_dir (Path): Experiment output directory.
    """

    # Set the filename for the algorithm config file
    algo_config_path = experiment_dir / "algorithm_config.yaml"

    # Save the algorithm config if the file does not exist
    if not algo_config_path.exists():
        with open(algo_config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"algorithm": algorithm_config.model_dump()},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"[INFO] Saved algorithm config to: {algo_config_path}")