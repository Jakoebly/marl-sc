"""
Data loaders for the plotting notebook.

Reads on-disk artifacts produced by a seed-evaluation run (training metrics,
final eval results, per-config summary, and rollout episodes) into plain dicts
suitable for downstream matplotlib code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.experiments.utils.experiment_utils import SEED_EVAL_SUMMARY_FILENAME
from src.experiments.utils.seed_evaluation import match_per_seed_run_dirname


def load_run(config_dir: str | Path) -> Dict[str, Any]:
    """
    Loads all per-seed artifacts for one seed-evaluation config.

    A "config directory" is the folder that contains the per-seed subfolders
    matching ``{name}_Seed{N}``. For single-mode seed eval this is the run's
    ``seed_evaluation/`` directory; for tune mode and baselines it is one of
    the per-config subfolders inside ``seed_evaluation/``.

    Heavy rollout arrays from ``episodes_data.npz`` are not loaded here. Use
    :func:`load_episodes` on a specific seed when you need them.

    Args:
        config_dir (str | Path): Path to the config directory.

    Returns:
        run (Dict[str, Any]): Mapping with keys ``name`` (config dir name),
            ``config_dir`` (Path), ``summary`` (parsed
            ``seed_evaluation_summary.yaml`` if present, else ``None``), and
            ``seeds`` (``{seed_number: {"training_metrics", "final_eval",
            "seed_run_dir"}}`` keyed by integer root seed).
    """

    # Resolve and validate the config directory
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # Load the per-config seed evaluation summary YAML if present
    summary_file_path = config_dir / SEED_EVAL_SUMMARY_FILENAME
    summary: Optional[dict] = None
    if summary_file_path.exists():
        with open(summary_file_path, encoding="utf-8") as f:
            summary = yaml.safe_load(f)

    # Walk per-seed subfolders and load the individual 
    # training metrics and final evaluation results YAML files
    seed_data: Dict[int, Dict[str, Any]] = {}
    for sub in sorted(config_dir.iterdir()):
        # Skip if not a directory
        if not sub.is_dir():
            continue
        # Skip if not a per-seed run directory
        matched = match_per_seed_run_dirname(sub.name)
        if matched is None:
            continue

        # Get the seed number
        _, seed_number = matched

        # Load the training metrics YAML file if present
        training_metrics: list = []
        training_metrics_file_path = sub / "training_metrics.yaml"
        if training_metrics_file_path.exists():
            with open(training_metrics_file_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, list):
                training_metrics = loaded

        # Load the final evaluation results YAML file if present
        final_eval: Optional[dict] = None
        final_eval_file_path = sub / "eval_results_final.yaml"
        if final_eval_file_path.exists():
            with open(final_eval_file_path, encoding="utf-8") as f:
                final_eval = yaml.safe_load(f)

        # Store the training metrics and final evaluation results in the seeds dictionary
        seed_data[seed_number] = {
            "training_metrics": training_metrics,
            "final_eval": final_eval,
            "seed_run_dir": sub,
        }

    # Create the run dictionary
    run = {
        "name": config_dir.name,
        "config_dir": config_dir,
        "summary": summary,
        "seed_data": seed_data,
    }

    return run

def load_episodes(seed_experiment_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Loads ``episodes_data.npz`` from a single ``{name}_Seed{N}`` directory and
    reconstructs the original ``List[Dict[str, np.ndarray]]`` shape returned
    by ``Algorithm.rollout``.

    The npz file is written by :func:`src.experiments.utils.experiment_utils.
    save_episodes` during evaluation when ``visualize=True``. Each list entry 
    has the keys consumed by :mod:`src.visualization.utils.eval_rollout_plots` 
    (``inventory``, ``order_quantities``, ``demand_per_region``, ...).

    Args:
        seed_run_dir (str | Path): Path to a per-seed run directory.

    Returns:
        episodes (List[Dict[str, Any]]): One dict per episode.
    """

    # Resolve and validate the seed experiment directory
    seed_experiment_dir = Path(seed_experiment_dir)
    npz_path = seed_experiment_dir / "episodes_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No episodes_data.npz file in {seed_experiment_dir}. Re-run "
            f"evaluation with --visualize to generate it."
        )

    # Load the episodes data from the npz file
    with np.load(npz_path, allow_pickle=False) as data:
        # Get the number of episodes and the feature config
        n_episodes = int(data["n_episodes"])
        feature_config = (
            json.loads(str(data["feature_config_json"]))
            if "feature_config_json" in data.files else None
        )

        # Get the array and scalar keys
        array_keys = [
            k for k in data.files
            if k not in ("feature_config_json", "n_episodes")
            and data[k].ndim > 0 and data[k].shape[0] == n_episodes
        ]
        scalar_keys = [
            k for k in data.files
            if k not in ("feature_config_json", "n_episodes")
            and k not in array_keys
        ]

        # Stack the arrays and scalars
        stacked = {k: data[k] for k in array_keys}
        scalars = {k: data[k] for k in scalar_keys}

    # Create the original episodes list
    episodes: List[Dict[str, Any]] = []
    for i in range(n_episodes):
        ep: Dict[str, Any] = {k: stacked[k][i] for k in array_keys}
        for k, v in scalars.items():
            ep[k] = v.item() if v.ndim == 0 else v
        if feature_config is not None:
            ep["feature_config"] = feature_config
        episodes.append(ep)

    return episodes
