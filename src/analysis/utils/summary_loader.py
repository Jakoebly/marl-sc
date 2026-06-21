"""Loader for seed-evaluation summary files used by the analysis notebook.

Reads the per-seed values of a single config out of a combined
``seed_evaluation_summary.yaml`` (or a renamed copy of one, as assembled under
``experiment_outputs/analysis_outputs/``). Loading is kept separate from the
paired statistics in :mod:`src.analysis.utils.paired_stats`, mirroring how
:mod:`src.visualization.utils.plot_data_loader` separates I/O from the plotting
code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml


def load_seed_values(
    summary_path: Path,
    config_name: Optional[str] = None,
) -> Dict[int, float]:
    """
    Loads the per-seed values of one config from a seed-evaluation summary.

    Reads the combined ``seed_evaluation_summary.yaml`` and returns the
    ``{seed: value}`` mapping for a single config. With ``config_name`` omitted
    the top-ranked config (``configs[0]``) is used; otherwise the config whose
    name matches exactly is returned.

    Args:
        summary_path (Path): Path to a seed-evaluation summary YAML.
        config_name (Optional[str]): Exact config name to select; defaults to
            the top-ranked config.

    Returns:
        seed_values (Dict[int, float]): ``{seed_number: metric_value}`` pairs.
    """

    # Resolve and validate the summary path
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Seed-evaluation summary not found: {summary_path}")

    # Read the combined summary written by the seed-evaluation pipeline
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = yaml.safe_load(f)

    # Guard against an empty or malformed summary before indexing configs[0]
    configs = summary.get("configs") if isinstance(summary, dict) else None
    if not configs:
        raise KeyError(f"No 'configs' entries in summary: {summary_path}")

    # Select the top-ranked config unless an explicit name is requested
    if config_name is None:
        entry = configs[0]
    else:
        entry = next((c for c in configs if c["name"] == config_name), None)
        if entry is None:
            raise KeyError(f"Config '{config_name}' not found in {summary_path}")

    # Get the seed values from the entry
    seed_values = {int(p["seed"]): float(p["value"]) for p in entry["per_seed"]}

    return seed_values
