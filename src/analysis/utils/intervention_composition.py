"""Combined-intervention composition for the heterogeneous environments.

Decides, per environment, which single interventions to carry into the combined
run. The union of all interventions is the default; an intervention is dropped
only when it both meaningfully and significantly harms the reference policy,
judged by the seed-paired difference in normalized score against the retuned
shared-parameter MAPPO baseline. Failing either test keeps the intervention, so
the rule is deliberately conservative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from src.analysis.utils.paired_stats import (
    anchor_mean,
    normalized_scores,
    paired_difference,
)
from src.analysis.utils.summary_loader import load_seed_values


# Summaries in an environment directory that are not single interventions
REFERENCE_FILENAME = "baseline_mappo_ps.yaml"
BASELINES_FILENAME = "baselines.yaml"
MATCHED_CONTROL_MARKER = "matched"

# Heuristic references reduced to the normalized-score anchors
LOWER_ANCHOR_NAME = "Random"
UPPER_ANCHOR_NAME = "BSOptimized"

# Margin below which a paired loss counts as meaningful (normalized score)
DEFAULT_DROP_THRESHOLD = -0.02


def discover_intervention_files(env_dir: Path) -> Dict[str, Path]:
    """
    Finds the single-intervention summaries in an environment directory.

    Treats every ``*.yaml`` as an intervention except the shared baselines, the
    reference policy, and any matched-capacity control. The label is the file
    stem, taken verbatim from the dropped data.

    Args:
        env_dir (Path): Directory holding one environment's summaries.

    Returns:
        files (Dict[str, Path]): ``{label: summary_path}`` per intervention,
            ordered by label.
    """

    # Get the env dir and define which files to skip
    env_dir = Path(env_dir)
    skip = {REFERENCE_FILENAME, BASELINES_FILENAME}

    # Keep only the intervention summaries and drop references and controls
    files: Dict[str, Path] = {}
    for path in sorted(env_dir.glob("*.yaml")):
        if path.name in skip or MATCHED_CONTROL_MARKER in path.stem:
            continue
        files[path.stem] = path
    return files


def load_anchors(env_dir: Path) -> Dict[str, float]:
    """
    Reduces the lower and upper reference policies to scalar anchors.

    Args:
        env_dir (Path): Directory holding ``baselines.yaml``.

    Returns:
        anchors (Dict[str, float]): Keys ``lower`` and ``upper`` with the mean
            value of the Random and BS-Optimized references.
    """
    
    # Get the baselines path and load the anchor values
    baselines_path = Path(env_dir) / BASELINES_FILENAME
    anchor_values = {
        "lower": anchor_mean(load_seed_values(baselines_path, LOWER_ANCHOR_NAME)),
        "upper": anchor_mean(load_seed_values(baselines_path, UPPER_ANCHOR_NAME)),
    }

    return anchor_values


def drop_decision(
    paired: Dict[str, object],
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> Dict[str, bool]:
    """
    Applies the two-part rule that retains an intervention by default.

    An intervention is dropped only when the paired mean falls below the margin
    (a meaningful loss) and the upper confidence bound stays below zero (a
    significant loss). Failing either test keeps it.

    Args:
        paired (Dict[str, object]): Result from
            :func:`~src.analysis.utils.paired_stats.paired_difference`.
        drop_threshold (float): Normalized-score margin (default -0.02).

    Returns:
        decision (Dict[str, bool]): Keys ``meaningful``, ``significant``, and
            ``drop``.
    """

    # Check if meaningsful or significant
    meaningful = bool(paired["delta"] < drop_threshold)
    significant = bool(paired["ci"][1] < 0)

    # Make a decision
    decision = {
        "meaningful": meaningful,
        "significant": significant,
        "drop": meaningful and significant,
    }

    return decision


def compose_environment(
    env_dir: Path,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """
    Computes the combined-intervention set for one environment.

    Normalizes every condition against the shared anchors, forms the seed-paired
    difference of each intervention against the reference policy, applies the
    drop rule, and returns the retained set alongside the full per-intervention
    statistics for auditing.

    Args:
        env_dir (Path): Directory holding one environment's summaries.
        drop_threshold (float): Normalized-score margin (default -0.02).
        confidence (float): Confidence level for the paired interval.

    Returns:
        result (Dict[str, object]): Keys ``env``, ``anchors``, ``interventions``
            (per-label ``paired`` stats and ``decision``), ``retained``, and
            ``dropped``.
    """

    # Get the environment directory and the anchors
    env_dir = Path(env_dir)
    anchors = load_anchors(env_dir)

    # Normalize the reference policy once for every comparison
    reference = normalized_scores(
        load_seed_values(env_dir / REFERENCE_FILENAME),
        anchors["lower"],
        anchors["upper"],
    )

    # Pair each normalized intervention against the normalized reference
    interventions: List[Dict[str, object]] = []
    retained: List[str] = []
    dropped: List[str] = []
    for label, path in discover_intervention_files(env_dir).items():
        condition = normalized_scores(
            load_seed_values(path), anchors["lower"], anchors["upper"]
        )
        paired = paired_difference(condition, reference, confidence)
        decision = drop_decision(paired, drop_threshold)

        interventions.append(
            {"label": label, "paired": paired, "decision": decision}
        )
        (dropped if decision["drop"] else retained).append(label)

    # Set the results
    results = {
        "env": env_dir.name,
        "anchors": anchors,
        "interventions": interventions,
        "retained": sorted(retained),
        "dropped": sorted(dropped),
    }

    return results


def format_environment_table(result: Dict[str, object]) -> str:
    """
    Renders one environment's composition decision as a text table.

    Args:
        result (Dict[str, object]): Output of :func:`compose_environment`.

    Returns:
        table (str): Console-ready block with per-intervention statistics.
    """

    anchors = result["anchors"]
    lines = [
        f"Environment: {result['env']}",
        f"Anchors: Random={anchors['lower']:.4f}  BS-Optimized={anchors['upper']:.4f}",
    ]

    # Lay out one row per intervention with the paired statistics
    header = f"{'Label':<14}{'delta':>10}{'95% CI':>24}{'d_z':>9}{'n':>4}  Verdict"
    lines.append(header)
    lines.append("-" * len(header))
    for item in result["interventions"]:
        paired, decision = item["paired"], item["decision"]
        ci = f"[{paired['ci'][0]:+.4f}, {paired['ci'][1]:+.4f}]"
        verdict = "DROP" if decision["drop"] else "keep"
        lines.append(
            f"{item['label']:<14}{paired['delta']:>+10.4f}{ci:>24}"
            f"{paired['cohen_dz']:>+9.2f}{paired['n']:>4}  {verdict}"
        )

    lines.append(
        f"Retained: {', '.join(result['retained']) or 'none'}"
        f"  |  Dropped: {', '.join(result['dropped']) or 'none'}"
    )
    return "\n".join(lines)
