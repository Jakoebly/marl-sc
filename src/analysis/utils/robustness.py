"""Robustness checks for the combined-intervention composition.

Re-tests every keep/drop verdict in two ways the seed-paired test cannot show on
its own: a Welch comparison that uses every reference seed instead of pairing on
the shared five, and a leave-one-seed-out jackknife that flags any verdict that
hinges on a single seed. Both reuse the normalized scores and the drop rule from
the composition driver, so the only new ingredient is the alternative estimator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats as scipy_stats

from src.analysis.utils.intervention_composition import (
    DEFAULT_DROP_THRESHOLD,
    REFERENCE_FILENAME,
    discover_intervention_files,
    drop_decision,
    load_anchors,
)
from src.analysis.utils.paired_stats import normalized_scores, paired_difference
from src.analysis.utils.summary_loader import load_seed_values


def welch_difference(
    values_a: Dict[int, float],
    values_b: Dict[int, float],
    confidence: float = 0.95,
) -> Dict[str, object]:
    """
    Computes the Welch (unpaired) difference between two conditions.

    Unlike the paired test, this keeps every seed of each condition rather than
    only the shared ones, so the reference retains all of its seeds. The returned
    keys match what the drop rule reads, so the same decision applies unchanged.

    Args:
        values_a (Dict[int, float]): ``{seed: value}`` for condition A.
        values_b (Dict[int, float]): ``{seed: value}`` for condition B.
        confidence (float): Confidence level for the interval (default 0.95).

    Returns:
        result (Dict[str, object]): Keys ``n_a``, ``n_b``, ``delta``, ``se``,
            ``df``, and ``ci`` (``[low, high]``).
    """

    # Take every seed of each condition (unpaired)
    a = np.array(list(values_a.values()))
    b = np.array(list(values_b.values()))
    n_a, n_b = len(a), len(b)
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))

    # Welch difference, standard error, and Satterthwaite degrees of freedom
    delta = float(a.mean() - b.mean())
    se = float(np.sqrt(var_a / n_a + var_b / n_b))
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_crit = float(scipy_stats.t.ppf(1.0 - (1.0 - confidence) / 2, df=df))

    return {
        "n_a": n_a,
        "n_b": n_b,
        "delta": delta,
        "se": se,
        "df": float(df),
        "ci": [delta - t_crit * se, delta + t_crit * se],
    }


def jackknife_paired(
    values_a: Dict[int, float],
    values_b: Dict[int, float],
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """
    Refits the paired drop decision with one seed left out at a time.

    Drops each shared seed in turn, re-runs the paired comparison and the drop
    rule on the seeds that remain, and counts how many of those refits disagree
    with the full-sample verdict. A verdict that survives every fold does not
    hinge on a single seed.

    Args:
        values_a (Dict[int, float]): ``{seed: value}`` for condition A.
        values_b (Dict[int, float]): ``{seed: value}`` for condition B.
        drop_threshold (float): Normalized-score margin (default -0.02).
        confidence (float): Confidence level for the interval (default 0.95).

    Returns:
        result (Dict[str, object]): Keys ``full_drop``, ``n_folds``,
            ``delta_min``, ``delta_max``, ``flips``, and ``fold_drops``.
    """

    # Establish the full-sample verdict to compare each fold against
    seeds = sorted(set(values_a) & set(values_b))
    full = paired_difference(values_a, values_b, confidence)
    full_drop = drop_decision(full, drop_threshold)["drop"]

    # Leave one seed out at a time and re-decide
    deltas: List[float] = []
    fold_drops: List[bool] = []
    for left_out in seeds:
        kept = [s for s in seeds if s != left_out]
        fold = paired_difference(
            {s: values_a[s] for s in kept},
            {s: values_b[s] for s in kept},
            confidence,
        )
        deltas.append(fold["delta"])
        fold_drops.append(drop_decision(fold, drop_threshold)["drop"])

    flips = sum(1 for d in fold_drops if d != full_drop)
    return {
        "full_drop": full_drop,
        "n_folds": len(seeds),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "flips": flips,
        "fold_drops": fold_drops,
    }


def environment_robustness(
    env_dir: Path,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """
    Re-tests every intervention's verdict in one environment two ways.

    For each intervention, compares the paired verdict against the Welch verdict
    that keeps all reference seeds, and runs the leave-one-seed-out jackknife.
    Both checks reuse the same normalized scores and drop rule as the composition
    itself, so a disagreement points at the paired test rather than the data.

    Args:
        env_dir (Path): Directory holding one environment's summaries.
        drop_threshold (float): Normalized-score margin (default -0.02).
        confidence (float): Confidence level for the intervals (default 0.95).

    Returns:
        result (Dict[str, object]): Keys ``env`` and ``rows`` (per-intervention
            ``label``, ``paired_drop``, ``welch_drop``, ``agree``, and
            ``jackknife``).
    """

    # Normalize the reference once against the shared anchors
    env_dir = Path(env_dir)
    anchors = load_anchors(env_dir)
    reference = normalized_scores(
        load_seed_values(env_dir / REFERENCE_FILENAME),
        anchors["lower"],
        anchors["upper"],
    )

    # Re-test each intervention against the reference
    rows: List[Dict[str, object]] = []
    for label, path in discover_intervention_files(env_dir).items():
        condition = normalized_scores(
            load_seed_values(path), anchors["lower"], anchors["upper"]
        )
        paired_drop = drop_decision(
            paired_difference(condition, reference, confidence), drop_threshold
        )["drop"]
        welch = welch_difference(condition, reference, confidence)
        welch_drop = drop_decision(welch, drop_threshold)["drop"]
        jackknife = jackknife_paired(condition, reference, drop_threshold, confidence)

        rows.append(
            {
                "label": label,
                "paired_drop": paired_drop,
                "welch_drop": welch_drop,
                "agree": paired_drop == welch_drop,
                "jackknife": jackknife,
            }
        )

    return {"env": env_dir.name, "rows": rows}


def format_robustness_table(robustness: Dict[str, object]) -> str:
    """
    Renders one environment's robustness checks as a text table.

    Args:
        robustness (Dict[str, object]): Output of :func:`environment_robustness`.

    Returns:
        table (str): Console-ready block, one row per intervention.
    """

    lines = [f"Environment: {robustness['env']}"]

    # Lay out one row per intervention with both alternative checks
    header = (
        f"{'Label':<8}{'paired':>8}{'Welch(all ref)':>16}{'agree':>7}"
        f"{'jk flips':>10}{'jk delta range':>24}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in robustness["rows"]:
        jk = row["jackknife"]
        paired = "DROP" if row["paired_drop"] else "keep"
        welch = "DROP" if row["welch_drop"] else "keep"
        agree = "yes" if row["agree"] else "NO"
        flips = f"{jk['flips']}/{jk['n_folds']}"
        rng = f"[{jk['delta_min']:+.4f}, {jk['delta_max']:+.4f}]"
        lines.append(
            f"{row['label']:<8}{paired:>8}{welch:>16}{agree:>7}{flips:>10}{rng:>24}"
        )
    return "\n".join(lines)
