"""Seed-paired difference statistics over evaluation summaries.

Implements the seed-matched comparison protocol: per-seed differences on the
common seed set, the paired mean, standard error, confidence interval,
t-statistic, two-sided p-value, and standardized effect size (Cohen's d_z).
The same primitives back the composition decision and any later paired analysis
(additivity, leave-one-out), so the math lives here once and is imported
elsewhere.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats as scipy_stats


def anchor_mean(seed_values: Dict[int, float]) -> float:
    """
    Reduces a reference policy to the scalar anchor of the normalized score.

    Args:
        seed_values (Dict[int, float]): ``{seed_number: metric_value}`` pairs.

    Returns:
        mean (float): Mean metric value across the provided seeds.
    """

    # Get the mean of the seed values
    mean = float(np.mean(list(seed_values.values())))
    
    return mean


def normalized_scores(
    seed_values: Dict[int, float],
    lower_anchor: float,
    upper_anchor: float,
) -> Dict[int, float]:
    """
    Rescales per-seed values onto the normalized-score axis.

    Applies the affine transform ``(value - lower) / (upper - lower)`` per seed,
    placing the lower anchor (Random) at 0 and the upper anchor (BS-Optimized)
    at 1. The transform is shared across compared conditions, so paired
    differences are merely rescaled and the t-statistic stays invariant.

    Args:
        seed_values (Dict[int, float]): ``{seed_number: metric_value}`` pairs.
        lower_anchor (float): Value mapped to 0 (Random reference).
        upper_anchor (float): Value mapped to 1 (BS-Optimized reference).

    Returns:
        normalized (Dict[int, float]): ``{seed_number: normalized_value}`` pairs.
    """

    # Get the span and check if it is zero
    span = upper_anchor - lower_anchor
    if span == 0:
        raise ValueError("Upper and lower anchors must differ.")

    # Normalize the seed values
    normalized = {s: (v - lower_anchor) / span for s, v in seed_values.items()}

    return normalized


def paired_difference(
    values_a: Dict[int, float],
    values_b: Dict[int, float],
    confidence: float = 0.95,
) -> Dict[str, object]:
    """
    Computes seed-paired difference statistics between two conditions.

    Restricts both conditions to their common seeds, forms per-seed differences
    ``a - b``, and returns the paired mean, standard error, confidence interval,
    t-statistic, two-sided p-value, and standardized effect size. The sample
    standard deviation of the per-seed differences equals
    ``sqrt(var(a) + var(b) - 2*cov(a, b))`` on matched seeds.

    Args:
        values_a (Dict[int, float]): ``{seed: value}`` for condition A.
        values_b (Dict[int, float]): ``{seed: value}`` for condition B.
        confidence (float): Confidence level for the interval (default 0.95).

    Returns:
        result (Dict[str, object]): Keys ``n``, ``seeds``, ``delta``,
            ``sd_paired``, ``se_paired``, ``ci`` (``[low, high]``), ``t_stat``,
            ``p_value``, ``cohen_dz``, and ``per_seed_diff``.
    """

    # Restrict both conditions to the seeds present in both
    seeds = sorted(set(values_a) & set(values_b))
    n = len(seeds)
    if n < 2:
        raise ValueError(f"Need at least two common seeds, got {n}.")

    # Form the per-seed differences on the matched seed set
    diffs = np.array([values_a[s] - values_b[s] for s in seeds])
    delta = float(diffs.mean())
    sd_paired = float(diffs.std(ddof=1))
    se_paired = sd_paired / np.sqrt(n)

    # Derive the interval, test statistic, and standardized effect size
    alpha = 1.0 - confidence
    t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2, df=n - 1))
    ci_low = delta - t_crit * se_paired
    ci_high = delta + t_crit * se_paired
    t_stat = delta / se_paired if se_paired > 0 else float("nan")
    p_value = (
        float(2 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
        if se_paired > 0
        else float("nan")
    )
    cohen_dz = delta / sd_paired if sd_paired > 0 else float("nan")

    # Return the results
    results = {
        "n": n,
        "seeds": seeds,
        "delta": delta,
        "sd_paired": sd_paired,
        "se_paired": float(se_paired),
        "ci": [float(ci_low), float(ci_high)],
        "t_stat": float(t_stat),
        "p_value": p_value,
        "cohen_dz": float(cohen_dz),
        "per_seed_diff": {int(s): float(d) for s, d in zip(seeds, diffs)},
    }

    return results
