"""
Optimality gap relative to the BS-Optimized ceiling.

The normalized score (``compute_normalized``) anchors at both Random and
BS-Optimized, so its denominator is the Random-to-ceiling range. That range is
large and *environment-dependent*: Random is far more catastrophic on the harder
heterogeneous envs, which both compresses genuine differences near the ceiling
and does so unevenly across envs (the same absolute shortfall reads as a smaller
normalized gap on a harder env). That makes the normalized score awkward for
*reporting* the magnitude of a heterogeneity-induced gap.

The optimality gap drops the Random anchor and measures suboptimality only
against the BS-Optimized ceiling, so a value means the same operational thing on
every env: the fractional excess cost over the strongest stationary baseline.
For a return ``R`` (negative = per-episode cost as negative reward) and the
locked BS-Optimized reference return ``R_BS``:

    gap = (R_BS - R) / |R_BS|

so ``gap = 0`` matches the ceiling, ``gap > 0`` is worse (more cost), and
``gap < 0`` beats the ceiling. Multiply by 100 to read it as "percent above
optimal cost".

Within one environment this is a strictly monotone (affine) re-expression of the
raw return, so it ranks conditions identically to the normalized score and to
the raw return; it differs across environments only by removing the Random
anchor. The exact link to the normalized score is

    1 - NS = gap * |R_BS| / (R_BS - R_Random).

Recommended division of labour: run the pre-registered statistical tests
(EXPERIMENTS.md 2) on the normalized score, and report magnitudes with the gap.
The per-seed outputs here mirror ``compute_normalized`` exactly, so either metric
can be fed to ``paired_difference``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np

from src.analysis.utils.compute_normalized import seed_returns

__all__ = ["optimality_gap", "gap_run", "gap_runs", "gap_curve"]


def optimality_gap(value, r_bs_optimized: float):
    """
    Optimality gap of a return (or array of returns) against the BS-Optimized
    ceiling, as a fraction of the ceiling magnitude.

    Args:
        value: Raw return(s); scalar or array-like.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        gap: ``(r_bs_optimized - value) / |r_bs_optimized|``. Positive means more
            cost than the ceiling (worse); ``* 100`` reads as percent above
            optimal cost.

    Raises:
        ValueError: If the BS-Optimized reference is zero.
    """

    denom = abs(float(r_bs_optimized))
    if denom == 0.0:
        raise ValueError(
            "BS-Optimized reference return is zero; cannot compute optimality gap"
        )
    return (float(r_bs_optimized) - np.asarray(value, dtype=float)) / denom


def gap_run(run: Mapping[str, Any], r_bs_optimized: float) -> Dict[str, Any]:
    """
    Computes the optimality gap for every seed of one condition and summarizes
    it across seeds.

    Mirrors ``compute_normalized.normalize_run``: the returned ``per_seed`` map
    can be handed straight to ``paired_difference`` for a gap-unit comparison,
    though the pre-registered tests use the normalized score.

    Args:
        run (Mapping[str, Any]): A condition run dict from ``load_run``.
        r_bs_optimized (float): Locked BS-Optimized reference return for the env.

    Returns:
        result (Dict[str, Any]): Keys ``per_seed`` (``{seed: gap}``), ``gaps``
            (np.ndarray ordered by seed), ``mean``, ``std``, ``sem``, and
            ``n_seeds``. Gaps are fractions (``* 100`` for percent).
    """

    returns = seed_returns(run)
    if not returns:
        raise ValueError(
            f"run '{run.get('name')}' has no per-seed final evals to score"
        )

    seeds_sorted = sorted(returns)
    raw = np.array([returns[s] for s in seeds_sorted], dtype=float)
    gaps = optimality_gap(raw, r_bs_optimized)

    n = len(gaps)
    std = float(np.std(gaps, ddof=1)) if n > 1 else 0.0
    return {
        "per_seed": {s: float(g) for s, g in zip(seeds_sorted, gaps)},
        "gaps": gaps,
        "mean": float(np.mean(gaps)),
        "std": std,
        "sem": std / np.sqrt(n) if n > 0 else 0.0,
        "n_seeds": n,
    }


def gap_runs(
    runs: Mapping[str, Mapping[str, Any]],
    r_bs_optimized: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper to score several conditions in one environment against
    the shared BS-Optimized anchor.

    Args:
        runs (Mapping[str, Mapping[str, Any]]): ``{label: run}`` for one env.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        result (Dict[str, Dict[str, Any]]): ``{label: gap_run(...)}``.
    """

    return {label: gap_run(run, r_bs_optimized) for label, run in runs.items()}


def gap_curve(
    iterations: Iterable[float],
    means: Iterable[float],
    stds: Iterable[float],
    r_bs_optimized: float,
) -> Dict[str, np.ndarray]:
    """
    Converts a raw-return training curve (mean +/- std across seeds) into an
    optimality-gap curve, for the gap figure where the y-axis is percent above
    optimal over training steps.

    The transform is affine with a fixed positive scale ``1 / |R_BS|``, so the
    mean maps through ``optimality_gap`` and the std band rescales by the same
    factor. BS-Optimized maps to the constant line 0.

    Args:
        iterations (Iterable[float]): Training-iteration x values (unchanged).
        means (Iterable[float]): Per-iteration across-seed mean return.
        stds (Iterable[float]): Per-iteration across-seed std of return.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        curve (Dict[str, np.ndarray]): Keys ``iterations``, ``means`` (gap units),
            and ``stds`` (gap units). Multiply by 100 for percent.
    """

    denom = abs(float(r_bs_optimized))
    if denom == 0.0:
        raise ValueError("zero BS-Optimized reference; cannot scale gap curve")

    return {
        "iterations": np.asarray(list(iterations), dtype=float),
        "means": optimality_gap(list(means), r_bs_optimized),
        "stds": np.asarray(list(stds), dtype=float) / denom,
    }
