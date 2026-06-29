"""
Normalized scores for cross-environment comparison.

Absolute returns are not comparable across the five environments because they
differ in absolute cost magnitude. The thesis therefore reports each policy's
return as the fraction of the Random-to-BS-Optimized range it recovers
(SCOPE.md 6.2, EXPERIMENTS.md 2):

    normalized = (R - R_Random) / (R_BS-Optimized - R_Random)

``R_Random`` and ``R_BS-Optimized`` are *locked* per-environment reference
values (the across-seed mean return of the Random and BS-Optimized baselines
from E0.6 / E1.2). They are treated as fixed scalars, so normalization is an
affine transform: a per-seed score is obtained by applying the same two anchors
to that seed's raw return. A score of 0 means no better than random; 1 means it
matches the BS-Optimized heuristic ceiling; above 1 is possible since learned
policies are not restricted to the stationary base-stock class.

These helpers consume the run dicts produced by
``src.visualization.utils.plot_data_loader.load_run``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np


def normalized_score(value, r_random: float, r_bs_optimized: float):
    """
    Applies the normalized-score affine transform.

    Args:
        value (float | np.ndarray): Raw return(s) to normalize.
        r_random (float): Locked Random reference return for the environment.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        score (float | np.ndarray): Normalized score(s), same shape as
            ``value``.
    """

    denom = float(r_bs_optimized) - float(r_random)
    if denom == 0:
        raise ValueError(
            "BS-Optimized and Random reference values are equal; the normalized "
            "score is undefined (zero range)"
        )
    return (np.asarray(value, dtype=float) - float(r_random)) / denom


def seed_returns(run: Mapping[str, Any]) -> Dict[int, float]:
    """
    Extracts the per-seed final evaluation return from a loaded run.

    Args:
        run (Mapping[str, Any]): A run dict from ``load_run``.

    Returns:
        returns (Dict[int, float]): ``{seed_number: episode_return_mean}`` for
            every seed that has a final evaluation result.
    """

    out: Dict[int, float] = {}
    for seed_number, seed in run.get("seed_data", {}).items():
        final_eval = seed.get("final_eval")
        if not final_eval:
            continue
        value = final_eval.get("episode_return_mean")
        if value is None:
            continue
        out[int(seed_number)] = float(value)
    return out


def reference_value(run: Mapping[str, Any]) -> float:
    """
    Returns the locked across-seed mean return for a baseline run, to be used
    as ``R_Random`` or ``R_BS-Optimized``.

    Prefers the per-seed final-eval mean (so the anchor matches the seeds the
    conditions are evaluated on) and falls back to the best-config mean stored
    in ``seed_evaluation_summary.yaml`` when no per-seed results are present.

    Args:
        run (Mapping[str, Any]): A baseline run dict from ``load_run``.

    Returns:
        value (float): The locked reference return.
    """

    returns = seed_returns(run)
    if returns:
        return float(np.mean(list(returns.values())))

    summary = run.get("summary")
    if summary and summary.get("configs"):
        return float(summary["configs"][0]["mean"])

    raise ValueError(
        f"run '{run.get('name')}' has neither per-seed final evals nor a summary "
        f"with configs; cannot derive a locked reference value"
    )


def normalize_run(
    run: Mapping[str, Any],
    r_random: float,
    r_bs_optimized: float,
) -> Dict[str, Any]:
    """
    Normalizes every seed's return for one condition against the locked anchors
    and summarizes the result across seeds.

    The standard error uses the across-seed standard deviation. For a single
    condition this is the per-condition SE; paired comparisons between two
    conditions should instead use the per-seed score arrays returned here under
    ``per_seed`` together with the paired protocol in EXPERIMENTS.md 2.

    Args:
        run (Mapping[str, Any]): A condition run dict from ``load_run``.
        r_random (float): Locked Random reference return for the environment.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        result (Dict[str, Any]): Keys ``per_seed`` (``{seed: score}``),
            ``scores`` (np.ndarray ordered by seed), ``mean``, ``std``, ``sem``,
            and ``n_seeds``.
    """

    returns = seed_returns(run)
    if not returns:
        raise ValueError(
            f"run '{run.get('name')}' has no per-seed final evals to normalize"
        )

    seeds_sorted = sorted(returns)
    raw = np.array([returns[s] for s in seeds_sorted], dtype=float)
    scores = normalized_score(raw, r_random, r_bs_optimized)

    n = len(scores)
    std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    return {
        "per_seed": {s: float(score) for s, score in zip(seeds_sorted, scores)},
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": std,
        "sem": std / np.sqrt(n) if n > 0 else 0.0,
        "n_seeds": n,
    }


def normalize_runs(
    runs: Mapping[str, Mapping[str, Any]],
    r_random: float,
    r_bs_optimized: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper to normalize several conditions in one environment
    against a shared pair of locked anchors.

    Args:
        runs (Mapping[str, Mapping[str, Any]]): ``{label: run}`` for the
            conditions in a single environment.
        r_random (float): Locked Random reference return.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        result (Dict[str, Dict[str, Any]]): ``{label: normalize_run(...)}``.
    """

    return {
        label: normalize_run(run, r_random, r_bs_optimized)
        for label, run in runs.items()
    }


def normalize_curve(
    iterations: Iterable[float],
    means: Iterable[float],
    stds: Iterable[float],
    r_random: float,
    r_bs_optimized: float,
) -> Dict[str, np.ndarray]:
    """
    Converts a raw-return training curve (mean +/- std band across seeds) into a
    normalized-score curve, for the gap figure where the y-axis is normalized
    score over training steps.

    Because normalization is affine with a fixed positive range, the band scales
    by the same factor and BS-Optimized / Random map to the constant lines 1 and
    0. The standard-deviation band is rescaled by ``1 / (R_BS - R_Random)``.

    Args:
        iterations (Iterable[float]): Training-iteration x values (passed
            through unchanged).
        means (Iterable[float]): Per-iteration across-seed mean return.
        stds (Iterable[float]): Per-iteration across-seed std of return.
        r_random (float): Locked Random reference return.
        r_bs_optimized (float): Locked BS-Optimized reference return.

    Returns:
        curve (Dict[str, np.ndarray]): Keys ``iterations``, ``means`` (in score
            units), and ``stds`` (in score units).
    """

    denom = float(r_bs_optimized) - float(r_random)
    if denom == 0:
        raise ValueError("zero Random-to-BS-Optimized range; cannot normalize curve")

    return {
        "iterations": np.asarray(list(iterations), dtype=float),
        "means": normalized_score(list(means), r_random, r_bs_optimized),
        "stds": np.asarray(list(stds), dtype=float) / denom,
    }
