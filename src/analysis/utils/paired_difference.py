"""
Paired-difference statistics for across-seed condition comparisons.

Any two conditions in this thesis are evaluated on the *same* set of root seeds,
so their per-seed scores are paired and the seed-induced noise shared by both
conditions can be removed with the paired (matched) protocol committed in
EXPERIMENTS.md 2. This module implements that protocol exactly. Given the
per-seed scores of a condition ``A`` and a reference condition ``B`` it reports
the paired mean difference, the paired standard error (with the shared-noise
covariance term subtracted), a two-sided paired t-test, the paired confidence
interval, and the paired effect size Cohen's d_z.

The natural inputs are the per-seed score maps produced by
``compute_normalized.normalize_run`` (its ``per_seed`` field), so a comparison
is typically::

    from src.analysis.utils.compute_normalized import normalize_run
    from src.analysis.utils.paired_difference import paired_difference

    a = normalize_run(retuned_run, r_random, r_bs)["per_seed"]
    b = normalize_run(transferred_run, r_random, r_bs)["per_seed"]
    res = paired_difference(a, b, label="MAPPO-PS: re-tuned vs transferred")
    print(res.delta, res.ci, res.p_value)

Sign convention: ``delta = mean(A) - mean(B)``, so a positive ``delta`` means
condition ``A`` scores higher than the reference ``B``.

When ``a`` and ``b`` are per-seed mappings the observations are paired on the
seed key and only the common seeds are used; when they are plain sequences they
are paired by position and must have equal length.

Multiple-comparison correction (Holm-Bonferroni over a family of these tests)
lives in the sibling module ``holm_bonferroni``; it consumes the ``p_value``
fields produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

__all__ = ["PairedDifference", "paired_difference", "paired_differences"]

# Either a {seed: score} mapping (paired on seed) or a score sequence (paired by
# position).
PairedInput = Union[Mapping[int, float], Sequence[float]]


@dataclass(frozen=True)
class PairedDifference:
    """
    Result of one paired-difference comparison of condition ``A`` against
    reference ``B`` across the shared seeds, following EXPERIMENTS.md 2.

    Attributes:
        label (Optional[str]): Human-readable name of the comparison.
        n (int): Number of paired observations (seeds) used.
        mean_a (float): Across-seed mean of condition ``A``.
        mean_b (float): Across-seed mean of reference ``B``.
        delta (float): Point estimate ``mean_a - mean_b``.
        var_a (float): Across-seed sample variance of ``A`` (ddof=1).
        var_b (float): Across-seed sample variance of ``B`` (ddof=1).
        cov (float): Across-seed sample covariance of ``A`` and ``B`` (ddof=1).
        var_paired (float): Paired variance ``var_a + var_b - 2 * cov``.
        se_paired (float): Paired standard error ``sqrt(var_paired) / sqrt(n)``.
        df (int): Degrees of freedom ``n - 1``.
        t_stat (float): Paired t-statistic ``delta / se_paired``.
        p_value (float): Two-sided p-value of the paired t-test.
        ci_level (float): Confidence level of the interval (``1 - alpha``).
        ci_low (float): Lower bound of the paired confidence interval on ``delta``.
        ci_high (float): Upper bound of the paired confidence interval on ``delta``.
        cohens_dz (float): Paired effect size ``delta / sqrt(var_paired)``.
    """

    label: Optional[str]
    n: int
    mean_a: float
    mean_b: float
    delta: float
    var_a: float
    var_b: float
    cov: float
    var_paired: float
    se_paired: float
    df: int
    t_stat: float
    p_value: float
    ci_level: float
    ci_low: float
    ci_high: float
    cohens_dz: float

    @property
    def ci(self) -> Tuple[float, float]:
        """The paired confidence interval as ``(ci_low, ci_high)``."""
        return (self.ci_low, self.ci_high)

    @property
    def significant(self) -> bool:
        """
        Whether the *uncorrected* two-sided test rejects at ``1 - ci_level``.

        For family-wise control across several comparisons use
        ``holm_bonferroni`` on the ``p_value`` fields instead.
        """
        return self.p_value <= (1.0 - self.ci_level)

    def as_dict(self) -> Dict[str, Any]:
        """Returns a plain ``dict`` of the fields for YAML/table serialization."""
        return {
            "label": self.label,
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "var_a": self.var_a,
            "var_b": self.var_b,
            "cov": self.cov,
            "var_paired": self.var_paired,
            "se_paired": self.se_paired,
            "df": self.df,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "ci_level": self.ci_level,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "cohens_dz": self.cohens_dz,
        }


def _aligned(a: PairedInput, b: PairedInput) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two paired inputs into seed-ordered (or position-ordered) arrays.

    Args:
        a (PairedInput): Condition ``A`` scores, as a ``{seed: score}`` mapping
            or a sequence.
        b (PairedInput): Reference ``B`` scores, of the same kind as ``a``.

    Returns:
        a_arr, b_arr (Tuple[np.ndarray, np.ndarray]): Aligned score arrays.

    Raises:
        TypeError: If exactly one of ``a``, ``b`` is a mapping (ambiguous pairing).
        ValueError: If mappings share no seed, or sequences differ in length.
    """

    a_is_map = isinstance(a, Mapping)
    b_is_map = isinstance(b, Mapping)
    if a_is_map != b_is_map:
        raise TypeError(
            "a and b must be the same kind of input: either both per-seed "
            "mappings (paired on seed) or both sequences (paired by position)"
        )

    if a_is_map:
        common = sorted(set(a) & set(b))
        if not common:
            raise ValueError("a and b share no common seeds; cannot pair them")
        a_arr = np.array([float(a[s]) for s in common], dtype=float)
        b_arr = np.array([float(b[s]) for s in common], dtype=float)
    else:
        a_arr = np.asarray(list(a), dtype=float)
        b_arr = np.asarray(list(b), dtype=float)
        if a_arr.shape != b_arr.shape:
            raise ValueError(
                f"sequence inputs must have equal length, got {a_arr.size} "
                f"and {b_arr.size}"
            )

    return a_arr, b_arr


def paired_difference(
    a: PairedInput,
    b: PairedInput,
    *,
    alpha: float = 0.05,
    label: Optional[str] = None,
) -> PairedDifference:
    """
    Computes the paired-difference statistics of condition ``A`` against
    reference ``B`` across shared seeds (EXPERIMENTS.md 2).

    The covariance term in ``var_paired = var_a + var_b - 2 * cov`` absorbs the
    seed-induced noise shared by both conditions, so the paired standard error
    is materially smaller than the unpaired one whenever the conditions are run
    on the same seeds.

    Args:
        a (PairedInput): Condition-of-interest per-seed scores, as a
            ``{seed: score}`` mapping (e.g. ``normalize_run(...)["per_seed"]``)
            or a sequence paired by position.
        b (PairedInput): Reference per-seed scores, of the same kind as ``a``.
        alpha (float): Significance level; the interval is at ``1 - alpha``.
        label (Optional[str]): Optional name carried into the result.

    Returns:
        result (PairedDifference): The full set of paired statistics.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``, fewer than two paired
            observations remain, or any input value is non-finite.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    a_arr, b_arr = _aligned(a, b)
    n = int(a_arr.size)
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        raise ValueError("inputs contain non-finite values")

    mean_a = float(np.mean(a_arr))
    mean_b = float(np.mean(b_arr))
    delta = mean_a - mean_b

    # 2x2 covariance matrix, normalized by (n - 1).
    cov_mat = np.cov(a_arr, b_arr)
    var_a = float(cov_mat[0, 0])
    var_b = float(cov_mat[1, 1])
    cov_ab = float(cov_mat[0, 1])

    var_paired = var_a + var_b - 2.0 * cov_ab
    if var_paired < 0.0:  # guard against a tiny negative from floating point
        var_paired = 0.0

    df = n - 1
    ci_level = 1.0 - alpha

    if var_paired == 0.0:
        # Every paired difference is identical: the test is degenerate.
        se_paired = 0.0
        ci_low = ci_high = delta
        if delta == 0.0:
            t_stat, p_value, cohens_dz = 0.0, 1.0, 0.0
        else:
            sign = 1.0 if delta > 0.0 else -1.0
            t_stat = sign * np.inf
            p_value = 0.0
            cohens_dz = sign * np.inf
    else:
        se_paired = float(np.sqrt(var_paired) / np.sqrt(n))
        t_stat = delta / se_paired
        # Survival function is accurate in the tail; two-sided test.
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        p_value = min(1.0, max(0.0, p_value))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        margin = t_crit * se_paired
        ci_low = delta - margin
        ci_high = delta + margin
        cohens_dz = delta / float(np.sqrt(var_paired))

    return PairedDifference(
        label=label,
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=float(delta),
        var_a=var_a,
        var_b=var_b,
        cov=cov_ab,
        var_paired=float(var_paired),
        se_paired=float(se_paired),
        df=df,
        t_stat=float(t_stat),
        p_value=float(p_value),
        ci_level=ci_level,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        cohens_dz=float(cohens_dz),
    )


def paired_differences(
    pairs: Mapping[str, Tuple[PairedInput, PairedInput]],
    *,
    alpha: float = 0.05,
) -> Dict[str, PairedDifference]:
    """
    Runs ``paired_difference`` over a family of named comparisons.

    Convenience wrapper for the pre-declared primary tests of a block (e.g. the
    four E1 ``re-tuned vs transferred`` comparisons). The returned p-values can
    be handed straight to ``holm_bonferroni`` for family-wise correction.

    Args:
        pairs (Mapping[str, Tuple[PairedInput, PairedInput]]): ``{label: (a, b)}``
            where ``a`` is the condition of interest and ``b`` the reference.
        alpha (float): Significance level passed through to each comparison.

    Returns:
        result (Dict[str, PairedDifference]): ``{label: PairedDifference}``.
    """

    return {
        label: paired_difference(a, b, alpha=alpha, label=label)
        for label, (a, b) in pairs.items()
    }
