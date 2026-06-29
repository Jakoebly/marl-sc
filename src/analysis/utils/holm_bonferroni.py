"""
Holm-Bonferroni step-down correction for a family of hypothesis tests.

EXPERIMENTS.md 2 controls the family-wise error rate of each experiment block's
pre-declared *primary* tests at alpha = 0.05 with the Holm-Bonferroni step-down
procedure, while secondary tests are reported with uncorrected p-values. This
module applies that correction to a set of named p-values, for example the four
E1 primary tests (each algorithm's re-tuned vs transferred comparison) or the
five E2 intervention tests.

It is deliberately decoupled from how the p-values were produced. Pass the
``p_value`` fields from ``paired_difference`` results, or any other p-values::

    from src.analysis.utils.paired_difference import paired_differences
    from src.analysis.utils.holm_bonferroni import holm_bonferroni

    results = paired_differences(primary_tests)          # {label: PairedDifference}
    corrected = holm_bonferroni({k: r.p_value for k, r in results.items()})
    for label, h in corrected.items():
        print(label, h.p_adjusted, h.reject)

The reported ``p_adjusted`` are the standard monotone Holm-adjusted p-values, so
``reject == (p_adjusted <= alpha)`` reproduces the step-down decision exactly:
the raw p-values are sorted ascending, the j-th smallest of ``m`` is scaled by
``(m - j + 1)``, and a running maximum enforces monotonicity so that once one
hypothesis is retained every larger p-value is retained too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np

__all__ = ["HolmResult", "holm_bonferroni"]


@dataclass(frozen=True)
class HolmResult:
    """
    Holm-Bonferroni outcome for a single hypothesis within the family.

    Attributes:
        p_value (float): The original uncorrected p-value.
        p_adjusted (float): Monotone Holm-adjusted p-value, capped at 1.
        reject (bool): Whether the hypothesis is rejected at the family-wise
            ``alpha`` (equivalently ``p_adjusted <= alpha``).
        rank (int): 1-based ascending rank of the raw p-value within the family
            (ties broken by input order).
        n_tests (int): Size of the family ``m``.
    """

    p_value: float
    p_adjusted: float
    reject: bool
    rank: int
    n_tests: int

    def as_dict(self) -> Dict[str, Any]:
        """Returns a plain ``dict`` of the fields for YAML/table serialization."""
        return {
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "reject": self.reject,
            "rank": self.rank,
            "n_tests": self.n_tests,
        }


def holm_bonferroni(
    pvalues: Union[Mapping[str, float], Sequence[float]],
    *,
    alpha: float = 0.05,
) -> Union[Dict[str, HolmResult], List[HolmResult]]:
    """
    Applies the Holm-Bonferroni step-down correction to a family of p-values.

    Args:
        pvalues (Union[Mapping[str, float], Sequence[float]]): Either a
            ``{label: p_value}`` mapping or a sequence of p-values. The family
            size ``m`` is the number of entries; pass only the tests that belong
            to the correction family.
        alpha (float): Family-wise significance level.

    Returns:
        result (Union[Dict[str, HolmResult], List[HolmResult]]): For a mapping
            input, ``{label: HolmResult}`` keyed as the input; for a sequence
            input, a list of ``HolmResult`` in the original input order.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``, the family is empty, or
            any p-value is non-finite or outside ``[0, 1]``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    is_map = isinstance(pvalues, Mapping)
    if is_map:
        labels = list(pvalues.keys())
        p = np.asarray([float(v) for v in pvalues.values()], dtype=float)
    else:
        labels = None
        p = np.asarray([float(v) for v in pvalues], dtype=float)

    m = int(p.size)
    if m == 0:
        raise ValueError("no p-values provided")
    if np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must be finite and in [0, 1]")

    # Ascending order of raw p-values (stable so ties keep input order).
    order = np.argsort(p, kind="stable")

    p_adjusted = np.empty(m, dtype=float)
    rank_of = np.empty(m, dtype=int)
    running_max = 0.0
    for i, idx in enumerate(order):
        # j-th smallest (1-based j = i + 1) is scaled by (m - i).
        scaled = (m - i) * p[idx]
        running_max = max(running_max, scaled)
        p_adjusted[idx] = min(1.0, running_max)
        rank_of[idx] = i + 1

    reject = p_adjusted <= alpha

    results = [
        HolmResult(
            p_value=float(p[i]),
            p_adjusted=float(p_adjusted[i]),
            reject=bool(reject[i]),
            rank=int(rank_of[i]),
            n_tests=m,
        )
        for i in range(m)
    ]

    if is_map:
        return {label: results[i] for i, label in enumerate(labels)}
    return results
