"""
Synthetic cost generator for inventory management environments.

Generates consistent cost structures (penalty costs, SKU weights, distances,
and shipment costs) based on environment dimensions (n_warehouses, n_skus, n_regions).

Used automatically when data_source.type == "synthetic" during config loading
to ensure cost matrix dimensions match the environment dimensions.
"""

import math
from typing import Any, Dict

import numpy as np


# =============================================================================
# Cost ranges — every cost is fully defined by [min, max].
# =============================================================================

# SKU weights (kg): lightest to heaviest item
WEIGHT_MIN, WEIGHT_MAX = 0.45, 3.46

# Penalty cost ($/unit): lost-sale penalty, increases with weight
PENALTY_MIN, PENALTY_MAX = 8.6, 15.2

# Inbound fixed cost ($/shipment): receiving/handling per shipment
IN_FIXED_MIN, IN_FIXED_MAX = 9.0, 18.0

# Inbound variable cost ($/unit): per-unit receiving, scales with weight
IN_VAR_MIN, IN_VAR_MAX = 0.12, 0.30

# Outbound fixed cost ($/shipment): home-region vs farthest-region
OUT_FIXED_CLOSE, OUT_FIXED_FAR = 22.0, 42.0

# Outbound variable cost ($/unit·km): home-region vs farthest-region
OUT_VAR_CLOSE, OUT_VAR_FAR = 0.18, 0.42

# Distance (km): home-region vs farthest-region
DIST_CLOSE, DIST_FAR = 150.0, 420.0


# =============================================================================
# Helpers
# =============================================================================

def _smooth_factor(i: int, j: int, amp: float = 0.06) -> float:
    """Small bounded deterministic variation in [1-amp, 1+amp]."""
    return 1.0 + amp * math.sin(0.9 * i + 1.7 * j + 0.3)


def _bounded_increasing(lo: float, hi: float, n: int,
                        amp: float = 0.02, seed: int = 0) -> np.ndarray:
    """
    Generate n strictly increasing values in [lo, hi].
    Uses log-spacing (denser at the low end) with small deterministic
    perturbation to avoid perfectly uniform spacing.
    """
    # If there is only one value, return the geometric mean
    if n == 1:
        return np.array([math.sqrt(lo * hi)])
    
    vals = np.exp(np.linspace(np.log(lo), np.log(hi), n))

    # Small perturbation on interior points (endpoints stay exact)
    for k in range(1, n - 1):
        vals[k] *= _smooth_factor(k, seed, amp=amp)

    # Clip, then enforce strict monotonicity
    vals = np.clip(vals, lo, hi)
    for k in range(1, n):
        if vals[k] <= vals[k - 1]:
            vals[k] = vals[k - 1] + (hi - lo) * 1e-4
    return vals


def _ring_distance(i: int, r: int, n: int) -> int:
    """Ring topology distance for warehouse-region proximity."""
    d = abs(i - r)
    return min(d, n - d)


# =============================================================================
# Main generator function
# =============================================================================

def generate_synthetic_costs(n_warehouses: int, n_skus: int, n_regions: int) -> Dict[str, Any]:
    """
    Generates a complete synthetic cost structure matching the given environment dimensions.

    Produces deterministic, plausible cost matrices for:
    - penalty_cost: per-SKU lost-sale penalty, shape (n_skus,)
    - sku_weights: per-SKU weights in kg, shape (n_skus,)
    - distances: warehouse-to-region distances in km, shape (n_warehouses, n_regions)
    - outbound_fixed: fixed outbound shipment cost, shape (n_warehouses, n_regions)
    - outbound_variable: variable outbound shipment cost, shape (n_warehouses, n_regions)
    - inbound_fixed: fixed inbound shipment cost, shape (n_skus, n_warehouses)
    - inbound_variable: variable inbound shipment cost, shape (n_skus, n_warehouses)

    Args:
        n_warehouses (int): Number of warehouses.
        n_skus (int): Number of SKUs.
        n_regions (int): Number of demand regions.

    Returns:
        costs (Dict[str, Any]): Dictionary containing all generated cost arrays as nested lists,
            ready to be merged into a config dict.
    """

    # SKU-level arrays: 
    # N strictly increasing values in [min, max]
    sku_weights = _bounded_increasing(WEIGHT_MIN, WEIGHT_MAX, n_skus, amp=0.02, seed=0)
    penalty_cost = _bounded_increasing(PENALTY_MIN, PENALTY_MAX, n_skus, amp=0.02, seed=1)

    # Inbound costs: 
    # Per-SKU level increases from lightest to heaviest within [min, max] 
    # with small per-warehouse variation via _smooth_factor
    in_fixed_level = _bounded_increasing(IN_FIXED_MIN, IN_FIXED_MAX, n_skus, amp=0.03, seed=2)
    in_var_level = _bounded_increasing(IN_VAR_MIN, IN_VAR_MAX, n_skus, amp=0.03, seed=3)
    inbound_fixed = np.zeros((n_skus, n_warehouses), dtype=float)
    inbound_var = np.zeros((n_skus, n_warehouses), dtype=float)

    for s in range(n_skus):
        for w in range(n_warehouses):
            inbound_fixed[s, w] = in_fixed_level[s] * _smooth_factor(s, w, amp=0.07)
            inbound_var[s, w] = in_var_level[s] * _smooth_factor(s, w, amp=0.05)

    # Outbound costs + distances: 
    # Smooth interpolation between "close" (ring dist = 0) and "far" (ring dist = max_ring) cost
    # with small per-warehouse variation via _smooth_factor
    max_ring = max(n_warehouses // 2, 1)
    distances = np.zeros((n_warehouses, n_regions), dtype=float)
    outbound_fixed = np.zeros((n_warehouses, n_regions), dtype=float)
    outbound_var = np.zeros((n_warehouses, n_regions), dtype=float)

    for w in range(n_warehouses):
        for r in range(n_regions):
            d = _ring_distance(w, r, n_warehouses)
            t = d / max_ring  # 0 = home, 1 = farthest
            dist = DIST_CLOSE + t * (DIST_FAR - DIST_CLOSE)
            of_ = OUT_FIXED_CLOSE + t * (OUT_FIXED_FAR - OUT_FIXED_CLOSE)
            ov_ = OUT_VAR_CLOSE + t * (OUT_VAR_FAR - OUT_VAR_CLOSE)
            distances[w, r] = dist * _smooth_factor(w, r, amp=0.06)
            outbound_fixed[w, r] = of_ * _smooth_factor(w, r, amp=0.08)
            outbound_var[w, r] = ov_ * _smooth_factor(w, r, amp=0.06)


    # Package results as nested Python lists (for YAML/config compat)
    return {
        "penalty_cost": [round(float(x), 3) for x in penalty_cost],
        "sku_weights": [round(float(x), 3) for x in sku_weights],
        "distances": distances.round(3).tolist(),
        "outbound_fixed": outbound_fixed.round(3).tolist(),
        "outbound_variable": outbound_var.round(5).tolist(),
        "inbound_fixed": inbound_fixed.round(3).tolist(),
        "inbound_variable": inbound_var.round(5).tolist(),
    }
