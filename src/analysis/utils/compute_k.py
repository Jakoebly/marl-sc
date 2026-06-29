"""
Operational coupling metric ``kappa`` (cross-region fulfillment fraction).

``kappa`` is the fraction of all shipped customer units that are delivered by a
warehouse other than the destination region's home warehouse. It quantifies how
much cross-region fulfillment (implicit transshipment) a policy actually uses,
which converts "coupling" from a descriptor into a measured environment
property (see SCOPE.md 5.4 and the E1.5 procedure in EXPERIMENTS.md).

The metric is computed from deterministic rollout episodes produced with
``--visualize`` (so that ``shipment_quantities`` and ``home_regions`` are saved
into ``episodes_data.npz``). For the thesis it is reported under the
BS-Optimized policy, but the reducers here are policy-agnostic and are reused on
the MAPPO-PS rollouts for the cost-axis failure-mode analysis.

A warehouse's home region is ``argmin`` over its distance row, which the
environment stores as ``env.home_regions`` and the rollout persists per episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _episode_home_regions(episode: Dict[str, Any]) -> np.ndarray:
    """
    Extracts and validates the per-warehouse home-region index vector.

    Args:
        episode (Dict[str, Any]): One rollout episode dict.

    Returns:
        home_regions (np.ndarray): Integer array of shape ``(n_warehouses,)``.
    """

    if "home_regions" not in episode:
        raise KeyError(
            "episode is missing 'home_regions'; re-run the rollout with "
            "--visualize so the env's home_regions are saved into episodes_data.npz"
        )
    home_regions = np.asarray(episode["home_regions"]).astype(int).ravel()
    return home_regions


def _split_home_cross(
    shipment_quantities: np.ndarray,
    home_regions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a per-step shipment tensor into home and cross-region unit totals
    per warehouse.

    Args:
        shipment_quantities (np.ndarray): Units shipped, shape
            ``(T, n_warehouses, n_regions)``.
        home_regions (np.ndarray): Home region index per warehouse, shape
            ``(n_warehouses,)``.

    Returns:
        home_units (np.ndarray): Per-warehouse home-region units, shape
            ``(n_warehouses,)``.
        cross_units (np.ndarray): Per-warehouse cross-region units, shape
            ``(n_warehouses,)``.
    """

    shipments = np.asarray(shipment_quantities, dtype=float)
    if shipments.ndim != 3:
        raise ValueError(
            f"expected shipment_quantities of shape (T, n_warehouses, n_regions), "
            f"got {shipments.shape}"
        )

    _, n_warehouses, _ = shipments.shape
    if home_regions.shape[0] != n_warehouses:
        raise ValueError(
            f"home_regions has {home_regions.shape[0]} entries but shipment "
            f"tensor has {n_warehouses} warehouses"
        )

    # Total units shipped per warehouse over the whole episode.
    total_per_wh = shipments.sum(axis=(0, 2))  # (n_warehouses,)

    # Units each warehouse shipped to its own home region (the diagonal pick).
    wh_idx = np.arange(n_warehouses)
    home_per_wh = shipments[:, wh_idx, home_regions].sum(axis=0)  # (n_warehouses,)

    cross_per_wh = total_per_wh - home_per_wh
    return home_per_wh, cross_per_wh


def episode_kappa(episode: Dict[str, Any], by_warehouse: bool = False):
    """
    Computes the cross-region fulfillment fraction for a single episode.

    Args:
        episode (Dict[str, Any]): Rollout episode dict carrying
            ``shipment_quantities`` (T, n_warehouses, n_regions) and
            ``home_regions`` (n_warehouses,).
        by_warehouse (bool): If True, return a per-warehouse fraction vector
            instead of the single pooled fraction.

    Returns:
        kappa (float | np.ndarray): Pooled cross-region fraction in ``[0, 1]``,
            or a per-warehouse vector of shape ``(n_warehouses,)`` when
            ``by_warehouse`` is True. Warehouses (or episodes) that shipped no
            units yield ``nan``.
    """

    home_regions = _episode_home_regions(episode)
    home_per_wh, cross_per_wh = _split_home_cross(
        episode["shipment_quantities"], home_regions
    )

    if by_warehouse:
        total_per_wh = home_per_wh + cross_per_wh
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total_per_wh > 0, cross_per_wh / total_per_wh, np.nan)
        return frac

    total = float(home_per_wh.sum() + cross_per_wh.sum())
    if total <= 0:
        return float("nan")
    return float(cross_per_wh.sum() / total)


def episode_kappa_per_sku(episode: Dict[str, Any]) -> np.ndarray:
    """
    Computes the cross-region fulfillment fraction per SKU for one episode.

    Uses ``shipment_quantities_by_sku`` (T, n_warehouses, n_regions, n_skus).
    Useful for inspecting whether cross-region fulfillment concentrates in
    particular SKUs (e.g. light, high-volume items) under heterogeneity.

    Args:
        episode (Dict[str, Any]): Rollout episode dict.

    Returns:
        kappa_per_sku (np.ndarray): Cross-region fraction per SKU, shape
            ``(n_skus,)``; SKUs with no shipments yield ``nan``.
    """

    if "shipment_quantities_by_sku" not in episode:
        raise KeyError("episode is missing 'shipment_quantities_by_sku'")

    home_regions = _episode_home_regions(episode)
    shipments = np.asarray(episode["shipment_quantities_by_sku"], dtype=float)
    if shipments.ndim != 4:
        raise ValueError(
            f"expected shipment_quantities_by_sku of shape "
            f"(T, n_warehouses, n_regions, n_skus), got {shipments.shape}"
        )

    _, n_warehouses, _, _ = shipments.shape
    wh_idx = np.arange(n_warehouses)

    total_per_sku = shipments.sum(axis=(0, 1, 2))                 # (n_skus,)
    home_per_sku = shipments[:, wh_idx, home_regions, :].sum(axis=(0, 1))  # (n_skus,)
    cross_per_sku = total_per_sku - home_per_sku

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total_per_sku > 0, cross_per_sku / total_per_sku, np.nan)
    return frac


def compute_kappa(
    episodes: Sequence[Dict[str, Any]],
    by_warehouse: bool = False,
) -> Dict[str, Any]:
    """
    Aggregates ``kappa`` across a set of rollout episodes (e.g. all evaluation
    episodes pooled across seeds for one environment).

    Two estimators are returned. ``kappa_pooled`` is the volume-weighted ratio
    of total cross-region units to total shipped units across every episode,
    which is the natural estimate of the *expected* cross-region fraction.
    ``kappa_mean`` / ``kappa_std`` summarise the distribution of the per-episode
    fraction and are useful for a spread indication.

    Args:
        episodes (Sequence[Dict[str, Any]]): Rollout episode dicts. Pool all
            episodes for one environment here (across seeds and eval episodes).
        by_warehouse (bool): If True, also include per-warehouse pooled
            fractions under ``kappa_pooled_by_warehouse``.

    Returns:
        result (Dict[str, Any]): Keys ``kappa_pooled`` (float), ``kappa_mean``
            (float), ``kappa_std`` (float), ``n_episodes`` (int), and optionally
            ``kappa_pooled_by_warehouse`` (np.ndarray).
    """

    if len(episodes) == 0:
        raise ValueError("episodes is empty; nothing to aggregate")

    total_home = 0.0
    total_cross = 0.0
    home_by_wh: np.ndarray | None = None
    cross_by_wh: np.ndarray | None = None
    per_episode: List[float] = []

    for episode in episodes:
        home_regions = _episode_home_regions(episode)
        home_per_wh, cross_per_wh = _split_home_cross(
            episode["shipment_quantities"], home_regions
        )

        ep_home = float(home_per_wh.sum())
        ep_cross = float(cross_per_wh.sum())
        ep_total = ep_home + ep_cross

        total_home += ep_home
        total_cross += ep_cross
        per_episode.append(ep_cross / ep_total if ep_total > 0 else np.nan)

        if by_warehouse:
            home_by_wh = home_per_wh if home_by_wh is None else home_by_wh + home_per_wh
            cross_by_wh = cross_per_wh if cross_by_wh is None else cross_by_wh + cross_per_wh

    grand_total = total_home + total_cross
    per_episode_arr = np.asarray(per_episode, dtype=float)

    result: Dict[str, Any] = {
        "kappa_pooled": total_cross / grand_total if grand_total > 0 else float("nan"),
        "kappa_mean": float(np.nanmean(per_episode_arr)),
        "kappa_std": float(np.nanstd(per_episode_arr, ddof=1))
        if np.sum(~np.isnan(per_episode_arr)) > 1
        else 0.0,
        "n_episodes": len(episodes),
    }

    if by_warehouse and cross_by_wh is not None and home_by_wh is not None:
        total_by_wh = home_by_wh + cross_by_wh
        with np.errstate(invalid="ignore", divide="ignore"):
            result["kappa_pooled_by_warehouse"] = np.where(
                total_by_wh > 0, cross_by_wh / total_by_wh, np.nan
            )

    return result
