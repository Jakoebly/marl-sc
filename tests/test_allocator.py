"""
Tests for ``GreedyDemandAllocator``: every region is served by its home warehouse first, and the
leftover demand is handed out with the regions taking turns in a random order and with random
tie-breaks between equally expensive warehouses.
"""

from typing import List, Tuple

import numpy as np
import pytest

from src.config.loader import load_environment_config
from src.config.schema import DemandAllocatorGreedy
from src.environment.components.demand_allocator import AllocationResult, GreedyDemandAllocator
from src.environment.components.demand_sampler import Order
from src.environment.context import EnvironmentContext, ShipmentCosts
from src.environment.envs.multi_env import InventoryEnvironment

SYMMETRIC_ENV = "config_files/environments/env_symmetric_3WH5SKU.yaml"


# ============================================================================
# Helpers
# ============================================================================

def make_allocator(
    distances: np.ndarray,
    outbound_variable: np.ndarray,
    sku_weights: np.ndarray,
    max_splits="default",
    seed: int = 0,
) -> GreedyDemandAllocator:
    """Builds a greedy allocator for a hand-made network with zero fixed outbound costs."""

    n_warehouses, n_regions = distances.shape
    n_skus = len(sku_weights)
    context = EnvironmentContext(
        n_warehouses=n_warehouses,
        n_skus=n_skus,
        n_regions=n_regions,
        episode_length=100,
        holding_cost=1.0,
        penalty_cost=5.0,
        shipment_cost=ShipmentCosts(
            outbound_fixed=np.zeros((n_warehouses, n_regions)),
            outbound_variable=np.asarray(outbound_variable, dtype=float),
            inbound_fixed=np.zeros((n_warehouses, n_skus)),
            inbound_variable=np.ones((n_warehouses, n_skus)),
        ),
        sku_weights=np.asarray(sku_weights, dtype=float),
        distances=np.asarray(distances, dtype=float),
    )
    config = DemandAllocatorGreedy(type="greedy", params={"max_splits": max_splits})
    allocator = GreedyDemandAllocator(context, config)
    allocator.reset(np.random.default_rng(seed))
    return allocator

def symmetric_network(n_warehouses: int = 3, n_skus: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distances, outbound rates and weights of the symmetric environment (home = same index)."""

    distances = np.full((n_warehouses, n_warehouses), 500.0)
    np.fill_diagonal(distances, 50.0)
    outbound_variable = np.full((n_warehouses, n_warehouses), 0.5)
    np.fill_diagonal(outbound_variable, 0.05)
    return distances, outbound_variable, np.ones(n_skus)

def random_network(
    rng: np.random.Generator, n_warehouses: int, n_skus: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random network in which every region has a distinct closest warehouse, assigned in shuffled
    order so that the home of region r is not warehouse r. Some foreign rates are equal so that
    the random tie-breaks are exercised.
    """

    home_of_region = rng.permutation(n_warehouses) # Shape: (n_regions,)
    regions = np.arange(n_warehouses)
    distances = rng.uniform(200.0, 1000.0, size=(n_warehouses, n_warehouses))
    distances[home_of_region, regions] = rng.uniform(10.0, 100.0, size=n_warehouses)
    outbound_variable = rng.choice([0.3, 0.5, 0.5, 0.8], size=(n_warehouses, n_warehouses))
    outbound_variable[home_of_region, regions] = 0.05
    sku_weights = rng.choice([0.3, 1.0, 3.0, 10.0, 30.0], size=n_skus)
    return distances, outbound_variable, sku_weights, home_of_region

def random_orders(rng: np.random.Generator, n_regions: int, n_skus: int) -> List[Order]:
    """Random multi-SKU orders (some all-zero), shuffled so they are not sorted by region."""

    orders = []
    for region_id in range(n_regions):
        for _ in range(rng.poisson(3.0)):
            demand = rng.poisson(2.0, size=n_skus) * (rng.random(n_skus) < 0.6)
            orders.append(Order(region_id=region_id, sku_demands=demand.astype(float)))
    rng.shuffle(orders)
    return orders

def demand_per_region(orders: List[Order], n_regions: int, n_skus: int) -> np.ndarray:
    """Total demand per (region, SKU)."""

    demand = np.zeros((n_regions, n_skus))
    for order in orders:
        demand[order.region_id] += order.sku_demands
    return demand

def results_equal(a: AllocationResult, b: AllocationResult) -> bool:
    """True if two allocation results are identical in every field."""

    return all(
        np.array_equal(getattr(a, field), getattr(b, field))
        for field in (
            "fulfillment_matrix", "unfulfilled_demands", "shipment_counts",
            "shipment_quantities", "shipment_quantities_by_sku", "lost_order_counts",
        )
    )


# ============================================================================
# Properties on random networks, orders and inventories
# ============================================================================

def test_random_allocations_are_consistent():
    """
    On random cases: demand is conserved, no warehouse ships more than it holds, orders respect
    max_splits, and the result fields agree with each other.
    """

    rng = np.random.default_rng(0)
    for _ in range(300):
        n_warehouses = int(rng.integers(2, 6))
        n_skus = int(rng.integers(1, 6))
        max_splits = int(rng.integers(0, n_warehouses))
        distances, outbound_variable, sku_weights, _ = random_network(rng, n_warehouses, n_skus)
        allocator = make_allocator(
            distances, outbound_variable, sku_weights, max_splits, seed=int(rng.integers(1 << 30))
        )
        orders = random_orders(rng, n_warehouses, n_skus)
        inventory = rng.integers(0, 15, size=(n_warehouses, n_skus)).astype(float)
        inventory_before = inventory.copy()

        result = allocator.allocate(orders, inventory)

        # The caller's inventory is not modified
        np.testing.assert_array_equal(inventory, inventory_before)

        # Shipped plus unfulfilled equals demand per (region, SKU)
        demand = demand_per_region(orders, n_warehouses, n_skus)
        shipped = result.shipment_quantities_by_sku.sum(axis=0) # Shape: (n_regions, n_skus)
        np.testing.assert_allclose(shipped + result.unfulfilled_demands, demand, atol=1e-9)

        # No warehouse ships more than it holds, and nothing negative is shipped
        assert np.all(result.fulfillment_matrix >= 0)
        assert np.all(result.fulfillment_matrix.sum(axis=0) <= inventory + 1e-9)

        # Every order is served by at most max_splits + 1 warehouses
        warehouses_per_order = (result.fulfillment_matrix.sum(axis=2) > 0).sum(axis=1)
        assert np.all(warehouses_per_order <= max_splits + 1)

        # The result fields agree with the per-order fulfillment matrix
        order_regions = np.array([order.region_id for order in orders], dtype=int)
        for region_id in range(n_warehouses):
            region_rows = result.fulfillment_matrix[order_regions == region_id] # Shape: (n_region_orders, W, K)
            np.testing.assert_allclose(
                result.shipment_quantities_by_sku[:, region_id, :], region_rows.sum(axis=0), atol=1e-9
            )
            np.testing.assert_array_equal(
                result.shipment_counts[:, region_id], (region_rows.sum(axis=2) > 0).sum(axis=0)
            )
            lost_orders = sum(
                1 for order, row in zip(orders, result.fulfillment_matrix)
                if order.region_id == region_id and np.any(order.sku_demands - row.sum(axis=0) > 1e-9)
            )
            assert result.lost_order_counts[region_id] == lost_orders
        np.testing.assert_allclose(
            result.shipment_quantities, result.shipment_quantities_by_sku.sum(axis=2), atol=1e-9
        )

def test_home_warehouse_serves_its_region_first():
    """
    A region whose home warehouse can cover its demand receives nothing from other warehouses,
    and a warehouse ships a SKU to a foreign region only after it has covered all of its home
    region's demand for that SKU.
    """

    rng = np.random.default_rng(1)
    n_covered = n_uncovered = n_foreign_shipments = 0
    for _ in range(300):
        n_warehouses = int(rng.integers(2, 6))
        n_skus = int(rng.integers(1, 6))
        max_splits = int(rng.integers(0, n_warehouses))
        distances, outbound_variable, sku_weights, home_of_region = random_network(
            rng, n_warehouses, n_skus
        )
        allocator = make_allocator(
            distances, outbound_variable, sku_weights, max_splits, seed=int(rng.integers(1 << 30))
        )
        np.testing.assert_array_equal(allocator.home_wh, home_of_region)
        orders = random_orders(rng, n_warehouses, n_skus)
        inventory = rng.integers(0, 25, size=(n_warehouses, n_skus)).astype(float)

        result = allocator.allocate(orders, inventory)
        demand = demand_per_region(orders, n_warehouses, n_skus)
        by_sku = result.shipment_quantities_by_sku # Shape: (n_warehouses, n_regions, n_skus)

        for region_id in range(n_warehouses):
            home = home_of_region[region_id]
            foreign = np.arange(n_warehouses) != home

            # A region covered by its home warehouse gets everything from home and nothing from others
            if np.all(demand[region_id] <= inventory[home]):
                n_covered += 1
                assert np.all(by_sku[foreign, region_id, :] == 0)
                np.testing.assert_allclose(by_sku[home, region_id, :], demand[region_id], atol=1e-9)
                assert np.all(result.unfulfilled_demands[region_id] == 0)
            else:
                n_uncovered += 1

            # The home warehouse ships SKU k abroad only if its home region's SKU k demand is fully met by it
            other_regions = np.arange(n_warehouses) != region_id
            shipped_abroad = by_sku[home, other_regions, :].sum(axis=0) # Shape: (n_skus,)
            for sku in np.flatnonzero(shipped_abroad > 0):
                n_foreign_shipments += 1
                assert by_sku[home, region_id, sku] == pytest.approx(demand[region_id, sku])

    # Make sure the random cases exercised every branch
    assert n_covered > 50 and n_uncovered > 50 and n_foreign_shipments > 50

def test_same_seed_gives_same_allocations():
    """Two allocators reset with the same seed produce identical sequences of allocations."""

    rng = np.random.default_rng(2)
    distances, outbound_variable, sku_weights, _ = random_network(rng, 4, 3)
    allocators = [make_allocator(distances, outbound_variable, sku_weights, seed=42) for _ in range(2)]
    other = make_allocator(distances, outbound_variable, sku_weights, seed=43)

    n_differences = 0
    for _ in range(100):
        orders = random_orders(rng, 4, 3)
        inventory = rng.integers(0, 8, size=(4, 3)).astype(float)
        first, second = (allocator.allocate(orders, inventory) for allocator in allocators)
        assert results_equal(first, second)
        n_differences += not results_equal(first, other.allocate(orders, inventory))

    # A different seed must change some allocations, otherwise the randomness is not used
    assert n_differences > 0


# ============================================================================
# Fairness of the random rules
# ============================================================================

def test_ties_between_foreign_warehouses_are_broken_at_random():
    """
    Region 0's home warehouse is empty and two equally expensive foreign warehouses have stock:
    each of them serves the order in about half of the seeds. A cheaper foreign warehouse always wins.
    """

    distances, outbound_variable, sku_weights = symmetric_network(n_warehouses=3, n_skus=1)
    orders = [Order(region_id=0, sku_demands=np.array([3.0]))]
    inventory = np.array([[0.0], [10.0], [10.0]])

    # Equal foreign rates: the serving warehouse is random
    served_by = []
    for seed in range(400):
        allocator = make_allocator(distances, outbound_variable, sku_weights, seed=seed)
        result = allocator.allocate(orders, inventory)
        served = np.flatnonzero(result.fulfillment_matrix[0, :, 0] > 0)
        assert len(served) == 1 and served[0] in (1, 2)
        served_by.append(served[0])
    share_wh1 = np.mean(np.array(served_by) == 1)
    assert 0.35 < share_wh1 < 0.65

    # Unequal foreign rates: the cheaper warehouse always serves first
    outbound_variable = outbound_variable.copy()
    outbound_variable[2, 0] = 0.8
    for seed in range(50):
        allocator = make_allocator(distances, outbound_variable, sku_weights, seed=seed)
        result = allocator.allocate(orders, inventory)
        assert result.fulfillment_matrix[0, 1, 0] == 3.0 and result.fulfillment_matrix[0, 2, 0] == 0.0

def test_regions_take_turns_in_random_order():
    """
    Regions 0 and 1 both run out at home and compete for the 5 units left at warehouse 2. Each
    region gets them in about half of the seeds (the thesis allocator always served region 0).
    """

    distances, outbound_variable, sku_weights = symmetric_network(n_warehouses=3, n_skus=1)
    orders = [
        Order(region_id=1, sku_demands=np.array([5.0])),
        Order(region_id=0, sku_demands=np.array([5.0])),
    ]
    inventory = np.array([[0.0], [0.0], [5.0]])

    region0_wins = 0
    for seed in range(400):
        allocator = make_allocator(distances, outbound_variable, sku_weights, seed=seed)
        result = allocator.allocate(orders, inventory)
        shipped_to = result.shipment_quantities_by_sku[2, :, 0] # Shape: (n_regions,)
        assert shipped_to.sum() == 5.0 and shipped_to.max() == 5.0
        region0_won = bool(shipped_to[0] == 5.0)
        np.testing.assert_array_equal(result.lost_order_counts, [0, 1, 0] if region0_won else [1, 0, 0])
        region0_wins += region0_won
    assert 0.35 < region0_wins / 400 < 0.65


# ============================================================================
# Wiring into the environment
# ============================================================================

def test_environment_seeds_the_allocator():
    """
    The environment resets the allocator with a generator from its SeedManager: environments
    with the same seed allocate identically, environments with different seeds do not.
    """

    env_config = load_environment_config(SYMMETRIC_ENV)
    envs = [InventoryEnvironment(env_config, seed=seed) for seed in (11, 11, 12)]
    for env in envs:
        env.reset()

    # Regions 0 and 1 compete for warehouse 2's stock, so the region order matters
    sku_demands = np.zeros(env_config.n_skus)
    sku_demands[0] = 5.0
    orders = [Order(region_id=0, sku_demands=sku_demands), Order(region_id=1, sku_demands=sku_demands.copy())]
    inventory = np.zeros((env_config.n_warehouses, env_config.n_skus))
    inventory[2, 0] = 5.0

    results = [[env.demand_allocator.allocate(orders, inventory) for _ in range(30)] for env in envs]
    assert all(results_equal(a, b) for a, b in zip(results[0], results[1]))
    assert not all(results_equal(a, c) for a, c in zip(results[0], results[2]))
