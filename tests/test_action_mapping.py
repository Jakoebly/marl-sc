"""
Tests for the demand-centered action mapping (``InventoryEnvironment._rescale_actions_to_quantities``)
and its inverse in the baselines (``_qty_to_action``). For last home demand D and range Delta
(``max_quantity_adjustment``): action -1 orders nothing, 0 orders D, and +1 orders D + Delta.
"""

import numpy as np
import pytest

from src.config.loader import load_environment_config
from src.environment.envs.multi_env import InventoryEnvironment
from src.experiments.run_baselines import _qty_to_action

# SKUHetero has the smallest ranges (max_quantity_adjustment = [60, 20, 7, 2, 2])
SKU_HETERO_ENV = "config_files/environments/env_sku_hetero_3WH5SKU.yaml"
LAST_DEMANDS = [0, 1, 2, 3, 5, 8, 13, 44, 61, 150]


@pytest.fixture
def env() -> InventoryEnvironment:
    """SKUHetero environment after a reset."""

    env = InventoryEnvironment(load_environment_config(SKU_HETERO_ENV), seed=0)
    env.reset()
    return env

def set_last_demand(env: InventoryEnvironment, demand) -> None:
    """Sets the last incoming home demand of every warehouse, shape (n_warehouses, n_skus)."""

    env._incoming_demand_home = np.broadcast_to(
        np.asarray(demand, dtype=np.float32), (env.n_warehouses, env.n_skus)
    ).copy()

def order_quantities(env: InventoryEnvironment, actions) -> np.ndarray:
    """Order quantities of every warehouse for actions of shape (n_warehouses, n_skus)."""

    actions = np.broadcast_to(np.asarray(actions, dtype=np.float32), (env.n_warehouses, env.n_skus))
    quantities = env._rescale_actions_to_quantities(
        {agent_id: actions[i] for i, agent_id in enumerate(env.agents)}
    )
    return np.stack([quantities[agent_id] for agent_id in env.agents])


def test_boundary_actions(env):
    """Action -1 orders nothing for any last demand, 0 orders the last demand, +1 adds Delta."""

    delta = env.max_quantity_adjustment
    for demand in LAST_DEMANDS:
        set_last_demand(env, demand)
        np.testing.assert_array_equal(order_quantities(env, -1.0), 0.0)
        np.testing.assert_array_equal(order_quantities(env, 0.0), demand)
        np.testing.assert_array_equal(order_quantities(env, 1.0), demand + np.broadcast_to(delta, (env.n_warehouses, env.n_skus)))

    # Different last demands per warehouse and SKU
    demand = np.random.default_rng(0).integers(0, 200, size=(env.n_warehouses, env.n_skus))
    set_last_demand(env, demand)
    np.testing.assert_array_equal(order_quantities(env, -1.0), 0.0)
    np.testing.assert_array_equal(order_quantities(env, 0.0), demand)
    np.testing.assert_array_equal(order_quantities(env, 1.0), demand + delta)

def test_every_quantity_up_to_demand_plus_range_is_reachable(env):
    """
    Over the action range, the reachable orders are exactly 0, 1, ..., D + Delta. In the thesis
    mapping the smallest order was max(0, D - Delta), e.g. 3 for Delta = 2 and D = 5.
    """

    delta = env.max_quantity_adjustment.astype(int)
    grid = np.linspace(-1.0, 1.0, 4001)
    for demand in LAST_DEMANDS:
        set_last_demand(env, demand)
        reachable = np.stack([order_quantities(env, a)[0] for a in grid]) # Shape: (n_grid, n_skus)
        for sku in range(env.n_skus):
            assert set(reachable[:, sku].astype(int)) == set(range(demand + delta[sku] + 1))

def test_orders_grow_with_the_action_and_are_never_negative(env):
    """Order quantities are non-negative integers and non-decreasing in the action."""

    grid = np.linspace(-1.0, 1.0, 801)
    for demand in LAST_DEMANDS:
        set_last_demand(env, demand)
        quantities = np.stack([order_quantities(env, a) for a in grid]) # Shape: (n_grid, n_warehouses, n_skus)
        assert np.all(quantities >= 0)
        np.testing.assert_array_equal(quantities, np.round(quantities))
        assert np.all(np.diff(quantities, axis=0) >= 0)

def test_inverse_returns_the_requested_quantity(env):
    """
    ``_qty_to_action`` followed by the env mapping returns the requested quantity whenever it
    lies in [0, D + Delta]. Larger quantities are capped at D + Delta, negative ones give 0.
    """

    delta = env.max_quantity_adjustment
    for demand in LAST_DEMANDS:
        set_last_demand(env, demand)
        upper = demand + delta # Shape: (n_skus,)
        for wh_idx, agent_id in enumerate(env.agents):
            for qty in range(int(upper.max()) + 3):
                requested = np.minimum(float(qty), upper)
                action = _qty_to_action(env, wh_idx, requested)
                assert np.all(np.abs(action) <= 1.0)
                ordered = env._rescale_actions_to_quantities({agent_id: action})[agent_id]
                np.testing.assert_array_equal(ordered, requested)

            # Out-of-range requests are capped at D + Delta and floored at zero
            for requested, expected in ((upper + 5.0, upper), (np.full(env.n_skus, -3.0), np.zeros(env.n_skus))):
                action = _qty_to_action(env, wh_idx, requested)
                ordered = env._rescale_actions_to_quantities({agent_id: action})[agent_id]
                np.testing.assert_array_equal(ordered, expected)

            # Fractional requests (base-stock gaps) are rounded to the nearest integer
            requested = np.minimum(np.array([0.3, 1.7, 2.2, 4.8, 6.6]), upper)
            action = _qty_to_action(env, wh_idx, requested)
            ordered = env._rescale_actions_to_quantities({agent_id: action})[agent_id]
            np.testing.assert_array_equal(ordered, np.round(requested))
