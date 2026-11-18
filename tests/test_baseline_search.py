"""
Tests for the simulation-optimized base-stock baselines in ``run_baselines.py``: the search
bounds ``ceil(3 * (L + 1) * mu + 10)``, the per-SKU costs that BS-PerSKU optimizes, the per-SKU
search itself, and the objective of BS-Independent.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import skopt

from src.config.loader import load_environment_config
from src.experiments.run_baselines import (
    base_stock_upper_bounds,
    expected_home_demand,
    make_bs_optimized_action_fn,
    make_train_env,
    run_bs_independent_optimization,
    run_bs_joint_optimization,
    run_bs_per_sku_optimization,
    run_rollout,
)
from src.utils.seed_manager import EXPERIMENT_SEEDS, SeedManager

SYMMETRIC_ENV = "config_files/environments/env_symmetric_3WH5SKU.yaml"
SKU_HETERO_ENV = "config_files/environments/env_sku_hetero_3WH5SKU.yaml"
COMBINED_HETERO_ENV = "config_files/environments/env_combined_hetero_3WH5SKU.yaml"
TEAM_ENV = "config_files/environments/E3.4_team/env_combined_hetero_team_mappo_ps.yaml"


def demand_over_lead_time(env_config) -> np.ndarray:
    """Expected home demand over the lead time plus one step, mu * (L + 1). Shape: (n_warehouses, n_skus)."""

    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times, dtype=float
    )
    return expected_home_demand(env_config) * (lead_times + 1.0)


def test_upper_bounds_follow_each_skus_demand():
    """
    The bound is ceil(3 * (L + 1) * mu + 10), so it is small for slow SKUs and above 200, the
    thesis' fixed bound, for the fast SKU of SKUHetero (expected demand 44.5 per step, lead time 3).
    """

    np.testing.assert_array_equal(
        base_stock_upper_bounds(load_environment_config(SYMMETRIC_ENV)), np.full((3, 5), 171.0)
    )
    np.testing.assert_array_equal(
        base_stock_upper_bounds(load_environment_config(SKU_HETERO_ENV)),
        np.tile([544.0, 171.0, 64.0, 27.0, 16.0], (3, 1)),
    )

    # CombinedHetero: every warehouse has its own demand level
    env_config = load_environment_config(COMBINED_HETERO_ENV)
    bounds = base_stock_upper_bounds(env_config)
    assert bounds.shape == (env_config.n_warehouses, env_config.n_skus)
    assert np.all(bounds > 2.0 * demand_over_lead_time(env_config))

def test_expected_home_demand_accepts_parameters_shared_by_all_regions():
    """Scalar demand parameters give the same expected demand as the per-region lists of the symmetric env."""

    env_config = load_environment_config(SYMMETRIC_ENV)
    shared = env_config.model_copy(deep=True)
    shared.components.demand_sampler.params = {
        "lambda_orders": 4.0, "probability_skus": 0.667, "lambda_quantity": 5.0,
    }
    np.testing.assert_allclose(expected_home_demand(shared), expected_home_demand(env_config))

@pytest.mark.parametrize("path", [SKU_HETERO_ENV, TEAM_ENV], ids=["agent_scope", "team_scope"])
def test_per_sku_costs_add_up_to_the_joint_objective(path):
    """
    With ``decompose`` set, ``run_rollout`` records the per-SKU reward channels. Summed over
    SKUs they equal the summed warehouse rewards that BS-Optimized maximizes.
    """

    env_config = load_environment_config(path)
    action_fn = make_bs_optimized_action_fn(1.1 * demand_over_lead_time(env_config))

    env = make_train_env(env_config, seed=5)
    assert "reward_per_sku" not in run_rollout(env, action_fn, 1, collect_step_info=False)[0]

    env.reward_calculator.decompose = True
    for episode in run_rollout(env, action_fn, 3, collect_step_info=False):
        assert episode["reward_per_sku"].shape == (env.episode_length, env.n_warehouses, env.n_skus)
        np.testing.assert_allclose(
            episode["reward_per_sku"].sum(axis=2), episode["rewards"], rtol=1e-9, atol=1e-9
        )

def test_per_sku_search_stays_within_the_bounds_and_is_reproducible():
    """A small per-SKU search returns one level per (warehouse, SKU) inside the bounds, deterministically."""

    env_config = load_environment_config(SYMMETRIC_ENV)
    upper_bounds = base_stock_upper_bounds(env_config)
    kwargs = dict(upper_bounds=upper_bounds, n_calls=6, n_random_starts=3, n_obj_episodes=2)

    S_star, convergence = run_bs_per_sku_optimization(env_config, optimization_seed=11, **kwargs)
    assert S_star.shape == upper_bounds.shape
    assert np.all((S_star >= 0.0) & (S_star <= upper_bounds))
    assert list(convergence) == [f"sku{k}" for k in range(env_config.n_skus)]
    for curve in convergence.values():
        assert len(curve) == 6 and np.all(np.diff(curve) <= 0.0)

    S_again, _ = run_bs_per_sku_optimization(env_config, optimization_seed=11, **kwargs)
    np.testing.assert_array_equal(S_again, S_star)

def test_joint_and_independent_searches_use_the_scaled_bounds(monkeypatch):
    """BS-Optimized and BS-Independent hand the per-(warehouse, SKU) bounds to the optimizer."""

    searched = []

    def fake_gp_minimize(objective, dimensions, **kwargs):
        searched.append(list(dimensions))
        x = [high for _, high in dimensions]
        return SimpleNamespace(x=x, fun=0.0, func_vals=np.zeros(1))

    monkeypatch.setattr(skopt, "gp_minimize", fake_gp_minimize)
    env_config = load_environment_config(COMBINED_HETERO_ENV)
    upper_bounds = base_stock_upper_bounds(env_config)

    run_bs_joint_optimization(env_config, 1, upper_bounds, n_calls=1, n_random_starts=1, n_obj_episodes=1)
    run_bs_independent_optimization(
        env_config, 1, demand_over_lead_time(env_config), upper_bounds,
        n_rounds=1, n_calls_per_wh=1, n_random_starts_per_wh=1, n_obj_episodes=1,
    )

    assert searched[0] == [(0.0, ub) for ub in upper_bounds.ravel()]
    for wh in range(env_config.n_warehouses):
        assert searched[1 + wh] == [(0.0, ub) for ub in upper_bounds[wh]]

def test_bs_independent_scores_each_warehouse_on_its_own_reward(monkeypatch):
    """
    Under agent-scoped rewards, each warehouse's search objective is its own reward, not the
    network's. A fake optimizer evaluates one candidate per warehouse; the test replays the same
    candidates on the same episodes and compares.
    """

    calls = []

    def fake_gp_minimize(objective, dimensions, **kwargs):
        x = [0.5 * high for _, high in dimensions]
        value = objective(x)
        calls.append((x, value))
        return SimpleNamespace(x=x, fun=value, func_vals=np.array([value]))

    monkeypatch.setattr(skopt, "gp_minimize", fake_gp_minimize)
    env_config = load_environment_config(COMBINED_HETERO_ENV)
    assert env_config.components.reward_calculator.params.scope == "agent"
    S_init = demand_over_lead_time(env_config)
    run_bs_independent_optimization(
        env_config, 3, S_init, base_stock_upper_bounds(env_config),
        n_rounds=1, n_calls_per_wh=1, n_random_starts_per_wh=1, n_obj_episodes=2,
    )

    wh_seeds = SeedManager(root_seed=3, seed_registry=EXPERIMENT_SEEDS).spawn_child_seeds(
        "train", env_config.n_warehouses
    )
    S = S_init.copy()
    for wh, (x, value) in enumerate(calls):
        S[wh] = x
        episodes = run_rollout(
            make_train_env(env_config, wh_seeds[wh]), make_bs_optimized_action_fn(S), 2,
            collect_step_info=False,
        )
        own_reward = np.mean([episode["rewards"][:, wh].sum() for episode in episodes])
        network_reward = np.mean([episode["rewards"].sum() for episode in episodes])
        assert value == pytest.approx(-own_reward)
        assert value != pytest.approx(-network_reward)
