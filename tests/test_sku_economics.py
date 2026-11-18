"""
Checks the stocking economics of the SKU-heterogeneous environment: for every SKU, holding some
stock must be cheaper than holding none. With the thesis penalty of 5 per unit for every SKU, the
two heaviest SKUs (weights 10 and 30) cost more to buy than to lose, so their best stock level
was zero, and the demand-centered action range could not even order zero for them.
"""

import numpy as np
import pytest

from src.config.loader import load_environment_config
from src.config.schema import EnvironmentConfig
from src.environment.envs.multi_env import InventoryEnvironment
from src.experiments.run_baselines import expected_home_demand, make_bs_optimized_action_fn

SKU_HETERO_ENV = "config_files/environments/env_sku_hetero_3WH5SKU.yaml"
MULTIPLIERS = np.arange(0.0, 2.01, 0.25)
N_EPISODES = 100
SEED = 31


def per_sku_cost(env_config: EnvironmentConfig, base_stock_levels: np.ndarray) -> np.ndarray:
    """
    Mean network cost per step of each SKU (summed over warehouses) under a base-stock policy,
    read from the reward calculator's per-SKU channels. Shape: (n_skus,).
    """

    env = InventoryEnvironment(env_config, seed=SEED)
    env.reward_calculator.decompose = True
    scale = env.reward_calculator.scale_factor or 1.0
    action_fn = make_bs_optimized_action_fn(base_stock_levels)

    cost = np.zeros(env.n_skus)
    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, _, truncations, _ = env.step(action_fn(env, obs))
            cost -= env.reward_calculator._reward_per_sku.sum(axis=0) / scale
            done = all(truncations.values())
    return cost / (N_EPISODES * env.episode_length)


@pytest.mark.slow
def test_every_sku_is_worth_stocking():
    """
    Sweeps one order-up-to level per SKU, m * mu * (L + 1) for m in 0, 0.25, ..., 2, and asserts
    that the cheapest level is above zero for every SKU. All SKUs are swept at once because they
    do not interact (no capacity limit, zero fixed costs).
    """

    env_config = load_environment_config(SKU_HETERO_ENV)
    mu = expected_home_demand(env_config)
    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times, dtype=float
    )

    costs = np.array([
        per_sku_cost(env_config, m * mu * (lead_times + 1)) for m in MULTIPLIERS
    ]) # Shape: (n_levels, n_skus)
    best_multipliers = MULTIPLIERS[costs.argmin(axis=0)]

    table = "\n".join(f"m={m:4.2f}: {np.round(row, 2)}" for m, row in zip(MULTIPLIERS, costs))
    assert np.all(best_multipliers > 0), (
        f"Best stock level is zero for SKUs {np.flatnonzero(best_multipliers == 0)}.\n"
        f"Per-SKU cost per step:\n{table}"
    )
