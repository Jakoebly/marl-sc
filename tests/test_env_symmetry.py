"""
Checks that the symmetric environment treats every warehouse and region alike. With the same
base-stock level for every warehouse and SKU, the regions must get the same fill rate and the
warehouses the same cost. With the thesis allocator, which served the regions in index order,
the same check gave fill rates of 0.998 / 0.977 / 0.915 and about 25% more cost at warehouse 2.
"""

import numpy as np
import pytest

from src.config.loader import load_environment_config
from src.environment.envs.multi_env import InventoryEnvironment
from src.experiments.run_baselines import make_bs_optimized_action_fn

SYMMETRIC_ENV = "config_files/environments/env_symmetric_3WH5SKU.yaml"
BASE_STOCK_LEVEL = 63.0
N_EPISODES = 1000
SEED = 2026


@pytest.mark.slow
def test_identical_base_stock_policies_give_identical_outcomes():
    """Per-region fill rates differ by < 0.5 percentage points, per-warehouse costs by < 1%."""

    # Build the environment and the same base-stock policy for every warehouse and SKU
    env_config = load_environment_config(SYMMETRIC_ENV)
    env = InventoryEnvironment(env_config, seed=SEED)
    env.collect_step_info = True
    action_fn = make_bs_optimized_action_fn(
        np.full((env.n_warehouses, env.n_skus), BASE_STOCK_LEVEL)
    )

    # Roll out and accumulate per-warehouse costs and per-region demand and lost demand
    costs = np.zeros((N_EPISODES, env.n_warehouses))
    demand = np.zeros(env.n_regions)
    lost = np.zeros(env.n_regions)
    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, _, truncations, infos = env.step(action_fn(env, obs))
            info = infos[env.agents[0]]
            costs[episode] += (
                info["holding_cost"] + info["penalty_cost"]
                + info["outbound_shipment_cost"] + info["inbound_shipment_cost"]
            )
            demand += info["demand_per_region"].sum(axis=1)
            lost += info["unfulfilled_demands"].sum(axis=1)
            done = all(truncations.values())

    fill_rates = 1.0 - lost / demand
    mean_costs = costs.mean(axis=0)

    assert fill_rates.max() - fill_rates.min() < 0.005, f"Region fill rates differ: {fill_rates}"
    assert mean_costs.max() / mean_costs.min() - 1.0 < 0.01, f"Warehouse costs differ: {mean_costs}"
