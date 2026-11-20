"""
Tests for the mixed reward scope of ``CostRewardCalculator``: each warehouse receives
``team_weight * team reward + (1 - team_weight) * own reward``. Also checks that the lost sales
handler changes only the agent part of the reward, never the team reward.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.config.loader import load_environment_config
from src.config.schema import LostSalesClosest, LostSalesShipment, RewardCostParams
from src.experiments.run_baselines import (
    expected_home_demand,
    make_bs_optimized_action_fn,
    make_train_env,
    run_rollout,
)

COMBINED_HETERO_ENV = "config_files/environments/env_combined_hetero_3WH5SKU.yaml"


def config_with(scope: str, team_weight=None, lost_sales_rule: str = "shipment"):
    """CombinedHetero config with the given reward scope and lost sales handler."""

    env_config = load_environment_config(COMBINED_HETERO_ENV).model_copy(deep=True)
    params = env_config.components.reward_calculator.params
    env_config.components.reward_calculator.params = RewardCostParams(
        **{**params.model_dump(), "scope": scope, "team_weight": team_weight}
    )
    env_config.components.lost_sales_handler = (
        LostSalesClosest(type="closest") if lost_sales_rule == "closest" else LostSalesShipment(type="shipment")
    )
    return env_config

def rollout_rewards(env_config, base_stock_levels, n_episodes: int = 2):
    """Per-step warehouse rewards and per-SKU channels of a base-stock policy on fixed episodes."""

    env = make_train_env(env_config, seed=3)
    env.reward_calculator.decompose = True
    episodes = run_rollout(
        env, make_bs_optimized_action_fn(base_stock_levels), n_episodes, collect_step_info=False
    )
    rewards = np.stack([episode["rewards"] for episode in episodes]) # Shape: (n_episodes, T, W)
    reward_per_sku = np.stack([episode["reward_per_sku"] for episode in episodes]) # Shape: (n_episodes, T, W, K)
    return rewards, reward_per_sku

def demand_over_lead_time(env_config) -> np.ndarray:
    """Expected home demand over the lead time plus one step. Shape: (n_warehouses, n_skus)."""

    lead_times = np.array(
        env_config.components.lead_time_sampler.params.expected_lead_times, dtype=float
    )
    return expected_home_demand(env_config) * (lead_times + 1.0)


def test_team_weight_is_required_for_mixed_and_only_there():
    """The schema asks for a team weight in [0, 1] with scope 'mixed' and rejects it otherwise."""

    base = dict(scale_factor=0.01, cost_weights=[0.25, 0.25, 0.25, 0.25])
    assert RewardCostParams(scope="mixed", team_weight=0.3, **base).team_weight == 0.3
    with pytest.raises(ValidationError):
        RewardCostParams(scope="mixed", **base)
    with pytest.raises(ValidationError):
        RewardCostParams(scope="mixed", team_weight=1.5, **base)
    with pytest.raises(ValidationError):
        RewardCostParams(scope="agent", team_weight=0.3, **base)

@pytest.mark.parametrize("team_weight", [0.0, 0.3, 1.0])
def test_mixed_reward_blends_own_and_team_rewards(team_weight):
    """
    On identical episodes, the mixed reward equals team_weight * team + (1 - team_weight) * own,
    for the warehouse rewards and for the per-SKU channels. Weight 0 gives the agent reward and
    weight 1 the team reward.
    """

    levels = 1.1 * demand_over_lead_time(load_environment_config(COMBINED_HETERO_ENV))
    agent, agent_per_sku = rollout_rewards(config_with("agent"), levels)
    team, _ = rollout_rewards(config_with("team"), levels)
    mixed, mixed_per_sku = rollout_rewards(config_with("mixed", team_weight), levels)

    np.testing.assert_allclose(mixed, team_weight * team + (1.0 - team_weight) * agent, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        mixed_per_sku,
        team_weight * agent_per_sku.sum(axis=2, keepdims=True) + (1.0 - team_weight) * agent_per_sku,
        rtol=1e-12, atol=1e-9,
    )
    np.testing.assert_allclose(mixed_per_sku.sum(axis=3), mixed, rtol=1e-12, atol=1e-9)

def test_lost_sales_rule_changes_only_the_agent_part():
    """
    The lost sales handler moves stock-out penalties between warehouses but never changes their
    total, so the team reward is the same under both rules while the agent rewards are not.
    Warehouse 0 stocks for everyone here, so most stock-outs happen in the other regions.
    """

    env_config = load_environment_config(COMBINED_HETERO_ENV)
    levels = demand_over_lead_time(env_config) * np.array([[3.0], [0.0], [0.0]])

    team_shipment, _ = rollout_rewards(config_with("team", lost_sales_rule="shipment"), levels)
    team_closest, _ = rollout_rewards(config_with("team", lost_sales_rule="closest"), levels)
    np.testing.assert_allclose(team_shipment, team_closest, rtol=1e-12, atol=1e-9)

    agent_shipment, _ = rollout_rewards(config_with("agent", lost_sales_rule="shipment"), levels)
    agent_closest, _ = rollout_rewards(config_with("agent", lost_sales_rule="closest"), levels)
    np.testing.assert_allclose(agent_shipment.sum(axis=2), agent_closest.sum(axis=2), rtol=1e-12, atol=1e-9)
    assert not np.allclose(agent_shipment, agent_closest)

    # Under the shipment rule the stocked warehouse pays for stock-outs in the regions it ships to;
    # under the closest rule the empty warehouses pay for the stock-outs in their own regions
    own_shipment = -agent_shipment.sum(axis=1).mean(axis=0) # Shape: (W,)
    own_closest = -agent_closest.sum(axis=1).mean(axis=0) # Shape: (W,)
    assert own_shipment[0] > own_closest[0]
    assert np.all(own_closest[1:] > own_shipment[1:])
