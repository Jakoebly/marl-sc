"""Tests for manual rollout and visualization: collect_step_info, rollout(), plot generation."""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from src.environment.environment import InventoryEnvironment
from src.experiments.visualization import (
    generate_visualizations,
    plot_inventory,
    plot_orders,
    plot_cost_breakdown,
    plot_demand_fulfillment,
    plot_shipment_heatmap,
    plot_episode_summary,
)


# ============================================================================
# 1. collect_step_info flag tests (no RLlib needed)
# ============================================================================

class TestCollectStepInfo:
    """Tests that the collect_step_info flag controls info population in step()."""

    def test_default_flag_is_false(self, env_config):
        """collect_step_info should default to False."""
        env = InventoryEnvironment(env_config)
        assert env.collect_step_info is False

    def test_infos_empty_when_flag_off(self, env_config):
        """When collect_step_info=False, infos should contain empty dicts."""
        env = InventoryEnvironment(env_config, seed=42)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent in env.agents:
            assert infos[agent] == {}

    def test_infos_populated_when_flag_on(self, env_config):
        """When collect_step_info=True, infos should contain detailed step data."""
        env = InventoryEnvironment(env_config, seed=42)
        env.collect_step_info = True
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, infos = env.step(actions)

        # All agents should have the same step_info reference
        step_info = infos[env.agents[0]]

        # Check that expected keys are present
        expected_keys = [
            "inventory", "pending_total", "order_quantities",
            "demand_per_region", "fulfilled_per_warehouse", "unfulfilled_demands",
            "shipment_counts", "shipment_quantities", "lost_order_counts",
            "lost_sales",
        ]
        for key in expected_keys:
            assert key in step_info, f"Missing key in step_info: {key}"

    def test_step_info_shapes(self, env_config):
        """Detailed step info arrays should have correct shapes."""
        env = InventoryEnvironment(env_config, seed=42)
        env.collect_step_info = True
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, infos = env.step(actions)
        info = infos[env.agents[0]]

        n_wh = env.n_warehouses
        n_skus = env.n_skus
        n_regions = env.n_regions

        assert info["inventory"].shape == (n_wh, n_skus)
        assert info["pending_total"].shape == (n_wh, n_skus)
        assert info["order_quantities"].shape == (n_wh, n_skus)
        assert info["demand_per_region"].shape == (n_regions, n_skus)
        assert info["fulfilled_per_warehouse"].shape == (n_wh, n_skus)
        assert info["unfulfilled_demands"].shape == (n_regions, n_skus)
        assert info["shipment_counts"].shape == (n_wh, n_regions)
        assert info["shipment_quantities"].shape == (n_wh, n_regions)
        assert info["lost_order_counts"].shape == (n_regions,)
        assert info["lost_sales"].shape == (n_wh, n_skus)

    def test_cost_breakdown_in_step_info(self, env_config):
        """Cost breakdown keys should appear in step_info when collect_step_info=True."""
        env = InventoryEnvironment(env_config, seed=42)
        env.collect_step_info = True
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, infos = env.step(actions)
        info = infos[env.agents[0]]

        cost_keys = ["holding_cost", "penalty_cost", "outbound_shipment_cost", "inbound_shipment_cost"]
        for key in cost_keys:
            assert key in info, f"Missing cost key: {key}"
            assert info[key].shape == (env.n_warehouses,)

    def test_inventory_before_is_pre_step(self, env_config):
        """The 'inventory' in step_info should be the pre-step inventory (before
        any arrivals or fulfillment modify it)."""
        env = InventoryEnvironment(env_config, seed=42)
        env.collect_step_info = True
        obs, _ = env.reset()

        inventory_at_reset = env.inventory.copy()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, infos = env.step(actions)
        info = infos[env.agents[0]]

        # The captured 'inventory' should match what was in env.inventory at the
        # start of step() (i.e., right after the previous reset)
        np.testing.assert_array_equal(info["inventory"], inventory_at_reset)


# ============================================================================
# 2. Manual rollout tests (needs trained algorithm)
# ============================================================================

class TestRollout:
    """Tests for the rollout() method on BaseAlgorithmWrapper."""

    def test_rollout_returns_list_of_episodes(self, trained_algorithm):
        """rollout() should return a list with one dict per episode."""
        ctx = trained_algorithm
        rollout_env = InventoryEnvironment(
            ctx["env_config"], seed=ctx["eval_seed"],
            env_meta={"data_mode": "val"},
        )
        episodes = ctx["algorithm"].rollout(rollout_env, num_episodes=2)
        assert isinstance(episodes, list)
        assert len(episodes) == 2

    def test_rollout_episode_has_expected_keys(self, trained_algorithm):
        """Each episode dict should contain the expected data keys."""
        ctx = trained_algorithm
        rollout_env = InventoryEnvironment(
            ctx["env_config"], seed=ctx["eval_seed"],
            env_meta={"data_mode": "val"},
        )
        episodes = ctx["algorithm"].rollout(rollout_env, num_episodes=1)
        ep = episodes[0]

        expected_keys = [
            "actions_raw", "rewards",
            "inventory", "pending_total", "order_quantities",
            "demand_per_region", "fulfilled_per_warehouse", "unfulfilled_demands",
            "shipment_counts", "shipment_quantities", "lost_order_counts",
            "lost_sales",
            "holding_cost", "penalty_cost", "outbound_shipment_cost", "inbound_shipment_cost",
        ]
        for key in expected_keys:
            assert key in ep, f"Missing key in episode data: {key}"

    def test_rollout_episode_length(self, trained_algorithm):
        """Episode arrays should have T = episode_length along the first axis."""
        ctx = trained_algorithm
        rollout_env = InventoryEnvironment(
            ctx["env_config"], seed=ctx["eval_seed"],
            env_meta={"data_mode": "val"},
        )
        episodes = ctx["algorithm"].rollout(rollout_env, num_episodes=1)
        ep = episodes[0]

        expected_T = ctx["env_config"].episode_length
        assert ep["rewards"].shape[0] == expected_T
        assert ep["inventory"].shape[0] == expected_T
        assert ep["actions_raw"].shape[0] == expected_T

    def test_rollout_array_shapes(self, trained_algorithm):
        """Rollout arrays should have correct (T, ...) shapes."""
        ctx = trained_algorithm
        cfg = ctx["env_config"]
        rollout_env = InventoryEnvironment(
            cfg, seed=ctx["eval_seed"], env_meta={"data_mode": "val"},
        )
        episodes = ctx["algorithm"].rollout(rollout_env, num_episodes=1)
        ep = episodes[0]

        T = cfg.episode_length
        n_wh = cfg.n_warehouses
        n_skus = cfg.n_skus
        n_reg = cfg.n_regions

        assert ep["inventory"].shape == (T, n_wh, n_skus)
        assert ep["actions_raw"].shape == (T, n_wh, n_skus)
        assert ep["order_quantities"].shape == (T, n_wh, n_skus)
        assert ep["rewards"].shape == (T, n_wh)
        assert ep["demand_per_region"].shape == (T, n_reg, n_skus)
        assert ep["shipment_counts"].shape == (T, n_wh, n_reg)
        assert ep["holding_cost"].shape == (T, n_wh)

    def test_rollout_deterministic_with_same_seed(self, trained_algorithm):
        """Two rollouts with the same eval_seed should produce identical data."""
        ctx = trained_algorithm

        all_rewards = []
        for _ in range(2):
            rollout_env = InventoryEnvironment(
                ctx["env_config"], seed=ctx["eval_seed"],
                env_meta={"data_mode": "val"},
            )
            episodes = ctx["algorithm"].rollout(rollout_env, num_episodes=1)
            all_rewards.append(episodes[0]["rewards"])

        np.testing.assert_array_equal(all_rewards[0], all_rewards[1])

    def test_rollout_disables_flag_after(self, trained_algorithm):
        """After rollout(), collect_step_info should be set back to False."""
        ctx = trained_algorithm
        rollout_env = InventoryEnvironment(
            ctx["env_config"], seed=ctx["eval_seed"],
            env_meta={"data_mode": "val"},
        )
        ctx["algorithm"].rollout(rollout_env, num_episodes=1)
        assert rollout_env.collect_step_info is False


# ============================================================================
# 3. Visualization plot generation tests
# ============================================================================

def _make_dummy_episode(T=20, n_wh=3, n_skus=5, n_regions=3):
    """Creates a synthetic episode data dict for testing plot functions."""
    rng = np.random.default_rng(0)
    return {
        "inventory": rng.random((T, n_wh, n_skus)) * 50,
        "pending_total": rng.random((T, n_wh, n_skus)) * 10,
        "actions_raw": rng.uniform(-1, 1, (T, n_wh, n_skus)),
        "order_quantities": rng.random((T, n_wh, n_skus)) * 25,
        "demand_per_region": rng.random((T, n_regions, n_skus)) * 5,
        "fulfilled_per_warehouse": rng.random((T, n_wh, n_skus)) * 5,
        "unfulfilled_demands": rng.random((T, n_regions, n_skus)) * 2,
        "shipment_counts": rng.integers(0, 5, (T, n_wh, n_regions)).astype(float),
        "shipment_quantities": rng.random((T, n_wh, n_regions)) * 10,
        "lost_order_counts": rng.integers(0, 3, (T, n_regions)).astype(float),
        "lost_sales": rng.random((T, n_wh, n_skus)) * 2,
        "holding_cost": rng.random((T, n_wh)) * 5,
        "penalty_cost": rng.random((T, n_wh)) * 10,
        "outbound_shipment_cost": rng.random((T, n_wh)) * 8,
        "inbound_shipment_cost": rng.random((T, n_wh)) * 6,
        "rewards": -rng.random((T, n_wh)) * 30,
    }


class TestVisualizationPlots:
    """Tests that visualization functions create plot files without errors."""

    def test_plot_inventory(self):
        """plot_inventory should create a PNG file."""
        episode = _make_dummy_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_inventory(episode, Path(tmp_dir))
            assert (Path(tmp_dir) / "01_inventory.png").exists()

    def test_plot_orders(self):
        """plot_orders should create a PNG file."""
        episode = _make_dummy_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_orders(episode, Path(tmp_dir))
            assert (Path(tmp_dir) / "02_orders.png").exists()

    def test_plot_cost_breakdown(self):
        """plot_cost_breakdown should create a PNG file."""
        episode = _make_dummy_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_cost_breakdown(episode, Path(tmp_dir))
            assert (Path(tmp_dir) / "03_cost_breakdown.png").exists()

    def test_plot_demand_fulfillment(self):
        """plot_demand_fulfillment should create a PNG file."""
        episode = _make_dummy_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_demand_fulfillment(episode, Path(tmp_dir))
            assert (Path(tmp_dir) / "04_demand_fulfillment.png").exists()

    def test_plot_shipment_heatmap(self):
        """plot_shipment_heatmap should create a PNG file."""
        episode = _make_dummy_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_shipment_heatmap(episode, Path(tmp_dir))
            assert (Path(tmp_dir) / "05_shipment_heatmap.png").exists()

    def test_plot_episode_summary(self):
        """plot_episode_summary should create a PNG file for multiple episodes."""
        episodes = [_make_dummy_episode() for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_episode_summary(episodes, Path(tmp_dir))
            assert (Path(tmp_dir) / "06_episode_summary.png").exists()

    def test_generate_visualizations_all_plots(self):
        """generate_visualizations should create all expected plot files."""
        episodes = [_make_dummy_episode() for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_visualizations(episodes, tmp_dir)
            expected = [
                "01_inventory.png", "02_orders.png", "03_cost_breakdown.png",
                "04_demand_fulfillment.png", "05_shipment_heatmap.png",
                "06_episode_summary.png",
            ]
            for name in expected:
                assert (Path(tmp_dir) / name).exists(), f"Missing: {name}"

    def test_generate_visualizations_single_episode(self):
        """With a single episode, the summary plot should be skipped."""
        episodes = [_make_dummy_episode()]
        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_visualizations(episodes, tmp_dir)
            # 5 detail plots should exist
            assert (Path(tmp_dir) / "01_inventory.png").exists()
            assert (Path(tmp_dir) / "05_shipment_heatmap.png").exists()
            # Summary plot should NOT exist for a single episode
            assert not (Path(tmp_dir) / "06_episode_summary.png").exists()

    def test_plot_files_are_nonempty(self):
        """Generated plot files should have non-zero size."""
        episode = _make_dummy_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_inventory(episode, Path(tmp_dir))
            file_path = Path(tmp_dir) / "01_inventory.png"
            assert file_path.stat().st_size > 0


# ============================================================================
# 4. End-to-end rollout + visualization test (needs trained algorithm)
# ============================================================================

class TestRolloutVisualizationIntegration:
    """End-to-end test: rollout real episodes, then visualize them."""

    def test_real_rollout_generates_valid_visualizations(self, trained_algorithm):
        """A real rollout (with trained model) should produce data that
        the visualization module can plot without errors."""
        ctx = trained_algorithm

        # Run rollout
        rollout_env = InventoryEnvironment(
            ctx["env_config"], seed=ctx["eval_seed"],
            env_meta={"data_mode": "val"},
        )
        episodes = ctx["algorithm"].rollout(rollout_env, num_episodes=2)

        # Generate visualizations from real data
        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_visualizations(episodes, tmp_dir)
            expected = [
                "01_inventory.png", "02_orders.png", "03_cost_breakdown.png",
                "04_demand_fulfillment.png", "05_shipment_heatmap.png",
                "06_episode_summary.png",
            ]
            for name in expected:
                path = Path(tmp_dir) / name
                assert path.exists(), f"Missing: {name}"
                assert path.stat().st_size > 0, f"Empty file: {name}"
