"""Tests for the evaluation system: Algorithm.evaluate(), EvaluationRunner, checkpoint save/load."""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from src.environment.environment import InventoryEnvironment
from src.algorithms.registry import get_algorithm
from src.experiments.runner import EvaluationRunner
from src.utils.seed_manager import split_seed


# ============================================================================
# 1. Algorithm-level evaluate() tests
# ============================================================================

class TestAlgorithmEvaluate:
    """Tests for the evaluate() method on the algorithm wrapper."""

    def test_evaluate_returns_dict(self, trained_algorithm):
        """evaluate() should return a dictionary of metrics."""
        algo = trained_algorithm["algorithm"]
        result = algo.evaluate(eval_episodes=2)
        assert isinstance(result, dict)

    def test_evaluate_custom_episodes(self, trained_algorithm):
        """evaluate(eval_episodes=N) should run the requested number of episodes."""
        algo = trained_algorithm["algorithm"]
        # Just verify it doesn't crash with a custom episode count
        result = algo.evaluate(eval_episodes=1)
        assert isinstance(result, dict)

    def test_evaluate_restores_duration(self, trained_algorithm):
        """evaluate() should restore the original evaluation_duration after running."""
        algo = trained_algorithm["algorithm"]
        original = algo.trainer.config.evaluation_duration
        algo.evaluate(eval_episodes=3)
        assert algo.trainer.config.evaluation_duration == original


# ============================================================================
# 2. Checkpoint save/load tests
# ============================================================================

class TestCheckpointSaveLoad:
    """Tests for saving and loading model checkpoints."""

    def test_checkpoint_directory_exists(self, trained_algorithm):
        """The saved checkpoint path should exist on disk."""
        ckpt_path = trained_algorithm["checkpoint_path"]
        assert Path(ckpt_path).exists()

    def test_load_checkpoint_and_evaluate(self, trained_algorithm):
        """Loading a checkpoint into a fresh algorithm should allow evaluation."""
        ctx = trained_algorithm
        fresh_env = InventoryEnvironment(ctx["env_config"])
        fresh_algo = get_algorithm(
            ctx["algo_config"].name, fresh_env, ctx["algo_config"],
            train_seed=None, eval_seed=ctx["eval_seed"],
        )
        fresh_algo.load_checkpoint(ctx["checkpoint_path"])
        result = fresh_algo.evaluate(eval_episodes=1)
        assert isinstance(result, dict)
        fresh_algo.trainer.stop()

    def test_loaded_checkpoint_produces_same_eval_results(self, trained_algorithm):
        """Two algorithms loaded from the same checkpoint with the same eval_seed
        should produce identical evaluation rewards (deterministic rollout)."""
        ctx = trained_algorithm

        results = []
        algorithms = []
        for _ in range(2):
            env = InventoryEnvironment(ctx["env_config"], seed=ctx["eval_seed"])
            algo = get_algorithm(
                ctx["algo_config"].name, env, ctx["algo_config"],
                train_seed=None, eval_seed=ctx["eval_seed"],
            )
            algo.load_checkpoint(ctx["checkpoint_path"])

            # Use manual rollout for deterministic comparison
            rollout_env = InventoryEnvironment(
                ctx["env_config"], seed=ctx["eval_seed"],
                env_meta={"data_mode": "val"},
            )
            episodes = algo.rollout(rollout_env, num_episodes=1)
            results.append(episodes[0]["rewards"].sum())
            algorithms.append(algo)

        # Both runs should produce the same total reward
        assert results[0] == pytest.approx(results[1])

        for algo in algorithms:
            algo.trainer.stop()


# ============================================================================
# 3. EvaluationRunner tests
# ============================================================================

class TestEvaluationRunner:
    """Tests for the EvaluationRunner orchestrator."""

    def test_runner_without_visualize(self, trained_algorithm):
        """EvaluationRunner with visualize=False should return standard RLlib metrics."""
        ctx = trained_algorithm
        tmp_dir = tempfile.mkdtemp(prefix="eval_test_")
        try:
            runner = EvaluationRunner(
                env_config=ctx["env_config"],
                algorithm_config=ctx["algo_config"],
                checkpoint_dir=ctx["checkpoint_path"],
                output_dir=tmp_dir,
                eval_episodes=1,
                root_seed=ctx["root_seed"],
                visualize=False,
            )
            result = runner.run()
            assert isinstance(result, dict)
            runner.algorithm.trainer.stop()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_runner_with_visualize(self, trained_algorithm):
        """EvaluationRunner with visualize=True should create visualization files."""
        ctx = trained_algorithm
        tmp_dir = tempfile.mkdtemp(prefix="eval_viz_test_")
        try:
            runner = EvaluationRunner(
                env_config=ctx["env_config"],
                algorithm_config=ctx["algo_config"],
                checkpoint_dir=ctx["checkpoint_path"],
                output_dir=tmp_dir,
                eval_episodes=2,
                root_seed=ctx["root_seed"],
                visualize=True,
            )
            result = runner.run()

            # Check result structure
            assert "evaluation" in result
            assert "episode_reward_mean" in result["evaluation"]
            assert "num_episodes" in result["evaluation"]
            assert result["evaluation"]["num_episodes"] == 2

            # Check visualization files were created
            viz_dir = Path(tmp_dir) / "visualizations"
            assert viz_dir.exists()
            expected_plots = [
                "01_inventory.png",
                "02_orders.png",
                "03_cost_breakdown.png",
                "04_demand_fulfillment.png",
                "05_shipment_heatmap.png",
                "06_episode_summary.png",
            ]
            for plot_name in expected_plots:
                assert (viz_dir / plot_name).exists(), f"Missing plot: {plot_name}"

            runner.algorithm.trainer.stop()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_runner_eval_seed_matches_split(self, trained_algorithm):
        """EvaluationRunner.eval_seed should match split_seed(root_seed)[1]."""
        ctx = trained_algorithm
        tmp_dir = tempfile.mkdtemp(prefix="eval_seed_test_")
        try:
            runner = EvaluationRunner(
                env_config=ctx["env_config"],
                algorithm_config=ctx["algo_config"],
                checkpoint_dir=ctx["checkpoint_path"],
                output_dir=tmp_dir,
                root_seed=ctx["root_seed"],
            )
            _, expected_eval_seed = split_seed(ctx["root_seed"], num_children=2)
            assert runner.eval_seed == expected_eval_seed
            runner.algorithm.trainer.stop()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_runner_none_seed(self, trained_algorithm):
        """EvaluationRunner with root_seed=None should have eval_seed=None."""
        ctx = trained_algorithm
        tmp_dir = tempfile.mkdtemp(prefix="eval_none_test_")
        try:
            runner = EvaluationRunner(
                env_config=ctx["env_config"],
                algorithm_config=ctx["algo_config"],
                checkpoint_dir=ctx["checkpoint_path"],
                output_dir=tmp_dir,
                root_seed=None,
            )
            assert runner.eval_seed is None
            runner.algorithm.trainer.stop()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
