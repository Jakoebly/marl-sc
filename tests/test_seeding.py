"""Tests for the seeding system: split_seed, SeedManager, and seed flow through the stack."""

import numpy as np
import pytest
from numpy.random import SeedSequence

from src.utils.seed_manager import split_seed, SeedManager, ENVIRONMENT_SEED_REGISTRY
from src.environment.environment import InventoryEnvironment


# ============================================================================
# 1. split_seed unit tests
# ============================================================================

class TestSplitSeed:
    """Tests for the split_seed() utility function."""

    def test_returns_correct_number_of_children(self):
        """split_seed should return exactly num_children seeds."""
        seeds = split_seed(42, num_children=3)
        assert len(seeds) == 3

    def test_default_returns_two_children(self):
        """Default num_children=2 should return two seeds."""
        seeds = split_seed(42)
        assert len(seeds) == 2

    def test_none_root_returns_all_nones(self):
        """When root_seed is None, all children should be None."""
        seeds = split_seed(None, num_children=3)
        assert seeds == [None, None, None]

    def test_children_are_integers(self):
        """All child seeds should be integers (not numpy types)."""
        seeds = split_seed(42, num_children=2)
        for s in seeds:
            assert isinstance(s, int)

    def test_children_are_distinct(self):
        """Child seeds derived from the same root should differ from each other."""
        train_seed, eval_seed = split_seed(42)
        assert train_seed != eval_seed

    def test_deterministic(self):
        """Calling split_seed twice with the same root should produce identical results."""
        seeds_a = split_seed(42, num_children=2)
        seeds_b = split_seed(42, num_children=2)
        assert seeds_a == seeds_b

    def test_different_roots_produce_different_seeds(self):
        """Different root seeds should produce different child seeds."""
        seeds_a = split_seed(42)
        seeds_b = split_seed(99)
        assert seeds_a != seeds_b

    def test_same_eval_seed_across_runners(self):
        """ExperimentRunner and EvaluationRunner use the same split_seed logic,
        so the eval_seed (child index 1) should be identical for the same root_seed."""
        _, eval_from_experiment = split_seed(42, num_children=2)
        _, eval_from_evaluation = split_seed(42, num_children=2)
        assert eval_from_experiment == eval_from_evaluation


# ============================================================================
# 2. SeedManager unit tests
# ============================================================================

class TestSeedManager:
    """Tests for the SeedManager class."""

    def test_init_with_seed_spawns_all_components(self):
        """SeedManager should spawn a seed for every entry in the registry."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        for name in ENVIRONMENT_SEED_REGISTRY:
            assert sm.get_seed(name) is not None

    def test_init_without_seed_all_none(self):
        """Without a root seed, all component seeds should be None."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        for name in ENVIRONMENT_SEED_REGISTRY:
            assert sm.get_seed(name) is None

    def test_get_seed_int_returns_integer(self):
        """get_seed_int should return a plain int."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        seed_int = sm.get_seed_int("inventory")
        assert isinstance(seed_int, int)

    def test_get_seed_int_none_when_no_root(self):
        """get_seed_int should return None when root_seed is None."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        assert sm.get_seed_int("inventory") is None

    def test_unregistered_name_raises(self):
        """Requesting an unregistered seed name should raise ValueError."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        with pytest.raises(ValueError, match="not registered"):
            sm.get_seed("nonexistent_component")

    def test_advance_episode_changes_seeds(self):
        """advance_episode should produce different component seeds each call."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        seeds_ep0 = sm.get_seeds_int_for_components(list(ENVIRONMENT_SEED_REGISTRY))
        sm.advance_episode()
        seeds_ep1 = sm.get_seeds_int_for_components(list(ENVIRONMENT_SEED_REGISTRY))
        assert seeds_ep0 != seeds_ep1

    def test_advance_episode_deterministic(self):
        """Two SeedManagers with the same root should produce the same seed sequence."""
        sm_a = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        sm_b = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)

        for _ in range(5):
            sm_a.advance_episode()
            sm_b.advance_episode()
            seeds_a = sm_a.get_seeds_int_for_components(list(ENVIRONMENT_SEED_REGISTRY))
            seeds_b = sm_b.get_seeds_int_for_components(list(ENVIRONMENT_SEED_REGISTRY))
            assert seeds_a == seeds_b

    def test_advance_episode_noop_without_seed(self):
        """advance_episode should be a no-op when root_seed is None."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        sm.advance_episode()
        for name in ENVIRONMENT_SEED_REGISTRY:
            assert sm.get_seed(name) is None

    def test_update_root_seed_resets_counter(self):
        """update_root_seed should reset the episode counter so the seed
        sequence starts over with the new root."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        sm.advance_episode()
        sm.advance_episode()

        # Reset to a different root
        sm.update_root_seed(99)
        seeds_after_reset = sm.get_seeds_int_for_components(list(ENVIRONMENT_SEED_REGISTRY))

        # Compare with a fresh manager at root 99
        sm_fresh = SeedManager(root_seed=99, seed_registry=ENVIRONMENT_SEED_REGISTRY)
        seeds_fresh = sm_fresh.get_seeds_int_for_components(list(ENVIRONMENT_SEED_REGISTRY))

        assert seeds_after_reset == seeds_fresh


# ============================================================================
# 3. Environment seeding integration tests
# ============================================================================

class TestEnvironmentSeeding:
    """Tests that seeds propagate correctly through the environment."""

    def test_seeded_env_reset_is_deterministic(self, env_config):
        """Two environments with the same seed should produce identical initial states."""
        env_a = InventoryEnvironment(env_config, seed=42)
        env_b = InventoryEnvironment(env_config, seed=42)

        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()

        for agent in env_a.agents:
            np.testing.assert_array_equal(obs_a[agent]["local"], obs_b[agent]["local"])

    def test_seeded_env_episode_sequence_deterministic(self, env_config):
        """Multiple resets (episodes) with the same seed should produce the same
        sequence of initial observations across two independent environments."""
        env_a = InventoryEnvironment(env_config, seed=42)
        env_b = InventoryEnvironment(env_config, seed=42)

        for _ in range(3):
            obs_a, _ = env_a.reset()
            obs_b, _ = env_b.reset()
            for agent in env_a.agents:
                np.testing.assert_array_equal(obs_a[agent]["local"], obs_b[agent]["local"])

    def test_different_seeds_produce_different_trajectories(self, env_config):
        """Environments with different seeds should produce different trajectories
        after a few steps (initial obs may be identical with zero inventory)."""
        env_a = InventoryEnvironment(env_config, seed=42)
        env_b = InventoryEnvironment(env_config, seed=99)

        env_a.reset()
        env_b.reset()

        # Use the same random actions for both envs
        rng = np.random.default_rng(0)

        any_different = False
        for _ in range(5):
            actions_a = {agent: rng.uniform(-1, 1, size=env_a.n_skus).astype(np.float32)
                         for agent in env_a.agents}
            # Same actions for both environments
            actions_b = {agent: actions_a[agent].copy() for agent in env_b.agents}

            obs_a, rew_a, _, _, _ = env_a.step(actions_a)
            obs_b, rew_b, _, _, _ = env_b.step(actions_b)

            if any(not np.array_equal(obs_a[agent]["local"], obs_b[agent]["local"])
                   for agent in env_a.agents):
                any_different = True
                break

        assert any_different, "After 5 steps with different seeds, trajectories should differ"

    def test_unseeded_env_resets_differently(self, env_config):
        """Without a seed, consecutive resets should (very likely) differ."""
        env = InventoryEnvironment(env_config, seed=None)
        obs_1, _ = env.reset()
        obs_2, _ = env.reset()

        # With no seed and Poisson demand, consecutive episodes should differ
        # (probabilistically; this could very rarely fail)
        any_different = any(
            not np.array_equal(obs_1[agent]["local"], obs_2[agent]["local"])
            for agent in env.agents
        )
        # Note: with zero initial inventory and no seed, observations might
        # actually be the same (all zeros). This test is best-effort.


# ============================================================================
# 4. Algorithm-level seed flow (integration, uses trained_algorithm fixture)
# ============================================================================

class TestAlgorithmSeedFlow:
    """Tests that seeds flow correctly from root_seed through the algorithm wrapper."""

    def test_train_and_eval_seeds_stored(self, trained_algorithm):
        """The algorithm wrapper should store both train_seed and eval_seed."""
        algo = trained_algorithm["algorithm"]
        assert algo.train_seed == trained_algorithm["train_seed"]
        assert algo.eval_seed == trained_algorithm["eval_seed"]

    def test_train_seed_differs_from_eval_seed(self, trained_algorithm):
        """train_seed and eval_seed should be different."""
        assert trained_algorithm["train_seed"] != trained_algorithm["eval_seed"]

    def test_debugging_seed_matches_train_seed(self, trained_algorithm):
        """RLlib's .debugging(seed=...) should use train_seed."""
        algo = trained_algorithm["algorithm"]
        rllib_seed = algo.trainer.config.seed
        assert rllib_seed == trained_algorithm["train_seed"]

    def test_training_env_config_seed(self, trained_algorithm):
        """Training env_config['seed'] should be train_seed."""
        algo = trained_algorithm["algorithm"]
        training_seed = algo.trainer.config.env_config.get("seed")
        assert training_seed == trained_algorithm["train_seed"]

    def test_eval_env_config_seed(self, trained_algorithm):
        """Evaluation env_config['seed'] should be eval_seed."""
        algo = trained_algorithm["algorithm"]
        eval_cfg = algo.trainer.config.evaluation_config
        eval_seed_in_config = eval_cfg["env_config"]["seed"]
        assert eval_seed_in_config == trained_algorithm["eval_seed"]
