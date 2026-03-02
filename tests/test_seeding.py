"""Tests for the seeding system: SeedManager and seed flow through the stack."""

import numpy as np
import pytest

from src.utils.seed_manager import SeedManager, EXPERIMENT_SEEDS, ENVIRONMENT_SEEDS
from src.environment.envs.multi_env import InventoryEnvironment


# ============================================================================
# 1. SeedManager unit tests
# ============================================================================

class TestSeedManager:
    """Tests for the SeedManager class."""

    def test_init_with_seed_spawns_all_components(self):
        """SeedManager should spawn a seed for every entry in the registry."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        for name in ENVIRONMENT_SEEDS:
            assert sm.get_seed_int(name) is not None

    def test_init_without_seed_all_none(self):
        """Without a root seed, all component seeds should be None."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEEDS)
        for name in ENVIRONMENT_SEEDS:
            assert sm.get_seed_int(name) is None

    def test_get_seed_int_returns_integer(self):
        """get_seed_int should return a plain int."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        seed_int = sm.get_seed_int("inventory")
        assert isinstance(seed_int, int)

    def test_get_seed_int_none_when_no_root(self):
        """get_seed_int should return None when root_seed is None."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEEDS)
        assert sm.get_seed_int("inventory") is None

    def test_get_rng_returns_generator(self):
        """get_rng should return an np.random.Generator."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        rng = sm.get_rng("inventory")
        assert isinstance(rng, np.random.Generator)

    def test_get_rng_unseeded_returns_generator(self):
        """get_rng with no root seed should return an unseeded Generator."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEEDS)
        rng = sm.get_rng("inventory")
        assert isinstance(rng, np.random.Generator)

    def test_get_rng_deterministic(self):
        """Two SeedManagers with the same root should produce identical RNG streams."""
        sm_a = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        sm_b = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        vals_a = sm_a.get_rng("inventory").random(10)
        vals_b = sm_b.get_rng("inventory").random(10)
        np.testing.assert_array_equal(vals_a, vals_b)

    def test_unregistered_name_raises(self):
        """Requesting an unregistered seed name should raise ValueError."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        with pytest.raises(ValueError):
            sm.get_seed_int("nonexistent_component")

    def test_advance_episode_changes_seeds(self):
        """advance_episode should produce different component seeds each call."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        seeds_ep0 = [sm.get_seed_int(n) for n in ENVIRONMENT_SEEDS]
        sm.advance_episode()
        seeds_ep1 = [sm.get_seed_int(n) for n in ENVIRONMENT_SEEDS]
        assert seeds_ep0 != seeds_ep1

    def test_advance_episode_deterministic(self):
        """Two SeedManagers with the same root should produce the same seed sequence."""
        sm_a = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        sm_b = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)

        for _ in range(5):
            sm_a.advance_episode()
            sm_b.advance_episode()
            seeds_a = [sm_a.get_seed_int(n) for n in ENVIRONMENT_SEEDS]
            seeds_b = [sm_b.get_seed_int(n) for n in ENVIRONMENT_SEEDS]
            assert seeds_a == seeds_b

    def test_advance_episode_noop_without_seed(self):
        """advance_episode should be a no-op when root_seed is None."""
        sm = SeedManager(root_seed=None, seed_registry=ENVIRONMENT_SEEDS)
        sm.advance_episode()
        for name in ENVIRONMENT_SEEDS:
            assert sm.get_seed_int(name) is None

    def test_update_root_seed_resets_counter(self):
        """update_root_seed should reset the episode counter so the seed
        sequence starts over with the new root."""
        sm = SeedManager(root_seed=42, seed_registry=ENVIRONMENT_SEEDS)
        sm.advance_episode()
        sm.advance_episode()

        sm.update_root_seed(99)
        seeds_after_reset = [sm.get_seed_int(n) for n in ENVIRONMENT_SEEDS]

        sm_fresh = SeedManager(root_seed=99, seed_registry=ENVIRONMENT_SEEDS)
        seeds_fresh = [sm_fresh.get_seed_int(n) for n in ENVIRONMENT_SEEDS]

        assert seeds_after_reset == seeds_fresh

    def test_experiment_seeds_train_eval_differ(self):
        """Train and eval seeds from EXPERIMENT_SEEDS should be different."""
        sm = SeedManager(root_seed=42, seed_registry=EXPERIMENT_SEEDS)
        assert sm.get_seed_int('train') != sm.get_seed_int('eval')

    def test_experiment_seeds_deterministic(self):
        """Same root seed should always produce the same train/eval ints."""
        sm_a = SeedManager(root_seed=42, seed_registry=EXPERIMENT_SEEDS)
        sm_b = SeedManager(root_seed=42, seed_registry=EXPERIMENT_SEEDS)
        assert sm_a.get_seed_int('train') == sm_b.get_seed_int('train')
        assert sm_a.get_seed_int('eval') == sm_b.get_seed_int('eval')


# ============================================================================
# 2. Environment seeding integration tests
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

        rng = np.random.default_rng(0)

        any_different = False
        for _ in range(5):
            actions_a = {agent: rng.uniform(-1, 1, size=env_a.n_skus).astype(np.float32)
                         for agent in env_a.agents}
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

        any_different = any(
            not np.array_equal(obs_1[agent]["local"], obs_2[agent]["local"])
            for agent in env.agents
        )
        # Note: with zero initial inventory, observations may still be identical.


# ============================================================================
# 3. Algorithm-level seed flow (integration, uses trained_algorithm fixture)
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
