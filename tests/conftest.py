"""Shared fixtures for all test modules."""

# Force non-interactive matplotlib backend before any other imports
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
from pathlib import Path

from src.config.loader import load_environment_config, load_algorithm_config
from src.environment.environment import InventoryEnvironment
from src.algorithms.registry import get_algorithm


# ---------------------------------------------------------------------------
# Paths to config files (small default configs)
# ---------------------------------------------------------------------------

ENV_CONFIG_PATH = "config_files/environments/base_env.yaml"
ALGO_CONFIG_PATH = "config_files/algorithms/ippo.yaml"


# ---------------------------------------------------------------------------
# Lightweight config fixtures (no RLlib, fast)
# ---------------------------------------------------------------------------

@pytest.fixture
def env_config():
    """Loads the base environment config."""
    return load_environment_config(ENV_CONFIG_PATH)


@pytest.fixture
def algo_config():
    """Loads the base IPPO algorithm config."""
    return load_algorithm_config(ALGO_CONFIG_PATH)


@pytest.fixture
def env(env_config):
    """Creates an unseeded InventoryEnvironment."""
    return InventoryEnvironment(env_config)


@pytest.fixture
def seeded_env(env_config):
    """Creates a seeded InventoryEnvironment (seed=42)."""
    return InventoryEnvironment(env_config, seed=42)


# ---------------------------------------------------------------------------
# Algorithm fixture (builds RLlib PPO — slower, use sparingly)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trained_algorithm():
    """
    Builds an IPPO algorithm, trains for 1 iteration, and saves a checkpoint.
    Session-scoped so it is only built once across all tests.
    
    Returns a dict with all the objects tests need.
    """
    import tempfile, shutil

    env_config = load_environment_config(ENV_CONFIG_PATH)
    algo_config = load_algorithm_config(ALGO_CONFIG_PATH)

    # Override to make training as fast as possible
    algo_config.shared.num_iterations = 1
    algo_config.shared.checkpoint_freq = 1
    algo_config.shared.eval_interval = 1
    algo_config.shared.num_eval_episodes = 2

    root_seed = 42
    from src.utils.seed_manager import split_seed
    train_seed, eval_seed = split_seed(root_seed, num_children=2)

    template_env = InventoryEnvironment(env_config)
    algorithm = get_algorithm(
        algo_config.name, template_env, algo_config,
        train_seed=train_seed, eval_seed=eval_seed,
    )

    # Train one iteration so the model has some weights
    algorithm.train()

    # Save checkpoint to a temp directory (create dir first, matching runner.py pattern)
    tmp_dir = tempfile.mkdtemp(prefix="marl_test_ckpt_")
    ckpt_dir = Path(tmp_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir)
    algorithm.save_checkpoint(ckpt_path)

    yield {
        "algorithm": algorithm,
        "env_config": env_config,
        "algo_config": algo_config,
        "template_env": template_env,
        "root_seed": root_seed,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "checkpoint_path": ckpt_path,
        "tmp_dir": tmp_dir,
    }

    # Cleanup
    algorithm.trainer.stop()
    shutil.rmtree(tmp_dir, ignore_errors=True)
