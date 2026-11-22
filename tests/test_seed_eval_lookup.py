"""
Tests that tune-mode seed evaluation finds the tune under the given storage directory, so tunes
stored outside ``experiment_outputs/Tuning`` (such as the paper's ``experiment_outputs/Paper``)
can be seed-evaluated.
"""

import yaml

from src.experiments.utils.experiment_utils import SEED_EVAL_SUBDIR, TUNE_BEST_TRIALS_FILENAME
from src.experiments.utils.seed_evaluation import build_seed_evaluation_configs


def test_tune_mode_uses_the_given_storage_dir(tmp_path):
    """The top-K trials are read from the tune found under ``storage_dir``."""

    tune_dir = tmp_path / "Paper" / "P2_MAPPO_PS_Test"
    trial_dir = tune_dir / "trainable_abc12345_7_lr=0.001"
    trial_dir.mkdir(parents=True)
    top_k_trials = [
        {"rank": 1, "short_id": "trainable_abc12345_7", "trial_path": str(trial_dir), "eval_metric": -410.0},
    ]
    (tune_dir / TUNE_BEST_TRIALS_FILENAME).write_text(yaml.safe_dump({"top_k_trials": top_k_trials}))

    configs = build_seed_evaluation_configs(storage_dir=str(tmp_path), tune_name="P2_MAPPO_PS_Test", top_k=5)

    assert configs == [{
        "config_name": "01_trainable_abc12345_7",
        "env_config_path": str(trial_dir / "env_config.yaml"),
        "algorithm_config_path": str(trial_dir / "algorithm_config.yaml"),
        "storage_dir": str(tune_dir / SEED_EVAL_SUBDIR / "01_trainable_abc12345_7"),
    }]
