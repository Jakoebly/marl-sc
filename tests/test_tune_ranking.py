"""
Tests for the trial ranking after a tune (``src/experiments/utils/ray_tune.py``): only trials that
ran to the end are ranked. A trial stopped early by ASHA last reported at its rung, so its value is
not comparable with the final value of a trial that ran all iterations.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from src.experiments.utils.experiment_utils import TUNE_BEST_TRIALS_FILENAME
from src.experiments.utils.ray_tune import analyze_tune_convergence, print_and_save_best_results

METRIC = "eval/episode_return_mean"


class FakeResultGrid(list):
    """List of results with the ``experiment_path`` attribute of a Ray ``ResultGrid``."""

    def __init__(self, results, experiment_path):
        super().__init__(results)
        self.experiment_path = str(experiment_path)

    def get_best_result(self, metric, mode, scope="last"):
        """Ray's choice: the best last reported value among all trials, finished or not."""
        candidates = [r for r in self if r.metrics and metric in r.metrics]
        return (max if mode == "max" else min)(candidates, key=lambda r: r.metrics[metric])

def fake_result(tmp_path: Path, index: int, last_iteration: int, final_value: float):
    """A Ray Tune result whose last report was at ``last_iteration`` with ``final_value``."""

    trial_dir = tmp_path / f"trainable_{index:08x}_{index:05d}_lr={index}_2026-11-20_10-00-00"
    trial_dir.mkdir()
    config = {
        "env_config": {"n_warehouses": 3, "n_skus": 5},
        "algorithm_config": {"shared": {}, "algorithm_specific": {}},
        "shared": {"learning_rate": 1e-4 * (index + 1)},
        "algorithm_specific": {"lam": [0.9, 0.95, 0.97][index % 3]},
    }
    metrics = {
        "training_iteration": last_iteration,
        METRIC: final_value,
        "train/episode_return_mean": final_value - 5.0,
    }
    history = pd.DataFrame({
        "training_iteration": [last_iteration],
        "train/episode_return_mean": [final_value - 5.0],
    })
    return SimpleNamespace(metrics=metrics, path=str(trial_dir), config=config, metrics_dataframe=history)

def asha_grid(tmp_path: Path) -> FakeResultGrid:
    """Five trials; trials 1 and 4 were stopped at the iteration-150 rung. Trial 1's rung value
    is higher than every final value, so ranking it would pick a trial without a final checkpoint."""

    specs = [(0, 300, -420.0), (1, 150, -380.0), (2, 300, -410.0), (3, 300, -430.0), (4, 150, -500.0)]
    return FakeResultGrid([fake_result(tmp_path, *spec) for spec in specs], tmp_path)


def test_best_and_top_k_use_only_trials_that_ran_to_the_end(tmp_path):
    """The best trial and the top-K list come from the three trials that reached iteration 300."""

    best = print_and_save_best_results(asha_grid(tmp_path), metric=METRIC, mode="max", top_k=10)

    assert Path(best["best_trial_path"]).name.startswith("trainable_00000002_00002")
    assert [trial["eval_metric"] for trial in best["top_k_trials"]] == [-410.0, -420.0, -430.0]
    assert (best["n_trials_ran_to_end"], best["n_trials_stopped_early"]) == (3, 2)

    saved = yaml.safe_load((tmp_path / TUNE_BEST_TRIALS_FILENAME).read_text())
    assert saved["top_k_trials"] == best["top_k_trials"]

def test_without_early_stopping_every_trial_is_ranked(tmp_path):
    """With a FIFO scheduler every trial reaches the last iteration and all are ranked."""

    specs = [(0, 300, -420.0), (1, 300, -380.0), (2, 300, -410.0)]
    grid = FakeResultGrid([fake_result(tmp_path, *spec) for spec in specs], tmp_path)
    best = print_and_save_best_results(grid, metric=METRIC, mode="max", top_k=10)

    assert [trial["eval_metric"] for trial in best["top_k_trials"]] == [-380.0, -410.0, -420.0]
    assert best["n_trials_stopped_early"] == 0

def test_convergence_report_uses_only_trials_that_ran_to_the_end(tmp_path):
    """The convergence report counts and ranks only the trials that reached iteration 300."""

    report = analyze_tune_convergence(asha_grid(tmp_path), metric=METRIC, mode="max")

    assert report["summary"]["total_trials"] == 3
    assert report["summary"]["trials_stopped_early"] == 2
    assert report["summary"]["best_metric"] == -410.0
