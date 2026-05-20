"""
Offline regeneration of convergence analysis for completed Ray Tune sweeps.

Rebuilds ``convergence_analysis.yaml`` from already-finished tune folders
without re-running any training. Each experiment's result grid is restored
from disk (no trials are executed) and :func:`analyze_tune_convergence` is
re-run against the eval metric (``CONVERGENCE_METRIC``).

Use this to backfill the report for sweeps that finished before the metric
fix, where the analysis was skipped because the last reported row stored
``train/episode_return_mean = None``.

Examples:
    # Regenerate for every restorable experiment under the storage dir
    python src/experiments/regenerate_convergence.py \
        --storage-dir experiment_outputs/Tuning

    # Regenerate for a single experiment
    python src/experiments/regenerate_convergence.py \
        --storage-dir experiment_outputs/Tuning \
        --experiment-name MAPPO_Tune_3WH_5SKUS_Symmetric_Optuna_FIFO

This is a read-only analysis pass and is safe to run on a login node; it does
not launch training, tune, or evaluation jobs.
"""

import argparse
from pathlib import Path

import ray
from ray import tune

from src.experiments.run_experiment import trainable, CONVERGENCE_METRIC, TUNE_MODE
from src.experiments.utils.ray_tune import analyze_tune_convergence
from src.experiments.utils.experiment_utils import DEFAULT_TUNE_STORAGE_DIR


def regenerate_one(experiment_path: str) -> bool:
    """
    Restores one experiment's results and writes ``convergence_analysis.yaml``.

    Args:
        experiment_path (str): Path to a tune experiment directory.

    Returns:
        bool: True if a report was written, False if skipped/empty.
    """

    # Only restorable tune experiments carry the state needed to rebuild results
    if not tune.Tuner.can_restore(experiment_path):
        print(f"[SKIP] Not a restorable tune experiment: {experiment_path}")
        return False

    # Restore the result grid from disk without running any trials
    print(f"\n[INFO] Restoring results from: {experiment_path}")
    tuner = tune.Tuner.restore(experiment_path, trainable=trainable)
    analysis = tuner.get_results()

    # Re-run the convergence analysis against the eval metric
    report = analyze_tune_convergence(
        analysis=analysis,
        metric=CONVERGENCE_METRIC,
        mode=TUNE_MODE,
    )

    return bool(report)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate convergence_analysis.yaml for completed tune experiments."
    )
    parser.add_argument(
        "--storage-dir",
        default=str(DEFAULT_TUNE_STORAGE_DIR),
        help="Directory containing tune experiment folders.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Regenerate only this experiment (folder name under --storage-dir). "
            "If omitted, every restorable experiment under --storage-dir is processed."
        ),
    )
    args = parser.parse_args()

    storage_dir = Path(args.storage_dir)

    # Resolve targets: a single named experiment, or all subdirectories
    if args.experiment_name:
        targets = [storage_dir / args.experiment_name]
    else:
        targets = sorted(p for p in storage_dir.iterdir() if p.is_dir())

    # Initialize Ray once for all restores. This is a read-only pass (no trials
    # are scheduled), so keep the footprint minimal to stay friendly on a shared
    # login node: no dashboard, few logical CPUs, quiet logging.
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        configure_logging=False,
        log_to_driver=False,
        num_cpus=2,
    )

    n_ok = 0
    for target in targets:
        try:
            if regenerate_one(str(target)):
                n_ok += 1
        except Exception as e:
            print(f"[ERROR] Failed to regenerate {target}: {e}")

    print(f"\n[INFO] Done. Regenerated convergence analysis for {n_ok} experiment(s).")


if __name__ == "__main__":
    main()
