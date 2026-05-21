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
    # Convergence report only (read-only, login-node safe)
    python src/experiments/utils/regenerate_convergence.py \
        --storage-dir experiment_outputs/Tuning \
        --experiment-name MAPPO_Tune_3WH_5SKUS_Symmetric_Optuna_FIFO

    # Also re-run the best-trial eval + update best_trial_results.yaml
    # (evaluates a checkpoint -> needs the runtime + a salloc allocation)
    python src/experiments/utils/regenerate_convergence.py \
        --storage-dir experiment_outputs/Tuning \
        --experiment-name CPPO_Tune_3WH_5SKUS_Symmetric_Optuna_FIFO \
        --with-best-eval

The default pass is read-only and safe on a login node. ``--with-best-eval``
builds an RLlib algorithm and rolls out episodes, so run it in a compute
allocation.
"""

import argparse
import json
from pathlib import Path

import ray
from ray import tune

from src.experiments.run_experiment import (
    trainable,
    CONVERGENCE_METRIC,
    TUNE_METRIC,
    TUNE_MODE,
    _run_post_tune_analysis,
)
from src.experiments.utils.args import DEFAULT_EVAL_SEED
from src.experiments.utils.ray_tune import analyze_tune_convergence
from src.experiments.utils.experiment_utils import DEFAULT_TUNE_STORAGE_DIR


def _read_seeds(experiment_path: str):
    """
    Reads ``root_seed`` / ``eval_seed`` from the first trial's ``params.json``.

    Mirrors the resolution used by ``resume_tune_experiment`` so the offline
    best-trial evaluation uses the same eval seed as the original run.

    Returns:
        Tuple[Optional[int], int]: ``(root_seed, eval_seed)``.
    """
    experiment_dir = Path(experiment_path)
    for trial_dir in sorted(experiment_dir.iterdir()):
        params_file = trial_dir / "params.json"
        if params_file.is_file():
            with open(params_file, "r") as f:
                cfg = json.load(f)
            return cfg.get("root_seed"), cfg.get("eval_seed", DEFAULT_EVAL_SEED)
    return None, DEFAULT_EVAL_SEED


def regenerate_one(experiment_path: str, with_best_eval: bool = False) -> bool:
    """
    Restores one experiment's results and regenerates its post-tune artifacts.

    By default this rewrites only ``convergence_analysis.yaml`` (read-only; no
    checkpoint is evaluated). With ``with_best_eval=True`` it re-runs the full
    post-tune analysis: best-trial selection, the best-trial benchmark
    evaluation (updating ``best_trial_results.yaml`` with
    ``best_trial_eval_metric`` and writing ``eval_results_final.yaml`` +
    visualizations), and the convergence report.

    Args:
        experiment_path (str): Path to a tune experiment directory.
        with_best_eval (bool): If True, also re-run the best-trial evaluation.
            This builds an RLlib algorithm and rolls out episodes, so it needs
            the runtime and real compute (use a salloc allocation).

    Returns:
        bool: True if artifacts were regenerated, False if skipped/empty.
    """

    # Only restorable tune experiments carry the state needed to rebuild results
    if not tune.Tuner.can_restore(experiment_path):
        print(f"[SKIP] Not a restorable tune experiment: {experiment_path}")
        return False

    # Restore the result grid from disk without running any trials
    print(f"\n[INFO] Restoring results from: {experiment_path}")
    tuner = tune.Tuner.restore(experiment_path, trainable=trainable)
    analysis = tuner.get_results()

    if with_best_eval:
        # Reuse the production post-tune path: best results + best-trial eval
        # (updates best_trial_results.yaml) + convergence report.
        root_seed, eval_seed = _read_seeds(experiment_path)
        _run_post_tune_analysis(
            analysis,
            metric=TUNE_METRIC,
            mode=TUNE_MODE,
            root_seed=root_seed,
            eval_seed=eval_seed,
        )
        return True

    # Default: convergence report only (read-only, no checkpoint evaluation)
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
    parser.add_argument(
        "--with-best-eval",
        action="store_true",
        help=(
            "Also re-run the best-trial benchmark evaluation and update "
            "best_trial_results.yaml (writes eval_results_final.yaml + "
            "visualizations). This evaluates a checkpoint, so it needs the RLlib "
            "runtime and real compute -- run it inside a salloc allocation, not "
            "on a login node."
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
            if regenerate_one(str(target), with_best_eval=args.with_best_eval):
                n_ok += 1
        except Exception as e:
            print(f"[ERROR] Failed to regenerate {target}: {e}")

    print(f"\n[INFO] Done. Regenerated artifacts for {n_ok} experiment(s).")


if __name__ == "__main__":
    main()
