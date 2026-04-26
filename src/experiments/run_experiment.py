"""
CLI entry point for running, resuming, and evaluating RL experiments.

Orchestrates single-seed runs, Ray Tune hyperparameter sweeps, seed
evaluation, and checkpoint-based evaluation via ``ExperimentRunner`` /
``EvaluationRunner``.
"""

import json
import yaml
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, Optional, List
import ray
from ray import tune
from ray.tune import RunConfig, CLIReporter

from src.experiments.utils.args import parse_args, DEFAULT_EVAL_SEED, DEFAULT_EVAL_EPISODES
from src.config.schema import EnvironmentConfig, AlgorithmConfig
from src.experiments.runner import ExperimentRunner, EvaluationRunner
from src.experiments.utils.wandb import setup_wandb
from src.config.loader import (
    load_environment_config,
    load_algorithm_config,
    validate_config,
)
from src.experiments.utils.ray_tune import (
    get_tune_scheduler,
    get_tune_search_algorithm,
    get_resources_per_trial,
    report_tune_metrics,
    prepare_tune_config,
    merge_tune_params,
    merge_env_tune_params,
    print_and_save_best_results,
    analyze_tune_convergence,
)
from src.utils.seed_manager import SeedManager, EXPERIMENT_SEEDS
from src.experiments.utils.experiment_utils import (
    save_run_metadata,
    save_tune_config_to_experiment_dir,
    load_root_seed_from_metadata,
    resolve_saved_config,
    find_experiment_dir,
    find_checkpoint_dir,
    find_experiment_dir_from_checkpoint,
    generate_experiment_name,
)
from src.experiments.utils.seed_evaluation import (
    build_seed_evaluation_configs,
    evaluate_config_across_seeds,
    save_combined_seed_evaluation_summary,
    print_seed_evaluation_table,
)


TUNE_METRIC = "eval/episode_return_mean"
TUNE_MODE = "max"
CONVERGENCE_METRIC = "train/episode_return_mean"


# ============================================================================
# Run Experiment Functions
# ============================================================================

def run_single_experiment(
    env_config_path: Optional[str] = None,
    algorithm_config_path: Optional[str] = None,
    storage_dir: str = "./experiment_outputs",
    experiment_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    root_seed: Optional[int] = None,
    eval_seed: Optional[int] = None,
    resume_from: Optional[str] = None,
):
    """
    Runs a single experiment without hyperparameter tuning.

    When ``resume_from`` is provided, configs are loaded from the saved
    ``env_config.yaml`` and ``algorithm_config.yaml`` in the experiment
    directory unless explicit config paths are given.

    Args:
        env_config_path (Optional[str]): Path to environment config.
            Required for new runs; optional when resuming.
        algorithm_config_path (Optional[str]): Path to algorithm config.
            Required for new runs; optional when resuming.
        storage_dir (str): Root directory for experiment outputs.
        experiment_name (Optional[str]): Name for the experiment.
        wandb_project (Optional[str]): WandB project name.
        wandb_name (Optional[str]): WandB run name.
        root_seed (Optional[int]): Root seed for reproducibility.
        eval_seed (Optional[int]): Optional fixed eval root seed forwarded
            to ``ExperimentRunner`` as ``eval_seed_override``. ``None`` keeps 
            the per-run ``eval`` slot derived from ``root_seed``.
        resume_from (Optional[str]): Path to checkpoint to resume from.

    Returns:
        Dict[str, Any]: Training result dictionary.
    """
    
    # When resuming, resolve configs and root_seed from the experiment directory
    if resume_from:
        experiment_dir = find_experiment_dir_from_checkpoint(resume_from)
        env_config_path = resolve_saved_config(
            experiment_dir, "env_config.yaml", env_config_path
        )
        algorithm_config_path = resolve_saved_config(
            experiment_dir, "algorithm_config.yaml", algorithm_config_path
        )
        if root_seed is None:
            saved_seed = load_root_seed_from_metadata(experiment_dir)
            if saved_seed is not None:
                root_seed = saved_seed
                print(f"[INFO] Loading root_seed={root_seed} from saved metadata")

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = generate_experiment_name(
            env_config_path=env_config_path,
            algorithm_config_path=algorithm_config_path,
            mode="single"
        )

    # Create the single top-level SeedManager
    seed_manager = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)

    # Load configs (seed_manager flows into DataGenerator for synthetic data)
    env_config = load_environment_config(env_config_path, seed_manager=seed_manager)
    algorithm_config = load_algorithm_config(algorithm_config_path)

    # Setup WandB to log metrics to WandB
    wandb_config, _ = setup_wandb(
        wandb_project=wandb_project,
        mode="single",
        wandb_name=wandb_name if wandb_name else experiment_name,
    )

    # Create experiment directory
    experiment_dir = Path(storage_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(experiment_dir)

    # Create experiment runner
    runner = ExperimentRunner(
        env_config=env_config,
        algorithm_config=algorithm_config,
        seed_manager=seed_manager,
        checkpoint_dir=checkpoint_dir,
        wandb_config=wandb_config,
        eval_seed_override=eval_seed,
    )

    # Save run metadata once before training starts
    save_run_metadata(
        output_dir=checkpoint_dir,
        runner=runner,
        root_seed=root_seed,
    )

    # Run the experiment and return the result
    result = runner.run(resume_from=resume_from)

    return result

def run_evaluation(
    checkpoint_dir: str,
    experiment_dir: str,
    env_config_path: Optional[str] = None,
    algorithm_config_path: Optional[str] = None,
    eval_episodes: Optional[int] = None,
    root_seed: Optional[int] = None,
    visualize: bool = False,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
):
    """
    Runs standalone evaluation on a trained checkpoint.

    Configs are loaded from the experiment directory unless explicit
    paths are provided.

    Args:
        checkpoint_dir (str): Path to checkpoint directory to evaluate.
        experiment_dir (str): Path to the experiment directory.
        env_config_path (Optional[str]): Path to environment config.
        algorithm_config_path (Optional[str]): Path to algorithm config.
        eval_episodes (Optional[int]): Number of evaluation episodes.
        root_seed (Optional[int]): Root seed for reproducibility.
        visualize (bool): If True, generate visualization plots.
        wandb_project (Optional[str]): WandB project name.
        wandb_name (Optional[str]): WandB run name.

    Returns:
        Dict[str, Any]: Evaluation result dictionary.
    """

    # Resolve configs from the experiment directory
    env_config_path = resolve_saved_config(
        Path(experiment_dir), "env_config.yaml", env_config_path
    )
    algorithm_config_path = resolve_saved_config(
        Path(experiment_dir), "algorithm_config.yaml", algorithm_config_path
    )

    # Create the single top-level SeedManager
    seed_manager = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)

    # Load configs
    env_config = load_environment_config(env_config_path, seed_manager=seed_manager)
    algorithm_config = load_algorithm_config(algorithm_config_path)

    # Setup WandB to log metrics
    wandb_config, _ = setup_wandb(
        wandb_project=wandb_project,
        mode="single",
        wandb_name=wandb_name if wandb_name else f"eval_{Path(checkpoint_dir).name}",
    )

    # Create evaluation runner
    runner = EvaluationRunner(
        env_config=env_config,
        algorithm_config=algorithm_config,
        checkpoint_dir=checkpoint_dir,
        experiment_dir=experiment_dir,
        eval_episodes=eval_episodes,
        seed_manager=seed_manager,
        visualize=visualize,
        wandb_config=wandb_config,
    )

    # Run evaluation
    result = runner.run()

    return result

def run_tune_experiment(
    env_config_path: str,
    algorithm_config_path: str,
    tune_config_path: str,
    num_samples: int,
    storage_dir: str,
    wandb_project: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_gpus: int = 0,
    num_cpus_per_env_runner: Optional[int] = None,
    experiment_name: Optional[str] = None,
    root_seed: Optional[int] = None,
    eval_seed: int = DEFAULT_EVAL_SEED,
    top_k: int = 10,
):
    """
    Runs a hyperparameter tuning experiment with Ray Tune.

    Args:
        env_config_path (str): Path to base environment config.
        algorithm_config_path (str): Path to base algorithm config.
        tune_config_path (str): Path to tune config (search space, scheduler, search algorithm).
        num_samples (int): Number of trials to run.
        storage_dir (str): Root directory for experiment outputs.
        wandb_project (Optional[str]): WandB project name.
        num_cpus (Optional[int]): CPUs per trial.
        num_gpus (int): GPUs per trial.
        num_cpus_per_env_runner (Optional[int]): CPUs per env runner.
        experiment_name (Optional[str]): Name for the experiment.
        root_seed (Optional[int]): Root seed for reproducibility.
        eval_seed (int): Fixed root seed for the post-tune best-trial benchmark
            evaluation (default: ``DEFAULT_EVAL_SEED``).
        top_k (int): Number of top trials to save in best_trial_results.yaml.

    Returns:
        tune.ResultGrid: Ray Tune results.
    """
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create the single top-level SeedManager for the tune experiment
    seed_manager = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)

    # Prepare configuration by merging environment, algorithm and tune configs
    search_config, scheduler_config, search_algorithm_config = prepare_tune_config(
        env_config_path=env_config_path,
        algorithm_config_path=algorithm_config_path,
        tune_config_path=tune_config_path,
        seed_manager=seed_manager,
    )

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = generate_experiment_name(
            env_config_path=env_config_path,
            algorithm_config_path=algorithm_config_path,
            mode="tune",
            search_type=search_algorithm_config.type if search_algorithm_config else None,
            scheduler_type=scheduler_config.type if scheduler_config else None,
        )

    # Setup WandB callback to log metrics to WandB
    _, callbacks = setup_wandb(
        wandb_project=wandb_project,
        mode="tune",
        experiment_name=experiment_name,
    )

    # Get required resources per trial
    resources_per_trial = get_resources_per_trial(
        num_cpus=num_cpus,
        num_gpus=num_gpus, 
        num_env_runners=search_config["algorithm_config"]["shared"]["num_env_runners"], 
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        algorithm_config_dict=search_config["algorithm_config"],
    )

    # Wrap trainable with resources
    trainable_with_resources = tune.with_resources(trainable, resources_per_trial)

    # Add root seed and eval seed to the search config
    search_config["root_seed"] = root_seed
    search_config["eval_seed"] = eval_seed

    # Setup scheduler and search algorithm from validated config objects
    scheduler = get_tune_scheduler(scheduler_config, metric=TUNE_METRIC, mode=TUNE_MODE)
    search_seed = seed_manager.get_seed_int('train')
    search_alg = get_tune_search_algorithm(search_algorithm_config, metric=TUNE_METRIC, mode=TUNE_MODE, seed=search_seed)

    # Create a tuner
    # search_config: base configs and hyperparameter search spaces (passed to trainable)
    # tune_config: controls how Ray Tune runs the search
    # run_config: controls experiment execution and output
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=storage_dir,
            callbacks=callbacks,
            progress_reporter=CLIReporter(),
        ),
    )

    # Save the original tune config into the experiment directory for traceability
    save_tune_config_to_experiment_dir(
        tune_config_path=tune_config_path,
        storage_dir=storage_dir,
        experiment_name=experiment_name,
    )

    # Run the tuner
    analysis = tuner.fit()

    # Run post-tune analysis including best results, evaluation, convergence
    _run_post_tune_analysis(
        analysis,
        metric=TUNE_METRIC,
        mode=TUNE_MODE,
        root_seed=root_seed,
        eval_seed=eval_seed,
        top_k=top_k,
    )

    return analysis

def resume_tune_experiment(
    experiment_path: str,
    num_cpus: Optional[int] = None,
    num_gpus: int = 0,
    num_cpus_per_env_runner: Optional[int] = None,
    wandb_project: Optional[str] = None,
    eval_seed: Optional[int] = None,
    top_k: int = 10,
):
    """
    Resumes an interrupted Ray Tune experiment via ``Tuner.restore``.

    All configurations (search space, scheduler, search algorithm, num_samples,
    env/algorithm configs) are loaded from the saved experiment state,
    only the trainable and resource allocation are re-specified.

    Args:
        experiment_path (str): Path to the existing experiment directory.
        num_cpus (Optional[int]): CPUs per trial.
        num_gpus (int): GPUs per trial.
        num_cpus_per_env_runner (Optional[int]): CPUs per env runner.
        wandb_project (Optional[str]): WandB project name.
        eval_seed (Optional[int]): Fixed root seed for the post-tune best-trial
            benchmark evaluation (default: ``DEFAULT_EVAL_SEED``). Might be read
            from ``params.json`` if the initial tune run stored it.
        top_k (int): Number of top trials to save in best_trial_results.yaml.

    Returns:
        analysis (tune.ResultGrid): Ray Tune results.
    """

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Validate that the experiment can be restored
    if not tune.Tuner.can_restore(experiment_path):
        raise FileNotFoundError(
            f"Cannot restore Tune experiment from '{experiment_path}'. "
            f"The directory does not contain a valid Tune experiment state. "
            f"Make sure the full experiment directory (including trial folders) is intact."
        )
    print(f"[INFO] Restoring Tune experiment from: {experiment_path}")

    # Get the tune config from the first trial's ``params.json``
    experiment_dir = Path(experiment_path)
    first_trial_config = None
    for trial_dir in sorted(experiment_dir.iterdir()):
        params_file = trial_dir / "params.json"
        if params_file.is_file():
            with open(params_file, "r") as f:
                first_trial_config = json.load(f)
            break
    if first_trial_config is None:
        raise RuntimeError(
            "Could not find any trial params.json in the experiment directory. "
            "The experiment may have no started trials."
        )

    # Get the algorithm config and number of environment runners from the tune config
    algorithm_config_dict = first_trial_config["algorithm_config"]
    num_env_runners = algorithm_config_dict["shared"]["num_env_runners"]

    # Get the root seed and eval seed from the first trial's params.json.
    # Resolution order: CLI override > stored value in params.json > DEFAULT_EVAL_SEED.
    root_seed = first_trial_config.get("root_seed")
    if eval_seed is None:
        eval_seed = first_trial_config.get("eval_seed", DEFAULT_EVAL_SEED)

    # Get required resources per trial
    resources_per_trial = get_resources_per_trial(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        algorithm_config_dict=algorithm_config_dict,
    )

    # Wrap trainable with resources
    trainable_with_resources = tune.with_resources(trainable, resources_per_trial)

    # Restore the tuner and re-run interrupted/unstarted trials
    tuner = tune.Tuner.restore(
        experiment_path,
        trainable=trainable_with_resources,
        resume_unfinished=True,
        resume_errored=True,
    )

    # Run the tuner (only unfinished/unstarted trials will execute)
    analysis = tuner.fit()

    # Run post-tune analysis including best results, evaluation, convergence
    metric = TUNE_METRIC
    mode = TUNE_MODE
    _run_post_tune_analysis(
        analysis,
        metric=metric,
        mode=mode,
        root_seed=root_seed,
        eval_seed=eval_seed,
        top_k=top_k,
    )

    return analysis

def run_seed_evaluation(
    n_seeds: int = 5,
    top_k: int = 10,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    eval_seed: int = DEFAULT_EVAL_SEED,
    env_config_path: Optional[str] = None,
    algorithm_config_path: Optional[str] = None,
    storage_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    num_iterations: Optional[int] = None,
    tune_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
):
    """
    Runs a full seed evaluation by training N seeds per config, evaluating each,
    computing aggregate statistics, and plotting training curves.

    Supports two modes:
        - single: explicit env/algorithm configs with a given experiment name.
        - tune:   reads the top-K trial configs from a completed tune experiment's
                  ``best_trial_results.yaml`` and evaluates each one.

    Args:
        n_seeds (int): Number of seeds (root seeds will be 100, 200, ..., N*100).
        top_k (int): Number of top trials to evaluate (tune mode).
        eval_episodes (int): Episodes for final deterministic evaluation.
        eval_seed (int): Fixed root seed used for all final evaluations
            (default: ``DEFAULT_EVAL_SEED``).
        env_config_path (Optional[str]): Path to environment config (single mode).
        algorithm_config_path (Optional[str]): Path to algorithm config (single mode).
        storage_dir (Optional[str]): Root directory for outputs (single mode).
        experiment_name (Optional[str]): Experiment name (single mode).
        num_iterations (Optional[int]): Override training iterations
            (``None`` keeps config default).
        tune_name (Optional[str]): Name of a completed tune experiment (tune mode).
        wandb_project (Optional[str]): Optional WandB project name.

    Returns:
        all_config_stats (List[dict]): Per-config statistics dicts, sorted best-first.
    """

    # Build list of configs to evaluate
    configs_to_eval = build_seed_evaluation_configs(
        env_config_path=env_config_path,
        algorithm_config_path=algorithm_config_path,
        storage_dir=storage_dir,
        experiment_name=experiment_name,
        tune_name=tune_name,
        top_k=top_k,
    )

    # Train + evaluate each config across seeds
    all_config_stats: List[dict] = []
    for cfg in configs_to_eval:
        stats = evaluate_config_across_seeds(
            cfg=cfg,
            n_seeds=n_seeds,
            eval_episodes=eval_episodes,
            eval_seed=eval_seed,
            num_iterations=num_iterations,
            wandb_project=wandb_project,
        )
        all_config_stats.append(stats)

    # Print overall summary table 
    all_config_stats.sort(key=lambda c: c["mean"], reverse=True)
    print_seed_evaluation_table(all_config_stats)

    # If multiple configs (tune mode), save an overall summary + combined plot
    if len(configs_to_eval) > 1 and tune_name is not None:
        save_combined_seed_evaluation_summary(tune_name, all_config_stats)

    return all_config_stats

def _run_post_tune_analysis(
    analysis: tune.ResultGrid,
    metric: str = TUNE_METRIC,
    mode: str = TUNE_MODE,
    root_seed: Optional[int] = None,
    eval_seed: int = DEFAULT_EVAL_SEED,
    top_k: int = 10,
):
    """
    Prints and saves best results, evaluates the best trial's checkpoint,
    and generates a convergence report.

    Args:
        analysis (tune.ResultGrid): Results returned by ``tuner.fit()``.
        metric (str): Metric to optimize.
        mode (str): Optimization mode (``"min"`` or ``"max"``).
        root_seed (Optional[int]): Tune-level root seed.
        eval_seed (int): Fixed root seed used for the best-trial benchmark
            evaluation.
        top_k (int): Number of top trials to save.
    """

    # Print and save best results 
    best_info = print_and_save_best_results(
        analysis=analysis,
        metric=metric,
        mode=mode,
        top_k=top_k,
        tune_root_seed=root_seed,
        eval_root_seed=eval_seed,
    )

    # Get the final checkpoint and trial directory
    final_checkpoint = best_info.get("best_trial_final_checkpoint")
    best_trial_dir = best_info.get("best_trial_path")

    # Run evaluation and update best trial results
    if final_checkpoint:
        print("\n" + "=" * 80)
        print("EVALUATING BEST TRIAL (final checkpoint)")
        print("=" * 80)

        # Run evaluation on the final checkpoint
        eval_result = run_evaluation(
            checkpoint_dir=final_checkpoint,
            experiment_dir=best_trial_dir,
            eval_episodes=DEFAULT_EVAL_EPISODES,
            root_seed=eval_seed,
            visualize=True,
        )

        # Update best trial results with the evaluation reward from the best checkpoint
        eval_reward = eval_result.get("evaluation", {}).get("episode_return_mean")
        if eval_reward is not None:
            results_yaml_path = Path(analysis.experiment_path) / "best_trial_results.yaml"
            if results_yaml_path.exists():
                with open(results_yaml_path, "r") as f:
                    saved = yaml.safe_load(f)
                updated = {}
                for k, v in saved.items():
                    updated[k] = v
                    if k == "best_trial_best_train_metric":
                        updated["best_trial_eval_metric"] = float(eval_reward)
                with open(results_yaml_path, "w") as f:
                    yaml.dump(updated, f, default_flow_style=False, sort_keys=False)
                print(f"[INFO] Eval metric ({eval_reward:.4f}) saved to: {results_yaml_path}")

    # Create a convergence report
    analyze_tune_convergence(
        analysis=analysis,
        metric=CONVERGENCE_METRIC,
        mode=mode,
    )


# ============================================================================
# Tune Trainable Function
# ============================================================================

def trainable(config: Dict[str, Any]):
    """
    Ray Tune trainable function.

    Merges sampled hyperparameters into base configs, builds an
    ``ExperimentRunner``, and reports metrics back to Tune.

    Args:
        config (Dict[str, Any]): Configuration dictionary from Ray Tune.
    """

    # Create the single top-level SeedManager for this trial
    root_seed = config.get("root_seed")
    seed_manager = SeedManager(root_seed=root_seed, seed_registry=EXPERIMENT_SEEDS)

    # Merge env tune params (environment + features) into env config and validate
    env_config_dict = merge_env_tune_params(config["env_config"], config)
    env_config = validate_config(env_config_dict, EnvironmentConfig)

    # Merge algorithm tune params (shared + algorithm_specific) and validate
    algorithm_config_dict = merge_tune_params(config["algorithm_config"], config)
    algorithm_config = validate_config(algorithm_config_dict, AlgorithmConfig)

    # Get WandB config (if provided)
    wandb_config = config.get("wandb_config")

    # Get trial directory for checkpoints
    checkpoint_dir = None
    try:
        trial_dir = tune.get_context().get_trial_dir()
        if trial_dir:
            checkpoint_dir = str(trial_dir)
    except (AttributeError, RuntimeError):
        checkpoint_dir = config.get("checkpoint_dir")

    # Create experiment runner
    runner = ExperimentRunner(
        env_config=env_config,
        algorithm_config=algorithm_config,
        seed_manager=seed_manager,
        checkpoint_dir=checkpoint_dir,
        wandb_config=wandb_config,
    )

    # Save run metadata once before training starts
    if checkpoint_dir:
        ray_trial_id = None
        try:
            ray_trial_id = tune.get_context().get_trial_id()
        except (AttributeError, RuntimeError):
            pass
        save_run_metadata(
            output_dir=checkpoint_dir,
            runner=runner,
            ray_trial_id=ray_trial_id,
            root_seed=root_seed,
        )

    # Run the experiment and report metrics each iteration via report_tune_metrics
    result = runner.run(tune_callback=report_tune_metrics)

    # End-of-training deterministic evaluation (DEFAULT_EVAL_EPISODES) to
    # produce a definitive eval metric for Optuna and trial selection.
    eval_result = runner.algorithm.evaluate(eval_episodes=DEFAULT_EVAL_EPISODES)
    eval_reward = (
        eval_result
        .get("env_runners", {})
        .get("episode_return_mean")
    )
    train_reward = (
        result
        .get("env_runners", {})
        .get("episode_return_mean")
    )

    # Final report with the definitive 100-episode eval reward.
    # This becomes the last entry in the trial's metrics, which is what
    # scope="last" and Optuna's on_trial_complete pick up.
    tune.report({
        "train/episode_return_mean": train_reward,
        "eval/episode_return_mean": eval_reward if eval_reward is not None else train_reward,
    })

    return None


# ============================================================================
# Dispatch and Main Function
# ============================================================================

def _dispatch_experiment(args: Namespace):
    """
    Routes parsed CLI arguments to the appropriate experiment function.

    Args:
        args (Namespace): Parsed command-line arguments.
    """

    # ----- Dispatch experiments based on mode -----
    # Run a single experiment if mode is "single"
    if args.mode == "single":
        run_single_experiment(
            env_config_path=args.env_config,
            algorithm_config_path=args.algorithm_config,
            storage_dir=args.storage_dir,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            root_seed=args.root_seed,
            eval_seed=args.eval_seed,
            resume_from=args.resume_from,
            experiment_name=args.experiment_name,
        )

    # Run hyperparameter tuning if mode is "tune"
    elif args.mode == "tune":
         # If resuming, get the experiment path and resume the experiment
        if args.resume_from is not None:
            experiment_path = str(find_experiment_dir(args.storage_dir, args.resume_from))
            resume_tune_experiment(
                experiment_path=experiment_path,
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus,
                num_cpus_per_env_runner=args.num_cpus_per_env_runner,
                wandb_project=args.wandb_project,
                eval_seed=args.eval_seed,
                top_k=args.top_k,
            )
        # If not resuming, start a new tune experiment
        else:
            run_tune_experiment(
                experiment_name=args.experiment_name,
                env_config_path=args.env_config,
                algorithm_config_path=args.algorithm_config,
                tune_config_path=args.tune_config,
                num_samples=args.num_samples,
                storage_dir=args.storage_dir,
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus,
                num_cpus_per_env_runner=args.num_cpus_per_env_runner,
                wandb_project=args.wandb_project,
                root_seed=args.root_seed,
                eval_seed=args.eval_seed,
                top_k=args.top_k,
            )

    # Run evaluation if mode is "evaluate"
    elif args.mode == "evaluate":
        # Resolve checkpoint and experiment directories by name-based lookup or 
        # explicit path and run the evaluation
        if args.experiment_name: 
            experiment_dir = str(
                find_experiment_dir(args.storage_dir, args.experiment_name)
            )
            checkpoint_dir = str(
                find_checkpoint_dir(Path(experiment_dir), args.checkpoint_number)
            )
        elif args.checkpoint_dir: 
            checkpoint_dir = args.checkpoint_dir
            experiment_dir = find_experiment_dir_from_checkpoint(args.checkpoint_dir)
        else:
            raise ValueError(
                "Either --experiment-name or --checkpoint-dir is required for evaluate mode"
            )
        if not Path(checkpoint_dir).is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_dir}\n"
                f"Checkpoints are only saved every N episodes, so make sure "
                f"the requested checkpoint number is valid."
            )
        run_evaluation(
            checkpoint_dir=checkpoint_dir,
            experiment_dir=experiment_dir,
            env_config_path=args.env_config,
            algorithm_config_path=args.algorithm_config,
            eval_episodes=args.eval_episodes,
            root_seed=args.root_seed,
            visualize=args.visualize,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
        )

    # Run seed evaluation if mode is "seed-eval"
    elif args.mode == "seed-eval":
        run_seed_evaluation(
            n_seeds=args.n_seeds,
            eval_episodes=args.eval_episodes or DEFAULT_EVAL_EPISODES,
            eval_seed=args.eval_seed,
            wandb_project=args.wandb_project,
            num_iterations=args.num_iterations,
            env_config_path=args.env_config,
            algorithm_config_path=args.algorithm_config,
            storage_dir=args.storage_dir,
            experiment_name=args.experiment_name,
            tune_name=args.tune_name,
            top_k=args.top_k,
        )

def main():
    """Main CLI entry point."""
    # Parse command-line arguments and dispatch the appropriate experiment function
    args = parse_args()
    _dispatch_experiment(args)

if __name__ == "__main__":
    main()

