import json
import yaml
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List
import ray
from ray import tune
from ray.tune import RunConfig, CLIReporter

from src.experiments.utils.args import parse_args
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
    load_root_seed_from_metadata,
    resolve_saved_config,
    find_experiment_dir,
    find_checkpoint_dir,
    find_experiment_dir_from_checkpoint,
    generate_experiment_name,
)


TUNE_METRIC = "env_runners/episode_return_mean"
TUNE_MODE = "max"


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
    )

    # Save run metadata once before training starts
    save_run_metadata(
        output_dir=checkpoint_dir,
        runner=runner,
        root_seed=root_seed,
    )

    # Resume from checkpoint if specified
    if resume_from:
        runner.algorithm.load_checkpoint(resume_from)

    # Run the experiment and return the result
    result = runner.run()

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

    # Set metric and mode and add it to the search config
    metric = TUNE_METRIC
    mode = TUNE_MODE
    search_config["tune_metric"] = metric
    search_config["tune_mode"] = mode

    # Add root seed to the search config
    search_config["root_seed"] = root_seed

    # Setup scheduler and search algorithm from validated config objects
    scheduler = get_tune_scheduler(scheduler_config, metric=metric, mode=mode)
    search_seed = seed_manager.get_seed_int('train')
    search_alg = get_tune_search_algorithm(search_algorithm_config, metric=metric, mode=mode, seed=search_seed)

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

    # Run the tuner
    analysis = tuner.fit()

    # Run post-tune analysis including best results, evaluation, convergence
    _run_post_tune_analysis(analysis, metric=metric, mode=mode, root_seed=root_seed)
    
    return analysis

def resume_tune_experiment(
    experiment_path: str,
    num_cpus: Optional[int] = None,
    num_gpus: int = 0,
    num_cpus_per_env_runner: Optional[int] = None,
    wandb_project: Optional[str] = None,
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
    root_seed = first_trial_config.get("root_seed")

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
    _run_post_tune_analysis(analysis, metric=metric, mode=mode, root_seed=root_seed)

    return analysis

def _run_post_tune_analysis(
    analysis: tune.ResultGrid,
    metric: str = TUNE_METRIC,
    mode: str = TUNE_MODE,
    root_seed: Optional[int] = None,
    last_k: Optional[int] = 10,
):
    """
    Prints and saves best results, evaluates the best trial's checkpoint,
    and generates a convergence report.

    Args:
        analysis (tune.ResultGrid): Results returned by ``tuner.fit()``.
        metric (str): Metric to optimize.
        mode (str): Optimization mode (``"min"`` or ``"max"``).
        root_seed (Optional[int]): Root seed forwarded to the evaluation runner.
        last_k (Optional[int]): Number of last reported values to average
            when selecting the best trial. Defaults to 10.
    """

    # Print and save best results
    best_info = print_and_save_best_results(
        analysis=analysis,
        metric=metric,
        mode=mode,
        last_k=last_k,
    )

    # Get the best checkpoint and trial directory
    best_checkpoint = best_info.get("best_trial_best_checkpoint")
    best_trial_dir = best_info.get("best_trial_path")

    # Run evaluation and update best trial results
    if best_checkpoint:
        print("\n" + "=" * 80)
        print("EVALUATING BEST TRIAL")
        print("=" * 80)

        # Run evaluation on the best checkpoint
        eval_result = run_evaluation(
            checkpoint_dir=best_checkpoint,
            experiment_dir=best_trial_dir,
            eval_episodes=50,
            root_seed=root_seed,
            visualize=True,
        )

        # Update best trial results with the evaluation reward from the best checkpoint
        eval_reward = eval_result.get("evaluation", {}).get("episode_reward_mean")
        if eval_reward is not None:
            results_yaml_path = Path(analysis.experiment_path) / "best_trial_results.yaml"
            if results_yaml_path.exists():
                with open(results_yaml_path, "r") as f:
                    saved = yaml.safe_load(f)
                updated = {}
                for k, v in saved.items():
                    updated[k] = v
                    if k == "best_trial_best_metric":
                        updated["best_trial_eval_metric"] = float(eval_reward)
                with open(results_yaml_path, "w") as f:
                    yaml.dump(updated, f, default_flow_style=False, sort_keys=False)
                print(f"[INFO] Eval metric ({eval_reward:.4f}) saved to: {results_yaml_path}")

    # Create a tuner convergence report
    analyze_tune_convergence(
        analysis=analysis,
        metric=metric,
        mode=mode,
        last_k=last_k,
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

    # Create callback for reporting single iterations to Ray Tune
    tune_callback = partial(report_tune_metrics, required_metrics=config["tune_metric"])

    # Run the experiment and return the result
    result = runner.run(tune_callback=tune_callback)

    return result


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

def main():
    """Main CLI entry point."""
    # Parse command-line arguments and dispatch the appropriate experiment function
    args = parse_args()
    _dispatch_experiment(args)

if __name__ == "__main__":
    main()

