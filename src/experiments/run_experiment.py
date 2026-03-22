import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import ray
from ray import tune
from ray.tune import RunConfig, CLIReporter
from ray.tune import CheckpointConfig

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


# ============================================================================
# Helpers
# ============================================================================

def _save_run_metadata(
    output_dir: str,
    runner: 'ExperimentRunner',
    ray_trial_id: Optional[str] = None,
):
    """
    Writes a one-time metadata file at the start of a run or trial.
    The file is written only if it does not already exist, so it is safe to
    call multiple times without overwriting.

    Args:
        output_dir (str): Directory to write the metadata file into.
        runner (ExperimentRunner): The experiment runner (used to access the
            underlying RLlib Algorithm for logdir and config).
        ray_trial_id (Optional[str]): Ray Tune trial ID, or None for single runs.
    """

    # Set the filename for the metadata file
    METADATA_FILENAME = "metadata.json"

    # Create the path to the metadata file and check if it already exists
    meta_path = Path(output_dir) / METADATA_FILENAME
    if meta_path.exists():
        return

    # Get the trainer from the runner instance
    trainer = runner.algorithm.trainer

    # Create the metadata dictionary
    meta = {
        "ray_trial_id": ray_trial_id or Path(trainer.logdir).name,
        "ray_logdir": str(trainer.logdir),
        "config": trainer.config.to_dict(),
    }

    # Create the directory if it does not exist
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the metadata to the file
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[INFO] Saved run metadata to: {meta_path}")

def find_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Searches recursively for a directory matching ``experiment_name``
    under *base_dir*.  Supports exact matches as well as prefix matching
    so that tune trial directories can be found with a short prefix
    (e.g. ``trainable_da92fc08``) instead of the full long name.

    Args:
        base_dir (str): Root directory to search under (e.g., "./experiment_outputs").
        experiment_name (str): Exact directory name or a unique prefix.

    Returns:
        experiment_dir (Path): Path to the found experiment directory.
    """

    # Get the base directory and check if it exists
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Base directory '{base_dir}' does not exist."
        )

    # Search for directories with exact matching first
    matches = [
        p for p in base.rglob(experiment_name)
        if p.is_dir() and p.name == experiment_name
    ]

    # If no exact matches are found, fall back to prefix match
    if len(matches) == 0:
        matches = [
            p for p in base.rglob(f"{experiment_name}*")
            if p.is_dir() and p.name.startswith(experiment_name)
        ]

    # If no exact or prefix matches are found, raise an error
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Experiment '{experiment_name}' not found under '{base_dir}'."
        )
    
    # If multiple matches are found, raise an error
    if len(matches) > 1:
        paths_str = "\n  ".join(str(m) for m in matches)
        raise ValueError(
            f"Multiple directories matching '{experiment_name}' found under '{base_dir}':\n  {paths_str}\n"
            f"Please provide a more specific prefix or the full name."
        )

    return matches[0]


def find_checkpoint_dir(
    experiment_dir: Path,
    checkpoint_number: Optional[int] = None,
) -> Path:
    """
    Resolves the checkpoint directory under an experiment run folder.

    If ``checkpoint_number`` is given, uses ``checkpoint_<n>`` or zero-padded Ray Tune
    naming when the unpadded folder is missing. Otherwise prefers ``checkpoint_best``,
    then ``checkpoint_final``, then the last sorted ``checkpoint_*`` directory, or
    ``checkpoint_final`` as a fallback name (may not exist yet).
    """

    # If checkpoint number is provided, use it to find the checkpoint directory
    if checkpoint_number is not None:
        checkpoint_folder = f"checkpoint_{checkpoint_number}"
        if not (experiment_dir / checkpoint_folder).is_dir():
            padded = f"checkpoint_{int(checkpoint_number):06d}"
            if (experiment_dir / padded).is_dir():
                checkpoint_folder = padded
    
    # If no checkpoint number is provided, use the best checkpoint directory first
    elif (experiment_dir / "checkpoint_best").is_dir():
        checkpoint_folder = "checkpoint_best"
    
    # If no best checkpoint directory is found, use the final checkpoint directory
    elif (experiment_dir / "checkpoint_final").is_dir():
        checkpoint_folder = "checkpoint_final"
    
    # If no best or final checkpoint directory is found, use the last sorted checkpoint directory
    else:
        tune_chkpts = sorted(
            p for p in experiment_dir.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint_")
        )
        if tune_chkpts:
            checkpoint_folder = tune_chkpts[-1].name
        else:
            checkpoint_folder = "checkpoint_final"

    # Assemble the checkpoint directory 
    checkpoint_dir = experiment_dir / checkpoint_folder

    return checkpoint_dir


def _find_experiment_dir_from_checkpoint(checkpoint_dir: str) -> Path:
    """
    Walks up from a checkpoint directory to find the experiment directory,
    identified by the presence of ``env_config.yaml`` (saved during training).
    Falls back to the immediate parent if no config file is found.

    Args:
        checkpoint_dir (str): Path to a checkpoint directory.

    Returns:
        experiment_dir (Path): Path to the resolved experiment directory.
    """
    
    # Get the given checkpoint directory
    current = Path(checkpoint_dir).resolve()

    # Walk up the directory tree until the experiment directory is found
    for parent in current.parents:
        if (parent / "env_config.yaml").exists():
            return parent
        if parent == parent.parent:
            break

    # Assemble the experiment dir
    experiment_dir = Path(checkpoint_dir).parent

    return experiment_dir

def generate_experiment_name(
    env_config_path: str,
    algorithm_config_path: str,
    mode: str = "single",
    search_type: Optional[str] = None,
    scheduler_type: Optional[str] = None,
) -> str:
    """
    Generates a default experiment name based on the algorithm name, search type, and scheduler type.
    
    Args:
        env_config_path (str): Path to environment config (to extract environment name)
        algorithm_config_path (str): Path to algorithm config (to extract algorithm name)
        mode (str): Experiment mode ("single" or "tune")
        search_type (Optional[str]): Search algorithm type string (e.g. "optuna")
        scheduler_type (Optional[str]): Scheduler type string (e.g. "asha")
        
    Returns:
        experiment_name (str): Generated experiment name
    """
    # Get algorithm name from config
    try:
        algorithm_config = load_algorithm_config(algorithm_config_path)
        algo_name = algorithm_config.name
    except Exception:
        # Fallback to config filename if loading fails
        algo_name = Path(algorithm_config_path).stem

    env_config = load_environment_config(env_config_path)
    n_warehouses = env_config.n_warehouses
    n_skus = env_config.n_skus
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Build name components
    name_parts = [algo_name.upper(), mode, f"{n_warehouses}WH", f"{n_skus}SKU"]
    if mode == "tune":
        if search_type and search_type != "random":
            name_parts.append(search_type)
        if scheduler_type and scheduler_type != "fifo":
            name_parts.append(scheduler_type)
    name_parts.append(timestamp)
    
    # Join with underscores
    experiment_name = "_".join(name_parts)
    
    return experiment_name


# ============================================================================
# Run Experiment Functions
# ============================================================================

def run_single_experiment(
    env_config_path: str,
    algorithm_config_path: str,
    storage_dir: str,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    root_seed: Optional[int] = None,
    resume_from: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    """
    Runs a single experiment without hyperparameter tuning.
    
    Args:
        env_config_path (str): Path to environment config
        algorithm_config_path (str): Path to algorithm config
        storage_dir (str): Root directory for experiment outputs (experiment folder
            is created as storage_dir / experiment_name).
        wandb_project (Optional[str]): WandB project name
        wandb_name (Optional[str]): WandB run name
        root_seed (Optional[int]): Root seed for reproducibility.
        resume_from (Optional[str]): Path to checkpoint to resume from
        experiment_name (Optional[str]): Name for the experiment (used in folder structure)
    """

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
    _save_run_metadata(
        output_dir=checkpoint_dir,
        runner=runner,
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
    Runs standalone evaluation of a trained checkpoint.
    
    Configs are loaded from the experiment directory (saved during training)
    if env_config_path / algorithm_config_path are not provided.
    
    Args:
        checkpoint_dir (str): Path to checkpoint directory to evaluate.
        experiment_dir (str): Path to the experiment directory (contains saved configs,
            receives eval outputs like eval_results.yaml and visualizations).
        env_config_path (Optional[str]): Path to environment config. If None, loaded from
            saved config in experiment_dir.
        algorithm_config_path (Optional[str]): Path to algorithm config. If None, loaded from
            saved config in experiment_dir.
        eval_episodes (Optional[int]): Number of evaluation episodes (overrides config default)
        root_seed (Optional[int]): Root seed for reproducibility. Split into train_seed and eval_seed 
            by EvaluationRunner (only eval_seed is used).
        visualize (bool): If True, run manual rollout and generate visualization plots.
        wandb_project (Optional[str]): WandB project name
        wandb_name (Optional[str]): WandB run name
    """

    # Resolve config paths: prefer provided paths, fall back to saved configs in experiment dir
    if env_config_path is None:
        env_config_path = str(Path(experiment_dir) / "env_config.yaml")
        if not Path(env_config_path).exists():
            raise FileNotFoundError(
                f"No saved environment config found at: {env_config_path}\n"
                f"Provide --env-config explicitly."
            )
        print(f"[INFO] Loading saved environment config from: {env_config_path}")
    if algorithm_config_path is None:
        algorithm_config_path = str(Path(experiment_dir) / "algorithm_config.yaml")
        if not Path(algorithm_config_path).exists():
            raise FileNotFoundError(
                f"No saved algorithm config found at: {algorithm_config_path}\n"
                f"Provide --algorithm-config explicitly."
            )
        print(f"[INFO] Loading saved algorithm config from: {algorithm_config_path}")

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
        env_config_path (str): Path to base environment config
        algorithm_config_path (str): Path to base algorithm config
        tune_config_path (str): Path to tune config (defines search space, scheduler, search algorithm)
        num_samples (int): Number of trials to run
        storage_dir (str): Root directory for experiment outputs (Ray Tune creates
            storage_dir / experiment_name / trial_dirs).
        wandb_project (Optional[str]): WandB project name
        num_cpus (Optional[int]): CPUs per trial.
        num_gpus (int): GPUs per trial
        num_cpus_per_env_runner (Optional[int]): CPUs per env runner
        experiment_name (Optional[str]): Name for the experiment (used in folder structure)
        root_seed (Optional[int]): Root seed for reproducibility. Defaults to None.
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
    metric = "env_runners/episode_return_mean"
    mode = "max"
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
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
            ),
        ),
    )

    # Run the tuner
    analysis = tuner.fit()
    
    # Print and save best results
    best_info = print_and_save_best_results(
        analysis=analysis,
        metric=metric,
        mode=mode,
    )

    # Get best checkpoint and trial directory
    best_checkpoint = best_info.get("best_checkpoint")
    if best_checkpoint:
        best_trial_dir = best_info.get("best_trial_path", str(Path(best_checkpoint).parent))

        # Print header
        print("\n" + "=" * 80)
        print("EVALUATING BEST TRIAL")
        print("=" * 80)

        # Evaluate best trial's best checkpoint with visualization
        eval_result = run_evaluation(
            checkpoint_dir=best_checkpoint,
            experiment_dir=best_trial_dir,
            eval_episodes=50,
            root_seed=root_seed,
            visualize=True,
            wandb_project=wandb_project,
            wandb_name=f"eval_best_{experiment_name}",
        )

        # Insert eval metric into best_trial_results.yaml
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

    # Analyze and save convergence report
    analyze_tune_convergence(
        analysis=analysis,
        metric=metric,
        mode=mode,
    )
    
    return analysis

def trainable(config: Dict[str, Any]):
    """
    Implements a Ray Tune trainable function.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary from Ray Tune
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
    # Fallback: use config if available, or let ExperimentRunner handle it
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
        _save_run_metadata(
            output_dir=checkpoint_dir,
            runner=runner,
            ray_trial_id=ray_trial_id,
        )

    # Create callback for reporting single iterations to Ray Tune
    from functools import partial
    tune_callback = partial(report_tune_metrics, required_metrics=config["tune_metric"])

    # Run training with callback
    result = runner.run(tune_callback=tune_callback)

    return result


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main CLI entry point."""

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description="Run RL experiments")
    
    # Experiment type
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "tune", "evaluate"],
        required=True,
        help="Experiment mode: 'single' for training, 'tune' for hyperparameter search, 'evaluate' for evaluation"
    )
    
    # Config paths
    parser.add_argument(
        "--env-config",
        type=str,
        help="Path to environment config YAML (required for single/tune modes; "
             "optional for evaluate when using --experiment-name)"
    )
    parser.add_argument(
        "--algorithm-config",
        type=str,
        help="Path to algorithm config YAML (required for single/tune modes; "
             "optional for evaluate when using --experiment-name)"
    )
    
    # Tune-specific
    parser.add_argument(
        "--tune-config",
        type=str,
        help="Path to tune config YAML (required for tune mode)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of trials for tune mode"
    )
    # Output
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="./experiment_outputs",
        help="Root directory for experiment outputs (experiment folders are created inside)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (used in folder structure). If not provided, auto-generated."
    )
    
    # WandB
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        help="WandB run name (for single mode)"
    )
    
    # Other
    parser.add_argument(
        "--root-seed",
        type=int,
        help="Root seed for reproducibility. Split into independent train_seed and eval_seed internally."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
    )

    # Evaluation-specific
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Path to checkpoint directory for evaluation. Mutually exclusive with --experiment-name."
    )
    parser.add_argument(
        "--checkpoint-number",
        type=str,
        help="Checkpoint identifier to evaluate (e.g. '50' for checkpoint_50, "
             "'000000' for checkpoint_000000). "
             "Only used with --experiment-name. If omitted, defaults to checkpoint_best "
             "(falls back to checkpoint_final if checkpoint_best does not exist)."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help="Number of evaluation episodes (overrides config default)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate visualization plots from manual rollout (evaluate mode only)"
    )
    
    # Resources
    parser.add_argument(
        "--num-cpus",
        type=int,
        help="Number of CPUs per trial (tune mode)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs per trial (tune mode)"
    )
    parser.add_argument(
        "--num-cpus-per-env-runner",
        type=int,
        help="Number of CPUs per environment runner (tune mode)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate that config paths are provided for modes that need them
    if args.mode in ("single", "tune"):
        if not args.env_config:
            raise ValueError("--env-config is required for single/tune mode")
        if not args.algorithm_config:
            raise ValueError("--algorithm-config is required for single/tune mode")

    # Validate evaluate mode: require either --checkpoint-dir or --experiment-name
    if args.mode == "evaluate":
        if args.checkpoint_dir and args.experiment_name:
            raise ValueError(
                "Cannot specify both --checkpoint-dir and --experiment-name. "
                "Use one or the other."
            )
        if not args.checkpoint_dir and not args.experiment_name:
            raise ValueError(
                "For evaluate mode, either --checkpoint-dir or --experiment-name "
                "(with optional --checkpoint-number) must be specified."
            )
        if args.checkpoint_number is not None and not args.experiment_name:
            raise ValueError(
                "--checkpoint-number can only be used with --experiment-name, "
                "not with --checkpoint-dir (which already points to a specific checkpoint)."
            )

    # Run a single experiment if mode is "single"
    if args.mode == "single":
        # Run the single experiment
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

    # Run a hyperparameter tuning if mode is "tune"
    elif args.mode == "tune":
        # Check if tune config is provided (since it has no parameter 'required' in the parser)
        if not args.tune_config:
            raise ValueError("--tune-config is required for tune mode")

        # Run the tune experiment
        run_tune_experiment(
            env_config_path=args.env_config,
            algorithm_config_path=args.algorithm_config,
            tune_config_path=args.tune_config,
            num_samples=args.num_samples,
            storage_dir=args.storage_dir,
            wandb_project=args.wandb_project,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            experiment_name=args.experiment_name,
            root_seed=args.root_seed,
        )

    # Run evaluation if mode is "evaluate"
    elif args.mode == "evaluate":
        # Option 1: Name-based lookup
        if args.experiment_name:
            # Get the experiment directory
            experiment_dir = str(find_experiment_dir(args.storage_dir, args.experiment_name))
            # Get the checkpoint directory
            checkpoint_dir = str(find_checkpoint_dir(Path(experiment_dir), args.checkpoint_number))

        # Option 2: Explicit checkpoint path
        elif args.checkpoint_dir:
            # Get the explicit path
            checkpoint_dir = args.checkpoint_dir
            # Get the experiment directory from the checkpoint directory
            experiment_dir = _find_experiment_dir_from_checkpoint(args.checkpoint_dir)
        else:
            raise ValueError(
                "Either --experiment-name or --checkpoint-dir is required for evaluate mode"
            )

        # Verify the checkpoint directory actually exists
        if not Path(checkpoint_dir).is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_dir}\n"
                f"Checkpoints are only saved every N episodes, so make sure "
                f"the requested checkpoint number is valid."
            )

        # Run evaluation
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


if __name__ == "__main__":
    main()

