import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import ray
from ray import tune
from ray.tune import RunConfig, CLIReporter

from src.config.schema import EnvironmentConfig, AlgorithmConfig
from src.experiments.runner import ExperimentRunner
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
    print_and_save_best_results,
)


def run_single_experiment(
    env_config_path: str,
    algorithm_config_path: str,
    output_dir: str,
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
        output_dir (str): Output directory for results
        wandb_project (Optional[str]): WandB project name
        wandb_name (Optional[str]): WandB run name
        root_seed (Optional[int]): Root seed for all components (env, RLlib, Ray Tune)
        resume_from (Optional[str]): Path to checkpoint to resume from
        experiment_name (Optional[str]): Name for the experiment (used in folder structure)
    """

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = generate_experiment_name(
            algorithm_config_path=algorithm_config_path,
            search_type="single",  # Indicate this is a single run
            scheduler_type=None,
        )

    # Load configs
    env_config = load_environment_config(env_config_path)
    algorithm_config = load_algorithm_config(algorithm_config_path)

    # Setup WandB to log metrics to WandB
    wandb_config, _ = setup_wandb(
        wandb_project=wandb_project,
        algorithm_config_path=algorithm_config_path,
        mode="single",
        wandb_name=wandb_name,
    )
    
    # Create experiment directory
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(experiment_dir)
    
    # Create experiment runner
    runner = ExperimentRunner(
        env_config=env_config,
        algorithm_config=algorithm_config,
        root_seed=root_seed,
        checkpoint_dir=checkpoint_dir,
        wandb_config=wandb_config,
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        runner.algorithm.load_checkpoint(resume_from)
    
    # Run the experiment and return the result
    result = runner.run()

    return result

def run_tune_experiment(
    env_config_path: str,
    algorithm_config_path: str,
    tune_config_path: str,
    num_samples: int,
    output_dir: str,
    wandb_project: Optional[str] = None,
    scheduler_type: str = "asha",
    search_type: str = "random",
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
        tune_config_path (str): Path to tune config (defines search space)
        num_samples (int): Number of trials to run
        output_dir (str): Output directory for results
        wandb_project (Optional[str]): WandB project name
        scheduler_type (str): Tune scheduler type
        search_type (str): Tune search algorithm type
        num_cpus (int): CPUs per trial. Must be at least 2 (1 for main process + 1 per env runner).
            Default: 2
        num_gpus (int): GPUs per trial
        experiment_name (Optional[str]): Name for the experiment (used in folder structure)
        root_seed (Optional[int]): Root seed for all components (env, RLlib, Ray Tune). Defaults to None.
    """
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = generate_experiment_name(
            algorithm_config_path=algorithm_config_path,
            search_type=search_type,
            scheduler_type=scheduler_type,
        )
    
    # Prepare configuration (i.e., merge environment, algorithm and tune configs)
    config = prepare_tune_config(
        env_config_path=env_config_path,
        algorithm_config_path=algorithm_config_path,
        tune_config_path=tune_config_path,
    )

    # Setup WandB callback to log metrics to WandB
    _, callbacks = setup_wandb(
        wandb_project=wandb_project,
        algorithm_config_path=algorithm_config_path,
        mode="tune",
    )

    # Get required resources per trial (i.e., number of CPUs and GPUs needed per trial)
    resources_per_trial = get_resources_per_trial(
        num_cpus=num_cpus,
        num_gpus=num_gpus, 
        num_env_runners=config["algorithm_config"]["shared"]["num_env_runners"], 
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        algorithm_config_dict=config["algorithm_config"],
    )

    # Wrap trainable with resources
    trainable_with_resources = tune.with_resources(trainable, resources_per_trial)

    # Set metric and mode and add it to the config
    metric = "env_runners/episode_return_mean"
    mode = "max"
    config["tune_metric"] = metric
    config["tune_mode"] = mode

    # Add root seed to the config
    config["root_seed"] = root_seed

    # Setup scheduler and search spaces
    scheduler = get_tune_scheduler(scheduler_type, metric=metric, mode="max")
    search_alg = get_tune_search_algorithm(search_type, metric=metric, mode="max", seed=root_seed)

    # Create a tuner
    # config: base configs and hyperparameter search spaces (passed to trainable)
    # tune_config: controls how Ray Tune runs the search
    # run_config: controls experiment execution and output
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            metric=metric,
            mode=mode,                    
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=output_dir,
            callbacks=callbacks,
            progress_reporter=CLIReporter(),
        ),
    )

    # Run the tuner
    analysis = tuner.fit()
    
    # Print and save best results
    print_and_save_best_results(
        analysis=analysis,
        output_dir=output_dir,
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

    # Extract root_seed from config
    root_seed = config.get("root_seed")

    # Get env config from config dict
    env_config = validate_config(config["env_config"], EnvironmentConfig)
    
    # Get algorithm config from config dict and merge tune parameters into it
    algorithm_config_dict = config["algorithm_config"].copy()
    algorithm_config_dict = merge_tune_params(algorithm_config_dict, config)
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
        root_seed=root_seed,
        checkpoint_dir=checkpoint_dir,
        wandb_config=wandb_config,
    )

    # Create callback for reporting single iterations to Ray Tune
    from functools import partial
    tune_callback = partial(report_tune_metrics, required_metric=config["tune_metric"])

    # Run training with callback
    result = runner.run(tune_callback=tune_callback)

    return result


def generate_experiment_name(
    algorithm_config_path: str,
    search_type: str = "random",
    scheduler_type: Optional[str] = None,
    mode: str = "single",
) -> str:
    """
    Generates a default experiment name based on the algorithm name, search type, and scheduler type.
    
    Args:
        algorithm_config_path (str): Path to algorithm config (to extract algorithm name)
        search_type (str): Search algorithm type
        scheduler_type (Optional[str]): Scheduler type (if not None)
        
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
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build name components
    name_parts = [algo_name, search_type]
    if scheduler_type and scheduler_type != "none":
        name_parts.append(scheduler_type)
    name_parts.append(timestamp)
    
    # Join with underscores
    experiment_name = "_".join(name_parts)
    
    return experiment_name


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run RL experiments")
    
    # Experiment type
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "tune"],
        required=True,
        help="Experiment mode: 'single' for single run, 'tune' for hyperparameter search"
    )
    
    # Config paths
    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="Path to environment config YAML"
    )
    parser.add_argument(
        "--algorithm-config",
        type=str,
        required=True,
        help="Path to algorithm config YAML"
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
    parser.add_argument(
        "--scheduler",
        type=str,
        default="asha",
        choices=["asha", "median_stopping", "hyperband", "fifo", "none"],
        help="Tune scheduler type. Options: 'asha', 'median_stopping', 'hyperband', 'fifo', 'none'"
    )
    parser.add_argument(
        "--search",
        type=str,
        default="random",
        choices=["random", "optuna", "bayesopt", "hyperopt", "ax", "nevergrad"],
        help="Tune search algorithm type. Options: 'random', 'optuna', 'bayesopt', 'hyperopt', 'ax', 'nevergrad'"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_outputs",
        help="Output directory for results and checkpoints"
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
        help="Root seed for all components (env, RLlib, Ray Tune)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
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
    
    # Run a single experiment if mode is "single"
    if args.mode == "single":
        # Run the single experiment
        run_single_experiment(
            env_config_path=args.env_config,
            algorithm_config_path=args.algorithm_config,
            output_dir=args.output_dir,
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

        # Get scheduler type to handle the None case (type(str) -> type(None))
        scheduler_type = None if args.scheduler == "none" else args.scheduler

        # Run the hyperparameter tuning experiment
        run_tune_experiment(
            env_config_path=args.env_config,
            algorithm_config_path=args.algorithm_config,
            tune_config_path=args.tune_config,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            scheduler_type=scheduler_type,
            search_type=args.search,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            experiment_name=args.experiment_name,
            root_seed=args.root_seed,
        )


if __name__ == "__main__":
    main()

