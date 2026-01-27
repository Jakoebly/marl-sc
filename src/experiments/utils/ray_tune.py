from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import yaml
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule, HyperBandScheduler, FIFOScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune import PlacementGroupFactory, ResultGrid
from ray import tune
import ray
import nevergrad as ng

from src.config.loader import load_environment_config, load_algorithm_config
from src.config.loader import load_tune_config

def create_tune_config(
    base_env_config_path: str,
    base_algorithm_config_path: str,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates a Ray Tune configuration from base configs and search space.
    
    Args:
        base_env_config_path (str): Path to base environment config
        base_algorithm_config_path (str): Path to base algorithm config
        search_space (Dict[str, Any]): Dictionary defining hyperparameter search space
        
    Returns:
        tune_config (Dict[str, Any]): Tune config dictionary
    """

    # Load base configs
    env_config = load_environment_config(base_env_config_path)
    algorithm_config = load_algorithm_config(base_algorithm_config_path)
    
    # Convert base configs from Pydantic models to dicts
    env_config_dict = env_config.model_dump(by_alias=True)
    algorithm_config_dict = algorithm_config.model_dump(by_alias=True)
    
    # Create tune config with base configs
    tune_config = {
        "env_config": env_config_dict,
        "algorithm_config": algorithm_config_dict,
    }
    
    # Merge search space (nested dicts for shared and algorithm_specific)
    if "shared" in search_space:
        tune_config["shared"] = search_space["shared"]
    if "algorithm_specific" in search_space:
        tune_config["algorithm_specific"] = search_space["algorithm_specific"]
    
    return tune_config

def _convert_single_search_spec(search_spec: Dict[str, Any]) -> Any:
    """
    Converts a single search space specification to a Ray Tune search space object.
    
    Args:
        search_spec (Dict[str, Any]): Dictionary with type and parameters
        
    Returns:
        tune_search (Any): Ray Tune search space object
    """

    # Get specified search type
    search_type = search_spec.get("type", "choice")
    
    # Return the corresponding Ray Tune search space object
    if search_type == "choice":
        return tune.choice(search_spec["values"])
    elif search_type == "uniform":
        return tune.uniform(search_spec["low"], search_spec["high"])
    elif search_type == "loguniform":
        return tune.loguniform(search_spec["low"], search_spec["high"])
    elif search_type == "randint":
        return tune.randint(search_spec["low"], search_spec["high"])
    elif search_type == "grid_search":
        return tune.grid_search(search_spec["values"])
    else:
        raise ValueError(f"Unknown search type: {search_type}")

def convert_to_tune_search(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a nested search space dictionary to Ray Tune format.
    
    Args:
        search_space (Dict[str, Any]): Dictionary with 'shared' and/or 'algorithm_specific' keys,
                                       each containing nested dictionaries of search specs
        
    Returns:
        tune_search_space (Dict[str, Any]): Ray Tune search space dictionary
    """

    # Initialize the tune search space dictionary
    tune_search_space = {}

    # Iterate over the nested dictionaries in the search space dictionary to convert each search space specification to Ray Tune format
    for key, value in search_space.items():
        if key in ["shared", "algorithm_specific"]:
            nested_space = {}
            for nested_key, nested_value in value.items():
                nested_space[nested_key] = _convert_single_search_spec(nested_value)
            tune_search_space[key] = nested_space
        else:
            tune_search_space[key] = _convert_single_search_spec(value)
    
    return tune_search_space

def merge_tune_params(algorithm_config_dict: Dict[str, Any], tune_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges tune parameters into algorithm config dictionary.
    
    Args:
        algorithm_config_dict (Dict[str, Any]): Base algorithm config dictionary
        tune_config (Dict[str, Any]): Tune config dictionary containing 'shared' and 'algorithm_specific' keys
        
    Returns:
        merged_config (Dict[str, Any]): Merged algorithm config dictionary
    """
    merged = algorithm_config_dict.copy()
    
    # Merge shared parameters
    if "shared" in tune_config:
        if "shared" not in merged:
            merged["shared"] = {}
        merged["shared"].update(tune_config["shared"])
    
    # Merge algorithm_specific parameters
    if "algorithm_specific" in tune_config:
        if "algorithm_specific" not in merged:
            merged["algorithm_specific"] = {}
        merged["algorithm_specific"].update(tune_config["algorithm_specific"])
    
    return merged

def prepare_tune_config(
    env_config_path: str,
    algorithm_config_path: str,
    tune_config_path: str,
) -> Dict[str, Any]:
    """
    Prepares the tune configuration by loading configs and building search spaces.
    
    Args:
        env_config_path (str): Path to environment config
        algorithm_config_path (str): Path to algorithm config
        tune_config_path (str): Path to tune config (defines search space)
        
    Returns:
        config (Dict[str, Any]): Complete tune configuration dictionary
    """

    # Load and validate tune config
    tune_config = load_tune_config(tune_config_path)
    
    # Build search space dict from direct fields
    tune_config_dict = tune_config.model_dump(exclude_none=True)
    search_space = {}
    if "shared" in tune_config_dict:
        search_space["shared"] = tune_config_dict["shared"]
    if "algorithm_specific" in tune_config_dict:
        search_space["algorithm_specific"] = tune_config_dict["algorithm_specific"]
    
    # Convert search space to Ray Tune format
    tune_search_space = convert_to_tune_search(search_space)
    
    # Create tune config
    config = create_tune_config(
        env_config_path,
        algorithm_config_path,
        tune_search_space
    )

    return config

def get_tune_scheduler(
    scheduler_type: str = "asha",
    metric: str = "env_runners/episode_return_mean",
    mode: str = "max",
    **kwargs,
) -> Any:
    """
    Gets a Ray Tune scheduler.
    
    Args:
        scheduler_type (str): Type of scheduler. Options:
            - "asha": Asynchronous Successive Halving Algorithm (default)
            - "median_stopping": Median Stopping Rule
            - "hyperband": HyperBand scheduler
            - "fifo": First-In-First-Out scheduler (no early stopping)
        metric (str): Metric to optimize. Default: "env_runners.episode_return_mean"
        mode (str): Optimization mode ("min" or "max"). Default: "max"
        **kwargs (Any): Scheduler-specific arguments (e.g., metric, mode, max_t, grace_period)
        
    Returns:
        tune_scheduler (Any): Tune scheduler instance or None
    """

    # Return the corresponding Ray Tune scheduler
    if scheduler_type == "asha":
        return ASHAScheduler(metric=metric, mode=mode, **kwargs)
    elif scheduler_type == "median_stopping":
        return MedianStoppingRule(metric=metric, mode=mode, **kwargs)
    elif scheduler_type == "hyperband":
        return HyperBandScheduler(metric=metric, mode=mode, **kwargs)
    elif scheduler_type == "fifo":
        return FIFOScheduler(**kwargs)
    else:
        return None

def get_tune_search_algorithm(
    search_type: str = "random",
    metric: str = "env_runners/episode_return_mean",
    mode: str = "max",
    seed: Optional[int] = None,
    **kwargs,
) -> Any:
    """
    Gets a Ray Tune search algorithm.
    
    Args:
        search_type (str): Type of search algorithm. Options:
            - "random": Random search (default, no algorithm needed)
            - "optuna": Bayesian optimization via Optuna
            - "bayesopt": Bayesian optimization via bayesian-optimization library
            - "hyperopt": Bayesian optimization via HyperOpt
            - "ax": Bayesian optimization via Facebook Ax
            - "nevergrad": Derivative-free optimization via Nevergrad
        metric (str): Metric to optimize. Default: "episode_reward_mean"
        mode (str): Optimization mode ("min" or "max"). Default: "max"
        seed (Optional[int]): Random seed for search algorithm reproducibility. Defaults to None.
        **kwargs (Any): Additional search-specific arguments
        
    Returns:
        tune_search_algorithm (Any): Tune search algorithm instance or None (for random search)
    """

    # Return the corresponding Ray Tune search algorithm
    if search_type == "random":
        return None  # Default random search
    elif search_type == "optuna":
        return OptunaSearch(metric=metric, mode=mode, seed=seed, **kwargs)
    elif search_type == "bayesopt":
        return BayesOptSearch(metric=metric, mode=mode, random_state=seed, **kwargs)
    elif search_type == "hyperopt":
        return HyperOptSearch(metric=metric, mode=mode, random_state_seed=seed, **kwargs)
    elif search_type == "ax":
        return AxSearch(metric=metric, mode=mode, random_seed=seed, **kwargs)
    elif search_type == "nevergrad":
        return NevergradSearch(optimizer=ng.optimizers.NGOpt, optimizer_kwargs={"seed": seed}, metric=metric, mode=mode, random_state=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown search_type: '{search_type}'. "
            f"Supported types: 'random', 'optuna', 'bayesopt', 'hyperopt', 'ax', 'nevergrad'"
        )

def get_resources_per_trial(
    num_cpus: int | None, 
    num_gpus: int, 
    num_env_runners: int, 
    num_cpus_per_env_runner: int | None,
    algorithm_config_dict: Dict[str, Any],
) -> Dict[str, int]:
    """
    Creates the placement group for the resources per trial.
    
    Args:
        num_cpus (int | None): Number of CPUs per trial. If not provided, set to 1 + (num_env_runners * num_cpus_per_env_runner)
        num_gpus (int): Number of GPUs per trial
        num_env_runners (int): Number of environment runners per trial
        num_cpus_per_env_runner (int | None): CPUs per environment runner. If not provided, set to the value from the algorithm config.
        algorithm_config_dict (Dict[str, Any]): Algorithm configuration dictionary

    Returns:
        placement_group_factory (PlacementGroupFactory): Placement group factory for resources per trial
    """

    # Validate and adjust resources per trial
    num_cpus, num_cpus_per_env_runner = _validate_and_adjust_resources(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        algorithm_config_dict=algorithm_config_dict,
    )

    # Create the placement group for the resources per trial
    placement_group_factory = _make_placement_group_factory(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
    )

    return placement_group_factory

def _validate_and_adjust_resources(
    num_cpus: int | None, 
    num_gpus: int, 
    num_env_runners: int, 
    num_cpus_per_env_runner: int | None,
    algorithm_config_dict: Dict[str, Any],
) -> Tuple[int, int, int]:
    """
    Validates and adjusts resources. Adjustments are made the following way:
    - num_cpus: If not provided in args, set to 1 + (num_env_runners * num_cpus_per_env_runner)
    - num_cpus_per_env_runner: If not provided in args, set to the value from the algorithm config.
    
    Args:
        num_cpus (int): Number of CPUs per trial
        num_gpus (int): Number of GPUs per trial
        num_env_runners (int): Number of environment runners per trial
        num_cpus_per_env_runner (int): CPUs per environment runner. Default: 1
        algorithm_config_dict (Dict[str, Any]): Algorithm configuration dictionary

    Returns:
        num_cpus (int): Validated and adjusted number of CPUs per trial
        num_gpus (int): Validated number of GPUs per trial
        num_cpus_per_env_runner (int): Validated and adjusted number of CPUs per environment runner
    """
    
    # Validate inputs
    if num_cpus is not None and num_cpus < 1:
        raise ValueError("num_cpus must be >= 1")
    if num_gpus is not None and num_gpus < 0:
        raise ValueError("num_gpus must be >= 0")
    if num_env_runners < 0:
        raise ValueError("num_env_runners must be >= 0")
    if num_cpus_per_env_runner is not None and num_cpus_per_env_runner < 1:
        raise ValueError("num_cpus_per_env_runner must be >= 1")

    # Get number of CPUs per environment runner from config if not provided in args
    if num_cpus_per_env_runner is None:
        num_cpus_per_env_runner = algorithm_config_dict["shared"]["num_cpus_per_env_runner"]

    # Compute required number of CPUs per trial if not provided in args
    if num_cpus is None:
        num_cpus = 1 + (num_env_runners * num_cpus_per_env_runner) # 1 main + num_env_runners * num_cpus_per_env_runner

    # Validate that the number of CPUs per trial is sufficient
    min_required_cpus = 1 + (num_env_runners * num_cpus_per_env_runner)
    if num_cpus < min_required_cpus:
        raise ValueError(
            f"num_cpus={num_cpus} is too small. "
            f"Need at least {min_required_cpus} CPUs "
            f"(1 for main actor + {num_env_runners} env runners × {num_cpus_per_env_runner} CPUs per runner)."
        )
    
    # Get available resources from Ray cluster
    cluster_resources = ray.cluster_resources()
    available_cpus = int(cluster_resources.get("CPU", 0))
    available_gpus = int(cluster_resources.get("GPU", 0))

    # Validate that the number of CPUs meets available resources
    if num_cpus > available_cpus:
        raise ValueError(
            f"Requested {num_cpus} CPUs per trial, but cluster has only {available_cpus} CPUs available. "
            f"Reduce num_cpus, num_env_runners, or num_cpus_per_env_runner."
        )

    # Validate that the number of GPUs meets available resources
    if num_gpus > available_gpus:
        raise ValueError(
            f"Requested {num_gpus} GPUs per trial, but cluster has only {available_gpus} GPUs available."
        )

    # Calculate max concurrent trials by CPU and GPU
    max_concurrent_by_cpu = available_cpus // num_cpus
    max_concurrent_by_gpu = available_gpus // num_gpus if num_gpus > 0 else float('inf')
    max_concurrent_trials = int(min(max_concurrent_by_cpu, max_concurrent_by_gpu))

    if max_concurrent_trials < 1:
        raise ValueError(
            f"Cannot run even 1 trial. Required: {num_cpus} CPUs, {num_gpus} GPUs. "
            f"Available: {available_cpus} CPUs, {available_gpus} GPUs."
        )

    print(f"[INFO] Resource allocation per trial:")
    print(f"  - CPUs: {num_cpus} (1 main actor + {num_env_runners} env runners × {num_cpus_per_env_runner} CPUs)")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Max concurrent trials: {max_concurrent_trials}")

    return num_cpus, num_cpus_per_env_runner
    
def _make_placement_group_factory(
    num_cpus: int,
    num_env_runners: int,
    num_cpus_per_env_runner: int,
    num_gpus: int,
) -> PlacementGroupFactory:
    """
    Creates a placement group factory for based on specified resources.
    
    Args:
        num_env_runners (int): Number of environment runners per trial
        num_cpus_per_env_runner (int): CPUs per environment runner
        num_gpus (int): Number of GPUs per trial
        
    Returns:
        placement_group_factory (PlacementGroupFactory): Placement group factory for resources per trial
    """

    # Create bundles: one per env runner
    main_cpus = max(1, num_cpus - (num_env_runners * num_cpus_per_env_runner))
    bundles = [{"CPU": main_cpus, "GPU": num_gpus}]  # bundle 0: main actor
    for _ in range(num_env_runners):
        bundles.append({"CPU": num_cpus_per_env_runner})
    
    return PlacementGroupFactory(bundles, strategy="PACK")

def extract_nested_metric(data: Dict[str, Any], metric_path: str) -> Any:
    """
    Extracts a metric value from a nested dictionary using a path-like string.
    Supports both nested paths (e.g., "env_runners/episode_return_mean") and 
    simple keys (e.g., "episode_return_mean").
    
    Args:
        data (Dict[str, Any]): Dictionary to extract metric from
        metric_path (str): Path to metric, using "/" as separator for nested keys
        
    Returns:
        metric_value (Any): Extracted metric value, or None if not found
    """
    
    # Check if the metric is a nested path
    if "/" in metric_path:
        parts = metric_path.split("/")
        nested_dict = data
        for part in parts:
            if isinstance(nested_dict, dict) and part in nested_dict:
                nested_dict = nested_dict[part]
            else:
                return None
        return nested_dict

    # Check if the metric is a simple key    
    else:
        return data.get(metric_path)

def report_tune_metrics(
    result: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    required_metrics: List[str] | str = ["env_runners/episode_return_mean"],
) -> None:
    """Report results and optionally a checkpoint to Ray Tune."""
    
    # Handle both single metric string and list of metrics strings by normalizing to list
    if isinstance(required_metrics, str):
        metrics_list = [required_metrics]
    else:
        metrics_list = required_metrics
    
    # Initialize tune metrics to report with training iteration
    tune_metrics = {
        "training_iteration": result.get("training_iteration")
    }

    # Extract the required metrics from nested result dictionary and add to tune metrics
    tune_metrics = {}
    for metric in metrics_list:
        metric_value = extract_nested_metric(result, metric)
        tune_metrics[metric] = metric_value

    # Report results including checkpoint if provided
    if checkpoint_path:
        tune.report(
            tune_metrics,
            checkpoint=tune.Checkpoint.from_directory(checkpoint_path)
        )
    else:
        tune.report(tune_metrics)

def print_and_save_best_results(
    analysis: ResultGrid,
    output_dir: str,
    metric: str = "env_runners/episode_reward_mean",
    mode: str = "max",
) -> Dict[str, Any]:
    """
    Prints and saves the best trial results from tuning.
    
    Args:
        analysis (ExperimentAnalysis): Ray Tune analysis object
        output_dir (str): Output directory to save the best results
        metric (str): Metric name to optimize. Default: "episode_reward_mean"
        mode (str): Optimization mode ("min" or "max"). Default: "max"
        
    Returns:
        best_trial_info (Dict[str, Any]): Dictionary with best trial information
    """
    
    # Get best trial based on average metric value across all iterations
    best_trial = analysis.get_best_result(metric=metric, mode=mode, scope="last")
    
    # Extract best trial information
    best_trial_dir = best_trial.path 
    best_trial_name = Path(best_trial_dir).name
    best_trial_config = best_trial.config
    best_trial_metrics_df = best_trial.metrics_dataframe
    
    # Latest metric of best trial
    best_trial_latest_metric = extract_nested_metric(best_trial.metrics, metric)

    # Best metric of best trial
    best_trial_best_metric = best_trial_metrics_df[metric].max() if mode == "max" else best_trial_metrics_df[metric].min()
    
    # Latest checkpoint of best trial
    best_trial_latest_checkpoint = best_trial.checkpoint

    # Best checkpoint of best trial
    best_trial_best_checkpoint = best_trial.get_best_checkpoint(metric=metric, mode=mode)

    # Print results
    print("\n" + "=" * 80)
    print("BEST TRIAL RESULTS")
    print("=" * 80)
    print(f"Best Trial Name: {best_trial_name}")
    print(f"Best Trial Path: {best_trial_dir}")
    print(f"Best Trial Latest Metric {metric}: {best_trial_latest_metric}")
    print(f"Best Trial Best Metric {metric}: {best_trial_best_metric}")
    print(f"\nBest Trial Hyperparameters (Tuned):")
    if "shared" in best_trial_config:
        print("  Shared:")
        for key, value in best_trial_config["shared"].items():
            print(f"    {key}: {value}")
    if "algorithm_specific" in best_trial_config:
        print("  Algorithm Specific:")
        for key, value in best_trial_config["algorithm_specific"].items():
            print(f"    {key}: {value}")
    
    # Print latest andbest checkpoints
    if best_trial_best_checkpoint and best_trial_latest_checkpoint:
        print(f"\nBest Checkpoint: {best_trial_best_checkpoint}")
        print(f"Latest Checkpoint: {best_trial_latest_checkpoint}")
    else:
        print("\nLatest and Best Checkpoint: None (no checkpoint saved)")
    
    print("=" * 80 + "\n")
    
    # Save best trial results to file
    output_path = Path(analysis.experiment_path)
    best_results = {
        "best_trial_name": best_trial_name,
        "best_trial_path": best_trial_dir,
        "best_trial_metric": metric,
        "best_trial_latest_metric": float(best_trial_latest_metric),
        "best_trial_best_metric": float(best_trial_best_metric),
        "best_config": best_trial_config,
        "latest_checkpoint": str(best_trial_latest_checkpoint) if best_trial_latest_checkpoint else None,
        "best_checkpoint": str(best_trial_best_checkpoint) if best_trial_best_checkpoint else None,
    }
    
    # Save to YAML
    best_results_path = output_path / "best_trial_results.yaml"
    with open(best_results_path, "w") as f:
        yaml.dump(best_results, f, default_flow_style=False, sort_keys=False)
    
    print(f"Best results saved to:")
    print(f"  YAML: {best_results_path}")
    
    return best_results