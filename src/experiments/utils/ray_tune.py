from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
import copy
import yaml
import numpy as np
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule, HyperBandScheduler, FIFOScheduler
from ray.tune import PlacementGroupFactory, ResultGrid
from ray import tune
import ray

if TYPE_CHECKING:
    from src.utils.seed_manager import SeedManager
    from src.config.schema import SchedulerConfig, SearchAlgorithmConfig

from src.config.loader import load_environment_config, load_algorithm_config, load_tune_config
from src.config.schema import TUNE_SEARCH_SPACE_SECTIONS


def create_tune_config(
    base_env_config_path: str,
    base_algorithm_config_path: str,
    search_space: Dict[str, Any],
    seed_manager: Optional['SeedManager'] = None,
) -> Dict[str, Any]:
    """
    Creates a Ray Tune configuration from base configs and search space.
    
    Args:
        base_env_config_path (str): Path to base environment config
        base_algorithm_config_path (str): Path to base algorithm config
        search_space (Dict[str, Any]): Dictionary defining hyperparameter search space.
            May contain keys: shared, algorithm_specific, environment, features.
        seed_manager (Optional['SeedManager']): Optional experiment-level ``SeedManager``.
        
    Returns:
        tune_config (Dict[str, Any]): Tune config dictionary
    """

    # Load base configs
    env_config = load_environment_config(base_env_config_path, seed_manager=seed_manager)
    algorithm_config = load_algorithm_config(base_algorithm_config_path)
    
    # Convert base configs from Pydantic models to dicts
    env_config_dict = env_config.model_dump(by_alias=True)
    algorithm_config_dict = algorithm_config.model_dump(by_alias=True)
    
    # Create tune config with base configs
    tune_config = {
        "env_config": env_config_dict,
        "algorithm_config": algorithm_config_dict,
    }
    
    # Merge search space sections
    for key in TUNE_SEARCH_SPACE_SECTIONS:
        if key in search_space:
            tune_config[key] = search_space[key]
    
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
        search_space (Dict[str, Any]): Dictionary with section keys (shared,
            algorithm_specific, environment, features), each containing nested
            dictionaries of search specs.
        
    Returns:
        tune_search_space (Dict[str, Any]): Ray Tune search space dictionary
    """

    # Initialize the tune search space dictionary
    tune_search_space = {}

    # Iterate over the nested dictionaries in the search space dictionary to convert each search space specification to Ray Tune format
    for key, value in search_space.items():
        if key in TUNE_SEARCH_SPACE_SECTIONS:
            nested_space = {}
            for nested_key, nested_value in value.items():
                nested_space[nested_key] = _convert_single_search_spec(nested_value)
            tune_search_space[key] = nested_space
        else:
            tune_search_space[key] = _convert_single_search_spec(value)
    
    return tune_search_space

def _parse_hidden_sizes(value) -> List[int]:
    """
    Parses a hidden-size spec into a list of ints.

    Supports int (64 -> [64]) and underscore-separated strings
    ("128_128" -> [128, 128]) for multi-layer architectures.

    Args:
        value (Any): Value to parse

    Returns:
        List[int]: List of hidden sizes
    """

    # Check if the value is an integer, float, string, or list
    if isinstance(value, int):
        return [value]
    if isinstance(value, float):
        return [int(value)]
    if isinstance(value, str):
        return [int(x) for x in value.split("_")]
    return list(value)


def _apply_network_size_overrides(algo_config_dict: Dict[str, Any]) -> None:
    """
    Pops synthetic network-size keys from *algorithm_specific* and inject
    them into the nested ``networks`` config before Pydantic validation.

    Args:
        algo_config_dict (Dict[str, Any]): Algorithm config dictionary
    """

    # Get the algorithm specific dictionary and the networks dictionary
    algo_specific = algo_config_dict.get("algorithm_specific", {})
    networks = algo_specific.get("networks", {})

    # Iterate over the actor and critic roles to parse and set the hidden sizes
    for role in ("actor", "critic"):
        key = f"{role}_hidden_size"
        value = algo_specific.pop(key, None)
        if value is not None:
            hidden_sizes = _parse_hidden_sizes(value)
            networks.setdefault(role, {}).setdefault("config", {})["hidden_sizes"] = hidden_sizes


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
    
    # Apply network size overrides
    _apply_network_size_overrides(merged)

    return merged


_ACTION_SPACE_PRESETS = {
    "demand_centered_15": {
        "type": "demand_centered",
        "params": {"max_quantity_adjustment": None},
    },
    "base_stock_100": {
        "type": "base_stock",
        "params": {"max_stock_level": None},
    },
    "direct_40": {
        "type": "direct",
        "params": {"max_order_quantities": None},
    },
}
_ACTION_SPACE_PARAM_VALUES = {
    "demand_centered_15": ("max_quantity_adjustment", 15),
    "base_stock_100": ("max_stock_level", 100),
    "direct_40": ("max_order_quantities", 40),
}


def _apply_env_overrides(env_config_dict: Dict[str, Any]) -> None:
    """
    Pops synthetic environment keys and replaces them with full nested
    config structures before Pydantic validation.

    Supported synthetic keys:

    * ``action_space_preset``  – string preset name that maps to a complete
      :class:`ActionSpaceConfig` dict (per-SKU lists are auto-sized).
    * ``initial_inventory_value`` – scalar integer that is broadcast into a
      uniform ``(n_warehouses, n_skus)`` custom inventory config.

    Args:
        env_config_dict (Dict[str, Any]): Environment config dictionary
            (modified in-place).
    """

    n_skus = env_config_dict.get("n_skus", 2)
    n_warehouses = env_config_dict.get("n_warehouses", 3)

    # --- action_space_preset → action_space ---
    preset = env_config_dict.pop("action_space_preset", None)
    if preset is not None:
        if preset not in _ACTION_SPACE_PRESETS:
            raise ValueError(
                f"Unknown action_space_preset '{preset}'. "
                f"Available: {list(_ACTION_SPACE_PRESETS)}"
            )
        param_key, param_val = _ACTION_SPACE_PARAM_VALUES[preset]
        env_config_dict["action_space"] = {
            "type": _ACTION_SPACE_PRESETS[preset]["type"],
            "params": {param_key: [param_val] * n_skus},
        }

    # --- initial_inventory_value → initial_inventory ---
    inv_val = env_config_dict.pop("initial_inventory_value", None)
    if inv_val is not None:
        env_config_dict["initial_inventory"] = {
            "type": "custom",
            "params": {
                "values": [[int(inv_val)] * n_skus for _ in range(n_warehouses)]
            },
        }


def merge_env_tune_params(
    env_config_dict: Dict[str, Any],
    tune_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merges sampled tune parameters into the environment config dictionary.
    Handles ``environment`` (top-level env params like episode_length) and
    ``features`` (individual feature toggles).

    Synthetic keys (``action_space_preset``, ``initial_inventory_value``)
    are expanded into full nested configs by :func:`_apply_env_overrides`.

    Args:
        env_config_dict (Dict[str, Any]): Base environment config dictionary.
        tune_config (Dict[str, Any]): Ray Tune config with sampled values.

    Returns:
        merged (Dict[str, Any]): Merged environment config dictionary.
    """

    # Deep copy the environment config dictionary
    merged = copy.deepcopy(env_config_dict)

    # Merge top-level environment parameters
    if "environment" in tune_config:
        for key, value in tune_config["environment"].items():
            merged[key] = value

    # Merge feature toggles
    if "features" in tune_config:
        merged.setdefault("features", {})
        merged["features"].update(tune_config["features"])

    # Expand synthetic keys into full nested configs
    _apply_env_overrides(merged)

    return merged

def prepare_tune_config(
    env_config_path: str,
    algorithm_config_path: str,
    tune_config_path: str,
    seed_manager: Optional['SeedManager'] = None,
) -> Tuple[Dict[str, Any], Optional['SchedulerConfig'], Optional['SearchAlgorithmConfig']]:
    """
    Prepares the tune configuration by loading configs and building search spaces.
    
    Args:
        env_config_path (str): Path to environment config.
        algorithm_config_path (str): Path to algorithm config.
        tune_config_path (str): Path to tune config (defines search space).
        seed_manager (Optional['SeedManager']): Optional experiment-level ``SeedManager``.
        
    Returns:
        config (Dict[str, Any]): Complete tune configuration dictionary.
        scheduler_config: Validated scheduler config (or None for ASHA default).
        search_algorithm_config: Validated search algorithm config (or None for random default).
    """

    # Load and validate tune config → validated TuneConfig instance
    tune_config = load_tune_config(tune_config_path)
    
    # Build search space dict from direct fields → dictionary converted from TuneConfig instance
    tune_config_dict = tune_config.model_dump(exclude_none=True)
    search_space = {
        key: tune_config_dict[key]
        for key in TUNE_SEARCH_SPACE_SECTIONS
        if key in tune_config_dict
    }
    
    # Convert search space to Ray Tune format → search spaces converted to Ray Tune format
    tune_search_space = convert_to_tune_search(search_space)
    
    # Create tune config → tune config dictionary with keys in [env_config, algorithm_config, 
    # shared, algorithm_specific, environment, features, tune_metric, tune_mode, root_seed]
    config = create_tune_config(
        env_config_path,
        algorithm_config_path,
        tune_search_space,
        seed_manager=seed_manager,
    )

    return config, tune_config.scheduler, tune_config.search_algorithm

def get_tune_scheduler(
    scheduler_config: Optional['SchedulerConfig'],
    metric: str = "env_runners/episode_return_mean",
    mode: str = "max",
) -> Any:
    """
    Gets a Ray Tune scheduler from a validated config object.
    
    Args:
        scheduler_config (Optional['SchedulerConfig']): Validated scheduler config, or None (defaults to ASHA).
        metric (str): Metric to optimize.
        mode (str): Optimization mode ("min" or "max").
        
    Returns:
        tune_scheduler (Any): Ray Tune scheduler instance.
    """

    # Default to ASHA scheduler
    if scheduler_config is None:
        return ASHAScheduler(metric=metric, mode=mode)

    # Get additional kwargs for the scheduler
    kwargs = scheduler_config.model_dump(exclude={"type"}, exclude_none=True)

    # Return the corresponding Ray Tune scheduler
    if scheduler_config.type == "fifo":
        return FIFOScheduler()
    elif scheduler_config.type == "asha":
        return ASHAScheduler(metric=metric, mode=mode, **kwargs)
    elif scheduler_config.type == "median_stopping":
        return MedianStoppingRule(metric=metric, mode=mode, **kwargs)
    elif scheduler_config.type == "hyperband":
        return HyperBandScheduler(metric=metric, mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: '{scheduler_config.type}'")

def get_tune_search_algorithm(
    search_config: Optional['SearchAlgorithmConfig'],
    metric: str = "env_runners/episode_return_mean",
    mode: str = "max",
    seed: Optional[int] = None,
) -> Any:
    """
    Gets a Ray Tune search algorithm from a validated config object.
    
    Args:
        search_config: Validated search algorithm config, or None (defaults to random).
        metric (str): Metric to optimize.
        mode (str): Optimization mode ("min" or "max").
        seed (Optional[int]): Random seed for search algorithm reproducibility.
        
    Returns:
        tune_search_algorithm (Any): Tune search algorithm instance or None (for random search).
    """


    # Default to random search
    if search_config is None or search_config.type == "random":
        return None

    # Get additional kwargs for the search algorithm
    kwargs = search_config.model_dump(exclude={"type"}, exclude_none=True)

    # Return the corresponding Ray Tune search algorithm
    if search_config.type == "optuna":
        from ray.tune.search.optuna import OptunaSearch
        return OptunaSearch(metric=metric, mode=mode, seed=seed, **kwargs)
    elif search_config.type == "bayesopt":
        from ray.tune.search.bayesopt import BayesOptSearch
        return BayesOptSearch(metric=metric, mode=mode, random_state=seed, **kwargs)
    elif search_config.type == "hyperopt":
        from ray.tune.search.hyperopt import HyperOptSearch
        return HyperOptSearch(metric=metric, mode=mode, random_state_seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown search_type: '{search_config.type}'. "
            f"Supported types: 'random', 'optuna', 'bayesopt', 'hyperopt'"
        )

def get_resources_per_trial(
    num_cpus: int | None, 
    num_gpus: int, 
    num_env_runners: int, 
    num_cpus_per_env_runner: int | None,
    algorithm_config_dict: Dict[str, Any],
) -> PlacementGroupFactory:
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
) -> Tuple[int, int]:
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
    num_gpus: int,
    num_env_runners: int,
    num_cpus_per_env_runner: int,
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
    Supports flat keys (e.g., Tune Result.metrics stores ``"env_runners/episode_return_mean"``
    as a literal key) and nested paths (e.g., RLlib result dicts use nested structure).
    
    Args:
        data (Dict[str, Any]): Dictionary to extract metric from
        metric_path (str): Path to metric, using "/" as separator for nested keys
        
    Returns:
        metric_value (Any): Extracted metric value, or None if not found
    """
    
    # Try flat key first (for Tune Result.metrics)
    if metric_path in data:
        return data[metric_path]

    # Then try nested traversal (for RLlib result dicts)
    if "/" in metric_path:
        parts = metric_path.split("/")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    # Set the get the metric value from the data
    metric_value = data.get(metric_path)

    return metric_value

def report_tune_metrics(
    result: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Reports training and evaluation metrics to Ray Tune each iteration.

    Always reports two metric keys:

    - ``train/episode_return_mean`` - training return (always available).
    - ``eval/episode_return_mean`` - evaluation return when available,
      otherwise falls back to the training return so that the key is
      always present (required by ``OptunaSearch.on_trial_result``).

    Args:
        result: Result dictionary from one RLlib training iteration.
        checkpoint_path: Optional path to a checkpoint directory to report.
    """

    # Extract training and evaluation return metrics
    train_return = extract_nested_metric(result, "env_runners/episode_return_mean")
    eval_return = extract_nested_metric(
        result, "evaluation/env_runners/episode_return_mean",
    )

    # Assemble all metrics to report
    tune_metrics = {
        "training_iteration": result.get("training_iteration"),
        "train/episode_return_mean": train_return,
        "eval/episode_return_mean": (
            eval_return if eval_return is not None else train_return
        ),
    }

    # Report results including checkpoint if provided
    if checkpoint_path:
        tune.report(
            tune_metrics,
            checkpoint=tune.Checkpoint.from_directory(checkpoint_path),
        )
    else:
        tune.report(tune_metrics)

def _extract_short_trial_id(trial_dir_name: str) -> str:
    """
    Extracts a short trial ID from a Ray Tune trial directory name.

    Ray Tune names directories as
    ``trainable_{hash}_{index}_{params}_{date}_{time}``.
    This function returns the ``trainable_{hash}_{index}`` prefix.
    """

    # Extract the short trial ID from the trial directory name
    parts = trial_dir_name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])

    return trial_dir_name


def print_and_save_best_results(
    analysis: ResultGrid,
    metric: str = "eval/episode_return_mean",
    mode: str = "max",
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Prints and saves the best trial results from tuning. Selects the best trial
    based on the final reported value of ``metric``. Also saves the top-K
    trials with short trial IDs for automated multi-seed validation.

    Args:
        analysis (ResultGrid): Ray Tune analysis object.
        metric (str): Metric name to optimize.
        mode (str): Optimization mode (``"min"`` or ``"max"``).
        top_k (int): Number of top trials to save for seed evaluation.

    Returns:
        best_trial_info (Dict[str, Any]): Dictionary with best trial information.
    """

    # Get the best trial based on the final reported value of metric
    best_trial = analysis.get_best_result(metric=metric, mode=mode, scope="last")

    # Extract best trial information
    best_trial_dir = best_trial.path
    best_trial_name = Path(best_trial_dir).name
    best_trial_metrics_df = best_trial.metrics_dataframe
    best_trial_config = best_trial.config
    best_trial_env_config = merge_env_tune_params(best_trial_config["env_config"], best_trial_config)
    best_trial_algorithm_config = merge_tune_params(best_trial_config["algorithm_config"], best_trial_config)
    
    # Get the final evaluation return of the best trial
    best_trial_latest_eval_metric = extract_nested_metric(best_trial.metrics, metric)

    # Get the best training return of the best trial
    best_trial_best_train_metric = (
        best_trial_metrics_df["train/episode_return_mean"].max()
        if mode == "max"
        else best_trial_metrics_df["train/episode_return_mean"].min()
    )

    # Resolve checkpoint_best saved by ExperimentRunner in the trial directory
    best_trial_best_checkpoint_path = Path(best_trial_dir) / "checkpoint_best"
    best_trial_best_checkpoint = str(best_trial_best_checkpoint_path) if best_trial_best_checkpoint_path.exists() else None

    # Collect and rank all completed trials by the metric for top-K selection
    trial_ranking: List[Tuple[float, str, str]] = []
    for r in analysis:
        val = r.metrics.get(metric) if r.metrics else None
        if val is not None and r.path:
            trial_name = Path(r.path).name
            trial_ranking.append((val, trial_name, r.path))
    trial_ranking.sort(key=lambda x: x[0], reverse=(mode == "max"))
    actual_k = min(top_k, len(trial_ranking))

    top_k_trials = []
    for rank, (val, name, path) in enumerate(trial_ranking[:actual_k]):
        short_id = _extract_short_trial_id(name)
        top_k_trials.append({
            "rank": rank + 1,
            "short_id": short_id,
            "trial_path": path,
            "eval_metric": round(float(val), 4),
        })

    # Print results
    print("\n" + "=" * 80)
    print("BEST TRIAL RESULTS")
    print("=" * 80)
    print(f"Best Trial Name: {best_trial_name}")
    print(f"Best Trial Path: {best_trial_dir}")
    print(f"Best Trial Latest Evaluation Metric {metric}: {best_trial_latest_eval_metric}")
    print(f"Best Trial Best Training Metric {'train/episode_return_mean'}: {best_trial_best_train_metric}")
    print(f"\nBest Trial Hyperparameters (Tuned):")
    if "shared" in best_trial_config:
        print("  Shared:")
        for key, value in best_trial_config["shared"].items():
            print(f"    {key}: {value}")
    if "algorithm_specific" in best_trial_config:
        print("  Algorithm Specific:")
        for key, value in best_trial_config["algorithm_specific"].items():
            print(f"    {key}: {value}")
    if "environment" in best_trial_config:
        print("  Environment:")
        for key, value in best_trial_config["environment"].items():
            print(f"    {key}: {value}")
    if "features" in best_trial_config:
        print("  Features:")
        for key, value in best_trial_config["features"].items():
            print(f"    {key}: {value}")

    # Print best checkpoint of best trial
    if best_trial_best_checkpoint:
        print(f"\nBest Checkpoint: {best_trial_best_checkpoint}")
    else:
        print("\nBest Checkpoint: None (no checkpoint_best found in trial directory)")

    # Print top-K trial summary
    print(f"\nTop-{actual_k} Trials (by {metric}):")
    for t in top_k_trials:
        print(f"  #{t['rank']}: {t['short_id']}  eval={t['eval_metric']}")

    print("=" * 80 + "\n")

    # Save best trial results to file
    output_path = Path(analysis.experiment_path)
    best_results = {
        "best_trial_selection_scope": "last",
        "best_trial_name": best_trial_name,
        "best_trial_path": best_trial_dir,
        "best_trial_metric": metric,
        "best_trial_latest_eval_metric": float(best_trial_latest_eval_metric),
        "best_trial_best_train_metric": float(best_trial_best_train_metric),
        "best_trial_env_config": best_trial_env_config,
        "best_trial_algorithm_config": best_trial_algorithm_config,
        "best_trial_best_checkpoint": best_trial_best_checkpoint,
        "top_k_trials": top_k_trials,
    }

    # Save to YAML
    best_results_path = output_path / "best_trial_results.yaml"
    with open(best_results_path, "w") as f:
        yaml.dump(best_results, f, default_flow_style=False, sort_keys=False)

    print(f"Best results saved to:")
    print(f"  YAML: {best_results_path}")

    return best_results


def _extract_trial_index(result) -> int:
    """
    Extract the sequential trial index from a Ray Tune Result's directory name.

    Ray Tune names trial directories as
    ``trainable_{trial_id}_{index}_{params}_{date}_{time}``
    where *index* is the 0-based creation order assigned by the scheduler.
    This order matches the sequence in which the search algorithm (e.g. Optuna
    TPE) proposed configurations, making it the correct chronological axis for
    running-best convergence curves.

    Args:
        result (Result): Ray Tune result object

    Returns:
        trial_index (int): Sequential trial index
    """
    try:
        trial_name = Path(result.path).name
        parts = trial_name.split("_")
        return int(parts[2])
    except (IndexError, ValueError, AttributeError):
        return 2**31


def analyze_tune_convergence(
    analysis: ResultGrid,
    metric: str = "train/episode_return_mean",
    mode: str = "max",
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Analyzes the convergence of a Ray Tune experiment and saves a report.

    Computes running-best curves, top-N agreement percentages per tuned
    parameter, and generates a recommendation (no further tuning, narrow
    refinement, more trials, etc.).

    Each trial's metric is the final reported value (``scope="last"``).

    Args:
        analysis (ResultGrid): Ray Tune results.
        metric (str): Metric to optimize.
        mode (str): ``"max"`` or ``"min"``.
        top_n (int): Number of top trials to analyze for agreement.

    Returns:
        report (Dict[str, Any]): Full convergence report dict.
    """

    # Get the tune sections
    tune_sections = list(TUNE_SEARCH_SPACE_SECTIONS)

    # 1. Collect final metric, config, and trial index from completed trials
    trial_data: List[Tuple[int, float, Dict[str, Any]]] = []
    for r in analysis:
        val = r.metrics.get(metric) if r.metrics else None
        if val is not None:
            trial_data.append((_extract_trial_index(r), val, r.config))

    n_trials = len(trial_data)
    if n_trials == 0:
        print("[WARNING] No completed trials with metrics found - skipping convergence analysis.")
        return {}

    # Sort by trial creation order so running-best reflects actual search progression
    trial_data.sort(key=lambda x: x[0])
    metrics_list = [d[1] for d in trial_data]
    configs_list = [d[2] for d in trial_data]
    metrics_arr = np.array(metrics_list)

    # 2. Running-best curve
    if mode == "max":
        running_best = np.maximum.accumulate(metrics_arr)
    else:
        running_best = np.minimum.accumulate(metrics_arr)

    pct_checkpoints = {}
    for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        idx = min(int(n_trials * pct / 100) - 1, n_trials - 1)
        pct_checkpoints[f"after_{pct}pct ({idx + 1} trials)"] = round(float(running_best[idx]), 4)

    last_20_start = max(0, int(n_trials * 0.80) - 1)
    last_10_start = max(0, int(n_trials * 0.90) - 1)
    improvement_last_20 = float(running_best[-1] - running_best[last_20_start])
    improvement_last_10 = float(running_best[-1] - running_best[last_10_start])

    # 3. Top-N trial selection
    if mode == "max":
        top_idx = np.argsort(metrics_arr)[-top_n:][::-1]
    else:
        top_idx = np.argsort(metrics_arr)[:top_n]

    # 4. Detect tuned parameters (> 1 unique value across all trials)
    all_param_keys: set = set()
    for section in tune_sections:
        for cfg in configs_list:
            if section in cfg and isinstance(cfg[section], dict):
                for key in cfg[section]:
                    all_param_keys.add((section, key))

    tuned_params: Dict[str, List[Any]] = {}
    for section, key in all_param_keys:
        all_vals = [cfg.get(section, {}).get(key) for cfg in configs_list]
        unique = set(str(v) for v in all_vals if v is not None)
        if len(unique) > 1:
            top_vals = [configs_list[i].get(section, {}).get(key) for i in top_idx]
            tuned_params[f"{section}/{key}"] = top_vals

    # 5. Agreement analysis
    agreement: Dict[str, Dict[str, Any]] = {}
    locked_params: Dict[str, Any] = {}
    variable_params: List[str] = []

    for param_name, top_vals in tuned_params.items():
        str_vals = [str(v) for v in top_vals]
        unique = set(str_vals)
        most_common_str = max(set(str_vals), key=lambda x: str_vals.count(x))
        most_common_val = top_vals[str_vals.index(most_common_str)]
        agree_pct = str_vals.count(most_common_str) / len(str_vals) * 100

        is_continuous = len(unique) == top_n
        entry: Dict[str, Any] = {
            "dominant_value": round(most_common_val, 6) if isinstance(most_common_val, float) else most_common_val,
            "agreement_pct": round(agree_pct, 1),
            "unique_values": len(unique),
            "is_continuous": is_continuous,
        }
        if is_continuous:
            nums = [v for v in top_vals if isinstance(v, (int, float))]
            if nums:
                entry["top_n_min"] = round(float(min(nums)), 6)
                entry["top_n_max"] = round(float(max(nums)), 6)
                entry["top_n_mean"] = round(float(np.mean(nums)), 6)
        agreement[param_name] = entry

        if agree_pct >= 80:
            locked_params[param_name] = entry["dominant_value"]
        else:
            variable_params.append(param_name)

    # 6. Recommendation
    total_tuned = len(tuned_params)
    locked_count = len(locked_params)
    locked_ratio = locked_count / total_tuned if total_tuned > 0 else 0
    abs_imp = abs(improvement_last_20)

    if abs_imp < 1.0 and locked_ratio >= 0.7:
        action = "no_further_tuning"
        reasoning = (
            f"Search has converged: improvement in last 20% of trials is only "
            f"{abs_imp:.2f}, and {locked_count}/{total_tuned} "
            f"({locked_ratio * 100:.0f}%) parameters are locked in top-{top_n}."
        )
    elif abs_imp < 3.0 and locked_ratio >= 0.5:
        action = "narrow_refinement"
        reasoning = (
            f"Search is near convergence: improvement in last 20% is {abs_imp:.2f}. "
            f"Lock the {locked_count} converged parameters and run a focused search "
            f"on the remaining {len(variable_params)} variable parameters."
        )
    elif abs_imp >= 3.0:
        action = "more_trials"
        reasoning = (
            f"Search is still improving ({abs_imp:.2f} in last 20% of trials). "
            f"Run more trials with the same search space."
        )
    else:
        action = "more_trials_or_narrow"
        reasoning = (
            f"Mixed signals: improvement is {abs_imp:.2f} with "
            f"{locked_ratio * 100:.0f}% params locked. "
            f"Consider more trials or a narrower search."
        )

    recommendation: Dict[str, Any] = {
        "action": action,
        "reasoning": reasoning,
    }
    if locked_params:
        recommendation["locked_params"] = locked_params
    if variable_params:
        recommendation["variable_params"] = variable_params
    # For variable continuous params, suggest narrowed ranges based on top-N
    suggested_ranges: Dict[str, Any] = {}
    for vp in variable_params:
        info = agreement.get(vp, {})
        if info.get("is_continuous") and "top_n_min" in info:
            suggested_ranges[vp] = {
                "suggested_low": info["top_n_min"],
                "suggested_high": info["top_n_max"],
            }
    if suggested_ranges:
        recommendation["suggested_narrow_ranges"] = suggested_ranges

    # 7. Build top-N trial details
    top_trials = []
    for rank, idx in enumerate(top_idx):
        trial = {"rank": rank + 1, "reward": round(float(metrics_arr[idx]), 4)}
        cfg = configs_list[idx]
        # Include only learning_rate explicitly (always useful)
        lr = cfg.get("shared", {}).get("learning_rate")
        if lr is not None:
            trial["learning_rate"] = round(lr, 6)
        for param_name in tuned_params:
            section, key = param_name.split("/", 1)
            val = cfg.get(section, {}).get(key)
            if isinstance(val, float):
                val = round(val, 6)
            trial[param_name] = val
        top_trials.append(trial)

    # 8. Assemble report
    report = {
        "summary": {
            "total_trials": n_trials,
            "best_metric": round(float(running_best[-1]), 4),
            "metric_name": metric,
            "metric_scope": "last",
            "mode": mode,
        },
        "convergence": {
            "running_best": pct_checkpoints,
            "improvement_last_20pct": round(improvement_last_20, 4),
            "improvement_last_10pct": round(improvement_last_10, 4),
        },
        "top_n_analysis": {
            "n": top_n,
            "trials": top_trials,
        },
        "agreement": agreement,
        "recommendation": recommendation,
    }

    # 9. Save to YAML
    output_path = Path(analysis.experiment_path)
    report_path = output_path / "convergence_analysis.yaml"
    with open(report_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    # 10. Print summary
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"Metric scope: last")
    print(f"Total trials: {n_trials}")
    print(f"Best metric: {running_best[-1]:.4f}")
    print(f"Improvement last 20%: {improvement_last_20:.4f}")
    print(f"Improvement last 10%: {improvement_last_10:.4f}")
    print(f"\nLocked parameters ({locked_count}/{total_tuned}):")
    for k, v in locked_params.items():
        print(f"  {k}: {v}")
    print(f"\nVariable parameters ({len(variable_params)}):")
    for vp in variable_params:
        info = agreement[vp]
        if info["is_continuous"]:
            print(f"  {vp}: continuous, top-{top_n} range [{info['top_n_min']}, {info['top_n_max']}]")
        else:
            print(f"  {vp}: {info['agreement_pct']}% agree on {info['dominant_value']} ({info['unique_values']} unique)")
    print(f"\nRECOMMENDATION: {action}")
    print(f"  {reasoning}")
    print("=" * 80)
    print(f"\nConvergence analysis saved to: {report_path}\n")

    return report