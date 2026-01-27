from typing import Optional, Callable, Any, Tuple, List, Dict
from ray.air.integrations.wandb import WandbLoggerCallback
import wandb

from src.config.loader import load_algorithm_config


def create_wandb_callback(
    project: str,
    name: Optional[Callable] = None,
    log_config: bool = True,
    upload_checkpoints: bool = False,
    **wandb_kwargs: Any
) -> WandbLoggerCallback:
    """
    Creates a WandB callback for Ray Tune with custom naming.
    
    Args:
        project (str): WandB project name
        name (Optional[Callable]): Optional callable that takes a trial and returns a run name.
              If None, uses default trial name.
        log_config (bool): Whether to log config/hyperparameters
        upload_checkpoints (bool): Whether to upload checkpoints as artifacts
        **wandb_kwargs: Additional WandB init arguments
        
    Returns:
        wandb_callback (WandbLoggerCallback): WandB logger callback instance
    """

    # Create WandB logger callback
    wandb_callback = WandbLoggerCallback(
        project=project,
        name=name,
        log_config=log_config,
        upload_checkpoints=upload_checkpoints,
        **wandb_kwargs
    )

    return wandb_callback

def create_wandb_run_name_fn(algorithm_name: Optional[str] = None) -> Callable:
    """
    Creates a function for custom WandB run naming that includes trial ID and key hyperparameters.
    
    Args:
        algorithm_name (Optional[str]): Optional algorithm name to include in run name
        
    Returns:
        wandb_run_name (Callable): Function that takes a trial and returns a run name string
    """

    def wandb_run_name(trial):
        """Generates a WandB run name from a trial."""

        # Extract trial ID from trial
        trial_id = trial.trial_id
        
        # Extract algorithm name from trial config if not provided
        algo_name = algorithm_name
        if algo_name is None:
            algo_config = trial.config.get("algorithm_config", {})
            algo_name = algo_config.get("name", "unknown")
        
        # Extract key hyperparameters for naming
        parts = [algo_name, trial_id]
        
        return "_".join(parts)
    
    return wandb_run_name

def setup_wandb(
    wandb_project: Optional[str],
    algorithm_config_path: str,
    mode: str = "single",
    wandb_name: Optional[str] = None,
) -> Tuple[Optional[Dict[str, str]], List]:
    """
    Sets up WandB configuration for a single or tune experiment.
    
    Args:
        wandb_project (Optional[str]): WandB project name
        algorithm_config_path (str): Path to algorithm config (for tune mode)
        mode (str): "single" for single experiments, "tune" for hyperparameter tuning
        wandb_name (Optional[str]): WandB run name (for single mode)
        
    Returns:
        wandb_config (Optional[Dict[str, str]]): WandB config dict for ExperimentRunner (single mode)
        callbacks (List): List of Ray Tune callbacks (tune mode)
    """

    # If no project is specified, return None and empty list
    if not wandb_project:
        return None, []

    # For single mode, return simple config dict for ExperimentRunner
    if mode == "single":
        wandb_config = {
            "project": wandb_project,
        }
        if wandb_name:
            wandb_config["name"] = wandb_name
        return wandb_config, []

    # For tune mode, return callbacks for Ray Tune
    elif mode == "tune":
        algorithm_config = load_algorithm_config(algorithm_config_path)
        wandb_name_fn = create_wandb_run_name_fn(algorithm_name=algorithm_config.name)
        
        callbacks = [create_wandb_callback(
            project=wandb_project,
            name=wandb_name_fn,
            log_config=True,
            upload_checkpoints=False,
        )]
        return None, callbacks
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'single' or 'tune'")

def log_wandb_metrics(result: Dict[str, Any], iteration: int) -> None:
    """
    Extracts and logs training and evaluation metrics to WandB.
    
    Args:
        result (Dict[str, Any]): Training result dictionary from RLlib
        iteration (int): Current training iteration
    """
    
    # Use training_iteration as step key for consistency
    step_key = result.get("training_iteration", iteration)
    metrics = {}
    
    # Extract metrics from env_runners (training episode metrics)
    env_runners = result.get("env_runners", {})
    if env_runners:
        # Episode return metrics (aggregate across all agents)
        if "episode_return_mean" in env_runners:
            metrics["train/episode_reward_mean"] = env_runners["episode_return_mean"]
        if "episode_return_max" in env_runners:
            metrics["train/episode_reward_max"] = env_runners["episode_return_max"]
        if "episode_return_min" in env_runners:
            metrics["train/episode_reward_min"] = env_runners["episode_return_min"]
        
        # Episode length metrics (aggregate across all agents)
        if "episode_len_mean" in env_runners:
            metrics["train/episode_len_mean"] = env_runners["episode_len_mean"]
        if "episode_len_max" in env_runners:
            metrics["train/episode_len_max"] = env_runners["episode_len_max"]
        if "episode_len_min" in env_runners:
            metrics["train/episode_len_min"] = env_runners["episode_len_min"]
        
        # Step counts (aggregate across all agents)
        if "num_env_steps_sampled" in env_runners:
            metrics["train/num_env_steps_sampled"] = env_runners["num_env_steps_sampled"]
        if "num_episodes" in env_runners:
            metrics["train/num_episodes"] = env_runners["num_episodes"]
        
        # Episode return metrics (per-agent)
        agent_returns = env_runners.get("agent_episode_returns_mean", {})
        if isinstance(agent_returns, dict):
            for agent_id, value in agent_returns.items():
                if isinstance(value, (int, float)):
                    metrics[f"train/agent/{agent_id}/episode_return_mean"] = value
        
        # Episode return metrics (per-policy)
        module_returns = env_runners.get("module_episode_returns_mean", {})
        if isinstance(module_returns, dict):
            for policy_id, value in module_returns.items():
                if isinstance(value, (int, float)):
                    metrics[f"train/policy/{policy_id}/episode_return_mean"] = value
        
        # Step counts (per-agent)
        agent_steps = env_runners.get("num_agent_steps_sampled", {})
        if isinstance(agent_steps, dict):
            for agent_id, value in agent_steps.items():
                if isinstance(value, (int, float)):
                    metrics[f"train/agent/{agent_id}/num_steps_sampled"] = value
    
    # Extract learner stats (per-policy training loss and statistics)
    learners = result.get("learners", {})
    if learners:
        for policy_id, policy_stats in learners.items():
            # Check for RLlib's key for aggregate statistics across all policies
            if policy_id == "__all_modules__":
                if isinstance(policy_stats, dict):
                    for key, value in policy_stats.items():
                        if key != "learner_connector" and isinstance(value, (int, float)):
                            metrics[f"train/learner/all/{key}"] = value
            
            # Otherwise, extract per-policy learner stats
            else:
                if isinstance(policy_stats, dict):
                    for key, value in policy_stats.items():
                        if isinstance(value, (int, float)):
                            metrics[f"train/learner/{policy_id}/{key}"] = value
    
    # Extract timing metrics
    timers = result.get("timers", {})
    if timers:
        for key, value in timers.items():
            if isinstance(value, (int, float)):
                metrics[f"train/timers/{key}"] = value
    
    
    # Extract top-level timing metrics
    if "time_this_iter_s" in result:
        metrics["train/time_this_iter_s"] = result["time_this_iter_s"]
    if "time_total_s" in result:
        metrics["train/time_total_s"] = result["time_total_s"]
    
    # Extract evaluation metrics if present
    eval_metrics = result.get("evaluation", {})
    if eval_metrics:
        # Handle evaluation metrics (similar structure to env_runners)
        eval_env_runners = eval_metrics.get("env_runners", {})
        if eval_env_runners:
            if "episode_return_mean" in eval_env_runners:
                metrics["eval/episode_reward_mean"] = eval_env_runners["episode_return_mean"]
            if "episode_return_max" in eval_env_runners:
                metrics["eval/episode_reward_max"] = eval_env_runners["episode_return_max"]
            if "episode_return_min" in eval_env_runners:
                metrics["eval/episode_reward_min"] = eval_env_runners["episode_return_min"]
            if "episode_len_mean" in eval_env_runners:
                metrics["eval/episode_len_mean"] = eval_env_runners["episode_len_mean"]
            if "num_episodes" in eval_env_runners:
                metrics["eval/num_episodes"] = eval_env_runners["num_episodes"]
    
    # Log all metrics with consistent step key
    if metrics:
        wandb.log(metrics, step=step_key)

