from typing import Optional, Any, Tuple, List, Dict
import numbers
from ray.air.integrations.wandb import WandbLoggerCallback
import wandb


def setup_wandb(
    wandb_project: Optional[str],
    mode: str = "single",
    wandb_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Tuple[Optional[Dict[str, str]], List]:
    """
    Sets up WandB configuration for a single or tune experiment.
    
    Args:
        wandb_project (Optional[str]): WandB project name
        mode (str): "single" for single experiments, "tune" for hyperparameter tuning
        wandb_name (Optional[str]): WandB run name (for single mode)
        experiment_name (Optional[str]): Experiment name used as WandB group (for tune mode).
            Each trial's run name defaults to its Ray Tune trial name.
        
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
        callbacks = [WandbLoggerCallback(
            project=wandb_project,
            group=experiment_name,
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
            metrics["train/episode_return_mean"] = env_runners["episode_return_mean"]
        if "episode_return_max" in env_runners:
            metrics["train/episode_return_max"] = env_runners["episode_return_max"]
        if "episode_return_min" in env_runners:
            metrics["train/episode_return_min"] = env_runners["episode_return_min"]
        
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
                        if key != "learner_connector" and isinstance(value, numbers.Number):
                            metrics[f"train/learner/all/{key}"] = float(value)
            
            # Otherwise, extract per-policy learner stats
            else:
                if isinstance(policy_stats, dict):
                    for key, value in policy_stats.items():
                        if isinstance(value, numbers.Number):
                            metrics[f"train/learner/{policy_id}/{key}"] = float(value)
    
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
                metrics["eval/episode_return_mean"] = eval_env_runners["episode_return_mean"]
            if "episode_return_max" in eval_env_runners:
                metrics["eval/episode_return_max"] = eval_env_runners["episode_return_max"]
            if "episode_return_min" in eval_env_runners:
                metrics["eval/episode_return_min"] = eval_env_runners["episode_return_min"]
            if "episode_len_mean" in eval_env_runners:
                metrics["eval/episode_len_mean"] = eval_env_runners["episode_len_mean"]
            if "num_episodes" in eval_env_runners:
                metrics["eval/num_episodes"] = eval_env_runners["num_episodes"]
    
    # Log all metrics with consistent step key
    if metrics:
        wandb.log(metrics, step=step_key)

