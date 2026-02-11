from typing import Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
import wandb
from ray import tune
from ray.air import session

from src.environment.environment import InventoryEnvironment
from src.algorithms.registry import get_algorithm
from src.config.schema import EnvironmentConfig, AlgorithmConfig
from src.experiments.utils.wandb import log_wandb_metrics
from src.utils.seed_manager import split_seed


class ExperimentRunner:
    """Implements an experiment runner that orchestrates the training loop for RLlib algorithms."""
    
    def __init__(
        self,
        env_config: EnvironmentConfig,
        algorithm_config: AlgorithmConfig,
        root_seed: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the experiment runner.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration
            algorithm_config (AlgorithmConfig): Algorithm configuration
            root_seed (Optional[int]): Root seed for reproducibility. If provided, split into
                independent train_seed and eval_seed via split_seed().
            checkpoint_dir (Optional[str]): Directory for saving checkpoints
            wandb_config (Optional[Dict[str, Any]]): WandB configuration dict (project, name, tags, etc.)
        """

        # Split root seed into independent train and eval seeds
        self.root_seed = root_seed
        self.train_seed, self.eval_seed = split_seed(root_seed, num_children=2)

        # Store configs
        self.env_config = env_config
        self.algorithm_config = algorithm_config

        # Store checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Initialize environment (template only; actual train/eval envs are created by RLlib)
        self.env = InventoryEnvironment(self.env_config)
        
        # Initialize algorithm with separate train and eval seeds
        self.algorithm = get_algorithm(
            self.algorithm_config.name, 
            self.env, 
            self.algorithm_config, 
            train_seed=self.train_seed,
            eval_seed=self.eval_seed,
        )
        
        # Initialize WandB (if provided)
        self.wandb_config = wandb_config
        if wandb_config:
            wandb.init(**wandb_config, config={
                "env": env_config.model_dump(),
                "algorithm": algorithm_config.model_dump(),
            })
    
    def run(
        self, 
        tune_callback: Optional[Callable[[Dict[str, Any], Optional[str]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full training loop.
        
        Args:
            callback (Callable[[str, Dict[str, Any], int], None]): Ray Tune callback function 
            to be called after each iteration to report metrics and optionally a checkpoint.
        
        Returns:
            result (Dict[str, Any]): Final training metrics
        """

        # Get number of iterations and checkpoint frequency
        num_iterations = self.algorithm_config.shared.num_iterations
        checkpoint_freq = self.algorithm_config.shared.checkpoint_freq
        
        # Train for the specified number of iterations
        for iteration in range(1, num_iterations + 1):
            print(f"[INFO] Training iteration {iteration} of {num_iterations}")
            # Train one iteration
            result = self.algorithm.train()
            
            # Log metrics to WandB (if WandB config is provided)
            if self.wandb_config:
                log_wandb_metrics(result, iteration)
            
            # Save and log a checkpoint (if checkpoint frequency is reached)
            checkpoint_path = None
            if iteration % checkpoint_freq == 0 and self.checkpoint_dir:
                print(f"[INFO] Saving checkpoint at iteration {iteration}")
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                checkpoint_path = str(checkpoint_path.resolve())
                self.algorithm.save_checkpoint(checkpoint_path)
                if self.wandb_config:
                    wandb.log({"checkpoint_iteration": iteration})
            
            # Report metrics and optionally a checkpoint back to Ray Tune
            if tune_callback:
                tune_callback(result, checkpoint_path)
        
        # Save final checkpoint
        final_checkpoint_path = None
        if self.checkpoint_dir:
            final_checkpoint_path = self.checkpoint_dir / "checkpoint_final"
            final_checkpoint_path.mkdir(parents=True, exist_ok=True)
            final_checkpoint_path = str(final_checkpoint_path.resolve())
            self.algorithm.save_checkpoint(str(final_checkpoint_path))
        
        # Report final metrics and the final checkpoint back to Ray Tune
        if tune_callback:
            tune_callback(result, final_checkpoint_path)

        # Finish WandB run (if WandB config is provided)
        if self.wandb_config:
            wandb.finish()

        return result


class EvaluationRunner:
    """Orchestrates standalone evaluation of a trained model from a checkpoint."""
    
    def __init__(
        self,
        env_config: EnvironmentConfig,
        algorithm_config: AlgorithmConfig,
        checkpoint_dir: str,
        output_dir: str,
        eval_episodes: Optional[int] = None,
        root_seed: Optional[int] = None,
        visualize: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the evaluation runner.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration.
            algorithm_config (AlgorithmConfig): Algorithm configuration.
            checkpoint_dir (str): Path to the checkpoint to evaluate (required).
            output_dir (str): Directory for saving visualizations.
            eval_episodes (Optional[int]): Number of evaluation episodes.
                If None, uses num_eval_episodes from algorithm config.
            root_seed (Optional[int]): Root seed for reproducibility. If provided, split into
                train_seed and eval_seed via split_seed(). Only eval_seed is used (no training).
            visualize (bool): If True, run manual rollout and generate visualization plots.
            wandb_config (Optional[Dict[str, Any]]): WandB configuration dict.
        """

        # Split root seed into independent train and eval seeds (only eval_seed is used)
        self.root_seed = root_seed
        _, self.eval_seed = split_seed(root_seed, num_children=2)

        # Store evaluation parameters
        self.eval_episodes = eval_episodes
        self.visualize = visualize

        # Store configs
        self.env_config = env_config
        self.algorithm_config = algorithm_config
        
        # Store checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)

        # Create template environment (seed=None; eval envs get eval_seed via evaluation_config)
        self.env = InventoryEnvironment(self.env_config)

        # Initialize algorithm with eval_seed only (no training in evaluation mode)
        self.algorithm = get_algorithm(
            self.algorithm_config.name, 
            self.env, 
            self.algorithm_config, 
            train_seed=None,
            eval_seed=self.eval_seed,
        )

        # Restore trained model weights from checkpoint
        self.algorithm.load_checkpoint(self.checkpoint_dir)

        # Initialize WandB (if provided)
        self.wandb_config = wandb_config
        if wandb_config:
            wandb.init(**wandb_config, config={
                "env": env_config.model_dump(),
                "algorithm": algorithm_config.model_dump(),
            })

    def run(self) -> Dict[str, Any]:
        """
        Runs evaluation and returns metrics. If visualize is True, also runs a manual 
        rollout to collect per-step data and generates visualization plots.
        
        Returns:
            result (Dict[str, Any]): Evaluation metrics dictionary.
        """

        # Determine number of evaluation episodes
        num_episodes = (
            self.eval_episodes 
            if self.eval_episodes is not None 
            else self.algorithm_config.shared.num_eval_episodes
        )

        # If visualize is True, run manual rollout and generate visualization plots
        if self.visualize:
            # Create a new evaluation environment with the eval_seed
            rollout_env = InventoryEnvironment(
                self.env_config, 
                seed=self.eval_seed,
                env_meta={"data_mode": "val"},
            )

            # Run manual rollout for detailed per-step data collection
            episodes_data = self.algorithm.rollout(rollout_env, num_episodes=num_episodes)

            # Generate and save visualizations
            from src.experiments.visualization import generate_visualizations
            viz_dir = self.output_dir / "visualizations"
            generate_visualizations(episodes_data, str(viz_dir))

            # Build a lightweight result dict from rollout data
            total_rewards = [ep["rewards"].sum() for ep in episodes_data]
            result = {
                "evaluation": {
                    "episode_reward_mean": float(np.mean(total_rewards)),
                    "episode_reward_min": float(np.min(total_rewards)),
                    "episode_reward_max": float(np.max(total_rewards)),
                    "num_episodes": num_episodes,
                    "visualizations_dir": str(viz_dir),
                }
            }
        
        # If visualize is False, run standard RLlib evaluation (aggregated metrics only)
        else:
            result = self.algorithm.evaluate(eval_episodes=num_episodes)

        # Log metrics and finish WandB run (if WandB config is provided)
        if self.wandb_config:
            log_wandb_metrics(result, step=0)
            wandb.finish()

        return result
