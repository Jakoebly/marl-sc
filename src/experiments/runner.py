from typing import Dict, Any, Optional, Callable
from pathlib import Path
import wandb
from ray import tune
from ray.air import session

from src.environment.environment import InventoryEnvironment
from src.algorithms.registry import get_algorithm
from src.config.schema import EnvironmentConfig, AlgorithmConfig
from src.experiments.utils.wandb import log_wandb_metrics


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
            root_seed (Optional[int]): Root seed for all components (env, RLlib, Ray Tune).
                If provided, used for environment initialization and passed to algorithm.
            checkpoint_dir (Optional[str]): Directory for saving checkpoints
            wandb_config (Optional[Dict[str, Any]]): WandB configuration dict (project, name, tags, etc.)
        """

        # Store root seed
        self.root_seed = root_seed

        # Store configs
        self.env_config = env_config
        self.algorithm_config = algorithm_config

        # Store checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Initialize environment with root seed
        self.env = InventoryEnvironment(self.env_config, seed=self.root_seed)
        
        # Initialize algorithm with root seed
        self.algorithm = get_algorithm(self.algorithm_config.name, self.env, self.algorithm_config, root_seed=self.root_seed)
        
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
            # Train one iteration
            result = self.algorithm.train()
            
            # Log metrics to WandB (if WandB config is provided)
            if self.wandb_config:
                log_wandb_metrics(result, iteration)
            
            # Save and log a checkpoint (if checkpoint frequency is reached)
            checkpoint_path = None
            if iteration % checkpoint_freq == 0 and self.checkpoint_dir:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                checkpoint_path = str(checkpoint_path)
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
            final_checkpoint_path = str(final_checkpoint_path)
            self.algorithm.save_checkpoint(str(final_checkpoint_path))
        
        # Report final metrics and the final checkpoint back to Ray Tune
        if tune_callback:
            tune_callback(result, final_checkpoint_path)

        # Finish WandB run (if WandB config is provided)
        if self.wandb_config:
            wandb.finish()

        return result

