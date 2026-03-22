from typing import Dict, Any, Optional, Callable
from pathlib import Path
import shutil
import numpy as np
import yaml
import wandb
from ray import tune
from ray.air import session

from src.environment.envs.multi_env import InventoryEnvironment
from src.algorithms.registry import get_algorithm
from src.config.schema import EnvironmentConfig, AlgorithmConfig
from src.experiments.utils.wandb import log_wandb_metrics
from src.utils.seed_manager import SeedManager
from src.utils.obs_stats import compute_obs_statistics


class ExperimentRunner:
    """Implements an experiment runner that orchestrates the training loop for RLlib algorithms."""
    
    def __init__(
        self,
        env_config: EnvironmentConfig,
        algorithm_config: AlgorithmConfig,
        seed_manager: Optional[SeedManager] = None,
        checkpoint_dir: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the experiment runner.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration
            algorithm_config (AlgorithmConfig): Algorithm configuration
            seed_manager (Optional[SeedManager]): Experiment-level seed manager.
            checkpoint_dir (Optional[str]): Directory for saving checkpoints
            wandb_config (Optional[Dict[str, Any]]): WandB configuration dict (project, name, tags, etc.)
        """

        # Derive train / eval / obs_stats seeds from the SeedManager
        self.seed_manager = seed_manager
        self.train_seed = seed_manager.get_seed_int('train') if seed_manager else None
        self.eval_seed = seed_manager.get_seed_int('eval') if seed_manager else None
        self.obs_stats_seed = seed_manager.get_seed_int('obs_stats') if seed_manager else None

        # Store configs
        self.env_config = env_config
        self.algorithm_config = algorithm_config

        # Store checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Initialize environment (template only; actual train/eval envs are created by RLlib)
        self.env = InventoryEnvironment(self.env_config)

        # Precompute observation statistics for meanstd_custom / meanstd_grouped
        obs_norm_mode = algorithm_config.algorithm_specific.obs_normalization
        if obs_norm_mode in ("meanstd_custom", "meanstd_grouped"):
            self.env.obs_stats = compute_obs_statistics(
                env_config, mode=obs_norm_mode, n_episodes=100, seed=self.obs_stats_seed,
            )
        
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
            tune_callback (Callable): Ray Tune callback function to be called after each
            iteration to report metrics and optionally a checkpoint. Defaults to None.
        
        Returns:
            result (Dict[str, Any]): Final training metrics
        """

        # Get number of iterations and checkpoint frequency
        num_iterations = self.algorithm_config.shared.num_iterations
        checkpoint_freq = self.algorithm_config.shared.checkpoint_freq

        # Track best metric for checkpoint_best
        best_metric_value = float('-inf')
        best_iteration = None
        
        # Train for the specified number of iterations
        for iteration in range(1, num_iterations + 1):
            # Train one iteration
            result = self.algorithm.train()
            
            # Log metrics to WandB (if WandB config is provided)
            if self.wandb_config:
                log_wandb_metrics(result, iteration)

            # Save checkpoint_best if current metric is a new best
            current_metric = result.get("env_runners", {}).get("episode_return_mean")
            if (
                current_metric is not None
                and current_metric > best_metric_value
                and self.checkpoint_dir
            ):
                best_metric_value = current_metric
                best_iteration = iteration
                best_checkpoint_path = self.checkpoint_dir / "checkpoint_best"
                if best_checkpoint_path.exists():
                    shutil.rmtree(best_checkpoint_path)
                best_checkpoint_path.mkdir(parents=True, exist_ok=True)
                self.algorithm.save_checkpoint(
                    str(best_checkpoint_path.resolve()),
                    env_config=self.env_config,
                    algorithm_config=self.algorithm_config,
                )
                print(
                    f"[INFO] New best checkpoint at iteration {iteration} with reward: {current_metric:.4f}"
                )
            
            # Save periodic checkpoint (if checkpoint frequency is reached)
            periodic_checkpoint_path = None
            if iteration % checkpoint_freq == 0 and self.checkpoint_dir:
                print(f"[INFO] Saving checkpoint at iteration {iteration}")
                periodic_checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}"
                periodic_checkpoint_path.mkdir(parents=True, exist_ok=True)
                periodic_checkpoint_path = str(periodic_checkpoint_path.resolve())
                self.algorithm.save_checkpoint(
                    periodic_checkpoint_path,
                    env_config=self.env_config,
                    algorithm_config=self.algorithm_config,
                )
                if self.wandb_config:
                    wandb.log({"checkpoint_iteration": iteration})
            
            # Report metrics and optionally a periodic checkpoint to Ray Tune
            if tune_callback:
                tune_callback(result, periodic_checkpoint_path)
            
            print(f"[INFO] Training iteration {iteration} of {num_iterations}: Reward: {current_metric:.4f}")

        if best_iteration is not None:
            print(
                f"[INFO] Best checkpoint: iteration {best_iteration} with reward: {best_metric_value:.4f}"
            )
        
        # Save final checkpoint
        final_checkpoint_path = None
        if self.checkpoint_dir:
            final_checkpoint_path = self.checkpoint_dir / "checkpoint_final"
            final_checkpoint_path.mkdir(parents=True, exist_ok=True)
            final_checkpoint_path = str(final_checkpoint_path.resolve())
            self.algorithm.save_checkpoint(
                final_checkpoint_path,
                env_config=self.env_config,
                algorithm_config=self.algorithm_config,
            )

            # Export RLModule weights as a standalone .pt file for curriculum warm-starts
            from src.utils.weight_transfer import export_module_weights
            ps = self.algorithm_config.algorithm_specific.parameter_sharing
            policy_id = "shared_policy" if ps else f"policy_{self.env.agents[0]}"
            weights_file = str(self.checkpoint_dir / "module_weights.pt")
            export_module_weights(self.algorithm.trainer, policy_id, weights_file)
        
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
        experiment_dir: str,
        eval_episodes: Optional[int] = None,
        seed_manager: Optional[SeedManager] = None,
        visualize: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the evaluation runner.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration.
            algorithm_config (AlgorithmConfig): Algorithm configuration.
            checkpoint_dir (str): Path to the checkpoint to evaluate (required).
            experiment_dir (str): Path to the experiment directory (receives eval
                outputs like eval_results.yaml and visualizations).
            eval_episodes (Optional[int]): Number of evaluation episodes.
                If None, uses num_eval_episodes from algorithm config.
            seed_manager (Optional[SeedManager]): Experiment-level seed manager.
            visualize (bool): If True, run manual rollout and generate visualization plots.
            wandb_config (Optional[Dict[str, Any]]): WandB configuration dict.
        """

        # Derive eval / obs_stats seeds from SeedManager
        self.seed_manager = seed_manager
        self.eval_seed = seed_manager.get_seed_int('eval') if seed_manager else None
        self.obs_stats_seed = seed_manager.get_seed_int('obs_stats') if seed_manager else None

        # Store evaluation parameters
        self.eval_episodes = eval_episodes
        self.visualize = visualize

        # Store configs
        self.env_config = env_config
        self.algorithm_config = algorithm_config
        
        # Store checkpoint and experiment directories
        self.checkpoint_dir = checkpoint_dir
        self.experiment_dir = Path(experiment_dir)

        # Create template environment (seed=None; eval envs get eval_seed via evaluation_config)
        self.env = InventoryEnvironment(self.env_config)

        # Precompute observation statistics for meanstd_custom / meanstd_grouped
        obs_norm_mode = algorithm_config.algorithm_specific.obs_normalization
        if obs_norm_mode in ("meanstd_custom", "meanstd_grouped"):
            self.env.obs_stats = compute_obs_statistics(
                env_config, mode=obs_norm_mode, n_episodes=100, seed=self.obs_stats_seed,
            )

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
            # Create the environment metadata for the rollout environment
            rollout_env_meta = {
                "data_mode": "val",
                "obs_normalization": self.algorithm_config.algorithm_specific.obs_normalization,
                "obs_stats": getattr(self.algorithm, "obs_stats", None),
            }
            if self.algorithm_config.algorithm_specific.parameter_sharing:
                rollout_env_meta["include_warehouse_id"] = True

            # Create a new evaluation environment with the eval_seed and environment metadata
            rollout_env = InventoryEnvironment(
                self.env_config, 
                seed=self.eval_seed,
                env_meta=rollout_env_meta,
            )

            # Run manual rollout for detailed per-step data collection
            episodes_data = self.algorithm.rollout(rollout_env, num_episodes=num_episodes)

            # Get the checkpoint name and subfolder name
            from src.experiments.visualization import generate_visualizations
            checkpoint_name = Path(self.checkpoint_dir).name
            if checkpoint_name == "checkpoint_final":
                viz_subfolder = "visualization_final"
                eval_suffix = "final"
            elif checkpoint_name == "checkpoint_best":
                viz_subfolder = "visualization_best"
                eval_suffix = "best"
            elif checkpoint_name.startswith("checkpoint_"):
                chkpt_num = checkpoint_name.replace("checkpoint_", "")
                viz_subfolder = f"visualization_chkpt{chkpt_num}"
                eval_suffix = f"chkpt{chkpt_num}"
            else:
                viz_subfolder = f"visualization_{checkpoint_name}"
                eval_suffix = checkpoint_name

            # Generate and save visualizations in a checkpoint-specific subfolder
            visualization_dir = self.experiment_dir / "visualizations" / viz_subfolder
            generate_visualizations(episodes_data, str(visualization_dir))

            # Build a lightweight result dict from rollout data
            total_rewards = [ep["rewards"].sum() for ep in episodes_data]
            result = {
                "evaluation": {
                    "episode_reward_mean": float(np.mean(total_rewards)),
                    "episode_reward_min": float(np.min(total_rewards)),
                    "episode_reward_max": float(np.max(total_rewards)),
                    "num_episodes": num_episodes,
                    "visualizations_dir": str(visualization_dir),
                }
            }

        # If visualize is False, run standard RLlib evaluation (aggregated metrics only)
        else:
            raw = self.algorithm.evaluate(eval_episodes=num_episodes)
            eval_env = raw.get("evaluation", {}).get("env_runners", {})

            # Get the checkpoint name
            checkpoint_name = Path(self.checkpoint_dir).name
            if checkpoint_name == "checkpoint_final":
                eval_suffix = "final"
            elif checkpoint_name == "checkpoint_best":
                eval_suffix = "best"
            elif checkpoint_name.startswith("checkpoint_"):
                chkpt_num = checkpoint_name.replace("checkpoint_", "")
                eval_suffix = f"chkpt{chkpt_num}"
            else:
                eval_suffix = checkpoint_name

            # Build a lightweight result dict from evaluation metrics
            result = {
                "evaluation": {
                    "episode_reward_mean": float(eval_env.get("episode_return_mean", 0)),
                    "episode_reward_min": float(eval_env.get("episode_return_min", 0)),
                    "episode_reward_max": float(eval_env.get("episode_return_max", 0)),
                    "num_episodes": num_episodes,
                }
            }

        # Print summary
        eval_metrics = result.get("evaluation", {})
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Checkpoint:  {Path(self.checkpoint_dir).name}")
        print(f"  Episodes:    {eval_metrics.get('num_episodes')}")
        print(f"  Mean Reward: {eval_metrics.get('episode_reward_mean', 0):.4f}")
        print(f"  Min  Reward: {eval_metrics.get('episode_reward_min', 0):.4f}")
        print(f"  Max  Reward: {eval_metrics.get('episode_reward_max', 0):.4f}")
        if "visualizations_dir" in eval_metrics:
            print(f"  Viz saved:   {eval_metrics['visualizations_dir']}")
        print("=" * 60 + "\n")

        # Save checkpoint-specific eval results (prevents overwrites across evaluations)
        eval_results_path = self.experiment_dir / f"eval_results_{eval_suffix}.yaml"
        eval_results = {
            "checkpoint": str(self.checkpoint_dir),
            **eval_metrics,
        }
        with open(eval_results_path, "w") as f:
            yaml.dump(eval_results, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Evaluation results saved to: {eval_results_path}")

        # Log metrics and finish WandB run (if WandB config is provided)
        if self.wandb_config:
            log_wandb_metrics(result, iteration=0)
            wandb.finish()

        return result
