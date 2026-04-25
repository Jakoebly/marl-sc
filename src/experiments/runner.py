"""
Core experiment runners for training and evaluation.

``ExperimentRunner`` wraps the train loop (algorithm build, training iterations,
checkpointing, W&B logging), while ``EvaluationRunner`` loads a checkpoint,
runs rollout episodes, and generates visualizations.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import gc
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
from src.experiments.utils.experiment_utils import (
    checkpoint_suffix,
    load_and_truncate_training_metrics,
    parse_checkpoint_iteration,
    save_training_metrics,
)


class ExperimentRunner:
    """Orchestrates the training loop for RLlib algorithms."""
    
    def __init__(
        self,
        env_config: EnvironmentConfig,
        algorithm_config: AlgorithmConfig,
        seed_manager: Optional[SeedManager] = None,
        checkpoint_dir: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            env_config (EnvironmentConfig): Environment configuration.
            algorithm_config (AlgorithmConfig): Algorithm configuration.
            seed_manager (Optional[SeedManager]): Experiment-level seed manager.
            checkpoint_dir (Optional[str]): Directory for saving checkpoints.
            wandb_config (Optional[Dict[str, Any]]): WandB configuration dict.
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
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full training loop.

        When ``tune_callback`` is provided (tune mode), only ``checkpoint_best`` is saved
        and metrics are reported without checkpoint tracking. Periodic checkpoints,
        ``checkpoint_final``, and ``module_weights.pt`` are skipped to minimize disk I/O.

        Per-iteration train/eval returns are always collected and saved to
        ``training_metrics.yaml`` in the checkpoint directory after every
        iteration, so a killed run can be resumed from the latest periodic
        checkpoint with a consistent metrics log.

        When ``resume_from`` points at a periodic ``checkpoint_<N>`` directory,
        the trainer state is restored and training continues from iteration
        ``N + 1``. The existing ``training_metrics.yaml`` is truncated to
        iteration ``N`` so no duplicate or stale entries remain.

        Args:
            tune_callback (Optional[Callable]): Callback for reporting metrics
                to Ray Tune each iteration.
            resume_from (Optional[str]): Path to a ``checkpoint_<N>`` directory
                to resume from.

        Returns:
            result (Dict[str, Any]): Final training metrics.
        """

        # Get number of iterations and checkpoint frequency
        num_iterations = self.algorithm_config.shared.num_iterations
        checkpoint_freq = self.algorithm_config.shared.checkpoint_freq

        # Initialize bookkeeping (resume may overwrite these)
        _internal_metrics: List[Dict[str, Any]] = []
        best_metric_value = float('-inf')
        best_iteration: Optional[int] = None
        start_iteration = 1

        # Resume from periodic checkpoint if requested
        if resume_from:
            start_iteration, _internal_metrics, best_metric_value, best_iteration = (
                self._prepare_resume(resume_from)
            )

        # Train for the number of iterations between start_iteration and num_iterations
        result: Dict[str, Any] = {}
        for iteration in range(start_iteration, num_iterations + 1):
            # Train one iteration
            result = self.algorithm.train()

            # Log metrics to WandB (if WandB config is provided)
            if self.wandb_config:
                log_wandb_metrics(result, iteration)

            # Collect per-iteration metrics
            metric_entry = self._build_metric_entry(iteration, result)
            _internal_metrics.append(metric_entry)
            train_return = metric_entry["train_return"]

            # Save checkpoint_best if train return is a new best
            best_metric_value, best_iteration = self._save_best_checkpoint(
                iteration, train_return, best_metric_value, best_iteration,
            )

            # Save metrics for each iteration
            if self.checkpoint_dir and _internal_metrics:
                save_training_metrics(self.checkpoint_dir, _internal_metrics)

            # If in tune mode, report metrics to Ray Tune
            if tune_callback:
                tune_callback(result, None)

            # If not in tune mode and checkpoint frequency is reached, save periodic checkpoint
            elif iteration % checkpoint_freq == 0 and self.checkpoint_dir:
                self._save_periodic_checkpoint(iteration)

            # Print training progress
            reward_str = f"{train_return:.4f}" if train_return is not None else "N/A"
            print(
                f"[INFO] Training iteration {iteration} of {num_iterations}: "
                f"Reward: {reward_str}"
            )

            # Release per-iteration memory to keep plasma/python pressure bounded
            # over long runs and avoid OOM issues
            del result
            result = {}
            self._release_iteration_memory()

        # Print best checkpoint
        if best_iteration is not None:
            print(
                f"[INFO] Best checkpoint: iteration {best_iteration} with reward: {best_metric_value:.4f}"
            )

        # If not in tune mode and checkpoint directory exists, save final checkpoint
        # and export module weights
        if not tune_callback and self.checkpoint_dir:
            self._save_final_checkpoint_and_weights()

        # Ensure the final metrics snapshot is on disk even if the loop did
        # not execute (e.g. resumed run already past num_iterations).
        if self.checkpoint_dir and _internal_metrics:
            save_training_metrics(self.checkpoint_dir, _internal_metrics)

        # Finish WandB run (if WandB config is provided)
        if self.wandb_config:
            wandb.finish()

        return result

    def _prepare_resume(
        self,
        resume_from: str,
    ) -> Tuple[int, List[Dict[str, Any]], float, Optional[int]]:
        """
        Loads checkpoint state and truncates the metrics log for a clean resume.

        The periodic checkpoint directory's name determines the completed
        iteration (``checkpoint_<N>``); the existing
        ``training_metrics.yaml`` is truncated to that iteration so the
        metrics log stays monotone. Best-so-far bookkeeping is restored
        from the surviving entries.

        Args:
            resume_from (str): Path to a ``checkpoint_<N>`` directory.

        Returns:
            start_iteration (int): Iteration to start the loop from.
            metrics (List[Dict[str, Any]]): Truncated metrics log.
            best_metric_value (float): Restored best train return.
            best_iteration (Optional[int]): Restored best iteration.
        """

        # Parse the completed iteration from the checkpoint directory name
        completed = parse_checkpoint_iteration(resume_from)
        if completed is None:
            raise ValueError(
                f"resume_from must point at a 'checkpoint_<N>' directory so "
                f"training can continue from iteration N+1. Got: {resume_from}"
            )

        # Restore algorithm state
        print(f"[INFO] Resuming from checkpoint: {resume_from} (iteration {completed})")
        self.algorithm.load_checkpoint(resume_from)

        # Load and truncate existing metrics log and restore best-so-far bookkeeping
        metrics: List[Dict[str, Any]] = []
        best_metric_value = float("-inf")
        best_iteration: Optional[int] = None
        if self.checkpoint_dir:
            metrics, best_metric_value, best_iteration = (
                load_and_truncate_training_metrics(self.checkpoint_dir, completed)
            )

        start_iteration = completed + 1
        if best_iteration is not None:
            print(
                f"[INFO] Resume: start_iteration={start_iteration}, "
                f"restored {len(metrics)} metric entries, "
                f"best_so_far={best_metric_value:.4f} @ iter {best_iteration}"
            )
        else:
            print(
                f"[INFO] Resume: start_iteration={start_iteration}, "
                f"restored {len(metrics)} metric entries (no best yet)"
            )

        return start_iteration, metrics, best_metric_value, best_iteration

    def _save_best_checkpoint(
        self,
        iteration: int,
        train_return: Optional[float],
        best_metric_value: float,
        best_iteration: Optional[int],
    ) -> Tuple[float, Optional[int]]:
        """
        Saves ``checkpoint_best`` when ``train_return`` sets a new high.

        A no-op if ``train_return`` is ``None``, not strictly greater than
        the current best, or if no ``checkpoint_dir`` is configured. The
        previous ``checkpoint_best`` directory is removed first so stale
        files do not leak into the new checkpoint.

        Args:
            iteration (int): Current iteration number.
            train_return (Optional[float]): Train return for this iteration.
            best_metric_value (float): Current best train return.
            best_iteration (Optional[int]): Current best iteration.

        Returns:
            best_metric_value (float): Possibly updated best train return.
            best_iteration (Optional[int]): Possibly updated best iteration.
        """

        # Do not save checkpoint_best if train return is None, not better than the 
        # current best, or no checkpoint_dir is configured
        if (
            train_return is None
            or train_return <= best_metric_value
            or not self.checkpoint_dir
        ):
            return best_metric_value, best_iteration

        # Save checkpoint_best
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
            f"[INFO] New best checkpoint at iteration {iteration}: "
            f"Reward: {train_return:.4f}"
        )
        return train_return, iteration

    def _save_periodic_checkpoint(self, iteration: int) -> None:
        """
        Saves ``checkpoint_<iteration>`` and logs the iteration to WandB.

        Args:
            iteration (int): Iteration number to use in the checkpoint name.
        """

        # Save periodic checkpoint and log to wandb
        print(f"[INFO] Saving checkpoint at iteration {iteration}")
        periodic_checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}"
        periodic_checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.algorithm.save_checkpoint(
            str(periodic_checkpoint_path.resolve()),
            env_config=self.env_config,
            algorithm_config=self.algorithm_config,
        )
        if self.wandb_config:
            wandb.log({"checkpoint_iteration": iteration})

    def _save_final_checkpoint_and_weights(self) -> None:
        """
        Saves ``checkpoint_final`` and exports ``module_weights.pt``.

        The module ID is derived via the algorithm's ``policy_mapping_fn`` so
        this works for IPPO, MAPPO and centralized PPO without a special case.
        """

        # Save final checkpoint
        final_checkpoint_path = self.checkpoint_dir / "checkpoint_final"
        final_checkpoint_path.mkdir(parents=True, exist_ok=True)
        final_checkpoint_path = str(final_checkpoint_path.resolve())
        self.algorithm.save_checkpoint(
            final_checkpoint_path,
            env_config=self.env_config,
            algorithm_config=self.algorithm_config,
        )

        # Export module weights using the algorithm's policy mapping to
        # derive the correct module ID (handles IPPO, MAPPO, centralized PPO)
        from src.utils.weight_transfer import export_module_weights
        policy_id = self.algorithm.policy_mapping_fn(self.env.agents[0])
        weights_file = str(self.checkpoint_dir / "module_weights.pt")
        export_module_weights(self.algorithm.trainer, policy_id, weights_file)

    @staticmethod
    def _build_metric_entry(
        iteration: int,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Builds a single per-iteration entry for ``training_metrics.yaml``.

        Args:
            iteration (int): Iteration number.
            result (Dict[str, Any]): Raw RLlib training result.

        Returns:
            metric_entry (Dict[str, Any]): Metrics entry with ``iteration``,
                ``train_return`` and ``eval_return``.
        """

        # Extract train and eval returns
        train_return = result.get("env_runners", {}).get("episode_return_mean")
        eval_return = (
            result.get("evaluation", {})
            .get("env_runners", {})
            .get("episode_return_mean")
        )
        
        # Build metric entry
        metric_entry = {
            "iteration": iteration,
            "train_return": train_return,
            "eval_return": eval_return,
        }

        return metric_entry

    @staticmethod
    def _release_iteration_memory() -> None:
        """
        Runs ``gc.collect()`` and clears the CUDA cache when available.

        Called after each training iteration to keep plasma / Python memory
        pressure bounded over long runs and avoid OOM issues.
        """

        # Collect garbage   
        gc.collect()

        # Empty CUDA cache if torch is available
        try:    
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
           pass


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
        Args:
            env_config (EnvironmentConfig): Environment configuration.
            algorithm_config (AlgorithmConfig): Algorithm configuration.
            checkpoint_dir (str): Path to the checkpoint to evaluate.
            experiment_dir (str): Path to the experiment directory.
            eval_episodes (Optional[int]): Number of evaluation episodes.
            seed_manager (Optional[SeedManager]): Experiment-level seed manager.
            visualize (bool): If True, generate visualization plots.
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
        Runs the evaluation loop. 
        
        When ``visualize`` is True, performs a manual
        rollout and generates per-step visualization plots.

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
            # Create environment metadata for the rollout environment
            rollout_env_meta = {
                "data_mode": "val",
                "obs_normalization": self.algorithm_config.algorithm_specific.obs_normalization,
                "obs_stats": getattr(self.algorithm, "obs_stats", None),
            }
            if getattr(self.algorithm_config.algorithm_specific, "parameter_sharing", False):
                rollout_env_meta["include_warehouse_id"] = True

            # Create a new evaluation environment with the eval_seed and environment metadata
            rollout_env = InventoryEnvironment(
                self.env_config, 
                seed=self.eval_seed,
                env_meta=rollout_env_meta,
            )

            # Run manual rollout for detailed per-step data collection
            episodes_data = self.algorithm.rollout(rollout_env, num_episodes=num_episodes)

            # Derive suffix and visualization subfolder from checkpoint name
            eval_suffix = checkpoint_suffix(self.checkpoint_dir)
            viz_subfolder = f"visualization_{eval_suffix}"
            visualization_dir = self.experiment_dir / "visualizations" / viz_subfolder

            # Generate and save visualizations
            from src.experiments.utils.visualization import generate_visualizations
            generate_visualizations(episodes_data, str(visualization_dir))

            # Build a lightweight result dict from rollout data
            total_rewards = [ep["rewards"].sum() for ep in episodes_data]
            result = {
                "evaluation": {
                    "episode_return_mean": float(np.mean(total_rewards)),
                    "episode_return_min": float(np.min(total_rewards)),
                    "episode_return_max": float(np.max(total_rewards)),
                    "num_episodes": num_episodes,
                    "visualizations_dir": str(visualization_dir),
                }
            }

        # If visualize is False, run standard RLlib evaluation (aggregated metrics only)
        else:
            # Run RLlib evaluation
            raw_results = self.algorithm.evaluate(eval_episodes=num_episodes)
            eval_results = raw_results.get("env_runners", {})

            # Derive suffix from checkpoint name
            eval_suffix = checkpoint_suffix(self.checkpoint_dir)

            # Build a lightweight result dict from evaluation metrics
            result = {
                "evaluation": {
                    "episode_return_mean": float(eval_results.get("episode_return_mean", 0)),
                    "episode_return_min": float(eval_results.get("episode_return_min", 0)),
                    "episode_return_max": float(eval_results.get("episode_return_max", 0)),
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
        print(f"  Mean Return: {eval_metrics.get('episode_return_mean', 0):.4f}")
        print(f"  Min  Return: {eval_metrics.get('episode_return_min', 0):.4f}")
        print(f"  Max  Return: {eval_metrics.get('episode_return_max', 0):.4f}")
        if "visualizations_dir" in eval_metrics:
            print(f"  Viz saved:   {eval_metrics['visualizations_dir']}")
        print("=" * 60 + "\n")

        # Save checkpoint-specific eval results
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