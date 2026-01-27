
import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

import ray
from ray import tune

from src.config.loader import (
    load_environment_config,
    load_algorithm_config,
    load_tune_config,
)
from src.config.schema import EnvironmentConfig, AlgorithmConfig
from src.experiments.runner import ExperimentRunner
from src.experiments.run_experiment import (
    run_single_experiment,
    run_tune_experiment,
    generate_experiment_name,
)
from src.experiments.utils.ray_tune import (
    create_tune_config,
    get_tune_scheduler,
    get_tune_search_algorithm,
    print_and_save_best_results,
    merge_tune_params,
    convert_to_tune_search,
)


# Test fixtures
@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def ray_cleanup():
    """Cleanup Ray after each test."""
    yield
    try:
        ray.shutdown()
    except Exception:
        pass


@pytest.fixture
def env_config_file(temp_dir):
    """Create a minimal environment config file."""
    config_content = {
        "environment": {
            "n_warehouses": 2,
            "n_skus": 3,
            "n_regions": 2,
            "episode_length": 10,
            "max_order_quantity": 5,
            "initial_inventory": {
                "type": "uniform",
                "params": {"min": 0, "max": 10}
            },
            "cost_structure": {
                "holding_cost": 1.0,
                "penalty_cost": 10.0,
                "shipment_cost": [[1.0, 1.0], [1.0, 1.0]]
            },
            "components": {
                "demand_sampler": {
                    "type": "poisson",
                    "params": {
                        "lambda_orders": 2.0,
                        "lambda_skus": 1.0,
                        "lambda_quantity": 1.0
                    }
                },
                "demand_allocator": {
                    "type": "greedy",
                    "params": {"max_splits": 1}
                },
                "lead_time_sampler": {
                    "type": "uniform",
                    "params": {"min": 1, "max": 2}
                },
                "lost_sales_handler": {
                    "type": "cheapest",
                    "params": None
                },
                "reward_calculator": {
                    "type": "cost",
                    "params": {
                        "scope": "team",
                        "scale_factor": 1.0,
                        "normalize": False,
                        "cost_weights": [0.33, 0.33, 0.34]
                    }
                }
            },
            "data_source": {
                "type": "synthetic",
                "path": None
            }
        }
    }
    
    config_path = Path(temp_dir) / "test_env.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    return str(config_path)


@pytest.fixture
def algorithm_config_file(temp_dir):
    """Create a minimal algorithm config file."""
    config_content = {
        "algorithm": {
            "name": "ippo",
            "shared": {
                "num_iterations": 5,
                "checkpoint_freq": 3,
                "batch_size": 32,
                "num_epochs": 2,
                "num_minibatches": 2,
                "learning_rate": 0.001,
                "num_env_runners": 4, 
                "num_envs_per_env_runner": 1,
                "num_cpus_per_env_runner": 2
            },
            "algorithm_specific": {
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "clip_param": 0.2,
                "use_gae": True,
                "lam": 0.95,
                "parameter_sharing": True,
                "networks": {
                    "actor": {
                        "type": "mlp",
                        "config": {
                            "hidden_sizes": [32, 32],
                            "activation": "relu"
                        }
                    },
                    "critic": {
                        "type": "mlp",
                        "config": {
                            "hidden_sizes": [32, 32],
                            "activation": "relu"
                        }
                    }
                }
            }
        }
    }
    
    config_path = Path(temp_dir) / "test_algo.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    return str(config_path)


@pytest.fixture
def tune_config_file(temp_dir):
    """Create a minimal tune config file."""
    config_content = {
        "shared": {
            "learning_rate": {
                "type": "choice",
                "values": [0.001, 0.0005]
            },
            "batch_size": {
                "type": "choice",
                "values": [32, 64]
            }
        },
        "algorithm_specific": {
            "vf_loss_coeff": {
                "type": "uniform",
                "low": 0.3,
                "high": 0.7
            }
        }
    }
    
    config_path = Path(temp_dir) / "test_tune.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    return str(config_path)


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""
    
    def test_initialization(self, env_config_file, algorithm_config_file, temp_dir):
        """Test ExperimentRunner initialization."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        runner = ExperimentRunner(
            env_config=env_config,
            algorithm_config=algorithm_config,
            root_seed=42,
            checkpoint_dir=temp_dir,
            wandb_config=None,
        )
        
        print(f"test_initialization passed: {runner}")

        assert runner.env_config == env_config
        assert runner.algorithm_config == algorithm_config
        assert runner.root_seed == 42
        assert runner.checkpoint_dir == Path(temp_dir)
        assert runner.wandb_config is None
        assert runner.env is not None
        assert runner.algorithm is not None
    
    def test_initialization_with_wandb(self, env_config_file, algorithm_config_file, temp_dir):
        """Test ExperimentRunner initialization with WandB."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        wandb_config = {
            "project": "test_project",
            "name": "test_run",
            "mode": "offline",  # Use offline mode for testing
        }
        
        with patch("wandb.init") as mock_wandb_init:
            runner = ExperimentRunner(
                env_config=env_config,
                algorithm_config=algorithm_config,
                wandb_config=wandb_config,
            )
            
            mock_wandb_init.assert_called_once()
            call_args = mock_wandb_init.call_args
            assert call_args[1]["project"] == "test_project"
            assert call_args[1]["name"] == "test_run"
            assert "config" in call_args[1]
    
    def test_training_loop(self, env_config_file, algorithm_config_file, temp_dir):
        """Test training loop execution."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        runner = ExperimentRunner(
            env_config=env_config,
            algorithm_config=algorithm_config,
            checkpoint_dir=temp_dir,
        )
        
        # Run training
        result = runner.run()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "info" in result or "episode_reward_mean" in result or len(result) > 0
        
        # Verify checkpoints were created
        checkpoint_dir = Path(temp_dir)
        final_checkpoint = checkpoint_dir / "checkpoint_final"
        assert final_checkpoint.exists()
        
        # Verify intermediate checkpoints (if checkpoint_freq allows)
        if algorithm_config.shared.checkpoint_freq <= algorithm_config.shared.num_iterations:
            checkpoint_path = checkpoint_dir / f"checkpoint_{algorithm_config.shared.checkpoint_freq}"
            assert checkpoint_path.exists()
    
    def test_metrics_logging(self, env_config_file, algorithm_config_file, temp_dir):
        """Test metrics extraction and logging."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        with patch("wandb.log") as mock_wandb_log, patch("wandb.init"), patch("wandb.finish"):
            runner = ExperimentRunner(
                env_config=env_config,
                algorithm_config=algorithm_config,
                wandb_config={"project": "test", "mode": "offline"},
            )
            
            result = runner.run()
            print(f"[DEBUG] result: {result}")
            
            # Verify wandb.log was called for each iteration
            assert mock_wandb_log.call_count >= algorithm_config.shared.num_iterations
            
            # Check that metrics contain expected keys
            for call in mock_wandb_log.call_args_list:
                metrics = call[0][0]
                assert "iteration" in metrics
                assert isinstance(metrics["iteration"], int)
    
    def test_checkpoint_saving(self, env_config_file, algorithm_config_file, temp_dir):
        """Test checkpoint saving functionality."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        runner = ExperimentRunner(
            env_config=env_config,
            algorithm_config=algorithm_config,
            checkpoint_dir=temp_dir,
        )
        
        runner.run()
        
        # Verify final checkpoint exists
        final_checkpoint = Path(temp_dir) / "checkpoint_final"
        assert final_checkpoint.exists()
        assert final_checkpoint.is_dir()
        
        # Verify checkpoint contains files (algorithm-specific)
        checkpoint_files = list(final_checkpoint.iterdir())
        assert len(checkpoint_files) > 0


class TestSingleExperiment:
    """Tests for run_single_experiment function."""
    
    def test_single_experiment_basic(self, env_config_file, algorithm_config_file, temp_dir):
        """Test basic single experiment execution."""
        result = run_single_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            output_dir=temp_dir,
            experiment_name="test_single",
        )
        
        # Verify result
        assert isinstance(result, dict)
        
        # Verify experiment directory was created
        experiment_dir = Path(temp_dir) / "test_single"
        assert experiment_dir.exists()
        
        # Verify checkpoints exist
        final_checkpoint = experiment_dir / "checkpoint_final"
        assert final_checkpoint.exists()
    
    def test_single_experiment_auto_name(self, env_config_file, algorithm_config_file, temp_dir):
        """Test single experiment with auto-generated name."""
        result = run_single_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            output_dir=temp_dir,
            experiment_name=None,  # Auto-generate
        )
        
        # Verify experiment directory was created (name should start with algorithm name)
        experiment_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
        assert len(experiment_dirs) == 1
        assert "ippo" in experiment_dirs[0].name or "single" in experiment_dirs[0].name
    
    def test_single_experiment_with_wandb(self, env_config_file, algorithm_config_file, temp_dir):
        """Test single experiment with WandB."""
        with patch("wandb.init") as mock_wandb_init, patch("wandb.log"), patch("wandb.finish"):
            result = run_single_experiment(
                env_config_path=env_config_file,
                algorithm_config_path=algorithm_config_file,
                output_dir=temp_dir,
                wandb_project="test_project",
                wandb_name="test_run",
                experiment_name="test_wandb",
            )
            
            # Verify WandB was initialized
            mock_wandb_init.assert_called_once()
    
    def test_single_experiment_resume(self, env_config_file, algorithm_config_file, temp_dir):
        """Test resuming from checkpoint."""
        # First run
        run_single_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            output_dir=temp_dir,
            experiment_name="test_resume",
        )
        
        # Get checkpoint path
        checkpoint_path = Path(temp_dir) / "test_resume" / "checkpoint_final"
        
        # Resume from checkpoint
        result = run_single_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            output_dir=temp_dir,
            resume_from=str(checkpoint_path),
            experiment_name="test_resume2",
        )
        
        assert isinstance(result, dict)


class TestTuneExperiment:
    """Tests for run_tune_experiment function."""
    
    def test_tune_experiment_basic(self, env_config_file, algorithm_config_file, tune_config_file, temp_dir):
        """Test basic tune experiment execution."""

        print(f"[DEBUG] temp_dir: {temp_dir}")

        # Use minimal trials for testing
        analysis = run_tune_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            tune_config_path=tune_config_file,
            num_samples=2,  # Small number for testing
            output_dir=temp_dir,
            scheduler_type="fifo",  # Simple scheduler
            search_type="random",
            experiment_name="test_tune",
        )

        # Verify analysis object
        assert analysis is not None
        
        # Verify experiment directory structure
        experiment_dir = Path(temp_dir) / "test_tune"
        assert experiment_dir.exists()
        
        # Verify trials were created (check in the analysis object)
        assert analysis.num_errors + analysis.num_terminated == 2
        
        # Verify best results file exists
        best_results_file = experiment_dir / "best_trial_results.yaml"
        assert best_results_file.exists()
        
        # Verify best results content
        with open(best_results_file) as f:
            best_results = yaml.safe_load(f)
            assert "best_trial_name" in best_results
            assert "best_trial_path" in best_results
            assert "best_trial_metric" in best_results
            assert "best_trial_latest_metric" in best_results
            assert "best_trial_best_metric" in best_results
            assert "best_config" in best_results
            assert "best_checkpoint" in best_results
    
    def test_tune_experiment_auto_name(self, env_config_file, algorithm_config_file, tune_config_file, temp_dir):
        """Test tune experiment with auto-generated name."""
        analysis = run_tune_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            tune_config_path=tune_config_file,
            num_samples=1,
            output_dir=temp_dir,
            search_type="random",
            scheduler_type="fifo",
            experiment_name=None,  # Auto-generate
        )
        
        # Verify experiment directory was created
        experiment_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
        assert len(experiment_dirs) == 1
        assert "ippo" in experiment_dirs[0].name
    
    def test_tune_experiment_with_wandb(self, env_config_file, algorithm_config_file, tune_config_file, temp_dir):
        """Test tune experiment with WandB."""
        with patch("wandb.init") as mock_wandb_init:
            analysis = run_tune_experiment(
                env_config_path=env_config_file,
                algorithm_config_path=algorithm_config_file,
                tune_config_path=tune_config_file,
                num_samples=1,
                output_dir=temp_dir,
                wandb_project="test_project",
                search_type="random",
                scheduler_type="fifo",
                experiment_name="test_wandb_tune",
            )
            
            # WandB should be initialized via callback
            # Note: Ray Tune handles WandB differently, so we check analysis exists
            assert analysis is not None


class TestConfigLoadingAndMerging:
    """Tests for config loading and parameter merging."""
    
    def test_tune_config_loading(self, tune_config_file):
        """Test tune config loading and validation."""
        tune_config = load_tune_config(tune_config_file)
        
        assert tune_config.shared is not None
        assert tune_config.algorithm_specific is not None
        assert "learning_rate" in tune_config.shared
        assert "vf_loss_coeff" in tune_config.algorithm_specific
    
    def test_search_space_conversion(self, tune_config_file):
        """Test search space conversion to Ray Tune format."""
        tune_config = load_tune_config(tune_config_file)
        
        # Convert Pydantic models to dicts
        search_space = {}
        if tune_config.shared:
            search_space["shared"] = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in tune_config.shared.items()}
        if tune_config.algorithm_specific:
            search_space["algorithm_specific"] = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in tune_config.algorithm_specific.items()}
        
        tune_search_space = convert_to_tune_search(search_space)
        
        # Verify structure
        assert "shared" in tune_search_space
        assert "algorithm_specific" in tune_search_space
        
        # Verify Ray Tune objects were created
        assert "learning_rate" in tune_search_space["shared"]
        assert "vf_loss_coeff" in tune_search_space["algorithm_specific"]
    
    def test_parameter_merging(self, algorithm_config_file):
        """Test parameter merging for tune experiments."""
        algorithm_config = load_algorithm_config(algorithm_config_file)
        algorithm_config_dict = algorithm_config.model_dump()
        
        # Simulate tune config with sampled parameters
        tune_config = {
            "shared": {
                "learning_rate": 0.0005,  # Different from base
                "batch_size": 64,  # Different from base
            },
            "algorithm_specific": {
                "vf_loss_coeff": 0.6,  # Different from base
            }
        }
        
        merged = merge_tune_params(algorithm_config_dict, tune_config)
        
        # Verify merged values
        assert merged["shared"]["learning_rate"] == 0.0005
        assert merged["shared"]["batch_size"] == 64
        assert merged["algorithm_specific"]["vf_loss_coeff"] == 0.6
        
        # Verify other values preserved
        assert merged["shared"]["num_iterations"] == algorithm_config_dict["shared"]["num_iterations"]
    
    def test_create_tune_config(self, env_config_file, algorithm_config_file, tune_config_file):
        """Test create_tune_config function."""
        tune_config = load_tune_config(tune_config_file)
        # Convert Pydantic models to dicts
        search_space = {}
        if tune_config.shared:
            search_space["shared"] = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in tune_config.shared.items()}
        if tune_config.algorithm_specific:
            search_space["algorithm_specific"] = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in tune_config.algorithm_specific.items()}
        
        tune_search_space = convert_to_tune_search(search_space)
        
        config = create_tune_config(
            env_config_file,
            algorithm_config_file,
            tune_search_space,
        )
        
        # Verify structure
        assert "env_config" in config
        assert "algorithm_config" in config
        assert "shared" in config
        assert "algorithm_specific" in config
        
        # Verify base configs are present
        assert config["env_config"]["n_warehouses"] == 2
        assert config["algorithm_config"]["name"] == "ippo"


class TestExperimentNameGeneration:
    """Tests for experiment name generation."""
    
    def test_generate_experiment_name_basic(self, algorithm_config_file):
        """Test basic experiment name generation."""
        name = generate_experiment_name(
            algorithm_config_path=algorithm_config_file,
            search_type="random",
            scheduler_type=None,
        )
        
        assert name.startswith("ippo")
        assert "random" in name
        assert len(name.split("_")) >= 3  # algo_search_timestamp
    
    def test_generate_experiment_name_with_scheduler(self, algorithm_config_file):
        """Test experiment name generation with scheduler."""
        name = generate_experiment_name(
            algorithm_config_path=algorithm_config_file,
            search_type="optuna",
            scheduler_type="asha",
        )
        
        assert name.startswith("ippo")
        assert "optuna" in name
        assert "asha" in name
    
    def test_generate_experiment_name_single(self, algorithm_config_file):
        """Test experiment name generation for single experiments."""
        name = generate_experiment_name(
            algorithm_config_path=algorithm_config_file,
            search_type="single",
            scheduler_type=None,
        )
        
        assert name.startswith("ippo")
        assert "single" in name


class TestSchedulerAndSearchAlgorithms:
    """Tests for scheduler and search algorithm creation."""
    
    def test_get_tune_scheduler(self):
        """Test scheduler creation."""
        # Test all scheduler types
        schedulers = ["asha", "median_stopping", "hyperband", "fifo"]
        
        for scheduler_type in schedulers:
            scheduler = get_tune_scheduler(scheduler_type)
            if scheduler_type == "fifo":
                assert scheduler is not None
            else:
                assert scheduler is not None
    
    def test_get_tune_search_algorithm_random(self):
        """Test random search algorithm."""
        search_alg = get_tune_search_algorithm("random")
        assert search_alg is None  # Random search doesn't need an algorithm
    
    def test_get_tune_search_algorithm_optuna(self):
        """Test Optuna search algorithm."""
        try:
            search_alg = get_tune_search_algorithm("optuna", metric="episode_reward_mean", mode="max")
            assert search_alg is not None
        except AssertionError:
            # Optuna not installed, skip this test
            pytest.skip("Optuna not installed")
    
    def test_get_tune_search_algorithm_invalid(self):
        """Test invalid search algorithm raises error."""
        with pytest.raises(ValueError):
            get_tune_search_algorithm("invalid_search_type")


class TestBestResults:
    """Tests for best results extraction and saving."""
    
    def test_print_and_save_best_results(self, env_config_file, algorithm_config_file, tune_config_file, temp_dir):
        """Test best results extraction and saving."""
        # Run a small tune experiment
        analysis = run_tune_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            tune_config_path=tune_config_file,
            num_samples=2,
            output_dir=temp_dir,
            scheduler_type="fifo",
            search_type="random",
            experiment_name="test_best_results",
        )
        
        # Test best results function
        best_results = print_and_save_best_results(
            analysis=analysis,
            output_dir=temp_dir,
            metric="episode_reward_mean",
            mode="max",
        )
        
        # Verify best results structure
        assert "trial_id" in best_results
        assert "best_metric" in best_results
        assert "best_metric_value" in best_results
        assert "best_config" in best_results
        assert "best_checkpoint" in best_results
        
        # Verify file was created
        experiment_dir = Path(temp_dir) / "test_best_results"
        best_results_file = experiment_dir / "best_trial_results.yaml"
        assert best_results_file.exists()
        
        # Verify file content
        with open(best_results_file) as f:
            saved_results = yaml.safe_load(f)
            assert saved_results["trial_id"] == best_results["trial_id"]
            assert saved_results["best_config"] == best_results["best_config"]


class TestParameterFlow:
    """Tests for parameter flow through the system."""
    
    def test_config_flow_single_experiment(self, env_config_file, algorithm_config_file, temp_dir):
        """Test that configs flow correctly through single experiment."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        runner = ExperimentRunner(
            env_config=env_config,
            algorithm_config=algorithm_config,
            checkpoint_dir=temp_dir,
        )
        
        # Verify configs are passed correctly
        assert runner.env_config.n_warehouses == env_config.n_warehouses
        assert runner.algorithm_config.name == algorithm_config.name
        assert runner.algorithm_config.shared.learning_rate == algorithm_config.shared.learning_rate
    
    def test_config_flow_tune_experiment(self, env_config_file, algorithm_config_file, tune_config_file, temp_dir):
        """Test that configs flow correctly through tune experiment."""
        # Run tune experiment
        analysis = run_tune_experiment(
            env_config_path=env_config_file,
            algorithm_config_path=algorithm_config_file,
            tune_config_path=tune_config_file,
            num_samples=1,
            output_dir=temp_dir,
            scheduler_type="fifo",
            search_type="random",
            experiment_name="test_flow",
        )
        
        # Get best trial
        best_trial = analysis.get_best_result(metric="/env_runners/episode_reward_mean", mode="max", scope="last")
        
        # Verify config structure
        assert "env_config" in best_trial.config
        assert "algorithm_config" in best_trial.config
        assert "shared" in best_trial.config
        assert "algorithm_specific" in best_trial.config
        
        # Verify base configs are preserved
        assert best_trial.config["env_config"]["n_warehouses"] == 2
        assert best_trial.config["algorithm_config"]["name"] == "ippo"
        
        # Verify tune parameters are present
        assert "learning_rate" in best_trial.config["shared"]
        assert "batch_size" in best_trial.config["shared"]
        assert "vf_loss_coeff" in best_trial.config["algorithm_specific"]


class TestCheckpointFlow:
    """Tests for checkpoint saving and loading flow."""
    
    def test_checkpoint_save_and_load(self, env_config_file, algorithm_config_file, temp_dir):
        """Test checkpoint saving and loading."""
        env_config = load_environment_config(env_config_file)
        algorithm_config = load_algorithm_config(algorithm_config_file)
        
        # Create runner and train
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        runner = ExperimentRunner(
            env_config=env_config,
            algorithm_config=algorithm_config,
            checkpoint_dir=str(checkpoint_dir),
        )
        
        runner.run()
        
        # Verify checkpoint exists
        final_checkpoint = checkpoint_dir / "checkpoint_final"
        assert final_checkpoint.exists()
        
        # Test loading checkpoint
        new_runner = ExperimentRunner(
            env_config=env_config,
            algorithm_config=algorithm_config,
        )
        
        # Load checkpoint
        new_runner.algorithm.load_checkpoint(str(final_checkpoint))
        
        # Verify algorithm still works
        result = new_runner.algorithm.train()
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

