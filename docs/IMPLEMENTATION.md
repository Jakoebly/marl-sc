 # Implementation Notes — marl-sc

This document describes how the marl-sc codebase is wired and operated. Specifically, it described the entry points and experiment paths, the core class contracts, the configuration system, the runtime objects, the RLlib integration, the seeding scheme, the visualization helpers, and the shell entry layer. It is the primary reference for modifying the codebase or debugging runtime behavior.

For thesis scope, research questions, contribution claims, environment design, algorithms, baselines, and interventions, see [SCOPE.md](SCOPE.md). For the full experimental program, statistical protocol, and supplementary appendix experiments, see [EXPERIMENTS.md](EXPERIMENTS.md). For what the system models and how its components behave, see [SYSTEM_MODEL.md](SYSTEM_MODEL.md).

## 1. Runtime Wiring

The runtime is layered. A bash script invokes the Python CLI. The CLI dispatches to a runner. The runner builds an algorithm wrapper. The algorithm wrapper builds an RLlib trainer. The trainer creates environments through a factory.

### 1.1 Entry Points and Experiment Paths

The codebase has two Python CLIs. [src/experiments/run_experiment.py](src/experiments/run_experiment.py) covers RL training, evaluation, tuning, and multi-seed evaluation. [src/experiments/run_baselines.py](src/experiments/run_baselines.py) covers non-RL baselines under the same multi-seed protocol. Both CLIs are invoked through bash scripts in [scripts/](scripts/).

Five experiment paths are supported.

| Path | CLI mode | Driver script |
|---|---|---|
| Single run | `single` on `run_experiment.py` | [run_experiment.sh](scripts/run_experiment.sh) |
| Hyperparameter sweep | `tune` on `run_experiment.py` | [run_tune_experiment.sh](scripts/run_tune_experiment.sh) |
| Standalone evaluation | `evaluate` on `run_experiment.py` | [run_evaluation.sh](scripts/run_evaluation.sh), [run_evaluation_batch.sh](scripts/run_evaluation_batch.sh) |
| RL multi-seed evaluation | `seed-eval` on `run_experiment.py` | [run_seed_evaluation.sh](scripts/run_seed_evaluation.sh) (parallel SLURM), [run_seed_evaluation_sequential.sh](scripts/run_seed_evaluation_sequential.sh) (sequential) |
| Baseline multi-seed evaluation | `single` or `all-seeds` on `run_baselines.py` | [run_single_baseline.sh](scripts/run_single_baseline.sh), [run_all_baselines.sh](scripts/run_all_baselines.sh) |

### 1.2 Trace per Path

The shared backbone is the same for every RL path. The script forwards arguments to the CLI. `_dispatch_experiment` routes to a mode handler. The handler loads and validates configs against the Pydantic schemas in [src/config/schema.py](src/config/schema.py). It creates a top-level `SeedManager` from [src/utils/seed_manager.py](src/utils/seed_manager.py). It then instantiates a runner from [src/experiments/runner.py](src/experiments/runner.py). Each path diverges from this point.

**Single run.** `run_single_experiment` builds one `ExperimentRunner`. The runner calls `get_algorithm` from [src/algorithms/registry.py](src/algorithms/registry.py) to obtain an `IPPOWrapper`, `MAPPOWrapper`, or `CPPOWrapper`. The wrapper builds an RLlib `PPOConfig` and registers a per-environment factory via `register_env`. `runner.run` then iterates `num_iterations` times and calls `self.algorithm.train()` each iteration.

**Hyperparameter sweep.** `run_tune_experiment` initializes Ray and merges the base configs with the tune config through `prepare_tune_config`. The Ray Tune `Tuner` runs `num_samples` trials. The trainable function builds a fresh `ExperimentRunner` per trial. After `tuner.fit`, `_run_post_tune_analysis` saves `best_trial_results.yaml` and evaluates the best trial's checkpoint.

**Standalone evaluation.** `run_evaluation` resolves the checkpoint and experiment directories. It builds an `EvaluationRunner`, loads the checkpoint, runs deterministic rollouts through `BaseAlgorithmWrapper.rollout`, and writes `eval_results_final.yaml` plus per-episode artifacts.

**RL multi-seed evaluation.** Two execution paths produce identical on-disk artifacts. The sequential path is invoked through [run_seed_evaluation_sequential.sh](scripts/run_seed_evaluation_sequential.sh). It calls `run_experiment.py --mode seed-eval` once. `run_seed_evaluation` builds the config list through `build_seed_evaluation_configs` (one entry in single-config mode, top-K trials in tune-best-K mode) and trains each config across `n_seeds` seeds in series. The default seed list is `100, 200, ..., n_seeds × 100`. The parallel SLURM path is invoked through [run_seed_evaluation.sh](scripts/run_seed_evaluation.sh). Without `--phase` the script submits a worker array (one task per (config, seed) pair) plus a dependent aggregate task. The aggregate task calls `aggregate_and_plot_seed_evaluation` from [src/experiments/utils/seed_evaluation.py](src/experiments/utils/seed_evaluation.py). The aggregate phase is self-healing. It re-submits failed indices until all `eval_results_final.yaml` files exist or `max-heal-rounds` is hit. Both paths write `seed_evaluation_summary.yaml` and plot into `seed_evaluation_plots/`.

**Baseline multi-seed evaluation.** The `single` mode calls `run_single_baseline` and rolls out one baseline policy on one seed without RLlib. The `all-seeds` mode calls `run_baselines_across_seeds` and performs seed evaluation on all baselines similar to the RL multi-seed evaluation. The full sweep [run_all_baselines.sh](scripts/run_all_baselines.sh) follows the same parallel SLURM pattern as the RL seed-eval path. The aggregate task reuses `aggregate_and_plot_seed_evaluation` with `mode="baseline"`.

### 1.3 Object Ownership

Several layers each own a distinct slice of state. Understanding the ownership avoids confusion when debugging.

| Layer | Owns |
|---|---|
| CLI | Argument parsing, mode dispatch, top-level `SeedManager`. |
| `ExperimentRunner` / `EvaluationRunner` | Training loop, periodic checkpointing, intra-training evaluation, W&B logging. |
| Algorithm wrapper (`IPPOWrapper` etc.) | RLlib `PPOConfig`, RLlib trainer, env factory, multi-agent policy mapping. |
| RLlib trainer | Worker processes, sample collection, policy updates, checkpoint state. |
| `InventoryEnvironment` | Supply chain state, pluggable components, episode counter, observation buffers. |

## 2. Core Types and Their Contracts

A small number of classes form the backbone of the codebase. Reading their definitions first will make the rest of the code easier to follow.

### 2.1 `InventoryEnvironment`

The main multi-agent environment is defined in [src/environment/envs/multi_env.py](src/environment/envs/multi_env.py). It is a `pettingzoo.ParallelEnv` in which each warehouse appears as one agent and is identified using the pattern `warehouse_<i>`. The action space has shape `Dict[agent_id, np.ndarray of shape (n_skus,)]` and defines each action in the interval `[-1, 1]`. The concrete action interpretation depends on `action_space.type` in the env config (see 7.3 Action Space). The observation shape is `Dict[agent_id, np.ndarray]`. Each per-agent observation is a flat vector consisting of the concatenation `[local obs | global obs]`. The split is consumed by the RLModule when it routes data to actor and critic. The rewards are produced by the reward calculator. Across all agents, the reward has shape `Dict[agent_id, float]` and follows either team or agent scope.

### 2.2 `BaseAlgorithmWrapper`

The algorithm contract is defined in [src/algorithms/base.py](src/algorithms/base.py). All algorithm wrappers inherit from `BaseAlgorithmWrapper`, which fixes the public surface that `ExperimentRunner` calls into. That surface consists of `train`, `evaluate`, `rollout`, `save_checkpoint`, `load_checkpoint`, and `get_policy`. In addition to the abstract methods, the base class owns four static helpers that every subclass reuses. The most important is `create_env_factory`, which produces the factory function that RLlib invokes to instantiate environments on each worker. Alongside it sits `noop_logger_creator`, a helper that redirects RLlib's default Trainable logger to a temp directory and so prevents HPC runs from writing to NFS-mounted home directories. The remaining two helpers both inspect a network config. Specifically, `extract_max_seq_len` reads the GRU sequence length and `has_gru_from_config` walks the config recursively to detect any GRU usage.

### 2.3 `EnvironmentContext`

The shared context object is defined in [src/environment/context.py](src/environment/context.py). It is built once per environment through `create_environment_context` and is then passed to every pluggable component, so the components do not need to duplicate environment-level parameters. The exposed fields fall into three groups. The basic dimensions are `n_warehouses`, `n_skus`, `n_regions`, and `episode_length`. The cost fields are `holding_cost`, `penalty_cost`, and `shipment_cost`. The geometry fields are `sku_weights` and `distances`. A final optional field, `preprocessed_data`, carries the train and validation demand DataFrames when the data source is real-world.

### 2.4 `StochasticComponent`

All random components inherit from `StochasticComponent` in [src/environment/components/base.py](src/environment/components/base.py). The contract is intentionally minimal and consists of a single `reset(rng: Optional[np.random.Generator])` method. The environment calls `reset` on each stochastic component at every episode reset and passes in an RNG drawn from the environment's `SeedManager`. This wiring guarantees that any randomness inside a component is reproducible from the root seed.

## 3. Configuration System

All configuration is fully typed through Pydantic models and persisted as YAML.

### 3.1 Schema

The schema in [src/config/schema.py](src/config/schema.py) is anchored by three top-level models: `EnvironmentConfig`, `FeatureConfig`, and `AlgorithmConfig`. The latter is a discriminated union over `IPPOConfig`, `MAPPOConfig`, and `CPPOConfig` and is resolved on the `name` field. Beneath them sits a layer of sub-models, one per pluggable subsystem (inventory setup, demand sampler, lead-time sampler, reward calculator, action space). A second cluster of sub-models defines tuning search spaces, schedulers, and search algorithms.

### 3.2 Loader

The [src/config/loader.py](src/config/loader.py) module exposes three entry points. `load_yaml` reads a YAML file. `validate_config` validates a dictionary against a Pydantic schema. `load_environment_config` and `load_algorithm_config` combine the two steps and resolve the discriminated algorithm union. When a config is rejected, the error message points back to a Pydantic validator in the schema module.

### 3.3 Configuration Files

YAML files live under [config_files/](config_files/) and are split across four directories. [config_files/algorithms/](config_files/algorithms/) holds per-algorithm hyperparameters. [config_files/environments/](config_files/environments/) holds environment configurations such as warehouses, SKUs, costs, and demand parameters. [config_files/experiments/](config_files/experiments/) holds Ray Tune sweep definitions. [config_files/features/](config_files/features/) holds the observation feature toggle file referenced by env configs through `feature_config_path`.

## 4. Registry Pattern

The environment is built from five interchangeable subsystems: demand sampling, demand allocation, lead times, lost sales handling, and rewards. Each has a base class and a string-keyed registry. At construction time, the registry reads the env config `type`, picks the concrete class, and instantiates it with shared `EnvironmentContext` plus that component’s sub-config.

There are three places in the code where registries appear: 

- **Environment**: [src/environment/registry.py](src/environment/registry.py) holds five `*_REGISTRY` maps (`DEMAND_SAMPLER`, `DEMAND_ALLOCATOR`, `LEAD_TIME_SAMPLER`, `LOST_SALES_HANDLER`, `REWARD_CALCULATOR`) with matching `register_*` / `get_*` helpers. Concrete classes register on import at the bottom of that file.
- **Algorithms**: Algorithms follow the same pattern. The algorithm registry lives in  [src/algorithms/registry.py](src/algorithms/registry.py) and maps `"ippo"`, `"mappo"`, and `"cppo"` to wrapper classes.
- **Architectures**: Architectures follow the same pattern. The architectures registry lives in  [src/algorithms/models/registry.py](src/algorithms/models/registry.py) and maps `"mlp"` and `"gru"` to `NetworkArchitecture` subclasses.

### 5. Experiment and Evaluation Runtime
The runtime objects in [src/experiments/runner.py](src/experiments/runner.py) are:

- **`ExperimentRunner`**: Builds the algorithm, runs `num_iterations` training iterations, periodically saves checkpoints (`checkpoint_freq`), runs intra-training evaluation every `eval_interval` iterations using `num_eval_episodes`, and (optionally) logs metrics to Weights & Biases.
- **`EvaluationRunner`**: Loads a checkpoint, runs deterministic rollouts (`num_eval_episodes` from config or an override), feeds the rollout episodes into the visualization pipeline, and writes evaluation metrics and rollout artifacts to disk.

Evaluation envs use a fixed `num_eval_episodes`  cycle: the env's episode counter resets to 0 every `num_eval_episodes` resets, so consecutive evaluation rounds replay the same fixed validation episodes drawn from the eval-seed sequence.

File I/O, run-name generation, checkpoint resolution, and metadata persistence are handled by helpers in [src/experiments/utils/experiment_utils.py](src/experiments/utils/experiment_utils.py). Optional Weights & Biases integration lives in [src/experiments/utils/wandb.py](src/experiments/utils/wandb.py) and is activated by passing `--wandb-project` (and optionally `--wandb-name`).

### 6. RLlib Integration Notes

The RLlib integration follows the new API stack. The algorithm wrapper builds a `PPOConfig` from `ray.rllib.algorithms.ppo` and registers an env factory via `ray.tune.registry.register_env`. The factory then wraps the PettingZoo `InventoryEnvironment` in `ParallelPettingZooEnv` for multi-agent algorithms. CPPO uses `CentralizedEnvWrapper` instead which passes a single `RLModuleSpec` per policy through `MultiRLModuleSpec`. The `MeanStdFilter` env-to-module connector is attached when `obs_normalization == "meanstd"` and any custom learners attach through the `learner_class` and `learner_config_dict` fields of the training config.

## 7. Seeding and Reproducibility

The central `SeedManager` in [src/utils/seed_manager.py](src/utils/seed_manager.py) spawns named child seeds from a single root seed using  NumPy's `SeedSequence`. Three registries scope the named seed slots.

- **`EXPERIMENT_SEEDS` registry**: Used by the experiment runner. Scopes named seed slots for `('data_weights', 'data_distances', 'data_costs', 'train', 'eval', 'obs_stats')`
- **`ENVIRONMENT_SEEDS` registry**: Used by the environment at construction time. Scopes named seed slots for `('preprocessing', 'inventory', 'demand_sampler', 'lead_time_sampler')`
- **`STOCHASTIC_SEEDS` registry**: Used by stochastic components which reset on every episode with an RNG obtained through `seed_manager.get_rng(name)`. Scopes named seed slots for `('demand_sampler', 'lead_time_sampler')`

Per-(worker, env) seeds are derived from the root seed inside the env factory. The derivation ensures that parallel envs across RLlib workers produce diverse but deterministic episodes. Just before the algorithm is built, the wrappers also seed Python's `random` module, NumPy, and PyTorch and call `torch.use_deterministic_algorithms(True)`.

## 8. Visualization

Visualization helpers live under [src/visualization/utils/](src/visualization/utils/) and consist of these three key parts:

- [src/visualization/utils/eval_rollout_plots.py](src/visualization/utils/eval_rollout_plots.py): generates per-episode plots from `EvaluationRunner` rollouts. The plots cover inventory levels, order quantities both per step and as a summary, the cost breakdown, demand fulfillment, the shipment heatmap, and raw and normalized observations. 
- [src/visualization/utils/seed_eval_plots.py](src/visualization/utils/seed_eval_plots.py): provides plotting helpers for seed-evaluation output. The helpers cover per-config mean ± std training and eval-return curves and baseline comparison bands. 
- [src/visualization/utils/plot_data_loader.py](src/visualization/utils/plot_data_loader.py): loads on-disk seed-evaluation artifacts for the [src/visualization/plots.ipynb](src/visualization/plots.ipynb) notebook such as training metrics, eval results, summaries, episodes.

## 9. Scripts

The [scripts/](scripts/) directory holds the shell entry layer which covers local invocation and SLURM submission:

| Script | Purpose |
|---|---|
| [scripts/run_experiment.sh](scripts/run_experiment.sh) | Runs a single experiment locally. |
| [scripts/run_experiment_batch.sh](scripts/run_experiment_batch.sh) | Submits an array batch of experiments to SLURM. |
| [scripts/run_evaluation.sh](scripts/run_evaluation.sh) | Runs evaluation on a single checkpoint. |
| [scripts/run_evaluation_batch.sh](scripts/run_evaluation_batch.sh) | Submits an array batch of evaluations to SLURM. |
| [scripts/run_single_baseline.sh](scripts/run_single_baseline.sh) | Runs one non-RL baseline. |
| [scripts/run_all_baselines.sh](scripts/run_all_baselines.sh) | Runs the full baseline sweep. |
| [scripts/run_seed_evaluation.sh](scripts/run_seed_evaluation.sh) | Runs the parallel SLURM seed-evaluation pipeline. |
| [scripts/run_seed_evaluation_sequential.sh](scripts/run_seed_evaluation_sequential.sh) | Runs the sequential single-node seed-evaluation pipeline. |
| [scripts/run_tune_experiment.sh](scripts/run_tune_experiment.sh) | Runs a Ray Tune sweep. |
| [scripts/lib/start_ray.sh](scripts/lib/start_ray.sh) | Bootstraps a Ray cluster for SLURM jobs. |