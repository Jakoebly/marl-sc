# marl-sc — Multi-Agent RL for Heterogeneous Supply Chain Inventory Management

A single-echelon distribution network of warehouses must reorder multiple SKUs each
timestep to minimise total supply-chain cost (holding + lost-sales + inbound/outbound
shipment) under stochastic, order-level demand. Each warehouse is one RL agent acting
from its own local observation. The thesis studies where cooperative **parameter-shared**
PPO breaks once the network stops being symmetric (different demand, SKU mix, and cost
geography per warehouse), and which architectural interventions recover the lost
performance.

**Note.** This is research code for a thesis, not a packaged library. Entry points are
the two Python CLIs and the shell scripts under [`scripts/`](scripts/). Some subsystems
are implemented but intentionally unused by the thesis configs (see
[Work in progress](#11-work-in-progress--not-fully-integrated)).

---

## Contents

1. [Documentation map](#1-documentation-map)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Running locally](#4-running-locally)
5. [Running on the cluster (SLURM)](#5-running-on-the-cluster-slurm)
6. [Configuration system](#6-configuration-system)
7. [What the environment models](#7-what-the-environment-models)
8. [Algorithms, architectures, and interventions](#8-algorithms-architectures-and-interventions)
9. [Baselines](#9-baselines)
10. [Analysis and visualization](#10-analysis-and-visualization)
11. [Work in progress / not fully integrated](#11-work-in-progress--not-fully-integrated)
12. [Reproducibility](#12-reproducibility)

---

## 1. Documentation map

The `docs/` directory is the authoritative reference. Read the relevant file before
changing behaviour.

| Document | Purpose |
|---|---|
| [docs/SCOPE.md](docs/SCOPE.md) | Research questions, contributions, environment design, algorithms, baselines, the five interventions (A1–A5), and limitations. |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Per-cell experimental program (E0–E4), statistical protocol, appendix experiments. |
| [docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md) | What the supply-chain system models and how each component behaves. |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Runtime wiring, entry points, RLlib integration, registries, seeding. |

---

## 2. Repository structure

```
marl-sc/
├── src/                        # All Python source
│   ├── algorithms/             # RL algorithm wrappers around RLlib PPO
│   │   ├── base.py             #   BaseAlgorithmWrapper contract
│   │   ├── ippo.py / mappo.py / cppo.py
│   │   ├── registry.py         #   "ippo" | "mappo" | "cppo" -> wrapper
│   │   ├── learners/           #   Custom PPO learners (hysteretic, per-SKU reward decomposition)
│   │   └── models/
│   │       ├── architectures/  #   MLP, GRU, per-SKU self-attention (+ registry)
│   │       └── rlmodules/      #   Shared ActorCriticRLModule
│   ├── config/                 # Pydantic schema (schema.py) + YAML loader (loader.py)
│   ├── data/                   # Synthetic generation + real-world preprocessing (see §11)
│   │   └── notebooks/          #   Offline data-generation notebooks
│   ├── environment/            # PettingZoo multi-agent env + pluggable components
│   │   ├── components/         #   Demand samplers, allocators, lead times, lost sales, rewards
│   │   ├── envs/               #   multi_env.py (main), single_env.py (CPPO wrapper)
│   │   ├── context.py          #   EnvironmentContext shared by components
│   │   └── registry.py         #   Five string-keyed component registries
│   ├── experiments/            # CLIs, runners, tune + seed-eval orchestration
│   │   ├── run_experiment.py   #   RL CLI (train / tune / evaluate / seed-eval)
│   │   ├── run_baselines.py    #   Non-RL baseline CLI
│   │   ├── runner.py           #   ExperimentRunner / EvaluationRunner
│   │   └── utils/              #   arg parsing, seed-eval, ray-tune, experiment I/O
│   ├── utils/                  # Seed manager, observation stats, weight transfer, cost generator
│   └── visualization/          # Rollout + seed-eval plots, plots.ipynb
├── config_files/               # YAML configs (see §6)
│   ├── algorithms/             #   Per-algorithm hyperparameters (E0–E3, A1–A5, ablations)
│   ├── environments/           #   Env configs (warehouses, SKUs, costs, demand)
│   ├── experiments/            #   Ray Tune sweep definitions
│   └── features/               #   Observation feature-toggle configs
├── scripts/                    # Bash entry points for local + SLURM runs
│   └── lib/start_ray.sh        #   Ray bootstrap for SLURM jobs
├── docs/                       # Project documentation (see §1)
├── tests/                      # Pytest suite
├── experiment_outputs/         # Run artifacts (checkpoints, eval YAML, plots)
├── requirements.txt
└── thesis/                     # LaTeX thesis source (not part of the runtime)
```

---

## 3. Installation

- **Python 3.12**
- Core stack: `ray[rllib]` 2.52, `torch` 2.9, `pettingzoo` 1.25, `gymnasium` 1.1,
  `pydantic` 2.12, `numpy`/`pandas`/`scipy`, `optuna` + `scikit-optimize` + `bayesian-optimization`
  (tuning and BO baselines), `wandb` (optional logging), `matplotlib`/`seaborn` (plots).

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

All commands below are run **from the repository root**, and the repo root must be on
`PYTHONPATH` because imports are absolute (`from src....`). The SLURM scripts export this
for you. Locally, set it yourself:

```bash
export PYTHONPATH="$PWD"          # bash
# $env:PYTHONPATH = (Get-Location).Path   # PowerShell
```

---

## 4. Running locally

Everything runs through one of two CLIs.

### 4.1 RL CLI (`src/experiments/run_experiment.py`)

Supported modes via `--mode` argument:

| Mode | What it does |
|---|---|
| `single` | Train one (env, algorithm) config for `num_iterations`. |
| `tune` | Ray Tune hyperparameter sweep over a search-space YAML. |
| `evaluate` | Load a checkpoint and run deterministic rollouts (optionally with plots). |
| `seed-eval` | Train + benchmark one config (or top-K tune trials) across several seeds. |

**Common arguments**

| Argument | Meaning |
|---|---|
| `--mode {single,tune,evaluate,seed-eval}` | Required, see above. |
| `--env-config PATH` | Environment YAML. Required for new `single`/`tune` runs. |
| `--algorithm-config PATH` | Algorithm YAML. Required for new `single`/`tune` runs. |
| `--storage-dir PATH` | Output root (default `./experiment_outputs`). |
| `--experiment-name NAME` | Run folder name (auto-generated if omitted). |
| `--root-seed INT` | Root seed for reproducibility (default 42). |
| `--resume-from PATH\|NAME` | Resume: checkpoint dir (`single`) or experiment name (`tune`). |
| `--wandb-project NAME` / `--wandb-name NAME` | Enable optional Weights & Biases logging. |

**Mode-specific arguments**

| Argument | Modes | Meaning |
|---|---|---|
| `--tune-config PATH` | `tune` | Search-space YAML (required for new sweeps). |
| `--num-samples INT` | `tune` | Number of trials (required for new sweeps). |
| `--num-cpus` / `--num-gpus` / `--num-cpus-per-env-runner` | `tune` | Per-trial resources (GPUs default 0). |
| `--top-k INT` | `tune`, `seed-eval` | Top trials to save / seed-evaluate (default 10). |
| `--checkpoint-dir PATH` | `evaluate` | Evaluate a specific checkpoint (exclusive with `--experiment-name`). |
| `--checkpoint-number N` | `evaluate` | Pick `checkpoint_N` under a named run (default: `final` → `best` → latest). |
| `--eval-episodes INT` | `evaluate`, `seed-eval` | Rollout episode count. |
| `--visualize` | `evaluate` | Save per-episode rollout plots. |
| `--n-seeds INT` | `seed-eval` | Number of seeds (default 5; seeds are 100, 200, …). |
| `--tune-name NAME` | `seed-eval` | Seed-evaluate the top-K trials of a finished tune run. |
| `--eval-seed INT` | `tune`, `seed-eval`, `single` | Held-out root seed for benchmark eval (default 123). |
| `--num-iterations INT` | `seed-eval` | Override training iterations. |
| `--eval-interval` / `--intra-eval-episodes` / `--final-eval-episodes` | `seed-eval` | Intra-training vs final benchmark eval controls. |

**Examples**

```bash
# 1) Train MAPPO with parameter sharing on the symmetric env
python src/experiments/run_experiment.py \
  --mode single \
  --env-config config_files/environments/env_symmetric_3WH5SKU.yaml \
  --algorithm-config config_files/algorithms/algorithms_E0/mappo_ps_E0.yaml \
  --storage-dir experiment_outputs/Runs \
  --experiment-name MAPPO_PS_Symmetric \
  --root-seed 42

# 2) Evaluate a trained run with rollout plots
python src/experiments/run_experiment.py \
  --mode evaluate \
  --storage-dir experiment_outputs/Runs \
  --experiment-name MAPPO_PS_Symmetric \
  --eval-episodes 100 --visualize --root-seed 42

# 3) Hyperparameter sweep (Ray Tune)
python src/experiments/run_experiment.py \
  --mode tune \
  --env-config config_files/environments/env_combined_hetero_3WH5SKU.yaml \
  --algorithm-config config_files/algorithms/algorithms_E0/mappo_ps_E0.yaml \
  --tune-config config_files/experiments/experiments_E1/tune_config_E1.yaml \
  --num-samples 50 --storage-dir experiment_outputs/Tuning \
  --experiment-name MAPPO_PS_CombinedHetero_Tune

# 4) Multi-seed evaluation of a fresh config (sequential, single machine)
python src/experiments/run_experiment.py \
  --mode seed-eval \
  --experiment-name MAPPO_PS_SkuHetero_SeedEval \
  --env-config config_files/environments/env_sku_hetero_3WH5SKU.yaml \
  --algorithm-config config_files/algorithms/algorithms_E2/A4/mappo_ps_sku_hetero_A4_E2.yaml \
  --n-seeds 5 --storage-dir experiment_outputs/Runs
```

### 4.2 Baseline CLI (`src/experiments/run_baselines.py`)

Runs the non-RL reference policies under the same multi-seed protocol.

Supported modes via `--mode` argument:

| Mode | What it does |
|---|---|
| `single` | One baseline/seed. |
| `all-seeds` | All baselines at once. |

| Argument | Meaning |
|---|---|
| `--mode {single,all-seeds}` | Required, see above. |
| `--env-config PATH` | Environment YAML (required). |
| `--baselines KEY [KEY ...]` | Subset of baselines (default: all — see §9 for keys). |
| `--n-seeds INT` | Number of seeds (default 5). |
| `--root-seed INT` | Root seed (`single` mode). |
| `--eval-episodes INT` / `--eval-seed INT` | Benchmark episodes (default 100) / held-out eval seed (default 123). |
| `--storage-dir PATH` / `--experiment-name NAME` | Output location. |
| `--visualize` | Save the sweep / BO-convergence and per-episode plots. |

```bash
# Evaluate the base-stock oracle on one seed with plots
python src/experiments/run_baselines.py \
  --mode single --baselines bs_optimized \
  --env-config config_files/environments/env_cost_hetero_3WH5SKU.yaml \
  --root-seed 100 --experiment-name Baselines_CostHetero --visualize
```

---

## 5. Running on the cluster (SLURM)

The [`scripts/`](scripts/) directory wraps the CLIs for HPC. Each script has a
`CLUSTER="..."` variable near the top that selects the environment (module load, venv
activation, working directory) for one of the preconfigured clusters. 

Before running on a new machine you must:

1. Set `CLUSTER` to the right value.
2. Adjust the `#SBATCH` lines marked `<-- ADJUST THIS!` (partition/cluster, `--chdir`,
   and any node `--exclude` list) and the venv path in the matching `CLUSTER` branch.

There are two kinds of scripts to use, depending on the type of task:
- Self-contained `sbatch` scripts
- Launcher scripts

### 5.1 Self-contained `sbatch` scripts

These are submitted with `sbatch`. The actual run command (config paths, storage dir,
experiment name) is written **inside the script body**, so you edit those lines and then
submit.

| Script | Submits |
|---|---|
| [scripts/run_experiment.sh](scripts/run_experiment.sh) | One training run (+ evaluation). |
| [scripts/run_tune_experiment.sh](scripts/run_tune_experiment.sh) | A Ray Tune sweep (large multi-core node). |
| [scripts/run_evaluation.sh](scripts/run_evaluation.sh) | Evaluation of one checkpoint with visualization. |
| [scripts/run_experiment_batch.sh](scripts/run_experiment_batch.sh) | An array batch of training runs. |
| [scripts/run_evaluation_batch.sh](scripts/run_evaluation_batch.sh) | An array batch of evaluations. |

```bash
# Edit the config paths inside the script, then:
sbatch scripts/run_experiment.sh
sbatch scripts/run_evaluation.sh --name MyRun --checkpoint-number 50
```

### 5.2 Launcher scripts (run on the login node)

These are executed directly (not via `sbatch`). They take CLI arguments, compute the
SLURM array size, and submit a **worker array (one task per (config, seed) or
(baseline, seed) pair) plus a dependent aggregate job. The aggregate is
self-healing and re-submits up to `--max-heal-rounds` times only the (config, seed) pairs whose
`eval_results_final.yaml` are still missing and workers
auto-resume from the latest periodic checkpoint. Completed pairs are skipped, so re-runs
are cheap.

| Script | Runs |
|---|---|
| [scripts/run_seed_evaluation.sh](scripts/run_seed_evaluation.sh) | Parallel RL multi-seed evaluation (`--mode single` from a config, or `--mode tune` over top-K trials). |
| [scripts/run_seed_evaluation_sequential.sh](scripts/run_seed_evaluation_sequential.sh) | Single-node sequential fallback for the above. |
| [scripts/run_all_baselines.sh](scripts/run_all_baselines.sh) | Parallel baseline sweep across seeds. |
| [scripts/run_single_baseline.sh](scripts/run_single_baseline.sh) | One baseline on one seed (submitted via `sbatch`). |

**Key launcher arguments** (see the header comment in each script for the full list):

`run_seed_evaluation.sh`
- `--mode {single,tune}` (required), `--name <run-or-tune-name>` (required)
- `--env-config` + `--algorithm-config` (single mode, to seed-evaluate a fresh config;
  otherwise the saved configs of an existing run are reused)
- `--n-seeds` (default 10), `--top-k` (tune, default 10), `--num-iterations`
- `--final-eval-episodes` (100), `--intra-eval-episodes` (50), `--eval-interval` (1),
  `--eval-seed` (123), `--checkpoint-freq` (25), `--wandb`

`run_all_baselines.sh`
- `--name` (required), `--env-config` (which heterogeneity type), `--n-seeds` (default 10),
  `--baselines <subset>`, `--eval-episodes` (100), `--eval-seed` (123)

```bash
# Parallel seed-eval of a fresh MAPPO+A4 config on SKU-heterogeneous env, 10 seeds
./scripts/run_seed_evaluation.sh \
  --mode single --name MAPPO_PS_A4_SkuHetero \
  --env-config config_files/environments/env_sku_hetero_3WH5SKU.yaml \
  --algorithm-config config_files/algorithms/algorithms_E2/A4/mappo_ps_sku_hetero_A4_E2.yaml \
  --n-seeds 10

# Seed-eval the top-10 trials of a finished tune run
./scripts/run_seed_evaluation.sh --mode tune --name MAPPO_PS_CombinedHetero_Tune --top-k 10

# All reference baselines on the combined-heterogeneity env, 10 seeds
./scripts/run_all_baselines.sh \
  --name Baselines_CombinedHetero \
  --env-config config_files/environments/env_combined_hetero_3WH5SKU.yaml \
  --n-seeds 10
```

### 5.3 Ray bootstrap — `scripts/lib/start_ray.sh`

Sourced (not executed) by the SLURM jobs. It starts a per-task Ray head with
deterministic port allocation and cgroup-aware memory limits so that many array tasks can
be packed onto one node without port collisions or OOM. Tunable via `RAY_*` environment
variables set before sourcing (see the header of the file). Not needed for local runs.

---

## 6. Configuration system

Configuration is fully typed (Pydantic) and persisted as YAML. Update the schema before
the YAML, otherwise validation errors point back at the schema.

- **Schema** ([src/config/schema.py](src/config/schema.py)): Three top-level models `EnvironmentConfig`, `FeatureConfig`, and `AlgorithmConfig` (a discriminated union over
  `IPPOConfig` / `MAPPOConfig` / `CPPOConfig`, resolved on `name`).
- **Loader**: ([src/config/loader.py](src/config/loader.py)):  Contains the config loading machinery (`load_environment_config`,
  `load_algorithm_config`).

Config files live under [`config_files/`](config_files/) in four directories:

| Directory | Holds |
|---|---|
| `algorithms/` | Per-algorithm hyperparameters, organised by experiment block (E0, E1, E2/A1–A5, E3 ablations). |
| `environments/` | Environment configs. The five main ones are `env_symmetric`, `env_sku_hetero`, `env_demand_hetero`, `env_cost_hetero`, `env_combined_hetero` (all `3WH5SKU`). |
| `experiments/` | Ray Tune search-space YAMLs. |
| `features/` | Observation feature-toggle files referenced by env configs via `feature_config_path`. |

**Registry pattern:** New env components, algorithms, and architectures are registered by
string key rather than imported directly. Do not bypass the registries:

- Environment components: [src/environment/registry.py](src/environment/registry.py)
  (demand sampler, allocator, lead-time sampler, lost-sales handler, reward calculator).
- Algorithms: [src/algorithms/registry.py](src/algorithms/registry.py).
- Architectures: [src/algorithms/models/registry.py](src/algorithms/models/registry.py).

---

## 7. Environment

The environment models a single-echelon distribution network, in which peer warehouses hold inventory and fulfil
customer demand across geographic regions, replenishing from an implicit uncapacitated
supplier with a lead time. Full detail in [docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md).

- **Multi-agent env**: `InventoryEnvironment` (a `pettingzoo.ParallelEnv`) in
  [src/environment/envs/multi_env.py](src/environment/envs/multi_env.py). Each warehouse is
  an agent `warehouse_<i>`; the action is a continuous per-SKU vector in `[-1, 1]`.
- **Action spaces** (`action_space.type`): `direct`, `demand_centered` (thesis default),
  `base_stock`. All continuous in `[-1, 1]`, differing in how actions map to integer orders.
- **Pluggable components**, each chosen by config `type`:

| Component | Options (thesis default in **bold**) |
|---|---|
| Demand sampler | **Poisson (synthetic, order-level)**, Empirical (real-world CSV replay) |
| Demand allocator | **Greedy** (cheapest-first with order splitting → implicit transshipment) |
| Lead-time sampler | **Fixed**, Stochastic (expected ± uniform deviation) |
| Lost-sales handler | Closest, **Shipment-proportional**, Cost-based softmax |
| Reward calculator | **Cost-based** (holding + penalty + inbound/outbound shipment; `team`/`agent` scope; optional per-SKU decomposition) |

- **Observations**: each agent gets a flat `[local | global]` vector built from toggleable
  feature blocks (inventory, pipeline, rolling demand mean, and several optional features).
  Under parameter sharing a one-hot warehouse id is prepended. Actor/critic each read the
  local or global slice per algorithm.
- **Observation normalization** (`obs_normalization`): `off`, `ratio`, `meanstd`,
  `meanstd_custom`, `meanstd_sku_grouped`, `meanstd_wh_grouped` (the last is intervention A1).
- **Heterogeneity axes** (the research variable): the five `3WH5SKU` env configs perturb one
  or more of SKU mix, per-region demand, and cost geography while holding total expected
  weighted network throughput constant (~200 weight-units/step) so problem size does not
  confound the comparison.

---

## 8. Algorithms, architectures, and interventions

**Algorithms** (all PPO-family, all built on a shared `ActorCriticRLModule`):

| Algorithm | Description |
|---|---|
| **IPPO** | Independent PPO; actor and critic both see local observations. |
| **MAPPO** | CTDE; local actor, global (centralized) critic. Primary target for interventions. |
| **CPPO** | Single centralized PPO over the joint observation/action (RL upper bound). |

IPPO and MAPPO support parameter sharing (`parameter_sharing`): one shared policy across
all warehouses, with a one-hot warehouse id added to each observation.

**Network architectures** (`networks` block; registered as `mlp`, `gru`, `attention`):
flat MLP, GRU (recurrent), and per-SKU self-attention (intervention A4 — tokenises the
per-SKU observation and attends across SKU tokens within each warehouse block).

**Training options**: state-independent log-std vs a mu-sigma head; optional per-agent
observation normalization; warm-start weight transfer (`warmstart_weights_path`); plus two
learner swaps described below.

**Interventions** (the prescriptive contribution, detailed in [docs/SCOPE.md](docs/SCOPE.md) §7):

| ID | Name | Mechanism |
|---|---|---|
| A1 | Per-agent observation normalization | `meanstd_wh_grouped` — per-warehouse, SKU-grouped statistics. |
| A2 | Static SKU-weight feature | Appends the SKU weight to each per-SKU observation block. |
| A3 | Per-SKU reward decomposition | Per-SKU reward channels + per-SKU value heads + factored PPO surrogate (`RewardDecompositionPPOTorchLearner`). |
| A4 | Per-SKU self-attention | Attention backbone for actor and critic (this thesis's proposed intervention). |
| A5 | Combined | Stacks the most effective subset of A1–A4. |

A matched-capacity MLP control additionally isolates A4's mechanism from its parameter count.

---

## 9. Baselines

Six non-RL reference policies (plus CPPO as an RL ceiling) run under the same multi-seed
protocol. CLI keys for `--baselines`:

| Key | Policy |
|---|---|
| `random` | Uniform random orders (lower bound). |
| `constant` | Per-(WH, SKU) constant order, calibrated from observed demand, swept over α. |
| `bs_newsvendor` | Analytical newsvendor base-stock from true demand parameters. |
| `bs_adaptive` | Rolling-mean base-stock from observed history. |
| `bs_optimized` | Simulation-optimized base-stock via Bayesian optimization (strong ceiling). |
| `bs_independent` | Per-warehouse iterated best-response BO (coordination-free ceiling). |

Returns are reported both raw and as a **normalized score** in `[0, 1]` between Random and
BS-Optimized, so degradation and recovery are comparable across the five environments.

---

## 10. Analysis and visualization

- **Rollout / seed-eval plots**: helpers under
  [src/visualization/utils/](src/visualization/utils/) generate per-episode rollout figures
  (inventory, orders, cost breakdown, fulfilment, shipment heatmaps) and per-config
  mean ± std training/eval curves with baseline bands.
- **Result notebooks**: the thesis tables and figures are produced by
  [src/analysis/numbers.ipynb](src/analysis/numbers.ipynb) (numerical results) and
  [src/visualization/plots.ipynb](src/visualization/plots.ipynb) (figures). New analysis
  logic belongs in [src/analysis/utils/](src/analysis/utils/) and is called from the
  notebooks.
- **Weights & Biases**: optional; pass `--wandb-project` (and `--wandb-name`) to the RL CLI,
  or `--wandb` to the seed-eval launcher.

Run artifacts land under `experiment_outputs/` (checkpoints, `eval_results_final.yaml`,
`seed_evaluation_summary.yaml`, and `seed_evaluation_plots/`).

---

## 11. Work in progress / not fully integrated

Several code paths are implemented and schema-validated but **not exercised by any current
thesis config**. Treat them as experimental. The thesis uses `data_source: custom`,
Poisson demand, greedy allocation, fixed lead times, shipment-proportional lost sales, and
the cost reward calculator exclusively.

- **Real-world data (`data_source: real_world`)** — not fully integrated. The CSV
  preprocessor ([src/data/preprocessor.py](src/data/preprocessor.py)) and the Empirical
  demand sampler it feeds are implemented but unused because the available demand data is
  of insufficient quality. The E4.2 transfer experiment uses a real-world-inspired but
  still synthetic instance instead.
- **In-run synthetic data generator (`data_source: synthetic`)** — experimental. The
  procedural generator ([src/data/data_generator.py](src/data/data_generator.py)) is not
  wired into any config; the transfer instance is generated offline via the notebooks in
  [src/data/notebooks/](src/data/notebooks/).
- **Stochastic lead times** — the stochastic lead-time sampler is implemented but all
  configs use `fixed` (lead-time stochasticity is out of scope for the main study).
- **Hysteretic PPO** (`hysteretic_beta` → `HystereticPPOTorchLearner`) — implemented and
  documented but enabled by no config.
- **Alternative lost-sales handlers** — the `closest` and cost-based `softmax` handlers are
  registered but unused (all configs use `shipment`-proportional).
- **No-op / inert config fields** — `max_wh_capacities` (validated, enforced by no code);
  `reward_calculator.params.cost_weights` (weighted-sum branch is commented out, so the
  unweighted sum is used); the `base_stock` action space (pilot relict; all configs use
  `demand_centered`); `use_mu_sigma_head` and `networks.shared_layers` (supported but off
  everywhere).
- **Legacy configs / backup scripts** — pilot env configs (`1WH2SKU`, `3WH2SKU`) and the
  sequential seed-eval fallback are kept for convenience.

See [docs/SYSTEM_MODEL.md](docs/SYSTEM_MODEL.md) §12 for the authoritative list.

---

## 12. Reproducibility

All randomness derives from a single root seed via NumPy `SeedSequence`. The central
`SeedManager` ([src/utils/seed_manager.py](src/utils/seed_manager.py)) spawns named child
seeds for data generation, training, evaluation, and stochastic components, and per-(worker,
env) seeds are derived inside the env factory so parallel RLlib workers stay deterministic
yet diverse. Before training, Python's `random`, NumPy, and PyTorch are seeded and
`torch.use_deterministic_algorithms(True)` is set. In multi-seed evaluation the training
seeds are 100, 200, … while the benchmark eval uses a fixed held-out `--eval-seed` (default
123) shared across all runs so comparisons are paired on identical evaluation episodes.
