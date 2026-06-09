# System Description And Modeling Decisions

This document describes what the marl-sc system models and how its main components behave. Specifically, it describes the supply chain model, the algorithms, the baselines, the data sources, and the training and evaluation protocols. Together with [SCOPE.md](SCOPE.md), it is the primary reference for thesis writing.

For thesis scope, research questions, contribution claims, environment design, algorithms, baselines, and interventions, see [SCOPE.md](SCOPE.md). For the full experimental program, statistical protocol, and supplementary appendix experiments, see [EXPERIMENTS.md](EXPERIMENTS.md). For runtime wiring, configuration, and implementation details, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## 1. Project Aim

Each warehouse is controlled by an independent RL agent. At every timestep the agent decides how many units of each SKU to reorder. The objective is to minimize total supply chain cost under stochastic demand and non-trivial fulfillment logistics. Total supply chain cost has four components. The components are inventory holding cost, lost-sales penalty, outbound shipment cost, and inbound shipment cost.

## 2. Repository Layout

The repository follows a flat top-level structure. Domain code lives in [src/](src/). Configuration lives in [config_files/](config_files/). Operational scripts live in [scripts/](scripts/).

```
marl-sc/
├── src/                       # All Python source code
│   ├── algorithms/            # RL algorithms and neural network modules
│   ├── config/                # Pydantic schemas and YAML loader
│   ├── data/                  # Synthetic data generation and real-world preprocessing
│   ├── environment/           # PettingZoo environment and pluggable components
│   ├── experiments/           # Training, evaluation, tune, and seed-eval pipelines
│   ├── utils/                 # Seed manager, observation statistics, weight transfer
│   └── visualization/         # Rollout plots, seed-eval plots, plotting notebook
├── config_files/              # YAML configuration files
│   ├── algorithms/            # Per-algorithm hyperparameter configs
│   ├── environments/          # Environment configs (warehouses, SKUs, costs)
│   ├── experiments/           # Ray Tune sweep configs
│   └── features/              # Observation feature toggle config
├── scripts/                   # Bash entry points for local and SLURM runs
│   └── lib/                   # Shared bash helpers
├── tests/                     # Pytest test suite
├── docs/                      # Project documentation including this file
└── thesis/                    # Thesis source (not part of the runtime)
```

The most important subdirectories of [src/](src/) are summarized below.

| Subdirectory | Purpose |
|---|---|
| [src/algorithms/](src/algorithms/) | Algorithm wrappers around RLlib's PPO. Subfolders cover learners, RLModules, and network architectures. |
| [src/config/](src/config/) | Pydantic schemas in [src/config/schema.py](src/config/schema.py) and the YAML loader in [src/config/loader.py](src/config/loader.py). |
| [src/data/](src/data/) | Synthetic supply chain generation in [src/data/data_generator.py](src/data/data_generator.py) and real-world CSV preprocessing in [src/data/preprocessor.py](src/data/preprocessor.py). |
| [src/environment/](src/environment/) | The multi-agent PettingZoo environment in [src/environment/envs/multi_env.py](src/environment/envs/multi_env.py) and a single-agent wrapper in [src/environment/envs/single_env.py](src/environment/envs/single_env.py). Pluggable components are under [src/environment/components/](src/environment/components/). |
| [src/experiments/](src/experiments/) | The CLI and dispatching in [src/experiments/run_experiment.py](src/experiments/run_experiment.py), the runners in [src/experiments/runner.py](src/experiments/runner.py), and helpers under [src/experiments/utils/](src/experiments/utils/). |
| [src/utils/](src/utils/) | Cross-cutting utilities. The seed manager lives in [src/utils/seed_manager.py](src/utils/seed_manager.py). Observation statistics live in [src/utils/obs_stats.py](src/utils/obs_stats.py). |
| [src/visualization/](src/visualization/) | Plotting code for evaluation rollouts and seed evaluations. The notebook [src/visualization/plots.ipynb](src/visualization/plots.ipynb) consumes these helpers. |

## 3. Pluggable Components

The environment is built from five interchangeable subsystems: demand sampling, demand allocation, lead times, lost sales handling, and rewards. Each has a base class and a string-keyed registry. At construction time, the registry reads the env config `type`, picks the concrete class, and instantiates it with shared `EnvironmentContext` plus that component’s sub-config.

### 3.1 Available Environment Components

Each environment has five interchangeable components:

| Component | Available Options |
|---|---|
| **Demand sampler** | Poisson (synthetic), Empirical (real-world data) |
| **Demand allocator** | Greedy (cheapest-first with order splitting) |
| **Lead time sampler** | Fixed (deterministic), Stochastic (expected ± uniform deviation) |
| **Lost sales handler** | Closest warehouse, Shipment-proportional, Cost-based softmax |
| **Reward calculator** | Cost-based (with configurable cost weights, scale factor, team or agent scope) |

## 4. The Supply Chain Model

This section describes the modeling decisions encoded in the environment.

### 4.1 Network Topology

The supply chain is a **single-echelon distribution network** consisting of:

- **Warehouses**: Peer-level facilities that hold inventory and fulfill customer demand. There is no upstream manufacturer or multi-tier hierarchy modeled explicitly. Replenishment orders are placed to an implicit external supplier with no capacity constraints. The number of warehouses is configured by the environment config parameter ``n_warehouses``, which can be any positive integer.

- **Demand regions**: Geographic areas where customer orders originate. Each warehouse has a "home region" (its closest region by distance), but any warehouse can ship to any region. The number of demand regions is configured by the environment config parameter ``n_regions``, but must be set to `n_regions == n_warehouses` according to config validation. 

- **SKUs**: Product types the network handles. Each SKU can have its own weights, demand parameters, and cost rates when those fields are configured as vectors or matrices. Configs may also use shared scalar parameters so that SKUs are not necessarily heterogeneous. The number of SKUs is configured by the environment config parameter ``n_skus``, which can be any positive integer.

Distances and shipment costs are configured per environment. If provided, distances must be a non‑negative `(n_warehouses × n_regions)` matrix. Outbound fixed/variable rates are the same non‑negative `(n_warehouses × n_regions)` shape whereas inbound fixed/variable are `(n_warehouses × n_skus)`. Variable blocks must have matching fixed blocks (same shape). With `data_source.type: custom` the distances and all four shipment-cost matrices are required. With other data sources those fields can stay optional.

Note that there are no explicit suppliers modelled. However, the implicit assumption is that each warehouse has exactly one supplier per SKU such that no sourcing decisions need to be made. SKU suppliers do not necessarily need to be shared such across warehouses. 

### 4.2 Agent Decisions

Each warehouse agent makes exactly one type of decision per timestep: **how many units of each SKU to order from the external supplier**. The action is a continuous vector of length `n_skus` in the range `[-1, 1]`. The environment converts the action into integer order quantities according to the configured action space type (see 7.3 Action Space). There are no production decisions, pricing, or routing decisions and the agent only controls replenishment.

### 4.3 Action Space

The environment supports three action space formulations, controlled by the `action_space.type` parameter. in the environment config. All three use continuous actions in `[-1, 1]` but differ in how the environment converts raw actions to integer order quantities.

- **Direct action space** (`type: direct`): Actions are linearly scaled from `[-1, 1]` to `[0, max_order_quantities]` and then rounded to integers. Thus, the action `0` maps to `max_order_quantities / 2`. The relevant config parameter is `max_order_quantities` with shape `(n_skus,)`.

- **Demand-centered action space** (`type: demand_centered`): Actions represent adjustments relative to the current timestep's incoming demand. The action is scaled to `[-max_quantity_adjustment, +max_quantity_adjustment]`, added to the observed home-region demand, and then clipped to a non-negative integer. Thus, the action `0` orders exactly the observed demand. The relevant config parameter is `max_quantity_adjustment` with shape `(1,)` or `(n_skus,)`.

- **Base-stock action space** (`type: base_stock`): Actions specify a target inventory level. The action is scaled from `[-1, 1]` to `[0, max_stock_level]`. The environment then computes the final order quantity as `order_qty = max(0, target − demand − pending_pipeline)` where the pending pipeline is the sum of all in-transit orders. Thus, the action `0` corresponds to a target level of `max_stock_level / 2`. The relevant config parameter is `max_stock_level` with shape `(1,)` or `(n_skus,)`.

### 4.4 Demand Model

Customer demand is stochastic and generated independently for each region at every timestep via a pluggable demand sampler. Available options:

- **Poisson (synthetic):** For each region, the number of customer orders per timestep is drawn from a Poisson distribution according to `lambda_orders`. Each order includes a random subset of SKUs (according to a configured probability `probability_skus`) and the quantity per SKU is drawn from a SKU-specific Poisson distribution with `lambda_quantity`. Rate parameters can vary across regions and SKUs to create heterogeneous demand patterns. A configurable `min_quantity_per_sku` floor clamps each per-SKU quantity from below. Production configs set it to `0` so that a Bernoulli-included SKU that draws a Poisson zero contributes zero units. Setting the floor above zero shifts probability mass upward and inflates the realized mean above `lambda_quantity`.

- **Empirical (real-world):** Historical order data is replayed from a CSV. Specifically, a random contiguous time window is sampled at the start of each episode and the corresponding historical orders are then fed to the environment timestep by timestep.

A modeling nuance matters here. Demand arrives as individual customer _orders_, not as aggregate demand per SKU. This matters because fulfillment and shipment costs are computed at the order level and each order can potentially request multiple SKUs.

### 4.5 Demand Fulfillment and Transshipment

When customer orders arrive, they are allocated to warehouses for fulfillment via a pluggable demand allocator. The current allocation mechanism follows a simple greedy approach.

- **Greedy allocator**: Processes each order sequentially. For each order, it ranks warehouses by their total shipment cost to the order's region (cheapest first) and then fulfills as much as possible from the cheapest warehouse. If demand cannot be completely fulfilled, it splits to the next cheapest warehouse. The process continues until the order is fully fulfilled or the configured `max_splits` limit is reached.

Note that the current allocation logic implies that lateral transshipment between warehouses occurs implicitly. A warehouse may ship to a non-home region if it has inventory and the cheaper warehouse for that region is stocked out. Moreover, order splitting allows partial fulfillment from multiple warehouses but incurs separate shipment costs for each split. Any demand that remains unfulfilled after exhausting all warehouses becomes lost sales.

### 4.6 Lead Times

Replenishment orders do not arrive instantly but have a lead time during which the ordered units are in transit and unavailable. Orders whose actual arrival timestep would fall after the episode ends never trigger an arrival. The episode simply stops to avoid end-of-episode behavior. Lead times are modelled as `actual_lead_time = expected_lead_time + deviation`. While the `expected_lead_time` for each SKU is exposed to the agent by the environment via the pipeline observation feature, lead time variation stems from the stochastic `deviation` parameter, which is drawn per (warehouse, SKU) per order. The lead time is determined by a pluggable lead time sampler.

- **Fixed sampler** (`type: "fixed"`): Covers deterministic lead times. Eliminates stochasticity by returning only the configured `expected_lead_times` per (warehouse, SKU) pair on every call without sampling any `deviation`. 

- **Stochastic sampler**(`type: "stochastic"`): Covers stochastic lead times. Returns `actual_lead_time` computed as `expected_lead_time + deviation` where `deviation` is sampled according to a distribution configurable through `DeviationConfig`. The only deviation type currently implemented is `uniform` that samples from `[-max_deviation, +max_deviation]` and clips the resulting actual lead times to a minimum of `1`. The `max_deviation` parameter can have shape `(1,)` or `(n_skus,)`.

### 4.7 Cost Structure

The reward signal is computed by a pluggable reward calculator. The only currently available one is a cost-based calculator that returns the negative of total supply chain costs. Total supply chain costs have four components:

1. **Holding cost**: Incurred per unit of inventory held at each warehouse at each timestep, scaled by SKU weight. Penalizes excess inventory.
2. **Penalty cost (lost sales)**: Incurred for each unit of demand that could not be fulfilled. Penalty rates are configured per-SKU such that higher-value or harder-to-source SKUs carry higher penalties. Lost sales are attributed back to warehouses via a pluggable lost sales handler. Currently, three assignment approaches are available: 
   - *Closest:* All lost sales in a region are assigned to the nearest warehouse.
   - *Shipment-proportional:* Lost sales are assigned to warehouses proportionally to how many units they shipped to that region. Falls back to the closest warehouse if no shipments occurred.
   - *Cost-based (softmax):* Lost sales are assigned based on shipment cost proximity using a softmax over inverse costs such that cheaper-to-reach warehouses bear more responsibility. The temperature parameter `alpha` controls how concentrated or uniform the assignment is.
3. **Outbound shipment cost:** Incurred when shipping fulfilled orders from warehouses to customer regions. Divided into a fixed per-order component and a variable per-weight component, both configured as matrices over (warehouse, region) pairs.
4. **Inbound shipment cost:** Incurred when receiving replenishment orders from the external supplier. Divided into a fixed per-order component and a variable per-weight component, both configured as matrices over (warehouse, SKU) pairs.

While the config field `cost_weights` can be used to control individual component weights, the current implementation uses the unweighted sum of the four components. A weighted version is present in `CostRewardCalculator.calculate` but commented out and reserved for later use. The reward calculator also supports a configurable `scale_factor` that multiplies the computed costs before converting them to rewards. Additionally, two reward scopes are available. While using `scope: team` returns the same reward to all warehouses equal to the negative sum of all warehouses' costs, `scope: agent` returns to each warehouse only its own negative cost.

### 4.8 Episode Structure

Each episode runs for a fixed number of timesteps `episode_length` as configured in the environment config. The internal sequence of events is implemented in `InventoryEnvironment.step` in [src/environment/envs/multi_env.py](src/environment/envs/multi_env.py). Specifically, that sequence is: 

| Step | What happens | Implementing code |
|---|---|---|
| 1 | Agents place replenishment orders. Actions are rescaled and applied. | `_rescale_actions_to_quantities` and `_apply_orders` in `multi_env.py`. |
| 2 | Previously ordered inventory whose lead time has elapsed arrives. | `_apply_arrivals` in `multi_env.py`. |
| 3 | New customer demand is sampled. | `self.demand_sampler.sample(self.timestep)`. |
| 4 | Demand is allocated across warehouses and fulfilled. | `self.demand_allocator.allocate(orders, self.inventory)`. |
| 5 | Inventory levels are updated. | `self.inventory = max(self.inventory − fulfillment_matrix.sum(axis=0), 0)`. |
| 6 | Observation features are updated from the allocation result. | `_update_observations` in `multi_env.py`. |
| 7 | Lost sales are assigned to warehouses. | `self.lost_sales_handler.calculate_lost_sales(...)`. |
| 8 | Costs are computed. Rewards are returned to agents. | `self.reward_calculator.calculate(...)`. |
| 9 | Final observations for the next decision are produced. | `self._get_observations()`. |
| 10 | Timestep counter advances. Termination and truncation flags are set. | `self.timestep += 1` and the truncations dict. |

Initial inventory at episode start is configured via `initial_inventory.type`:
- `uniform`: samples initial inventory per (warehouse, SKU) from `[min, max]`.
- `custom`: initial inventory is a fixed scalar or 2D list of shape `(n_warehouses, n_skus)`.
- `zero`: initial inventory is zero for all (warehouse, SKU) pairs.

### 4.9 Agent Observations

Each warehouse builds a flat local observation by concatenating the feature blocks enabled in the environment's `features` config. While per-SKU feature blocks contain one value per SKU, aggregate ones add a single scalar to the same block.

Two feature blocks are always on:
- **Inventory** (`inventory`): on-hand stock per SKU and warehouse.
- **Aggregate Inventory** (`inventory_aggregate`): total inventory across SKUs per warehouse.
- **Pipeline** (`pipeline`): per-SKU expected quantity arriving in each of the next `max_expected_lead_time` arrival slots. Late deliveries for which the expected arrival has already passed but the order has not yet actually arrived, are absorbed into slot 0. Expands to `max_expected_lead_time × n_skus` values per warehouse. Optional aggregate `pipeline_aggregate` (not always on).
- **Rolling Demand Mean** (`rolling_demand_mean`): per-SKU average of realized home-region demand over the trailing `rolling_window` steps.


Optional features (each toggleable via the `feature` config):

- **Incoming demand** (`incoming_demand_home`): total demand originating from the warehouse's home region. Optional aggregate `incoming_demand_home_aggregate`.
- **Units shipped to home region** (`units_shipped_home`): units this warehouse shipped to its home region.
- **Units shipped to other regions** (`units_shipped_away`) — units this warehouse shipped to non-home regions (transshipment). Optional aggregate `units_shipped_away_aggregate`.
- **Demand forecast** (`demand_forecast`): exponential moving average of home-region demand with smoothing factor `ema_alpha`. Optional aggregate `demand_forecast_aggregate`.
- **Days of supply** (`days_of_supply`): computed as `inventory / max(rolling_demand_mean, 1)` per SKU.
- **Net inventory position** (`net_inventory_position`): computed as `inventory + pipeline_total − demand_forecast × expected_lead_time` per SKU.
- **Demand history** (`demand_history`): last `rolling_window` raw home-region demand vectors. Most-recent first and zero-padded.
- **Demand variability** (`demand_variability`): rolling std of home-region demand over the rolling window.

Features that can technically be toggled but are decided to be turned off for all experiments:
- **Stockout** (`stockout`): unmet demand in the home region.

When parameter sharing is enabled the environment prepends a one-hot warehouse identifier `warehouse_id_onehot` to each warehouse's local observation so the shared policy can distinguish agents.

The global observation is the concatenation of all warehouses' local observations. The flat vector handed to the RLModule is `[local | global]` per agent. The actor and the critic each select either the local or the global slice based on their `actor_obs_type` and `critic_obs_type` configuration. IPPO uses local for both. MAPPO uses local for the actor and global for the critic. CPPO uses the full joint observation.

### 4.10 Observation Normalization

Five normalization modes are supported, selected via `algorithm_specific.obs_normalization`.
- **Off** (`"off"`): Raw observation values, no normalization applied.
- **Ratio** (`"ratio"`): Per-SKU features within a feature block are divided by the block's per-SKU total (so the block sums to 1). Aggregate scalars are appended in raw form.
- **Mean-std** (`"meanstd"`): Raw features are passed to RLlib's built-in running mean/std filter (`MeanStdFilter` env-to-module connector).
- **Custom mean-std** (`"meanstd_custom"`): At the start of training, [src/utils/obs_stats.py](src/utils/obs_stats.py) rollouts a random-policy to estimate per-feature means and standard deviations. These fixed statistics are then used to normalize observations as `(obs − mean) / std` throughout training.
- **SKU-grouped mean-std** (`"meanstd_sku_grouped"`): Same data-collection procedure as `meanstd_custom`, but per-SKU dimensions inside each feature block share a single `(mean, std)`; aggregate dimensions get their own independent statistics. Pooled across warehouses.
- **Warehouse-grouped mean-std** (`"meanstd_wh_grouped"`): Statistics are computed independently per warehouse with the same SKU-grouping rule as `meanstd_sku_grouped` applied inside each warehouse's bucket. This is intervention A1 (see SCOPE.md §7).

## 5. Algorithms

### 5.1 Implemented Algorithms

Since all algorithms used are from the actor-critic family, they all share the single neural network module `ActorCriticRLModule` in [src/algorithms/models/rlmodules/base.py](src/algorithms/models/rlmodules/base.py). While the module builds an actor and a critic from the algorithm config's `networks` block for all algorithms, they differ in how the environment factory is configured and which observation slice the actor and the critic see. Three algorithms are implemented:

- **IPPO (Independent PPO)**: Each warehouse agent learns its own policy using only local information with no explicit coordination mechanism beyond the shared environment dynamics. Both actor and critic see local observations. Uses the `IPPOWrapper` in [src/algorithms/ippo.py](src/algorithms/ippo.py).
- **MAPPO (Multi-Agent PPO):** A centralized-training, decentralized-execution (CTDE) approach where agents act based on local observations at execution time, but during training each agent's value function (critic) sees the global observation. Uses the `MAPPOWrapper` in [src/algorithms/mappo.py](src/algorithms/mappo.py).
- **CPPO (Centralized PPO):** Single-agent baseline. A single PPO policy receives the full global observation and outputs a joint action vector covering all warehouses. The multi-agent env is wrapped as a single-agent `gymnasium.Env` via `CentralizedEnvWrapper` (`src/environment/envs/single_env.py`), which splits the joint action and aggregates rewards.

IPPO and MAPPO support **parameter sharing** via the `algorithm_specific.parameter_sharing` flag. When enabled, all warehouse agents share a single `shared_policy` RLModule and a one-hot warehouse identifier is prepended to each warehouse's local observation so the shared policy can distinguish agents.

### 5.2 Neural Network Architecture

The actor and critic each use their own backbone. While the actor backbone outputs one value per action dimension (the mean of a diagonal Gaussian distribution continuous action spaces), the critic backbone outputs a single scalar value estimate. 

Three concrete neural network architectures are registered: MLP, GRU, and per-SKU self-attention. All are implemented as a subclass of `NetworkArchitecture` in [src/algorithms/models/architectures/](src/algorithms/models/architectures/).

The per-SKU self-attention architecture (`type: "attention"`) implements intervention A4 (see [SCOPE.md](SCOPE.md) §7). It interprets the flat observation as one or more repeated warehouse blocks (one for a local actor, `1 + n_warehouses` for a MAPPO global critic, `n_warehouses` for the CPPO joint observation), gathers each block's per-SKU columns into one token per SKU (handling both contiguous and strided feature blocks such as the pipeline), projects the tokens to `d_model`, and applies a Transformer encoder whose attention runs across the SKU tokens *within each warehouse block* (per-warehouse attention with shared weights). The attended tokens are flattened, concatenated with the non-per-SKU context features (warehouse-id one-hot and aggregates), and passed through an MLP head (`head_hidden_sizes`). The per-SKU token layout is derived at runtime from the observation feature config via `InventoryEnvironment._compute_obs_layout()` and passed to the RLModule through `model_config["obs_layout"]`, so it adapts automatically to the enabled features and to parameter sharing. Attention is incompatible with `shared_layers`, `use_mu_sigma_head`, and GRU backbones, which are rejected at config-validation time.

The Gaussian policy's standard deviation can be parameterized in one of two ways. The default is to use a state-independent learnable parameter vector with one entry per action dimension for log-std. This parameter is governed by the algorithm config entries `logstd_init` and `logstd_floor`. Alternatively, a mu-sigma head can be used by setting `use_mu_sigma_head: true` in the algorithm config. This mu-sigma head routes the actor backbone through two separate linear heads, where one produces the mean and the other produces the log standard deviation.

An optional `output_activation` field in the algorithm config can apply an activation to the actor backbone output before Gaussian parameterization. When set to `null` the raw linear output is used and RLlib handles action clipping. Shared layers between actor and critic are architecturally supported through the `networks.shared_layers` block.

### 5.3 Training Options

Two algorithm-level training options are available to algorithm wrappers:

- **Hysteretic PPO**: This learner downweights negative advantages by `hysteretic_beta` while leaving non-negative advantages at their original weight of `1.0`. Enabled by setting `algorithm_specific.hysteretic_beta` to a value in `(0, 1]`, which causes the wrapper to swap the default RLlib PPO learner for `HystereticPPOTorchLearner` in [src/algorithms/learners/hysteretic_learner.py](src/algorithms/learners/hysteretic_learner.py). Available for IPPO and MAPPO.
- **Warm-start weights**: Transfers existing weight parameters with matching shapes into the new policy at run-time while mismatched parameters retain their random initialization. Export and load helpers live in [src/utils/weight_transfer.py](src/utils/weight_transfer.py). Enabled via `algorithm_specific.warmstart_weights_path`, which points to a previously exported RLModule `state_dict`. Available for IPPO, MAPPO, and CPPO.

## 6. Baselines

Non-RL baselines are run under the same multi-seed evaluation protocol as the RL algorithms via [src/experiments/run_baselines.py](src/experiments/run_baselines.py). Six baselines are implemented:

- **Random**: Uniform random orders in `[-1, 1]`.
- **Constant Order (calibrated)**: Per-(warehouse, SKU) constant quantities derived from observed demand in pilot episodes. Swept over a shared multiplier `alpha`.
- **BS-Newsvendor**: Analytical newsvendor base-stock using the *true* demand parameters. Swept over safety factor `z`.
- **BS-Adaptive**: Rolling-mean base-stock using *observed* demand history. Swept over safety factor `z` and window width `H`.
- **BS-Optimized**: Simulation-optimized base-stock via Bayesian optimization.
- **BS-Independent**: Independently optimized base-stock via iterated best response (per-warehouse Bayesian optimization).

## 7. Data Sources

The env config's `data_source.type` determines where SKU weights, distances, and shipment-cost matrices come from, and which demand samplers are usable:

- **`custom`**: All cost matrices, `sku_weights`, `distances`, and demand-sampler parameters are read directly from the env config. Compatible with the Poisson demand sampler.
- **`synthetic`**: Cost matrices, weights, and distances are generated procedurally via [src/data/data_generator.py](src/data/data_generator.py). Generation uses helpers in [src/utils/cost_generator.py](src/utils/cost_generator.py) and a fitted log-normal mixture in `weight_gmm.pkl` for SKU weights. Compatible with the Poisson demand sampler.
- **`real_world`**: Uses data from raw CSV files. Loads and validates CSVs (warehouses, regions, suppliers, SKUs, orders, per-SKU demands), extracts shipment costs, and produces a `PreprocessedData` object containing train and (optional) validation demand DataFrames. The train-validation split is configured via `data_source.data_split` and can either be `ratio` (split the timesteps by a `train_ratio`) or `explicit` (lists of train and val timesteps). Loading and validation live in [src/data/preprocessor.py](src/data/preprocessor.py). Required by the Empirical demand sampler.

The thesis uses `custom` exclusively. The `synthetic` and `real_world` paths are implemented but not exercised by any current config.

## 8 Available Modes

[src/experiments/run_experiment.py](src/experiments/run_experiment.py) is the CLI entry point. It dispatches between four modes selected via `--mode`:

- `single`: Run one training experiment with a given env + algorithm config.
- `tune`: Run a Ray Tune hyperparameter sweep (see Hyperparameter Tuning).
- `evaluate`: Load a checkpoint and run an evaluation rollout (with optional visualization).
- `seed-eval`: Run multi-seed evaluation across a configurable list of root seeds.

### 9 Multi-Seed Evaluation

[src/experiments/utils/seed_evaluation.py](src/experiments/utils/seed_evaluation.py) orchestrates evaluation across multiple seeds and supports both a sequential single-node path (called from `run_experiment.py`'s `seed-eval` mode) and a parallel SLURM path (one task per (config, seed) pair, then a single aggregate task). The aggregate phase produces per-config training-curve plots, baseline comparison bands, and a `seed_evaluation_summary.yaml` containing per-seed and pooled statistics. The same protocol is reused by [src/experiments/run_baselines.py](src/experiments/run_baselines.py). 

## 10. Hyperparameter Tuning

Ray Tune integration in [src/experiments/utils/ray_tune.py](src/experiments/utils/ray_tune.py) allows hyperparameter search across four sections of the config: `shared`, `algorithm_specific`, `environment`, and `features`. Search spaces are declared per-key in a tune config YAML in [config_files/experiments/](config_files/experiments/), where supported space types are `uniform`, `loguniform`, `choice`, `randint`, and `grid_search`. There are multiple schedulers and search algorithms available:

- **Schedulers**: `asha`, `median_stopping`, `hyperband`, `fifo`.
- **Search algorithms**: `random`, `optuna`, `bayesopt` , `hyperopt`.

Note that `bayesopt` is continuous-only. Non-continuous spaces are rejected at validation time.

The tune driver builds the base env + algorithm config, merges per-trial sampled values, and reports `eval/episode_return_mean` (maximize) as the optimization metric. The shell entry point is [scripts/run_tune_experiment.sh](scripts/run_tune_experiment.sh).

## 11. Seeding and Reproducibility

Reproducibility is enforced by deriving all randomness from a single root seed via NumPy's `SeedSequence`. The central `SeedManager` in [src/utils/seed_manager.py](src/utils/seed_manager.py) spawns named child seeds for data generation, training, evaluation, and stochastic environment components, and per-(worker, env) seeds are derived inside the env factory. Before training starts, Python's `random`, NumPy, and PyTorch are also seeded and `torch.use_deterministic_algorithms(True)` is set.

## 12. Inert / Unused Code Paths

Several code paths are implemented and schema-validated but are not exercised by any current thesis config or experiment. Agents must not treat any of these as part of the active thesis pipeline unless the user says so.

- **Legacy configs:** The pilot env configs `env_symmetric_1WH2SKU.yaml` and `env_symmetric_3WH2SKU.yaml` (single-warehouse and 3-warehouse 2-SKU) are superseded by the five 3WH5SKU configs (SCOPE.md §5.2); likely removed before submission.

- **Unused environment-component options:** The component registries register more options than the thesis uses; each config uses exactly one option per component (Poisson demand, greedy allocation, fixed lead times, shipment-proportional lost sales, cost reward). Unused but registered: the **stochastic lead-time sampler** (all configs use `fixed`; stochastic lead times are out of scope per SCOPE.md §4) and the **`closest` and `cost` lost-sales handlers** (all configs use `shipment`).

- **Unused data sources and samplers:** The thesis uses `data_source.type: custom` exclusively (SYSTEM_MODEL.md §7). The **`synthetic` and `real_world`** paths — including the **`empirical` demand sampler** that `real_world` requires and the preprocessing in `src/data/preprocessor.py` — are implemented but not exercised. The **in-run data generator** (`src/data/data_generator.py`) is also unused; the real-world-inspired transfer instance (E4.2) is generated offline via the notebooks in `src/data/notebooks/`.

- **No-op config fields (validated, no enforcing logic):** `max_wh_capacities` (per-warehouse capacity caps) is validated but enforced by no code, and all configs set it effectively unbounded; removed or left as a documented no-op before submission. `reward_calculator.params.cost_weights` is read but its weighted-sum branch is commented out in `CostRewardCalculator.calculate` (the unweighted sum is used), so it is a no-op today — intended to be activated for intervention A3 (SCOPE.md §7).

- **Unused algorithm / training options:** The `base_stock` action space is a pilot relict (all configs use `demand_centered`, SCOPE.md §5.1). The Hysteretic PPO learner (`src/algorithms/learners/hysteretic_learner.py`, enabled via `hysteretic_beta`) is documented in §5.3 but set by no config and used by no experiment. The actor/critic config knobs `use_mu_sigma_head` (all configs `false`) and `networks.shared_layers (all configs `null`) are supported but disabled everywhere.

- **Backup scripts:** `scripts/run_seed_evaluation_sequential.sh` and the single-node branches it drives in `src/experiments/run_experiment.py` and `src/experiments/utils/seed_evaluation.py` are a fallback for the parallel SLURM baseline path (`scripts/run_all_baselines.sh`); likely removed before submission.
