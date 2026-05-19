# Thesis Scope

This document describes the thesis scope, the research questions, the contribution claims, the out-of-scope items, the environment design, the algorithms and baselines, the interventions, and the limitations acknowledged by the thesis. A high-level summary of the experimental program is included for orientation; the full per-cell program, the statistical protocol, and the appendix experiments are kept in a companion document.

For the full experimental program, statistical protocol, and supplementary appendix experiments, see [EXPERIMENTS.md](EXPERIMENTS.md). For what the system models and how its components behave, see [SYSTEM_MODEL.md](SYSTEM_MODEL.md). For runtime wiring, configuration, and implementation details, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

---

## 1. Working Title

There are multiple current options: 

1. *Architectural Interventions for Multi-Agent Reinforcement Learning in Heterogeneous Supply Chain Inventory Management*
2. *Beyond Symmetry: Diagnosing and Addressing MARL Fragility in Heterogeneous Distribution Networks*
3. *Matching Architectural Interventions to Heterogeneity Types in Multi-Warehouse Reinforcement Learning*

---

## 2. Research Questions

The thesis is organised around two research questions:

- **RQ1 (Diagnostic).** While cooperative multi-agent reinforcement learning with parameter sharing achieves strong results on symmetric multi-warehouse environments, real-world supply chain settings are inherently heterogeneous. Under such heterogeneity, does parameter-shared multi-agent PPO degrade relative to a symmetric baseline of matched network throughput and if so, what failure modes explain the gap? Can environment-specific hyperparameter tuning recover this lost performance?

- **RQ2 (Prescriptive).** Parameter sharing imposes a single inductive bias across all agents, which becomes a liability when agents operate in heterogeneous conditions. Which architectural modifications relax this constraint sufficiently to recover performance and are the same modifications effective across all heterogeneity types or does each heterogeneity type demand a distinct architectural response?

---

## 3. Contributions and Takeaways

### 3.1 Core Contributions

The thesis makes four contributions to the literature on multi-agent reinforcement learning for inventory control.

- **C1. An extensible benchmark that brings MARL inventory control closer to operational practice.** The thesis contributes a simulation environment that additionally supports controlled heterogeneity across multiple network dimensions and adopts an order-level multi-SKU demand model. The result is a richer evaluation setting in which heterogeneity-driven phenomena can be studied systematically.

- **C2. A diagnostic characterisation of MARL failure modes under heterogeneity.** The thesis quantifies the performance degradation of cooperative parameter-shared MARL across distinct heterogeneity types, identifies the specific failure modes responsible, and demonstrates that hyperparameter tuning alone is insufficient to close the gap. The result is a precise account of where and how standard MARL breaks once the inventory setting moves beyond its symmetric idealisation.

- **C3. A toolkit of architectural interventions and an analysis of their interactions.** The thesis introduces a set of architectural modifications that address the diagnosed failure modes, and reports both their individual efficacy and the effects of combining them. The result is a concrete set of remediation techniques for the research community together with empirical evidence on how those techniques compose.

- **C4. A prescriptive decision rule for practitioners.** The thesis derives a decision matrix that maps an observed heterogeneity profile of an inventory network to a recommended combination of architectural modifications. The result is practitioner-facing guidance that converts the thesis's findings into a directly applicable design rule.

### 3.2 Key Takeaways

Two abstract-style summary statements are kept open so the writer can pick the framing that best matches the final results.

> **(A)** *In cooperative parameter-shared multi-agent reinforcement learning for multi-warehouse inventory management, network heterogeneity produces distinct performance degradations that hyperparameter tuning alone does not close. The thesis characterises these degradations together with the architectural interventions that recover them and finds that the relationship between intervention and heterogeneity is non-uniform such that a modification that recovers one degradation can be neutral or even counterproductive for another.*

> **(B)** *Heterogeneity does not uniformly degrade multi-agent reinforcement learning for inventory control. Different forms of heterogeneity break different aspects of learning, and no single architectural modification recovers all of them. The thesis shows that the modification which closes one performance gap can be irrelevant or harmful for another, and derives a heterogeneity-to-intervention map that provides practitioners with a diagnostic-first design rule.*

---

## 4. Out-of-Scope

These items are explicitly excluded and will not be part of the thesis for the reasons given.

| Excluded | Why |
|---|---|
| Multi-echelon networks (manufacturer → DC → retailer) | The thesis studies a single-echelon distribution network. Multi-echelon adds a structural heterogeneity dimension orthogonal to the three studied here and would dilute the diagnostic framework. Mentioned in future work. |
| Correlated demand across SKUs | The Poisson model treats SKUs as independent within an order. Adding correlation would confound the SKU-heterogeneity axis with a separate generative-model axis. Mentioned in future work. |
| Non-Poisson demand distributions | Poisson is the standard generative model in stochastic inventory control. Changing the distribution would confound heterogeneity effects with demand-process effects. Mentioned in future work. |
| Stochastic lead times | Fixed lead times of 3 steps isolate the heterogeneity axes. Stochastic lead times are implemented in the codebase but not exercised in the main experiments. Mentioned in future work. |
| Real-world demand data | The thesis uses synthetic Poisson demand throughout. The `real_world` data path is implemented but no current config uses it because the available demand data is of insufficient quality. Section E4.2 instead uses a real-world-inspired synthetic instance for transfer evaluation. |
| Fixed shipment costs (inbound and outbound) | The codebase supports both fixed and variable shipment costs, but all main configs set fixed costs to zero. Including a fixed component would introduce an order-batching incentive that interacts with the lead-time decision and the SKU-mix decision. This would create a confound because the agent could improve performance by consolidating orders rather than by responding to the heterogeneity-driven failure modes the thesis studies. Mentioned in future work. |
| Learned entity embeddings (`nn.Embedding` over warehouse or SKU indices) | At three warehouses and five SKUs there are too few distinct entities for a learned embedding space to outperform hand-coded static features. Interventions A1 and A2 instead inject identity information through static observation features. Learned embeddings become defensible at roughly ten or more entities and are mentioned in [EXPERIMENTS.md](EXPERIMENTS.md) §3 as future work. |
| HAPPO, MAA2C, MASIA, and other non-PPO MARL algorithms | The PPO family (IPPO, MAPPO, CPPO) is the standard MARL backbone for cooperative continuous control. Adding non-PPO baselines would diffuse the contribution because the thesis is about interventions on a fixed algorithmic backbone rather than a horse race across algorithms. A single non-PPO method may appear as an appendix sanity check. Mentioned in future work. |
| Coupling regime as a research dimension | The cross-region outbound cost ratio is calibrated against real freight cost data at 10× and is treated as an environment property rather than a research axis. The multi-agent challenge in this thesis is generalization of a single shared policy across heterogeneous local contexts, which is non-trivial at any coupling strength. A single 3× regime is run as an appendix sensitivity check in [EXPERIMENTS.md](EXPERIMENTS.md) §3.4 to confirm that the findings hold under stronger coupling. |
| Curriculum learning as a primary intervention | Curriculum modifies the training trajectory rather than the policy representation. Promoting it to a primary intervention would dilute the architectural-contribution claim of C3. Curriculum is nevertheless run as a single sub-experiment in E3.7 against the best architectural combination so that the thesis can state whether curriculum adds value on top of the architecture. It is therefore not excluded from the study, only from the primary-contribution set. |
| Entity-encoder critic at the main scale | At three warehouses the flat critic input is on the order of 162 dimensions, where the entity-encoder critic provides no measurable benefit over a flat MLP. The entity-encoder critic is introduced only in the scalability experiment E4.1 where the flat input becomes large. |

---

## 5. Environment Design

### 5.1 Topology and Base Parameters

The default symmetric environment uses the parameters listed below. The five environment variants in §5.2 inherit these values and perturb only the parameters that define the heterogeneity axis under study.

- **Warehouses:** 3
- **Regions:** 3 (one home region per warehouse)
- **SKUs:** 5
- **Action space:** `demand_centered` with per-SKU `max_quantity_adjustment = max(2, ⌈1.5 · max_w E[D]^{SKU}_w⌉)`, where the maximum is taken over warehouses and `E[D]^{SKU}_w` is the expected per-step home-region demand for SKU at warehouse `w`
- **Initial inventory:** per-(warehouse, SKU), set to the analytical steady-state base stock `I_0^{SKU} = E[D]^{SKU} · L^{SKU} + z · √(E[D]^{SKU} · L^{SKU})` with `z = 3.09` (99.9th percentile of the standard normal)
- **Distances:** warehouse-to-region distances, set to 50 km from each warehouse to its home region and 500 km from each warehouse to each non-home region
- **Inbound variable cost:** 1.0 per unit weight, uniform across all (warehouse, SKU) pairs
- **Outbound variable cost:** 0.05 per unit weight from each warehouse to its home region and 0.5 per unit weight from each warehouse to each non-home region
- **λ_orders per region:** `[4, 4, 4]` (orders per region per step)
- **probability_skus per region:** `[0.667, 0.667, 0.667]` (per-SKU Bernoulli inclusion probability inside each order)
- **λ_quantity per (region, SKU):** 5 uniformly (Poisson rate for the per-SKU quantity inside an included order)
- **SKU weights:** `[1, 1, 1, 1, 1]`
- **Holding cost:** 1.0 per unit weight per timestep, uniform across all (warehouse, SKU) pairs
- **Penalty cost:** 5.0 per unit of unmet demand, uniform across SKUs
- **Episode length:** 100 timesteps
- **Lead times:** fixed at 3 timesteps for all (warehouse, SKU) pairs
- **Reward scope:** `agent`. Each warehouse receives the negative of its own per-step cost as reward.

#### Why 3 warehouses

Three is the smallest number that makes parameter sharing a non-trivial generalization problem because a shared policy must adapt to three distinct local contexts using only its observation. Three is also enough for the greedy demand allocator to produce non-trivial transshipment because each warehouse has one home region and two foreign regions to serve. Scaling to five and seven warehouses is tested in E4.1 as a scalability check and is not a research axis in the main study.

#### Why 5 SKUs

Five is the smallest portfolio that makes the per-SKU self-attention architecture a non-trivial mechanism. With two SKUs, self-attention degenerates to a learned weighted average that a flat MLP can match. With five, the attention has enough tokens to learn a non-trivial pattern over the SKUs about which to attend to, which to ignore, and how to weight their relationships. Five also spans a realistic light-to-heavy portfolio of items in the SKU-heterogeneity variant (§5.2).

#### Why the action space is demand-centered and not direct

Both action-space types are continuous and bounded in `[-1, 1]`, and they differ in what the boundary values represent. A `direct` action space scales `[-1, 1]` linearly to `[0, max_order_quantities]`, so the agent must discover the appropriate ordering magnitude starting from a uniform-random initialization. A `demand_centered` action space scales `[-1, 1]` to `[-max_quantity_adjustment, +max_quantity_adjustment]` and adds the observed home-region demand. Under this mapping, `action = 0` corresponds to ordering exactly the observed demand. The thesis uses `demand_centered` for two reasons. First, the order-up-to-mean policy is a strong analytical baseline under stationary stochastic demand, so the agent's initial policy is informative rather than random, and the learning task reduces to estimating the deviation from demand-tracking. Second, the centering point automatically adapts to the demand level the warehouse currently observes, which keeps the action signal scale-comparable across heterogeneous demand regimes.

#### Why this max_quantity_adjustment formula

The formula for determining the max_quantitiy_adjustment values is 

```
`max_quantity_adjustment = max(2, ⌈1.5 · max_w E[D]^{SKU}_w⌉)`
```

This rule is chosen based on multiple observations. First, note that the lower bound `max_quantity_adjustment ≥ E[D]^{SKU}` is the necessary condition for the agent to be able to order zero units under the `demand_centered` mapping. Specifically, the order quantity is `observed_demand + action · max_quantity_adjustment` with `action ∈ [-1, 1]`. Setting this expression to zero gives `action = -observed_demand / max_quantity_adjustment`, which lies in `[-1, 1]` if and only if `max_quantity_adjustment ≥ observed_demand`. The configuration time then substitutes the realised `observed_demand` with the expected value `E[D]^{SKU}`, which yields the necessary condition stated above. Second, the factor 1.5 extends this minimum upward by 50% so the agent retains meaningful headroom for stockout recovery without making most of the action space unused. Third, the maximum over warehouses applies because `max_quantity_adjustment` is shared across all warehouses, so the binding case is the warehouse with the highest home-region expected demand. In the heterogeneous-demand variants this is the high-volume warehouse, while in the homogeneous-demand variants every warehouse is equivalent. Finally, the floor of 2 protects the heaviest SKUs in the SKU-heterogeneous variants, where `E[D]^{SKU}` can fall well below 1 and the unconstrained 1.5-times rule would leave the agent with insufficient integer-level granularity to control those SKUs.

#### Why this initial inventory

The formula for determining the initial inventory is 

```
`I_0^{SKU} = E[D]^{SKU} · L^{SKU} + z · √(E[D]^{SKU} · L^{SKU})`
```

This is the standard newsvendor base-stock approximation for Poisson demand with a normal-CDF approximation of the safety-stock term. The first term tracks the expected lead-time demand. The second term adds safety stock proportional to the lead-time-demand standard deviation, which for Poisson is equal to the square root of the mean. The choice `z = 3.09` corresponds to a service-level target of approximately 99.9%, which is intentionally conservative. The reason is operational rather than statistical. A lower initial inventory would force the agent to spend the first lead-time worth of steps stocking up to its operating range. Such transient behavior is an artifact of the cold start rather than of the inventory policy and would dominate the reported per-episode return. A high initial inventory removes this artifact so the episode starts near the steady-state operating point, and the reported return reflects the policy's behavior in operation rather than its ability to recover from a cold start.

#### Why these distances, inbound variable costs, and outbound variable costs

The design principle for the symmetric environment is to use the simplest values that produce a clear home-versus-non-home cost differential while remaining consistent with a plausible real-world geography. Two distances suffice for this purpose. Each warehouse is placed 50 km from its home region and 500 km from each non-home region. The corresponding outbound variable costs are 0.05 per unit weight on home shipments and 0.5 per unit weight on non-home shipments, which gives a 10× home-versus-non-home ratio.

The chosen distances are symmetric simplifications of a layout that could plausibly occur in practice rather than free parameters tuned to fit the experiment. Concretely, they correspond to a stylised continental layout in which the three (warehouse, home-region) pairs sit at the vertices of an approximately equilateral triangle of side 500 km, and inside each vertex the warehouse and its co-located home region are 50 km apart. Continental distribution networks frequently arrange three or more regional hubs at distances on the order of several hundred kilometres, and the local warehouse-to-served-region distance is commonly on the order of tens of kilometres. The chosen 50 km and 500 km therefore capture realistic local-haul and regional-haul distances in a maximally symmetric fashion.

The cost values 0.05 and 0.5 are similarly grounded in commercial freight data rather than chosen for convenience. The real-world data available in the repository shows variable per-weight transport cost on the order of 0.05 per unit at the local scale of around 50 km and on the order of 0.5 per unit at the regional scale of around 500 km. The resulting 10× ratio is therefore an empirical property of the modelled geography rather than a free design parameter. The implications of this ratio for the multi-agent learning problem are discussed in §5.3.

When warehouse `w` serves a non-home region `r`, the simulator treats the shipment as a direct transit over the longer distance and charges the corresponding outbound variable cost. There is no intermediate relay through the destination region's home warehouse. This is a deliberate modelling choice that preserves the home-versus-non-home cost differential without introducing multi-hop logistics inside the simulator.

The inbound variable cost is held uniform at 1.0 per unit weight across all (warehouse, SKU) pairs because inbound variability is not a heterogeneity axis the thesis studies. A value of 1.0 places inbound replenishment at the dominant per-unit cost when an order is fulfilled, which keeps the replenishment decision economically meaningful. The value is also of the same order of magnitude as the per-unit holding cost and as a single penalty event, so no single component dominates the reward signal in the symmetric baseline.

#### Why these λ_orders, probability_skus, and λ_quantity per SKU values

The order-level demand model introduces a three-level decomposition of customer demand rather than the single aggregate per-(region, SKU) demand rate used in most prior MARL inventory work. At every step the model draws (a) a number of customer orders per region according to `λ_orders`, (b) a per-SKU Bernoulli inclusion drawn with probability `probability_skus` that determines which SKUs appear in each order, and (c) a per-line Poisson quantity drawn with rate `λ_quantity` for each included SKU. The three demand parameters in the base configuration correspond directly to the three levels of this decomposition. This structural separation is what makes the benchmark order-level and ties directly to the C1 contribution.

The specific numerical values `λ_orders = 4`, `probability_skus = 0.667`, and `λ_quantity = 5` are determined by two constraints. First, the joint product fixes the total expected weighted network throughput at the 200-unit target of §5.2: `3 · 4 · 0.667 · 5 · 1 · 5 ≈ 200.1`. Second, within that constraint the values are chosen so that the resulting per-(region, SKU) expected demand sits at approximately 13 units per step, computed as `4 · 0.667 · 5 ≈ 13.3`. This is the regime in which the Poisson approximation is statistically informative. At this rate the lead-time demand has mean 40 and standard deviation approximately 6.3, so demand fluctuates meaningfully from one lead-time cycle to the next. The system is therefore neither near-deterministic, which would make the inventory decision trivial, nor near-Gaussian, which would make individual integer-unit decisions invisible. The Poisson regime is also the regime in which the newsvendor approximation used for the initial inventory and for the base-stock baselines is most accurate. All derived quantities, including the initial inventory in the range 60 to 90 units, the `max_quantity_adjustment` value of 20, and the lead-time mean demand near 40 units, inherit their human-interpretable magnitudes from this 13-unit anchor.

#### Why these holding and penalty costs

The relationship between the holding cost `c_h` and the penalty cost `c_p` is what determines how aggressively the optimal policy should hold inventory. The intuition is simplest if read as a per-unit comparison. A holding cost of 1.0 means each unit of inventory carried forward to the next step costs one unit of reward. A penalty cost of 5.0 means each unit of unmet demand costs five units of reward. The penalty-to-holding ratio of 5:1 therefore says that missing one unit of demand is as costly as carrying five extra units of inventory for one step.

This ratio determines the optimal service level under the newsvendor approximation. A high ratio such as 50:1 makes the optimal policy push the service level close to 100% by carrying large safety stocks because every avoided stockout pays for fifty units of holding. A very low ratio close to 1:1 makes the optimal policy tolerate frequent stockouts because each avoided stockout pays for only one unit of holding. The interesting regime lies between these extremes, where neither effect dominates. The thesis uses 5:1, which under the newsvendor formula yields an optimal service level of `c_p / (c_p + c_h) = 5 / 6 ≈ 0.83`. Even the optimal stationary policy therefore leaves approximately 17% of demand unmet on average.

The choice of an 83% optimal service level rather than the more common 90% or 95% targets is deliberate and methodological. A 95% service level corresponds to a penalty-to-holding ratio of 19:1 and a 90% service level to 9:1. At those ratios the optimal policy leans heavily toward stocking up so that a simple "carry large safety stocks" heuristic already achieves near-optimal performance because penalty cost dominates the reward signal. The inventory problem effectively collapses to "always have inventory," and there is little for a learned policy to discover beyond a high replenishment rate. At 83%, holding cost is approximately 17% of the per-step cost on the optimal policy, so under-stocking and over-stocking are both visibly costly. The agent cannot solve the problem by over-stocking because holding cost remains a meaningful fraction of reward, and it cannot solve it by under-stocking because penalty cost remains meaningful. This places the inventory decision in a regime where a learned policy must actively trade the two errors off rather than gravitating to a corner solution, which is the precondition for measuring the value of the architectural interventions the thesis proposes.

Holding and penalty costs are uniform across (warehouse, SKU) in the default environment so the cost-rate dimension does not become an implicit fourth heterogeneity axis.

#### Why these SKU weights, episode length, lead times, and reward scope

The SKU weights are uniform in the default environment to be able to isolate the SKU-heterogeneity axis from other variations later. The lead time is fixed at 3 steps to isolate the heterogeneity axes from lead-time stochasticity, which is a separate axis the thesis does not study. The episode length of 100 steps is long enough for the lead-time-3 dynamics to play out repeatedly with approximately 33 lead-time cycles per episode, while remaining short enough to keep training cost manageable. The `agent` reward scope gives each warehouse credit only for its own cost, which produces clean per-agent credit assignment under heterogeneity. Clean credit assignment is essential for the diagnostic study because it makes per-agent failure modes attributable to per-agent policy decisions. A `team` scope and a mixed `agent`/`team` scope are tested as secondary axes in E3.5 and E3.6.

### 5.2 The Five 5-SKU Environments For The Experiments

Experiments are centered around heterogeneous environments that change one or more default parameters:

| Config | Max. Quantity Adjustment | Initial Inventory | Distances | Inbound Variable Costs | Outbound Variable Costs | SKU weights | Demand λ_orders | probability_skus | λ_quantity per SKU |
|---|---|---|---|---|---|---|---|---|---|
| `env_symmetric` | `[20, 20, 20, 20, 20]` | WH0: `[60, 60, 60, 60, 60]` <br>WH1: `[60, 60, 60, 60, 60]` <br>WH2: `[60, 60, 60, 60, 60]` | Symmetric (50 home / 500 away) | `[1, 1, 1, 1, 1]` for all warehouses | Symmetric (0.05 home / 0.5 away) | `[1, 1, 1, 1, 1]` |  `[4, 4, 4]` | `[0.667, 0.667, 0.667]` |  `[5, 5, 5, 5, 5]` for all regions |
| `env_sku_hetero` | `[60, 20, 7, 2, 2]` | WH0: `[169, 60, 20, 10, 5]` <br>WH1: `[169, 60, 20, 10, 5]` <br>WH2: `[169, 60, 20, 10, 5]` | Symmetric (50 home / 500 away) | `[1, 1, 1, 1, 1]` for all warehouses | Symmetric (0.05 home / 0.5 away) | `[0.3, 1, 3, 10, 30]` | `[4, 4, 4]` | `[0.667, 0.667, 0.667]` | `[16.67, 5, 1.67, 0.5, 0.167]` for all regions |
| `env_demand_hetero` | `[35, 35, 35, 35, 35]` | WH0: `[96, 96, 96, 96, 96]` <br>WH1: `[60, 60, 60, 60, 60]` <br>WH2: `[20, 20, 20, 20, 20]` | Symmetric (50 home / 500 away) | `[1, 1, 1, 1, 1]` for all warehouses | Symmetric (0.05 home / 0.5 away) | `[1, 1, 1, 1, 1]` | `[7, 4, 1]` | `[0.667, 0.667, 0.667]` | `[5, 5, 5, 5, 5]` for all regions |
| `env_cost_hetero` | `[20, 20, 20, 20, 20]` | WH0: `[60, 60, 60, 60, 60]` <br>WH1: `[60, 60, 60, 60, 60]` <br>WH2: `[60, 60, 60, 60, 60]` |  Heterogenous (50 home / 200 close / 5000 distant) | `[1, 1, 1, 1, 1]` for all warehouses | Heterogenous (0.05 home / 0.15 close / 1.5 distant) | `[1, 1, 1, 1, 1]` | `[4, 4, 4]` | `[0.667, 0.667, 0.667]` | `[5, 5, 5, 5, 5]` for all regions |
| `env_combined_hetero` | `[117, 35, 12, 4, 2]` | WH0: `[281, 96, 38, 15, 7]` <br>WH1: `[169, 60, 25, 10, 5]` <br>WH2: `[51, 20, 9, 4, 2]` |  Heterogenous (50 home / 200 close / 5000 distant) | `[1, 1, 1, 1, 1]` for all warehouses | Heterogenous (0.05 home / 0.15 close / 1.5 distant) | `[0.3, 1, 3, 10, 30]` | `[7, 4, 1]` | `[0.667, 0.667, 0.667]` | `[16.67, 5, 1.67, 0.5, 0.167]` for all regions |


#### Why are all environment variants comparable

The total expected weighted network throughput per step is held at approximately 200 weight-units across all five environments. This invariant is the central methodological device underlying C1 because it isolates the active heterogeneity axis from any concurrent change in problem size. The construction follows directly from the demand model. The expected per-step demand for SKU `s` at region `r` is

```
E[D]^{r,s} = λ_orders^{r} · p_{s}^{r} · λ_qty^{r,s}.
```

The total expected weighted network throughput per step is therefore

```
D = Σ_{r} Σ_{s} w_{s} · E[D]^{r,s} = Σ_{r} Σ_{s} w_{s} · λ_orders^{r} · p_{s}^{r} · λ_qty^{r,s}.
```

All five environments are constructed such that `D ≈ 200.1` per step. The mechanism by which each variant preserves this value while perturbing one heterogeneity axis is documented in the subsections that follow.

#### Why these heterogeneous λ_quantity values

The heterogeneous λ_quantity values in `env_sku_hetero` and `env_combined_hetero` are inversely proportional to the SKU weights. Specifically, the rule is

```
λ_qty^{r,s} = c / w_{s},
```

uniformly across regions. The constant `c` is chosen so that the total expected weighted network throughput remains at 200.1. Substituting the rule into the throughput formula above gives

```
D = Σ_{r} Σ_{s} w_{s} · λ_orders^{r} · p_{s}^{r} · (c / w_{s}) = c · Σ_{r} Σ_{s} λ_orders^{r} · p_{s}^{r},
```

so the weight terms cancel and the throughput becomes independent of the weights. With `λ_orders^{r}` summing to 12 across the three regions (which holds for both `env_sku_hetero` with `[4,4,4]` and `env_combined_hetero` with `[7,4,1]`) and per-SKU inclusion probabilities `p_{s}^{r} = 0.667` uniformly, the sum `Σ_{r} Σ_{s} λ_orders^{r} · p_{s}^{r} = 12 · (5 · 0.667) = 12 · 3.335 = 40.02`. Solving `c · 40.02 = 200.1` yields `c = 5.0`. The choice `c = 5` therefore satisfies the throughput invariant while realising the inverse-weight–demand relationship typical of distribution catalogues where heavier items move at lower per-order quantities.

The hypothesized failure mode under this variant concerns SKU-level credit assignment. Because holding cost is proportional to weight, the heaviest SKUs contribute the largest cost component per unit of inventory and dominate the per-step cost vector. A shared policy that minimises total reward can reduce it most quickly by under-stocking the heavy SKUs, which leads to large lost-sales penalties on those SKUs while keeping holding cost low. The intended test is whether MAPPO-PS with parameter sharing exhibits this under-stocking pattern and whether the per-SKU attention architecture corrects it.

#### Why these heterogeneous λ_orders values

The heterogeneous λ_orders values `[7, 4, 1]` in `env_demand_hetero` and `env_combined_hetero` are chosen so that their sum equals 12, matching the sum of the symmetric `[4, 4, 4]`. Since the remaining factors in the throughput formula are unchanged in `env_demand_hetero` (uniform weights and uniform λ_qty), this choice of values keeps the total expected weighted network throughput at 200.1.

Moreover, the triple `[7, 4, 1]` represents a setting where one region accounts for the majority of demand while the remaining regions handle progressively smaller shares (primary-versus-tail distribution pattern). This pattern is typical of continental distribution networks where population, purchasing power, or market maturity concentrates demand in one primary region and leaves a long tail of smaller regions. The 7× ratio between the highest- and lowest-volume region is large enough to expose any inability of a shared policy to differentiate warehouses by demand scale, and the sparsest region with `λ_orders = 1` is low enough that its home warehouse faces frequent zero-demand periods, which stresses the overstock-versus-stockout trade-off.

The hypothesized failure mode is that a parameter-shared policy must produce different ordering behavior for each warehouse based only on local observation, yet receives no direct signal that its identity differs from the others. The policy may therefore converge on an "average-warehouse" behavior that overstocks the low-volume warehouse and understocks the high-volume one. The intended test is whether per-agent observation normalization removes the scale confound and allows the shared policy to differentiate.

#### Why these heterogeneous weights

The `env_sku_hetero` and `env_combined_hetero` weights `[0.3, 1.0, 3.0, 10.0, 30.0]` are log-spaced over two orders of magnitude. Three properties motivate this choice. First, the range from 0.3 kg to 30 kg matches the practical span of distribution catalogues, with small consumer goods at the lower end and large appliances at the upper end. Second, log-spacing means there is no scale at which a single global normalization (such as the default `meanstd_grouped` policy) can recover the right magnitude for both the lightest and the heaviest SKUs simultaneously. This is a deliberate stressor for the normalization choice in the default observation pipeline and motivates intervention A1. Third, the 30× weight ratio between extremes ensures that holding cost (which is weight-proportional) becomes the dominant cost component for the heaviest SKUs, which is the precondition for the SKU-abandonment failure mode described above.

#### Why these heterogeneous distances

The same design principle applies as in the symmetric environment. The heterogeneous distance matrix uses the simplest values that produce a clear three-way cost structure while remaining consistent with a plausible real-world geography. Three distinct distances are used: 50 km for the local warehouse-to-home-region distance, 200 km for the intermediate same-continent cross-region distance, and 5000 km for the long-haul cross-continent distance. The corresponding outbound variable costs are 0.05, 0.15, and 1.5 per unit weight respectively, which gives 3× and 30× ratios over the home cost.

These distances correspond to a plausible multi-continental layout in which two of the three (warehouse, home-region) pairs sit close together on one continent and the third sits on a different continent. A 200 km separation between the two same-continent pairs is on the order of the distance between a pair of regional hubs inside a country or a region, and a 5000 km separation between the two continents is on the order of a trans-Atlantic distance. The 50 km local distance from §5.1 is preserved inside each pair. The chosen values are therefore symmetric simplifications of a real-world geography that could occur in practice. A possible instance is two European hubs about 200 km apart together with one North-American hub roughly 5000 km away, but the simulator does not depend on this specific labelling.

The cost values 0.15 and 1.5 follow the same freight-cost-with-distance calibration used for the symmetric environment. Continental variable per-weight transport cost on the order of 0.15 per unit at the regional scale of around 200 km, and intercontinental variable per-weight transport runs on the order of 1.5 per unit weight at the intercontinental scale of around 5000 km. As in §5.1, the simulator treats cross-region fulfillment as a direct shipment over the longer distance, with no intermediate relay through the destination region's home warehouse.

The hypothesized failure mode concerns asymmetric credit assignment under parameter sharing. The off-continent warehouse can only economically serve its home region because the cross-continent shipment cost is prohibitive, while the two same-continent warehouses can cross-fulfill at moderate cost. The shared policy must therefore learn three distinct cost geographies distinguished only by each warehouse's position in its local observation. The intended test is whether parameter sharing collapses these distinct geographies onto a single averaged policy, and whether cost-aware entity features (intervention A2) make the cost structure legible to the policy.

### 5.3 Coupling Regime

The main experiments are run at a 10× cross-region cost ratio. In the symmetric environment and in the SKU-heterogeneous variant, the outbound variable cost is 0.05 per unit weight to the home region and 0.5 per unit weight to each non-home region. In the cost-heterogeneous variants, the same calibration extends to the trans-Atlantic regime, where the cross-Atlantic cost of 1.5 per unit weight is 30× the home cost, consistent with the 5000 km distance.

The 10× ratio is not a free design parameter. The available real world data shows a variable per-weight cost on the order of 0.05 per unit at the cluster scale of around 50 km and on the order of 0.5 per unit at the cross-cluster scale of around 500 km, so the ratio is empirically calibrated rather than chosen for convenience. For this reason, the coupling regime is treated as an environment property rather than as a research dimension.

Nevertheless, a 10× cross-region cost makes transshipment a last resort, which weakens the operational coupling between agents and raises the question of whether the problem is truly multi-agent or instead three near-decomposable single-warehouse problems. The thesis addresses this objection in two ways. First, the multi-agent learning challenge studied here is not coordination across strongly coupled agents but generalization of a single shared policy across heterogeneous local contexts. A parameter-shared policy must produce different ordering behavior for warehouses with different home-region demand, different SKU portfolios, and different cost geographies, distinguished only by the local observation. This is a non-trivial generalization problem at any coupling strength. Second, operational coupling is measured explicitly rather than asserted. The metric κ defined in §5.4 reports the fraction of demand fulfilled cross-region under the BS-Optimized policy, and is reported per environment so the reader can see exactly how much cross-region fulfillment a near optimal policy actually uses. The 3× coupling regime is run as an appendix sensitivity check in [EXPERIMENTS.md](EXPERIMENTS.md) §3.4 on `env_combined_hetero` so the thesis can state whether the qualitative findings hold under stronger coupling.

### 5.4 Operational Coupling Metric κ

To convert "coupling" from a vague descriptor into a measured property, the thesis reports `κ` defined as the expected fraction of demand fulfilled cross-region under the BS-Optimized policy. The value is computed once in E1.5 from existing baseline rollouts and reported as a column in the environment-summary table. A higher `κ` indicates stronger operational coupling between agents.

The BS-Optimized policy is chosen as the reference for `κ` because it is the strongest stationary baseline available in this thesis. It is optimised end-to-end on the same evaluation distribution, knows the dynamics, and is not subject to a learning constraint. Computing `κ` under BS-Optimized therefore measures how much cross-region fulfillment the problem structurally demands rather than how much a specific learning algorithm happens to use. Using a learned policy such as MAPPO-PS would conflate the problem's structural coupling with the algorithm's specific learned behaviour, and MAPPO-PS in particular is itself among the policies the thesis evaluates, which would make the metric circular. BS-Optimized is also deterministic given its base-stock parameters, so `κ` is a stable and reproducible environment-level measurement.

### 5.5 Zero Minimum Quantity Per SKU

The Poisson demand sampler supports a `min_quantity_per_sku` floor that clamps each per-SKU quantity from below. Earlier experiments enforced a floor of 1 to guarantee that any Bernoulli-included SKU contributes at least one unit. However, under the five 5-SKU environments studied here, this floor is set to 0.

The reason is that a floor of 1 inflates the effective per-SKU mean by approximately `e^{-λ_qty}` relative to the unmodified Poisson mean. While this bias is negligible at `λ_qty = 5` in the symmetric environment (approximately 0.13%), it becomes a real problem for lower `λ_qty` values. Specifically, at `λ_qty ≤ 0.5`, which is the regime of the heaviest two SKUs in `env_sku_hetero` and `env_combined_hetero`, the bias exceeds 100%. Retaining the floor would therefore destroy the intended inverse-weight-demand relationship for the heaviest SKUs and break the throughput invariant of §5.2. Setting `min_quantity_per_sku = 0` uniformly across the five 5-SKU environments preserves both properties.

Removing the floor introduces a semantic case that does not arise under the legacy setting. A SKU may be Bernoulli-included in an order by the inclusion step but draw a quantity of zero from the per-SKU Poisson sampler. The SKU then contributes zero units to that order. If every Bernoulli-included SKU in an order draws zero, the order is generated with all-zero quantities and is handled gracefully by the greedy demand allocator, which schedules no shipments for it. Order counts therefore remain consistent across the floor-on and floor-off configurations and the inverse-weight-demand relationship is preserved for the heaviest SKUs.

---

## 6. Algorithms and Baselines

### 6.1 Algorithms

| Algorithm | Role | Why included |
|---|---|---|
| **MAPPO-PS** | Primary target for interventions | Centralized training, decentralized execution + parameter sharing → the most constrained but most scalable setting. Solving this is the most practically relevant result. |
| **IPPO-PS** | Reference baseline | Decentralized critic + parameter sharing. The IPPO ↔ MAPPO comparison isolates the value of the centralized critic. |
| **IPPO-noPS, MAPPO-noPS** | 2×2 grid completion | Separate policies per agent under each information regime. Tuned on `env_symmetric` (E0.2, E0.4) as the diagnostic-control baseline that empirically confirms PS ≈ noPS under symmetry, then transferred to heterogeneous envs in E1.1 and re-tuned in E1.4 as the localization probe for whether parameter sharing or the centralized critic is the binding constraint under heterogeneity. |

Excluded from main study: HAPPO, MASIA, MAA2C, off-policy MARL algorithms. Rationale in §4.

### 6.2 Reference Baselines

All baselines run under the same multi-seed evaluation protocol as the algorithms in §6.1. The base-stock baselines compute their order quantities as

```
order[w,k] = clip(S[w,k] − inventory[w,k] − pipeline[w,k], 0, max_qty).
```

Seven reference baselines are available. Six are non-RL and one is a centralized single-agent RL ceiling.

| Baseline | Description | Tuning | Role |
|---|---|---|---|
| **Random** | Uniform random orders | None | Lower-bound sanity check |
| **ConstantOrder** | Per-(WH, SKU) constant order quantity calibrated from observed demand, scaled by α | α | Naive non-stationary heuristic |
| **BS-Newsvendor** | Oracle base-stock from true demand parameters | safety factor z | Theory-derived reference |
| **BS-Adaptive** | Rolling-mean base-stock from observed history | z, window H | Adaptive heuristic without analytical demand |
| **BS-Optimized** | Joint GP-BO over all (n_WH × n_SKU) base-stocks | n_calls = 300 | Strong simulation-optimized reference |
| **BS-Independent** | Per-warehouse iterated best-response BO | n_rounds = 3, n_calls = 100 | Coordination-free strong reference |
| **CPPO** | A single PPO over the concatenated joint observation and joint action | Optuna tune (E0.5) | Centralized RL upper bound. Removes every MARL challenge (non-stationarity, partial observability, credit assignment). The gap from MAPPO-PS to CPPO quantifies the cost of decentralisation under heterogeneity. |

Two baselines anchor the reference scale from different sides. BS-Optimized is the practical heuristic ceiling. It knows the dynamics, has no learning constraint, and is optimised end-to-end on the same evaluation distribution. CPPO is the practical RL ceiling. It removes every multi-agent challenge by collapsing the network into a single joint observation and a single joint action and so represents the highest performance any RL agent could reach if granted full information and a single joint controller.

#### Normalized score for cross-environment comparison

The thesis reports each policy's return as the fraction of the BS-Optimized-to-Random performance range that the policy recovers. Concretely, for an environment in which Random achieves return `R_rand` and BS-Optimized achieves return `R_bs`, a policy with return `R` has a normalized score of `(R − R_rand) / (R_bs − R_rand)`. A score of zero means the policy is no better than uniform random ordering. A score of one means the policy matches the simulation-optimised heuristic ceiling. A score above one means the policy exceeds that ceiling, which is possible because BS-Optimized is restricted to a stationary base-stock policy class while learned policies are not.

The reason for reporting this normalized score rather than raw returns is that absolute returns are not comparable across the five environments studied here. The environments differ substantially in their absolute cost magnitudes because some incur larger shipment costs, larger holding costs, or larger throughput than others. A 100-unit improvement in raw return on the symmetric environment is not the same achievement as a 100-unit improvement on a heterogeneous environment with much higher absolute costs. The normalized score puts every environment on a common zero-to-one scale, which makes degradation and recovery directly comparable across the experimental suite.

#### Granularity of the base-stock baselines

The base-stock baselines use a per-(warehouse, SKU) parameterization rather than a single shared base-stock or a per-SKU-only base-stock. The reason is that under heterogeneity the optimal base-stock level depends jointly on the warehouse and the SKU. A warehouse facing high home-region demand requires more inventory than one facing low demand, and a heavy SKU with low order quantity requires a different inventory level than a light SKU with high order quantity. A coarser baseline that forces all warehouses or all SKUs to share a single base-stock cannot represent this variation and therefore understates the true heuristic ceiling. Using such a coarser baseline as the reference would artifactually inflate the apparent gain of any learned policy.

---

## 7. Interventions

The thesis evaluates five modifications to the cooperative parameter-shared MARL pipeline. A4 is positioned as the architectural intervention proposed by this thesis and is the centerpiece of contribution C3. A1, A2, and A3 are standard-practice modifications that the thesis tests as controls rather than as novelty claims. Each one targets a specific candidate explanation for the heterogeneity-induced performance gap, namely a per-agent scale mismatch (A1), a missing observation feature (A2), or a reward-component dominance pattern (A3). If any of these controls alone closes the gap on its targeted heterogeneity type, the result narrows the necessary intervention space and weakens the architectural claim of A4. If none of them does, the result strengthens the case for an architectural fix and motivates A4. A5 stacks the most effective subset of A1 through A4 into a single condition, and its decomposition by drop-one ablation is reported in E3.1. A separate matched-capacity MLP control isolates A4's mechanism from its parameter count.

### A1 — Per-Agent Observation Normalization (Control)

**What.** A1 computes the running mean and standard deviation of the observation per warehouse rather than pooling all warehouses' observations into a single statistic. Each agent's observation is then normalized by its own statistics before being fed to the policy network. Under parameter sharing the policy itself remains shared across agents and only the normalization pre-step is agent-specific.

**Why it helps.** Pooled-statistics normalization collapses warehouses with different demand scales onto a single mean and standard deviation, which makes a low-demand warehouse's features look like persistent negative outliers and a high-demand warehouse's features look like persistent positive outliers. Per-agent statistics remove this scale confound by giving each warehouse its own reference distribution. Every input value is then presented to the shared policy in a consistent semantic frame, namely whether it is above or below this agent's own typical level, regardless of absolute scale.

**Primary target.** Demand heterogeneity and cost asymmetry.

**Expected outcome.** A1 is expected to help on `env_demand_hetero` and `env_cost_hetero` because both regimes produce per-warehouse scale differences that pooled statistics cannot absorb. A1 is expected to be approximately neutral on `env_sku_hetero` because the warehouses are symmetric in that regime and per-agent statistics reduce to pooled statistics. A1 is expected to interact additively with A2 and with A4 because all three target orthogonal axes of the heterogeneity problem.

### A2 — Cost-Aware Entity Features (Control)

**What.** A2 appends a static per-SKU feature vector to each warehouse's observation. The vector contains the SKU weight, the holding cost, and the penalty cost. These features carry no temporal information because they are properties of the environment that do not change across timesteps, but they let the shared policy condition on SKU identity through explicit numeric attributes rather than through latent inference from the temporal features.

**Why it helps.** Without A2 the policy can only distinguish SKUs by the magnitudes of their inventory levels, pipeline contents, and demand history. Under SKU heterogeneity these magnitudes correlate with SKU identity, but the correlation is indirect and the policy must infer the identity from it. A2 provides the identity directly, which lets the policy map from "this is the heavy SKU" to "order rarely" without first inferring weight from observed demand magnitude. A2 is the cheapest possible intervention that injects SKU-aware information into the network.

**Primary target.** SKU heterogeneity and cost asymmetry.

**Expected outcome.** A2 is expected to help on `env_sku_hetero` and `env_cost_hetero` because both create per-SKU or per-warehouse cost structure that the policy must learn. The size of the gain on `env_sku_hetero` is the key control variable for A4. If A2 alone recovers the SKU-heterogeneity gap, then explicit feature engineering suffices and A4's attention mechanism is not necessary. A2 is expected to be approximately neutral on `env_demand_hetero` because SKU identity is not the binding constraint in that regime.

### A3 — Reward Component Balancing (Control)

**What.** A3 applies fixed scalar weights to the four cost components (holding, penalty, outbound shipment, inbound shipment) before summing them into the reward. The weights are calibrated so the per-component contribution to the reward magnitude is approximately equal at a baseline operating point. Under the default unweighted reward, the holding component dominates because it scales with weight, and A3 dampens the holding contribution and raises the others to equalise their per-step pressure on the policy gradient.

**Why it helps.** Without weighting, a policy that minimises the unweighted sum of costs has a strong incentive to reduce the largest component first because that produces the steepest gradient signal. Under SKU heterogeneity the heaviest SKUs dominate the holding-cost component by an order of magnitude, and a shared policy can most quickly reduce reward by under-stocking those SKUs at the price of large lost-sales penalties on them. A3 equalises the per-component pressure so the policy must balance all four cost components rather than chase the dominant one.

**Primary target.** SKU heterogeneity and cost asymmetry.

**Expected outcome.** A3 is expected to help on `env_sku_hetero` because the weight-scaled holding cost is the dominant gradient signal in that regime. A3 is expected to be approximately neutral on `env_demand_hetero` because the failure mode there is cross-warehouse magnitude mismatch rather than cross-component imbalance, which A1 is designed to address instead. A3 may interact substitutively with A4 because both target SKU heterogeneity through different surfaces, A4 through the policy network and A3 through the reward signal. If A3 alone closes the SKU-heterogeneity gap, the failure mode reduces to reward domination and A4's architectural contribution weakens.

### A4 — Per-SKU Self-Attention (Architectural Intervention)

**What.** A4 replaces the flat-MLP actor and critic backbones with an architecture that explicitly represents per-SKU information as a sequence of tokens. The per-SKU portion of each agent's observation is reshaped into a matrix with one row per SKU, and each row is projected through a shared linear layer to a common embedding dimension `d_model`. Multi-head self-attention is then applied across the SKU tokens, so each SKU's representation is updated by a learned weighted combination of all SKUs' representations. The attended tokens are flattened, concatenated with the warehouse identity and the aggregate features, and processed through a small MLP head that produces per-SKU action means in the actor or a scalar value in the critic.

**Why it helps.** A flat MLP receives the observation as a single vector and must discover from gradient signal alone that certain input dimensions form one block per SKU. A4 builds this structure into the network through the shared projection and the per-SKU tokens, which forces the policy to treat each SKU symmetrically before any cross-SKU interaction and allows it to learn explicit inter-SKU relationships through attention. Under SKU heterogeneity, where the correct behaviour for each SKU is qualitatively different, this inductive bias gives the policy the right vocabulary to differentiate between SKUs without inferring SKU identity indirectly from feature magnitudes. The parameter count of the attention module does not grow with the number of SKUs, which makes the intervention scale-friendly.

**Primary target.** SKU heterogeneity and combined heterogeneity.

**Expected outcome.** A4 is expected to produce its largest gain on `env_sku_hetero` and `env_combined_hetero`, where the SKU structure is the dominant axis of variation. A modest gain is expected on `env_cost_hetero` because the per-SKU tokenisation also exposes per-SKU shipment cost structure more cleanly. A4 is expected to be approximately neutral on `env_demand_hetero` because the dominant variation in that regime is across warehouses rather than across SKUs. A4 is expected to interact substitutively with A2 because both inject SKU-aware information into the policy, with A4 dominating when both are active, and to interact additively with A1 because the two address orthogonal heterogeneity axes.

### A5 — Combined

**What.** A5 stacks the interventions identified as most effective from A1 through A4 individually into a single condition. The default expectation is the full combination of A1, A2, A3, and A4, and the exact composition is finalised after the individual-intervention results in E2.1 through E2.4. A5 is run as one condition per heterogeneous environment alongside the individual interventions.

**Why it helps.** The four individual interventions target distinct axes of heterogeneity through distinct mechanisms. If their gains are additive or synergistic, A5 produces the strongest recovery available within the intervention space. If their gains are substitutive, A5 reveals which interventions dominate and which become redundant in combination. The drop-one ablation in E3.1 decomposes A5 into the marginal contribution of each component, which is the empirical basis for the C4 decision matrix.

**Primary target.** Combined heterogeneity (`env_combined_hetero`) is the primary stress test, with the other three heterogeneous environments included for completeness.

**Expected outcome.** A5 is expected to produce the strongest recovery on every heterogeneous environment, with the largest gain on `env_combined_hetero` where multiple heterogeneity axes are simultaneously active. The interactions across A1 through A4 are the central empirical question of E3, and the drop-one ablation in E3.1 is the primary tool to answer them.

### Matched-Capacity MLP Control for A4 (E2.4b)

**What.** The matched-capacity MLP control replaces the per-SKU attention architecture of A4 with a flat MLP whose total trainable parameter count matches the attention architecture exactly. The flat MLP has no inductive bias toward per-SKU token structure and treats the observation as an unstructured vector, but it has the same number of parameters available as A4.

**Why it helps as a control.** A4 introduces both new parameters and a new architectural mechanism. The matched-capacity MLP isolates the contribution of the mechanism by holding the parameter count constant. Any performance gap between A4 and the matched MLP is therefore attributable to the architectural mechanism rather than to the larger parameter budget. This control is non-negotiable because a reviewer will otherwise ask whether the gains of A4 are simply due to capacity, and an empirical answer is much stronger than a theoretical defence.

**Primary target.** `env_sku_hetero` and `env_combined_hetero`, the two environments on which A4 is expected to produce its largest gains.

**Expected outcome.** The matched-capacity MLP is expected to underperform A4 on `env_sku_hetero` and `env_combined_hetero` by a meaningful margin. If it matches or exceeds A4, the architectural-mechanism claim of A4 collapses and the contribution C3 must be reframed around capacity rather than mechanism. The control is run only on the two environments where A4 is the proposed intervention.

---

## 8. Experimental Program Summary

The full experimental program (per-cell procedures, deliverables, expected outcomes, falsifiers, the statistical protocol that governs all comparisons, and the supplementary appendix experiments) is in [EXPERIMENTS.md](EXPERIMENTS.md). The table below gives a single high-level matrix of every block for navigation and budget-planning.

Counts in the rightmost column are seed-evaluation training runs only and exclude (a) Optuna trial slots inside each tune and (b) non-RL baseline runs.

| Block | Purpose | Envs | Main outcomes | Seeds | Approx. new runs |
|---|---|---|---|---|---|
| **E0** | Symmetric baseline | `env_symmetric` | IPPO-PS, IPPO-noPS, MAPPO-PS, MAPPO-noPS, CPPO tuned on symmetric env, 6 reference baselines on symmetric env | 10 | ~500 |
| **E1** | Diagnostic | 4 hetero | 5 symmetric-tuned algos from E0 transferred to each of the 4 hetero envs, IPPO-PS, IPPO-noPS, MAPPO-PS, MAPPO-noPS, CPPO re-tuned on each of the 4 hetero envs; 6 reference baselines on each of the 4 hetero envs; coupling metric κ computed | 10 | ~2200 |
| **E2** | Interventions | 4 hetero | MAPPO-PS + A1/A2/A3/A4/A5 and matched-MLP control re-tuned on each of the 4 hetero envs | 10 | ~1200 |
| **E3** | Ablation & robustness | 4 hetero | A5 drop-one ablation for each of the 4 hetero envs, per-SKU rollouts of MAPPO-PS variants for all 5 envs, obs- and reward-side sharing variants for each of the 4 hetero envs, warm-start + curriculum training for MAPPO-PS and MAPPO-PS+A5 on combined_hetero env | 10 | ~900 |
| **E4** | Scalability & transfer | 5WH / 7WH, real-world-inspired | scalability test to 5WH and 7WH, zero-shot transfer of MAPPO-PS+A5 on real-world-inspired instance  | 10 | ~80 |
| **Total** | | | | | **~4900 new training runs** |

---

## 9. Limitations (Acknowledged in the Thesis)

The thesis acknowledges these limitations upfront so they do not become reviewer attack surfaces.

- **Single-echelon.** The model omits multi-tier supply chains (manufacturer → DC → retailer). Heterogeneity within an echelon is the contribution; cross-echelon heterogeneity is future work.
- **Independent demand.** SKUs are independent in the Poisson model. Empirical demand often exhibits cross-SKU correlation (substitution, complementarity, basket structure). Future work.
- **Fixed lead times.** Lead-time stochasticity is implemented in the codebase but not exercised in the main study. Future work.
- **Synthetic demand.** Main experiments use Poisson; E4.2 uses a real-world-inspired but still synthetic instance. No real demand traces are used.
- **Three warehouses, five SKUs.** Scalability is tested in E4.1 (5–7 warehouses) but the heterogeneity-axis findings are not re-validated at scale axis-by-axis.
- **Single coupling regime in main.** 10× is empirically grounded but a single choice. [EXPERIMENTS.md](EXPERIMENTS.md) §3.4 verifies one alternative (3×).
- **No algorithm-family comparison.** HAPPO, MASIA, off-policy MARL are not compared. The contribution is the intervention toolkit on a canonical PPO backbone, not a horse race.
- **Parameter-sharing only at the main scale.** No-PS variants appear only in the Phase 1 diagnostic grid (E1.4); the main intervention study is PS-only.

These limitations are referenced from the conclusion as concrete future-work avenues.
