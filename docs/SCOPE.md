# Thesis Scope and Experiment Roadmap

## Working Title

*"Architectural Interventions for Multi-Agent Reinforcement Learning in Heterogeneous Supply Chain Inventory Management"*

---

## Research Question

When warehouses differ in demand volume, SKU portfolios, and cost structures, vanilla MARL algorithms with parameter sharing learn pathological policies (e.g., abandoning entire product categories). What architectural modifications to the observation space, network structure, and reward signal enable effective policy learning under heterogeneity?

## Core Thesis

Standard MARL algorithms (IPPO/MAPPO with parameter sharing) degrade significantly when supply chain environments exhibit realistic heterogeneity in demand, SKU weights, and cost/geographic asymmetry. Targeted architectural interventions can recover and surpass performance. The paper frames the contribution as studying **policy generalization across heterogeneous agents** — a shared policy must adapt to heterogeneous local conditions through observation alone.

## Target Venue

EJOR or similar operations research journal.

---

## Core Contributions

1. **Diagnostic:** Systematic characterization of how three types of heterogeneity (SKU weight, demand volume, cost/geographic asymmetry) cause specific failure modes in MAPPO with parameter sharing. Includes the 2×2 information sharing analysis (IPPO/MAPPO × PS/no-PS) showing how centralized training and parameter sharing interact with heterogeneity.
2. **Interventions:** A set of targeted architectural modifications — per-agent normalization, cost-aware observations, SKU attention, entity embeddings, reward component balancing — that address these failure modes. The **per-SKU attention architecture** is the main architectural contribution.
3. **Empirical evaluation:** Comprehensive experiments on 5-SKU instances with 4 heterogeneity configurations, including ablation studies, information sharing analysis, and scalability tests. Culminates in a practitioner-oriented **guidelines matrix** showing which intervention helps which heterogeneity type and by how much.

## Key Takeaway

> *"In multi-agent supply chain management with parameter sharing, heterogeneity in SKU characteristics creates specific, diagnosable failure modes — most notably, agents learn to abandon high-cost product categories entirely. Standard remedies (hyperparameter re-tuning, reward rebalancing) are insufficient. We show that a combination of per-agent observation normalization, cost-aware features, and SKU-level attention enables MAPPO to learn effective differentiated policies across heterogeneous products and warehouses, closing the gap to per-SKU-tuned heuristic baselines by X%."*

---

## Environment Setup

### Base Environment

- **Topology:** Single-echelon, 3 warehouses, 3 regions, 5 SKUs
- **Symmetric baseline for tuning:** `env_symmetric_3WH5SKU.yaml`
- **Demand:** Poisson (standard in stochastic inventory management literature)
- **Lead times:** Fixed at 3 timesteps
- **Action space:** `demand_centered` with `max_quantity_adjustment` per SKU
- **Reward scope:** `agent` (each warehouse receives its own cost as reward)
- **Episode length:** 100 timesteps

### Why 3 Warehouses

The paper's central contribution is about SKU heterogeneity and architectural interventions, not warehouse scaling. Three warehouses provide:

- The minimum for parameter sharing to be non-trivial (shared network must adapt to 3 different local contexts)
- Enough structure for demand allocation dynamics (home region vs. 2 foreign regions)
- Computational efficiency for large HP search spaces

Warehouse scaling is tested in Phase 4 (5WH or 7WH).

### Why 5 SKUs

- **Practical argument:** Five SKUs is the minimum to represent a meaningfully diverse product portfolio ranging from lightweight high-turnover items to heavy low-turnover items. Fewer than 5 makes the heterogeneity trivial (2 SKUs is just "heavy vs. light").
- **Technical argument:** Self-attention over per-SKU feature groups requires enough "tokens" for the attention weights to be meaningful. With 2 SKUs, attention degenerates to a learned weighted average — a regular MLP can learn this. With 5 SKUs, the attention must learn a richer pattern: which SKUs to attend to, which to ignore, how to weight relationships. This is the regime where attention adds genuine value over a flat MLP.

### Heterogeneous Environment Configurations (all 5-SKU)

| Config | Demand | Geography | SKU Weights | SKU Demand |
|--------|--------|-----------|-------------|------------|
| `env_sku_hetero` | Symmetric | Symmetric | Varied (e.g., `[0.3, 1.0, 3.0, 10.0, 30.0]`) | `lambda_qty` inversely related to weight |
| `env_demand_hetero` | `lambda_orders: [8, 4, 1]` | Symmetric | `[1, 1, 1, 1, 1]` | Uniform |
| `env_cost_asymm` | Symmetric | 2EU+1US distances/costs | `[1, 1, 1, 1, 1]` | Uniform |
| `env_combined_hetero` | Heterogeneous | 2EU+1US | Heterogeneous | Heterogeneous |

### Coupling Strength

Create two cost regimes for heterogeneous experiments:

1. **Weak coupling (current):** outbound_variable home=0.05, away=0.50 (10x ratio). Transshipment is a last resort. Realistic for geographically dispersed networks.
2. **Moderate coupling:** outbound_variable home=0.05, away=0.15 (3x ratio). Transshipment is viable but not free. Realistic for regionally clustered warehouses.

The comparison becomes a finding: "Coupling strength moderates the benefit of centralized training (MAPPO vs IPPO)."

### Parameter Justification

Use **Option A (parametric study framing)** in the main text:

> *"We study two cost regimes representing different network topologies: a dispersed network where cross-region fulfillment costs 10x more than local delivery (typical for continental-scale distribution), and a clustered network with a 3x cost differential (typical for regional distribution within a country). Cost magnitudes are calibrated such that holding, penalty, and shipping costs are of comparable magnitude under the optimal base-stock policy."*

Use **Option B** in the appendix: reference the real-world industry data to show that the stylized parameters are in the right ballpark (e.g., "the ratio of cross-country to local shipping cost ranges from 3x to 15x in our industry dataset; we study 3x and 10x to span this range").

### Per-SKU Action Scaling (for heterogeneous cases)

In heterogeneous SKU settings, a static `max_quantity_adjustment` may not fit all SKUs. Two options:

- **Option A:** Per-SKU `max_quantity_adjustment` proportional to expected demand (e.g., `ceil(E[D] * 0.75)`)
- **Option B:** Relative/percentage action space where `order = demand * (1 + a * max_pct_adjustment)`, naturally handling heterogeneous demand scales

This is a design choice, not a main intervention. Mention briefly in the paper.

---

## Algorithms

### Primary Algorithm: MAPPO-PS

MAPPO with parameter sharing is the primary target for interventions because it is the most constrained but most scalable setting:

- **Centralized Training, Decentralized Execution (CTDE):** Critic sees all agents' states during training; actor uses only local observations at inference
- **Parameter sharing:** A single policy network must produce different ordering behavior for agents facing different demand levels, cost structures, and SKU mixes — distinguished only by local observations and warehouse ID
- **Why primary:** Under heterogeneity, parameter sharing creates the hardest learning challenge (one network must generalize across heterogeneous agents). Solving this is the most practically relevant result, since separate policies per agent don't scale.

### Reference Algorithm: IPPO-PS

Independent PPO with parameter sharing serves as a baseline throughout. The IPPO vs. MAPPO comparison is itself an information sharing experiment (local vs. global critic).

### 2×2 Diagnostic Grid (Phase 1)

| | No Parameter Sharing | Parameter Sharing |
|---|---|---|
| **IPPO** (local critic) | Separate policies, local info only | Shared policy, local info only |
| **MAPPO** (global critic) | Separate policies, global critic | Shared policy, global critic |

This grid answers three questions:
1. Does centralized training (MAPPO) help under heterogeneity?
2. Does parameter sharing hurt under heterogeneity?
3. Does MAPPO + PS still work? (If not, interventions must fix this specific configuration.)

---

## Baselines

| Baseline | Description | Status |
|----------|-------------|--------|
| **Constant order** | Sweep a constant order quantity per SKU (grid search over per-SKU constants) | Needs per-SKU sweep |
| **Oracle base-stock** | Newsvendor-style base-stock using true demand parameters (z-factor per SKU) | Needs per-SKU sweep |
| **Adaptive/reactive base-stock** | Capped base-stock that estimates demand from recent observations (moving average of last K fulfilled orders). No knowledge of true demand parameters. | Needs implementation |
| **Factored policy (per-WH-SKU)** | Each (warehouse, SKU) pair has its own network with per-SKU-only observations and a single action output. Under parameter sharing, a single network shared across all pairs. Scalable but loses inter-SKU information. | Phase 3 comparison |

The per-SKU baselines are critical: current baselines use one parameter for all SKUs and are misleadingly bad on heterogeneous settings.

---

## Interventions

### A1: Per-Agent Normalization

**What it does:** Compute observation mean/std statistics separately per agent (or per agent-type) rather than pooling all warehouses together. Agents with different scales get appropriate normalization.

**Why it helps:** Under demand heterogeneity, a high-demand warehouse has much larger feature magnitudes than a low-demand one. With pooled statistics, the low-demand warehouse's features look like large negative outliers. Per-agent normalization makes all agents see semantically consistent inputs: "this agent is slightly above/below its own typical demand."

**Implementation:** Modify `obs_stats.py` to store observations per warehouse separately, compute `(mean_i, std_i)` per warehouse, and pass a dict to the environment. In `_build_local_obs`, use `obs_stats[warehouse_idx]` instead of global stats.

**Primary target:** Demand heterogeneity, cost asymmetry.

### A2: Entity Embeddings / Cost-Aware Observations

**What it does:** Append static feature vectors encoding warehouse identity and SKU characteristics (weight, cost parameters) to each observation. With 3 warehouses and 5 SKUs, use static features (not learned embeddings) — append `sku_weight`, `holding_cost`, `penalty_cost` to the observation vector.

**Why it helps:** Gives the shared policy explicit knowledge about the agent it controls ("this SKU weighs 15 kg while the other weighs 0.5 kg") instead of forcing it to infer identity from feature magnitudes alone.

**Implementation:** Add cost profile and SKU weight features to `_build_local_obs()`. Register new features in `feature_config.yaml`.

**Note on learned vs. static embeddings:** At 3 warehouses and 5 SKUs, learned embeddings (`nn.Embedding`) provide minimal benefit over static features (there are too few entities for the embedding space to learn latent structure). Learned embeddings become justified at 10+ entities. This is a clean, defensible design choice.

**Primary target:** SKU heterogeneity, cost asymmetry.

### A3: SKU-Level Attention

**What it does:** Replace the flat MLP actor/critic with a network that reshapes per-SKU features into tokens, applies multi-head self-attention, then aggregates. The architecture processes each SKU independently through a shared projection, then uses attention to capture inter-SKU relationships.

**Data flow:**

1. Split the flat observation into `warehouse_id`, `per_sku_matrix (n_skus, features_per_sku)`, and `aggregate_features`
2. Project each SKU's features to attention dimension: `input_proj: Linear(features_per_sku, d_model)`, shared across SKUs
3. Apply multi-head self-attention: `nn.MultiheadAttention(embed_dim=d_model, num_heads=2)` over the n_skus tokens
4. Flatten attention output, concatenate with aggregate and warehouse ID features
5. Feed through actor MLP head to produce per-SKU action means

**For MAPPO critic:** Apply the same SKU attention to each agent's local observation independently, flatten each to a vector, concatenate all agents' vectors, feed to critic MLP.

**Why it helps:** The model's parameter count does not grow with n_skus (shared projection + attention weights), providing scalability. Per-SKU output maps directly to per-SKU actions. Under heterogeneity, the attention weights learn to differentiate how to process heavy vs. light SKUs.

**Implementation:** Create a new `SKUAttentionArchitecture` class. Register in `registry.py`.

**Primary target:** SKU heterogeneity. This is the **main architectural contribution** of the paper.

### A4: Reward Component Balancing

**What it does:** Apply fixed weights to the cost components (holding, penalty, shipping) in the reward via `cost_weights` in `CostRewardCalculator`, so no single component dominates the gradient.

**Why it helps:**

| Heterogeneity type | Benefit | Mechanism |
|---|---|---|
| SKU heterogeneity | **High** | Heavy SKU holding cost dominates reward → balancing equalizes component contributions |
| Cost asymmetry | **Medium** | Different agents have different component ratios → balancing ensures all components get attention |
| Demand heterogeneity | **Low** | The issue is cross-agent magnitude, not cross-component imbalance → A1 helps here instead |

**Implementation:** Calibrate `cost_weights` in `CostRewardCalculator` (already partially implemented).

### A5: Combined

Stack the best interventions from A1–A4. Run the best combination identified from individual intervention results. Includes ablation to show which components contribute most.

### Entity-Encoder Critic (Phase 4 scalability)

**What it does:** Replace MAPPO's flat concatenation critic with a structured architecture: shared encoder MLP per agent's local observation → mean-pooling (or attention-pooling) across agent embeddings → value head.

**Why deferred:** At 3 warehouses, the flat critic input is ~162 dimensions — an MLP handles this fine. The entity-encoder's structural benefit (permutation awareness, compositional processing) becomes meaningful at 5+ warehouses where the flat input grows large. Introduce in Phase 4 as a scalability enabler.

---

## Additional Experimental Axes (Phase 3)

### Information Sharing via Network-Level Aggregate Observations

Three feature config variants:

1. **Local only:** All aggregates off. Each agent sees only its own warehouse data.
2. **Local + own aggregates (current):** `inventory_aggregate: true`. Agent sees its own total inventory across SKUs.
3. **Local + global aggregates (new):** Network-level aggregates — total inventory across ALL warehouses per SKU, total demand across ALL regions per SKU.

Run IPPO-PS and MAPPO-PS with each variant on heterogeneous envs. This answers:
- Does adding global information to IPPO make it competitive with MAPPO (which gets global info through the critic)?
- Does MAPPO also benefit from global features in the observation (redundant with the critic, or complementary)?

**Implementation:** Modify `_build_local_obs()` to accept and append network-level features computed once in `_get_observations()`.

### Weighted Reward Scope for IPPO

Test `r_i = alpha * (-cost_i) + (1-alpha) * (-sum(cost_j))` for IPPO as a form of information sharing through the reward signal. Test `alpha ∈ {0.0, 0.3, 0.5}` on 1-2 heterogeneous envs. Compare to MAPPO with agent scope.

Keep separate from network aggregate experiments to avoid confounding.

### Factored Policy (Per-WH-SKU) Comparison

Each (warehouse, SKU) pair has its own network with per-SKU-only observations and a single action output. Under parameter sharing, a single network shared across all pairs.

**Purpose:** Addresses the reviewer question "why not just decompose the problem per SKU?" Shows that the factored approach is insufficient under SKU heterogeneity because it loses inter-SKU information — directly motivating the attention architecture.

### Agent vs. Team Reward Scope

Test agent vs. team reward on `env_combined_hetero` only (the hardest setting). Hypothesis: agent reward works better under heterogeneity because credit assignment is clearer.

---

## Design Decisions Made

| Decision | Choice | Justification |
|----------|--------|---------------|
| SKU count | 5 | Attention needs sufficient tokens; 5 spans realistic portfolio diversity |
| Warehouse count | 3 (scale to 5+ in Phase 4) | Sufficient for core contribution; scaling is a separate experiment |
| Primary algorithm | MAPPO-PS | Most constrained + most scalable = highest practical relevance |
| Action space | `demand_centered` (not searched) | `a=0` orders expected demand; natural default; easier learning than `direct` |
| Reward scope | `agent` (default) | Cleaner credit assignment; weighted/team tested as secondary axis |
| Normalization | `meanstd_custom` (searched via tune) | Per-feature-dimension normalization; per-agent variant is an intervention |
| Feature config | `inventory_aggregate`, `rolling_demand_mean` fixed on; `stockout` fixed off; `demand_variability` + 8 others searched | Based on IPPO tune evidence (always on/off in top-10 trials) |
| Demand distribution | Poisson only | Standard in stochastic inventory literature; isolates heterogeneity from demand model complexity |
| SKU correlations | Not included | Independent Poisson per SKU; correlations mentioned as future work |
| GRU / framestacking | Not in main study | MLP with temporal features sufficient; optional brief GRU comparison post-Phase 2 |
| Data generator | Used only for parameter justification in appendix; not in experiment pipeline | Demand generation incomplete; domain randomization is scope creep |
| Learned embeddings | Static features at 3WH/5SKU; learned embeddings at 10+ entities | Too few entities for learned representations to outperform static features |
| Entity-encoder critic | Deferred to Phase 4 | Benefit marginal at 3 warehouses; valuable for scalability |
| HAPPO | Excluded | Related work mention only; doesn't fit the problem structure |

---

## Experiment Roadmap

### Phase 0: Symmetric Tune

**Goal:** Establish baseline IPPO-PS and MAPPO-PS performance on the symmetric 3WH5SKU environment.

| Step | What | Status |
|------|------|--------|
| 0.1 | IPPO-PS tune on `env_symmetric_3WH5SKU`: 1000-sample Optuna, FIFO scheduler | Done |
| 0.2 | IPPO critic size diagnostic (512 vs 1024 vs 512×512) | Done — all equivalent for 54-dim local critic |
| 0.3 | Seed evaluation of top-10 IPPO trials (3 seeds each) | Pending |
| 0.4 | MAPPO-PS tune on `env_symmetric_3WH5SKU`: 1000-sample Optuna, same search space + expanded critic sizes (add 1024 and "512_512") | Next step |
| 0.5 | Seed evaluation of top-10 MAPPO trials (3 seeds each) | Pending |

**Tune search space:**
- Shared: `learning_rate` (loguniform 5e-5 to 3e-3), `batch_size` ({4000, 8000, 16000}), `num_epochs` ({3, 5, 10, 15, 20}), `num_minibatches` ({1, 4, 10, 20})
- Algorithm-specific: `lam` ({0.9, 0.95, 0.97, 0.99}), `gamma` ({0.95, 0.99, 0.995}), `use_kl_loss`, `grad_clip` ({5, 10, 20, 40, 100000}), `entropy_coeff` (loguniform 0.001 to 0.1), `vf_loss_coeff` (uniform 0.1 to 1.2), `clip_param` ({0.1, 0.2, 0.3}), `vf_clip_param` (loguniform 10 to 10000), `logstd_init` (uniform -1.5 to 0), `logstd_floor` (uniform -4.0 to -1.5)
- Network sizes: `actor_hidden_size` and `critic_hidden_size` ({64, 128, 256, 512, 1024, "64_64", "128_128", "256_256", "128_256", "256_128", "512_512"})
- Features: 9 binary toggles (`incoming_demand_home`, `units_shipped_home`, `units_shipped_away`, `demand_forecast`, `days_of_supply`, `net_inventory_position`, `demand_history`, `demand_variability`, `pipeline_aggregate`)

**Note on MAPPO critic size:** The IPPO tune showed 512 as optimal for the local critic (54-dim input), with larger sizes providing no benefit. For MAPPO, the global critic has 162-dim input — larger sizes may genuinely help. The expanded search space (adding 1024, "512_512") allows MAPPO to find its optimal capacity. This is fair because MAPPO inherently has a larger critic input and may need more capacity.

### Phase 0.5: Infrastructure

**Goal:** Build the remaining infrastructure needed for the full study.

| Step | What |
|------|------|
| 0.5.1 | Implement **per-SKU baselines**: sweep a separate constant (or z-factor) for each SKU independently. Grid search over `(z_sku0, z_sku1, ..., z_sku4)` for the base-stock heuristic. |
| 0.5.2 | Implement **adaptive/reactive baseline**: capped base-stock policy that estimates demand from recent observations (moving average of last K fulfilled orders) and orders up to a base-stock level. No knowledge of true demand parameters. |
| 0.5.3 | Create the **4 heterogeneous environment configs** (all 5-SKU): SKU hetero, demand hetero, cost asymmetry (2EU+1US), combined. |

### Phase 1: Diagnostic — "What Breaks and Why?"

**Goal:** Systematically characterize failure modes under each heterogeneity type. This becomes Section 4 of the paper.

| Step | What | Why |
|------|------|-----|
| 1.1 | **Run MAPPO-PS (symmetric HPs) on all 4 heterogeneous envs.** 3-5 seeds. Collect: reward, per-agent reward, per-SKU inventory levels, per-SKU order quantities, stockout rates, cost breakdowns, entropy trajectories, VF explained variance. | Quantifies degradation. Identifies specific failure modes per heterogeneity type. |
| 1.2 | **Re-tune MAPPO-PS on each heterogeneous env.** Smaller search space (key HPs only). 3-5 seeds on best config. | Controls for "maybe it just needed different HPs." If re-tuned MAPPO still abandons heavy SKUs, the problem is architectural, not HP-related. |
| 1.3 | **Run the full 2×2 grid** (IPPO/MAPPO × PS/no-PS) on each heterogeneous env. Two strategies per cell: from-scratch (with per-env re-tuned HPs) + warm-start from Phase 0 checkpoint. | Information sharing analysis. Tests whether parameter sharing or centralized training is the issue. |
| 1.4 | **Run all baselines** (per-SKU tuned) on all envs. | Provides the heuristic benchmark. |
| 1.5 | **Analyze.** For each env: which SKUs are neglected? Which agents underperform? Where do cost breakdowns show imbalance? Build the diagnostic table. | Motivates Phase 2 interventions. |

**Pre-registered hypotheses:**

| Config | Expected degradation | Key predicted failure mode |
|--------|---------------------|--------------------------|
| `env_demand_hetero` | Moderate (10-20%) | Low-demand warehouse overstocks; MAPPO helps via global critic |
| `env_cost_asymm` | Moderate (15-25%) | Credit assignment errors; MAPPO helps; PS may hurt |
| `env_sku_hetero` | Large (20-40%) | Heavy SKU dominates reward; light SKU under-served or abandoned |
| `env_combined_hetero` | Large (30-50%) | All failure modes compound |

**Runs:** 4 envs × 4 algo configs × 2 strategies × 3-5 seeds + 4 small re-tunes.

### Phase 2: Interventions — "What Helps?"

**Goal:** Implement and test each intervention on all 4 heterogeneous envs with MAPPO-PS. 3-5 seeds per experiment. MAPPO-noPS included as reference for key experiments.

**Ordering:** Normalization first (low-hanging fruit) → embeddings → attention → curriculum. Each layer builds on the previous. If normalization alone closes the gap, attention is unnecessary.

| Phase | Intervention | Test on | Implementation |
|-------|-------------|---------|----------------|
| 2.1 | Per-agent normalization (A1) + cost-aware observations (A2) + reward balancing (A4) | `env_sku_hetero` + `env_combined_hetero` | Modify `obs_stats.py`, `_build_local_obs()`, calibrate `cost_weights` |
| 2.2 | Entity embeddings (static features for warehouse identity + SKU characteristics) | All 4 heterogeneous configs | Add to observation vector, register in `feature_config.yaml` |
| 2.3 | SKU attention (main architectural contribution) | All 4 heterogeneous configs | New `SKUAttentionArchitecture` class, register in `registry.py` |
| 2.4 | Curriculum strategies: from-scratch vs simple warm-start vs progressive (symmetric → mild hetero → full hetero) | `env_combined_hetero` with best architecture from 2.1-2.3 | Tune fine-tuning LR multiplier ∈ {1.0, 0.5, 0.1, 0.03} |
| 2.5 | Combined (A5): stack best combination from A1-A4 | All 4 heterogeneous configs | |

### Phase 3: Ablation and Analysis

**Goal:** Dissect contribution of each component. Produce the guidelines matrix.

| Step | What |
|------|------|
| 3.1 | **Fractional factorial ablation** on `env_combined_hetero`: Full combination vs drop-one for each component. 5 seeds, MAPPO-PS only. |
| 3.2 | **Per-SKU behavior analysis:** Does the agent now maintain inventory for all SKUs? Show inventory trajectories, ordering patterns, cost breakdowns. |
| 3.3 | **IPPO-MAPPO gap analysis** across complexity levels: Plot performance gap with/without interventions. Does the gap shrink with interventions? ("Representation design partially substitutes for centralized training.") |
| 3.4 | **Information sharing experiments:** Network-level aggregate features (3 variants: local-only, own-aggregates, global-aggregates) with IPPO-PS and MAPPO-PS on heterogeneous envs. |
| 3.5 | **Weighted reward scope for IPPO:** Test alpha ∈ {0.0, 0.3, 0.5} on 1-2 heterogeneous envs. |
| 3.6 | **Agent vs team reward scope** on `env_combined_hetero`. |
| 3.7 | **Factored policy (per-WH-SKU) comparison** as a scaling reference and to address "why not decompose per SKU?" |

**The guidelines matrix** (example structure):

| | Demand hetero | Cost asymm | SKU hetero | Combined |
|---|:---:|:---:|:---:|:---:|
| A1: Per-agent normalization | +X% | +X% | +X% | +X% |
| A2: Entity embeddings | +X% | +X% | +X% | +X% |
| A3: SKU attention | +X% | +X% | +X% | +X% |
| A4: Reward balancing | +X% | +X% | +X% | +X% |
| All combined | +X% | +X% | +X% | +X% |

### Phase 4: Scalability and Validation

**Goal:** Demonstrate that interventions scale and transfer.

| Step | What |
|------|------|
| 4.1 | **Larger instances:** Test on 5WH5SKU or 7WH5SKU. Introduce entity-encoder critic here (replacing flat concatenation in MAPPO's critic). |
| 4.2 | **Real-world-inspired instance:** Use `DataGenerator` for realistic distances, weights, and costs (not demand/lead times). Run best intervention combination without re-tuning — tests zero-shot transfer of the approach. |
| 4.3 | Optional: brief GRU comparison (MLP vs GRU) on the most challenging heterogeneous env with the best intervention config. |

### Phase 5: Paper Writing (overlaps with Phase 4)

| Section | Content |
|---------|---------|
| Introduction | Motivation (heterogeneous supply chains + MARL), research question, contributions. Optional: condensed pilot result as motivating example ("agents learn to abandon heavy SKUs entirely"). |
| Related Work | MARL in supply chains, heterogeneous MARL, attention architectures, entity embeddings |
| Problem Formulation | Supply chain model, MARL formulation, heterogeneity dimensions |
| Diagnostic Study (Phase 1 results) | Failure modes under heterogeneity, 2×2 information sharing analysis, comparison with baselines |
| Proposed Interventions (Phase 2 design) | Architecture descriptions, intuition, implementation |
| Experiments (Phase 2+3 results) | Main results, ablations, guidelines matrix, information sharing analysis |
| Scalability (Phase 4 results) | Larger instances, entity-encoder critic, real-world-inspired validation |
| Conclusion | Summary, key takeaway, limitations, future work |

**Limitations/future work to mention:** Correlated demand across SKUs, learned embeddings at larger entity counts, domain randomization for robustness, GRU/temporal architectures, non-Poisson demand processes.

---

## Normalization Strategy

The normalization design resolves a tension between preserving inter-SKU relationships and handling scale differences:

- **`meanstd_grouped`:** Preserves inter-SKU relationships (agent sees "SKU 0 inventory is much higher than SKU 1") but fails when one SKU's scale dominates the shared statistics.
- **`meanstd_custom`:** Eliminates scale dominance (each SKU normalized independently) but destroys inter-SKU relationships.

**Resolved approach:** Use `meanstd_grouped` with per-agent stats (A1) + explicit SKU context features:

1. Keep `meanstd_grouped` as normalization mode (preserves relative magnitudes between SKUs) but compute grouped statistics per-agent (A1), so each warehouse's statistics reflect its own demand regime.
2. Add SKU characteristic features (weight, cost parameters) as static features to each SKU's observation block.
3. Retain aggregate features (`inventory_aggregate`, `pipeline_aggregate`) for total scale context.
4. Optionally add log-compressed raw features (`log(1 + raw_inventory)`) alongside grouped-normalized values for absolute scale information.

---

## Coupling and Multi-Agent Justification

The paper frames the multi-agent aspect through two mechanisms:

1. **Demand spillover through the allocator:** When a warehouse stocks out, unfulfilled demand is re-allocated to others at higher shipping cost. Each agent's inventory decision directly affects other agents' workloads and costs.
2. **Parameter sharing under heterogeneity:** A single-agent approach would require separate policies per warehouse. Parameter sharing forces one network to handle all warehouse types — a fundamentally multi-agent generalization challenge.

Paper framing:

> *"While each warehouse primarily serves its local region, the decentralized nature of the problem and the use of parameter sharing create a multi-agent learning challenge: a shared policy must adapt to heterogeneous local conditions through observation alone. We study how heterogeneity in SKU portfolios and demand patterns affects the ability of MARL algorithms to learn such adaptive policies."*

This sidesteps the "weak coupling" critique by reframing the contribution as **policy generalization across heterogeneous agents**, not cooperation/coordination per se.

---

## Pilot Study (Completed)

The 3WH2SKU pilot validated the premise:

- **SKU heterogeneity** caused IPPO/MAPPO to learn pathological policies (abandoning heavy SKUs entirely — maintaining zero inventory and accepting chronic stockouts)
- **Demand heterogeneity** caused only modest degradation
- Baselines were found inadequate for heterogeneous SKUs (use one parameter for all SKUs)

The pilot was for internal go/no-go. It is NOT a formal paper section, but a condensed version (2-3 sentences + one figure) can serve as a motivating example in the introduction.

---

## Current Status

| Item | Status |
|------|--------|
| Environment implementation (PettingZoo, allocators, costs, observations) | Done |
| IPPO and MAPPO wrappers with RLlib new API | Done |
| Parameter sharing, observation normalization modes | Done |
| Symmetric 3WH2SKU environment config + tunes | Done |
| Pilot study (3WH2SKU, 3 heterogeneous configs) | Done |
| Symmetric 3WH5SKU environment config | Done |
| Feature config with fixed/searched feature split | Done |
| IPPO-PS tune on symmetric 3WH5SKU (1000 samples) | Done |
| IPPO critic size diagnostic (512 vs 1024 vs 512×512) | Done — all equivalent |
| Seed evaluation of top-10 IPPO trials | Pending |
| **MAPPO-PS tune on symmetric 3WH5SKU** | **Next step** |
| Per-SKU baselines | Not started |
| Adaptive/reactive baseline | Not started |
| Heterogeneous environment configs (5-SKU) | Not started |
| Phase 1 diagnostic experiments | Not started |
| Phase 2 intervention implementations | Not started |
| Phase 3 ablation and analysis | Not started |
| Phase 4 scalability and validation | Not started |

---

## Timeline Estimate

| Phase | Duration | Can overlap with |
|-------|----------|-----------------|
| Phase 0 (Symmetric Tune) | 2-3 weeks | — |
| Phase 0.5 (Infrastructure) | 1-2 weeks | Phase 0 (configs + baselines while tune runs) |
| Phase 1 (Diagnostic) | 2-3 weeks | — |
| Phase 2 (Interventions) | 3-4 weeks | — |
| Phase 3 (Ablation) | 1-2 weeks | Phase 2 (later runs) |
| Phase 4 (Scalability) | 1-2 weeks | Phase 3 |
| Phase 5 (Writing) | 2-3 weeks | Phase 4 |
| **Total** | **~12-16 weeks** | |

---

## Metrics to Record (All Experiments)

- Final eval reward (mean ± std over seeds)
- Per-agent reward breakdown
- Per-SKU inventory levels, order quantities, stockout rates
- Per-cost-component breakdown (holding, penalty, inbound shipping, outbound shipping)
- Entropy trajectory over training
- VF explained variance over training
- Convergence speed (iterations to reach X% of final performance)
- Attention weight heatmaps (for SKU attention experiments)
