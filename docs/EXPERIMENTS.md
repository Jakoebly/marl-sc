# Experiment Log: Making IPPO Learn in a Simplified Environment

## Overview and Approach

The experimental strategy follows a **simplify-first** principle: start with a highly simplified, symmetric environment where the optimal policy is known analytically (a roughly constant order quantity), make the IPPO algorithm learn this policy, and only then add complexity (heterogeneous demand, geographic imbalance, multi-agent coupling, MAPPO).

The project has progressed through **Phases 1.1–1.8**. A critical evaluation bug (weight synchronization) was discovered in Phase 1.4, meaning all custom evaluation rewards before that point were measured with randomly initialized weights. After fixing this, all prior phases were re-evaluated, revealing that IPPO was learning effectively much earlier than originally believed. Phases 1.6–1.7 further improved IPPO through observation space enrichment and hyperparameter tuning. Phase 1.8 introduced MAPPO (centralized critic), uncovering a critical observation routing bug and identifying entropy collapse as the primary remaining challenge for continuous-action multi-agent PPO.

---

## Simplified Environment

The simplified environment (`env_simplified_symmetric.yaml`) is designed to minimize confounding factors:

- **3 warehouses, 2 SKUs, 3 regions** (symmetric: identical demand, costs, distances across warehouses).
- **Symmetric distances:** Each warehouse is close to its home region (distance 50) and far from others (distance 500), making transshipment rare.
- **Uniform demand:** All warehouses/regions/SKUs have identical Poisson demand parameters (`lambda_orders=4`, `probability_skus=0.667`, `lambda_quantity=5` per SKU), yielding expected demand of ~8 units per SKU per warehouse per timestep.
- **Fixed lead times:** 3 timesteps for all (warehouse, SKU) pairs.
- **Cost structure:** `holding_cost=1.0`, `penalty_cost=5` per SKU, `inbound_variable=1.0`, symmetric outbound costs (`0.05` for home regions, `0.5` for others).
- **Agent-scope rewards:** Each warehouse receives its own cost as reward (no team reward to complicate credit assignment).
- **Episode length:** 100 timesteps.

A single-agent variant with 1 warehouse, 1 SKU, 1 region also exists for isolation testing.

### Baselines (Phase 0)

Three baseline policies were evaluated to establish performance bounds:

| Baseline | Best Config | Mean Episode Reward (unscaled) | Approx. Reward (scale=0.01) |
|---|---|---|---|
| **Random** | Uniform `[-1, 1]` | −343,291 | ~−3,433 |
| **Constant order** | qty=12 per SKU per timestep | **−15,628** | **~−156** |
| **Heuristic (newsvendor)** | base-stock, safety factor z=2.0 | −15,641 | ~−156 |

The constant order policy of 12 units per SKU effectively matches the newsvendor optimal. The goal for IPPO is to approach this reward level. For the 1WH/1SKU single-agent variant, the baseline is approximately −38.1 (scaled).

> **Note on baselines:** Exact baseline values vary slightly by seed. Phase 1.5 uses a 3WH/2SKU baseline of −154.5 (scaled, base-stock z=2). Phase 1.4 uses a 1WH/1SKU baseline of −38.1 (scaled).

---

## Critical Discovery: Evaluation Bug (Identified in Phase 1.4)

During Phase 1.4, a major evaluation discrepancy was discovered: RLlib's internal evaluation during training showed near-optimal rewards, but the custom rollout evaluation (used for visualization and final assessment) showed significantly worse rewards.

**Root cause:** `self.trainer.restore_from_path()` in `BaseAlgorithmWrapper.load_checkpoint()` restored the *learner's* module weights but did NOT synchronize them to the local `env_runner`. Since `self.trainer.get_module()` (used by the custom rollout) returns the module from the `env_runner`, the custom rollout was running with **randomly initialized weights** instead of trained weights.

**Fix:** After `restore_from_path()`, explicitly sync weights:
```python
weights = self.trainer.learner_group.get_weights()
self.trainer.env_runner.set_weights(weights)
```

**Impact:** All custom evaluation rewards reported in Phases 1.1–1.3 were incorrect (measured with random weights). After fixing, re-evaluation revealed that IPPO was learning effectively much earlier than believed. The conclusions and diagnoses from those phases have been revised accordingly below.

---

## Phase 1.1: Initial IPPO Experiments

### Setup

- **Action space:** `direct` with `max_order_quantities` varied across {20, 30, 40}.
- **Observation normalization:** Varied across {off, ratio, meanstd}.
- **Actor:** MLP [256, 256] with tanh output activation.
- **Log_std:** Free parameter initialized at 0.0 (std = 1.0), no floor.
- **PPO config:** `entropy_coeff=0.01`, `use_kl_loss=false` (not passed to RLlib — see bugs), `vf_clip_param=1000000`, `clip_param=0.2`.
- **Total runs:** 9 (3 max_qtys × 3 normalizations).

### Results (Corrected)

After fixing the evaluation bug, re-evaluation showed that **all runs learned to some degree** — contradicting the original conclusion of complete failure. The actor mean was not stuck at zero but had moved to between −0.5 and −1.0, with many runs outputting mean ≈ −1.0 (the tanh saturation boundary).

### Revised Key Observations

1. **Tanh saturation was the primary bottleneck:** The actor mean was pushed toward −1.0 (the tanh boundary) across nearly all runs, confirming that tanh output activation trapped the policy at the boundary where gradients vanish. This was misdiagnosed as "mean does not move" when using the buggy evaluation.

2. **Off normalization and larger action spaces worked better than originally believed:** With corrected eval, these configurations showed meaningful learning, reversing the original "initialization bias" conclusion.

3. **Entropy collapse was present but not the root cause:** The free log_std parameter still collapsed, but the policy had actually learned useful mean actions despite this.

### Bugs Discovered

- **KL loss not disabled:** Config had `use_kl_loss: false`, but the parameter was not passed to RLlib's `PPOConfig.training()`. RLlib defaulted to `use_kl_loss=True` with `kl_coeff=0.2`.
- **Gradient clipping not passed:** `grad_clip: 0.5` was not forwarded to RLlib (default: no clipping).

---

## Phase 1.2: Addressing Entropy Collapse and Initialization Bias

### Interventions

1. **Std floor:** Initialized `log_std = -1.0` (std ≈ 0.37) and clamped at `min = -2.0` (std ≥ 0.14).
2. **Removed tanh output activation:** Set `output_activation: null`.
3. **Fixed KL loss bug:** Added `use_kl_loss=ippo_params.use_kl_loss` to `.training()` call.
4. **Gradient clipping:** Set `grad_clip: null` (RLlib default).

### Experimental Design

- **Action space:** `direct`, max_order_quantities ∈ {20, 30, 40}.
- **Learning rate:** ∈ {3e-4, 1e-3}.
- **VF clip param:** ∈ {30000, 100000, 1e10 (off)}.
- **Fixed:** `entropy_coeff=0.01`, ratio normalization, no tanh, std floor, `use_kl_loss=false`.
- **Total runs:** 18 (3 × 2 × 3 factorial).

### Results (Corrected)

After fixing the evaluation bug, re-evaluation showed that the Phase 1.2 interventions (particularly removing tanh and fixing KL loss) were **highly effective**, reducing the gap to baseline to approximately 8% for the best configuration (stochastic evaluation). All MaxQty settings showed meaningful learning, not just MaxQty=20.

### Revised Key Observations

1. **Removing tanh was the most impactful fix:** With tanh removed, the actor mean was no longer trapped at ±1.0 and could move freely toward the optimal action.

2. **Higher LR (1e-3) beneficial:** LR=1e-3 produced better results than 3e-4, especially for larger MaxQty where the mean needs to travel further from initialization.

3. **Moderate VF clipping (30,000) helped:** VFC=30,000 outperformed both VFC=100,000 and VFC=off (1e10) when combined with LR=1e-3, providing stability without over-constraining VF learning.

4. **The original diagnosis of "catastrophic failure" for MaxQty=30/40 was incorrect** — these runs were also learning, just measured with random weights.

---

## Phase 1.3: Reward Scaling, Action Space Designs, and Structural Variations

### Interventions

1. **Reward scaling:** `scale_factor=0.01` (per-step rewards from ~−150 to ~−1.5).
2. **Demand-centered action space:** Actions as adjustments relative to observed demand.
3. **Holding cost increase:** Testing `holding_cost` ∈ {1.0, 3.0}.
4. **Custom mean-std normalization:** Fixed normalization from random-policy rollout statistics.
5. **Mu-sigma head:** State-dependent standard deviation.

### Experimental Design

Two parallel arrays (10 direct + 10 demand-centered runs), varying scale_factor, holding_cost, obs_normalization, std_type. All use LR=3e-4, entropy_coeff=0.01, vf_clip_param=1e10, use_kl_loss=false, 300 iterations.

### Results (Corrected)

After fixing the evaluation bug, re-evaluation showed most runs achieving meaningful learning.

### Revised Key Observations

1. **Custom meanstd normalization was the most significant improvement** for the multi-agent case, outperforming both ratio and RLlib's running meanstd.

2. **Mu-sigma head was consistently bad:** State-dependent σ led to instability across all configurations. The free log_std parameter with floor is strictly better.

3. **Reward scaling (scale=0.01) worked as intended:** VF losses dropped from millions to manageable levels, enabling stable VF learning.

4. **Demand-centered action space provided no significant advantage** over direct once the other fixes (no tanh, reward scaling) were in place.

5. **Holding cost variation (1 vs 3) had minimal effect:** The gradient asymmetry between holding and penalty costs was not the primary bottleneck.

---

## Phase 1.4: Single-Agent Isolation and Hyperparameter Tuning

### Motivation

To isolate whether remaining performance gaps were due to multi-agent dynamics or fundamental algorithm issues, a series of single-agent experiments (1WH, 1SKU) was conducted. This phase also led to the discovery of the evaluation bug.

### Experimental Design

- **Environment:** 1 warehouse, 1 SKU, 1 region, MaxQty=30, scale_factor=0.01.
- **Baseline:** −38.1 (base-stock z=2, scaled).
- **Variables tested:**
  - `entropy_coeff` ∈ {0.01, 0.1, 0.2}
  - `log_std_floor` ∈ {-2.0, -0.7}
  - Network architecture ∈ {[64], [128], linear (no hidden layers)}
  - `explore` = True vs False for evaluation

### Key Results

| Config | Eval Reward | Gap to Baseline |
|---|---|---|
| [64], floor=-2, entropy=0.01 | **−37.2** | **−2.4% (beats baseline)** |
| [128], floor=-2, entropy=0.01 | −37.4 | −1.8% (beats baseline) |
| [64], floor=-0.7, entropy=0.01 | −37.7 | −1.0% (beats baseline) |
| [128], floor=-0.7, entropy=0.01 | −38.3 | +0.5% |
| entropy=0.1 runs | ~−45 to −189 | Significantly worse |
| entropy=0.2 runs | ~−256 | Catastrophically worse |
| **Baseline (base-stock z=2)** | **−38.1** | — |

### Key Observations

1. **Single-agent achieves near-optimal performance**, validating the core algorithm (PPO + reward scaling + no tanh + log_std floor). The best configuration actually *beats* the heuristic baseline by 2.4%.

2. **entropy_coeff=0.01 is optimal.** Higher values (0.1, 0.2) cause entropy explosion — the entropy bonus overpowers the policy gradient, pushing σ upward and degrading the policy.

3. **log_std floor=-2 slightly outperforms floor=-0.7.** Floor=-2 (σ ≥ 0.14) allows more exploitation once the mean is well-learned, while floor=-0.7 (σ ≥ 0.50) forces excessive exploration throughout.

4. **log_std init=-1 is effective.** Starting at σ=0.37 provides adequate initial exploration without wasting early training on very noisy data.

5. **Network size [64] vs [128] barely matters in single-agent.** Both achieve near-optimal.

6. **explore=False for evaluation** produces more stable and interpretable eval rewards.

---

## Phase 1.5: Multi-Agent with Optimized Configuration

### Motivation

With single-agent performance validated, Phase 1.5 applies the winning configuration to the full 3WH/2SKU multi-agent environment, testing parameter sharing, network sizes, observation normalization, and batch size.

### Experimental Design

**Base configuration (all runs):** scale=0.01, holding_cost=1, LR=1e-3, gamma=0.99, lambda=0.95, entropy_coeff=0.01, log_std init=-1, log_std floor=-2, use_kl_loss=false, output_activation=null, vf_clip_param=300, explore=False for eval.

**Small batch:** num_iterations=300–500, batch_size=8000, num_epochs=10, num_minibatches=20.
**Large batch:** num_iterations=500, batch_size=40000, num_epochs=3, num_minibatches=4.

### Results

#### Main Runs (1–8): Systematic Comparison

| Run | Action Space | MaxQty | PS | NN | Obs Norm | Batch | Eval Reward | Gap |
|---|---|---|---|---|---|---|---|---|
| **2** | Direct | 40 | **True** | **[64]** | **meanstd_custom** | small | **−176.6** | **14.3%** |
| 1 | Direct | 40 | False | [64] | meanstd_custom | small | −207.2 | 34.1% |
| 6 | Direct | 40 | True | [64] | ratio | small | −216.9 | 40.4% |
| 5 | Direct | 40 | False | [64] | ratio | small | −221.0 | 43.1% |
| 7 | Direct | 40 | False | [64] | meanstd_custom | large | −256.1 | 65.8% |
| 4 | Direct | 40 | True | [128] | meanstd_custom | small | −369.8 | 139.4% |
| 8 | Direct | 40 | True | [64] | meanstd_custom | large | −384.5 | 149.0% |
| 3 | Direct | 40 | False | [128] | meanstd_custom | small | −481.1 | 211.4% |

**Baseline (base-stock z=2): −154.5**

#### Additional Runs (9–15): Ablations Against Run 5 Baseline

| Run | Variation | Eval Reward | Gap | vs Run 5 |
|---|---|---|---|---|
| 14 | Higher initial inventory (40) | −211.2 | 36.7% | +4.4% better |
| 5 | *(base: Direct, MQ40, [64], ratio)* | −221.0 | 43.1% | — |
| 10 | Demand-centered, MaxAdj15 | −223.3 | 44.5% | −1.0% |
| 12 | Two hidden layers [64,64] | −225.9 | 46.2% | −2.2% |
| 11 | log_std init=0 (σ=1.0) | −233.9 | 51.4% | −5.8% |
| 13 | Off normalization, [128], DC | −249.2 | 61.3% | −12.8% |
| 9 | MaxQty=30 | −255.5 | 65.4% | −15.6% |
| 15 | VF clip off (1e10) | −259.4 | 67.9% | −17.4% |

### Key Learnings

**1. Parameter sharing is the single most impactful intervention (with small batch + small network).**

PS=True improves eval by 15% with meanstd_custom (−207→−177), 23% with [128] (−481→−370), but only 2% with ratio normalization. However, PS *hurts* with large batch (−256→−385), because entropy explodes. PS eliminates multi-agent non-stationarity by forcing symmetric behavior and provides 3× more diverse training data for the shared policy.

**2. Smaller networks [64] are categorically better than [128].**

[128] is 2.1–2.3× worse than [64] across all settings. WandB shows [128] networks suffer from VF instability (warehouse 2 VF loss spikes to 160,000 in Run 3) and entropy explosion. With only 2 action dimensions and a near-constant optimal policy, a single [64] layer provides sufficient representational capacity.

**3. meanstd_custom consistently outperforms ratio normalization.**

The advantage is 6% without PS and 19% with PS. Custom normalization provides well-conditioned zero-mean, unit-variance inputs using pre-computed statistics, while ratio normalization creates bounded features with non-zero means.

**4. Large batch configuration was fundamentally flawed (configuration error, not a negative finding about batch size).**

Small batch: 200 gradient updates/iteration (10 epochs × 20 minibatches, minibatch=400). Large batch: 12 gradient updates/iteration (3 epochs × 4 minibatches, minibatch=10,000). Over 500 iterations: 100,000 vs 6,000 total gradient updates — 17× fewer. The 5× larger batch does not compensate for 17× fewer updates. A corrected configuration would use `num_ep=10, num_mb=20` with `bs=40000` (200 updates/iter, minibatch=2000).

**5. Entropy dynamics are the key differentiator between success and failure.**

- **Healthy (Run 2, best):** Entropy decreases to floor (−1), explained variance reaches 0.7 — the policy narrows exploration as the VF provides clean advantages.
- **Unhealthy (Runs 4, 6, 8):** Entropy *increases* (up to 6.5 in Run 8) despite explained variance increasing — a self-reinforcing loop where noisy advantages → wider σ → more random actions → returns converge to random-policy level → VF's job becomes trivially easy → explained variance rises while the policy deteriorates.
- **Critical diagnostic insight:** High explained variance alone does NOT indicate good learning. It must be paired with *decreasing* entropy to confirm genuine policy improvement. Increasing entropy + increasing explained variance = the VF is tracking a bad policy whose returns are becoming predictably bad.

**6. Warehouse asymmetry in PS=False runs reveals multi-agent non-stationarity.**

In Run 1 (PS=False), VF losses differ by 700× across warehouses (WH0=200, WH1=1.3, WH2=900) despite the symmetric environment. This is not caused by allocator tie-breaking (the home warehouse is always cheapest by 10×), but by independent agents' learning trajectories diverging due to different random initializations and cascading non-stationarity.

**7. VF clip at 300 is effective but very active in the best run.**

Run 5 (VFC=300) vs Run 15 (VFC=off): −221 vs −259, a 17% improvement. Without VFC, initial VF losses spike to 10,000–20,000. However, in Run 2 (best), unclipped VF loss=200 vs clipped=50 — VFC reduces VF learning by 75%, which may be overly constraining. WandB for Run 15 shows that without VFC, explained variance is paradoxically slightly higher (0.10–0.15 vs ~0) but advantages are biased from VF overshoot, leading to worse policy performance.

**8. Additional ablation findings (Runs 9–15):**

- **MaxQty40 > MaxQty30** (−221 vs −255): Larger action range is better once tanh/log_std floor are fixed. The Phase 1.1 initialization bias is reversed.
- **Demand-centered ≈ Direct** (−223 vs −221): Action space design is no longer a bottleneck.
- **init=-1 > init=0** (−221 vs −234): σ=0.37 is better than σ=1.0; excessive initial exploration wastes early training.
- **[64] ≈ [64,64]** (−221 vs −226): Depth adds nothing for this problem.
- **Normalization helps** (−221 ratio vs −249 off): Confirmed.
- **Higher initial inventory helps modestly** (−211 vs −221): Avoids early stockout penalties while learning, but WandB shows identical VF explained variance patterns — the improvement is purely from the stockout buffer, not from better learning dynamics.

### Convergence Analysis

The evaluation reward curve for Run 2 (best) shows a clear plateau from step ~300 to 500 — more training iterations will not close the 14% gap. The agent has converged to a suboptimal equilibrium. The remaining gap is due to the VF explained variance ceiling (0.7) creating noisy advantages that limit policy precision.

---

## Phase 1.6: Observation Space Improvements

### Motivation

Phase 1.5's best IPPO run (Run 2, −176.6) left a 14% gap to baseline. Before further hyperparameter tuning, this phase investigates whether enriching the agent's observation space can improve performance by providing more informative state representations.

### Experimental Design

**Base configuration:** Best IPPO config from Phase 1.5 (PS=True, [64], lr=1e-3, entropy_coeff=0.01, vf_clip=300, vf_loss_coeff=0.5).

Four observation space features were tested:

1. **Timestep in observation space:** Include the current timestep as a normalized feature, enabling the agent to condition its ordering policy on the episode phase (e.g., ordering more conservatively near episode end).
2. **Slot-based pending orders:** Replace aggregate pending order counts with a per-slot representation that tracks each in-transit order by its remaining lead time, providing richer temporal information about upcoming arrivals.
3. **Charging of non-arriving orders:** Charge inbound costs for orders placed near episode end that will not arrive within the episode horizon, penalizing wasteful end-of-episode ordering.
4. **Grouped MeanStd normalization:** Replace meanstd_custom with meanstd_grouped, which normalizes semantically related observation feature groups independently rather than using pre-computed statistics.

### Results

| Timestep | Slot-Based Pending | Charging Non-Arriving | Grouped MeanStd | Eval Reward | Baseline (BS z=2.0) |
|---|---|---|---|---|---|
| yes | no | no | no | −267.7 | −154.5 |
| no | yes | no | no | −218.3 | −154.5 |
| **yes** | **yes** | **no** | **no** | **−184.5** | **−154.5** |
| yes | yes | yes | no | −281.0 | −155.3 |
| yes | no | no | yes | −197.5 | −154.5 |

### Key Observations

1. **Slot-based pending orders is the most impactful single feature** (−267.7 → −218.3 when replacing timestep with slot-based). Providing per-slot lead-time information gives the agent a much clearer picture of upcoming inventory arrivals than the aggregate count.

2. **Timestep + slot-based together achieve the best result** (−184.5, 19.4% gap to baseline), indicating the two features are complementary — the agent benefits from knowing both when orders will arrive and how much of the episode remains.

3. **Charging non-arriving orders hurts significantly** (−184.5 → −281.0). The penalty for end-of-episode ordering creates a conflicting signal where ordering is simultaneously needed for inventory but penalized near episode end. The slightly different baseline (−155.3 vs −154.5) confirms this changes the reward structure itself.

4. **Grouped MeanStd normalization improves over baseline with timestep alone** (−267.7 → −197.5), suggesting better-conditioned inputs when normalizing feature groups independently, but does not outperform the timestep + slot-based combination without it.

5. **Final decision: slot-based pending orders adopted; timestep and charging dropped.** Although the timestep + slot-based combination achieved the best finite-episode reward (−184.5), timestep was deliberately excluded to prevent the agent from learning episode-boundary behavior (e.g., reducing orders near step 100) — real supply chains operate continuously, and the learned policy should be stationary (time-invariant). Charging for non-arriving orders was likewise dropped to avoid any end-of-episode cost dynamics. The chosen configuration ensures the episode end is a simple external cutoff: the agent does not know where it is in the episode, and no special cost accounting occurs at the boundary. This design prioritizes learning a steady-state ordering policy that generalizes beyond the finite training horizon.

---

## Phase 1.7: IPPO Hyperparameter Tuning (Updated Observation Space)

### Motivation

With the improved observation space from Phase 1.6 (timestep + slot-based pending orders adopted), this phase performs a targeted hyperparameter search over network architecture, normalization method, entropy coefficient, VF clipping, and VF loss coefficient to find the best IPPO configuration on the updated environment.

### Experimental Design

**Environment:** Updated observation space with timestep and slot-based pending orders from Phase 1.6. The heuristic baseline on this configuration is **−157.4** (base-stock z=2.0).

**Base configuration:** PS=True, lr=1e-3, gamma=0.99, lam=0.95, batch_size=8000, num_epochs=10, num_minibatches=20, log_std init=−1.0, floor=−2.0.

**Variables tested:** NN architecture ∈ {[64], [128]}, normalization ∈ {meanstd_grouped, meanstd_cust}, entropy_coeff ∈ {0, 0.01}, vf_clip_param ∈ {300, 1000}, vf_loss_coeff ∈ {0.5, 1}.

### Results

| NN | Normalization | Entropy Coeff | VF Clip | VF Loss Coeff | Eval Reward | Baseline |
|---|---|---|---|---|---|---|
| [64] | meanstd_grouped | 0 | 300 | 0.5 | −241.9 | −157.4 |
| **[64]** | **meanstd_grouped** | **0.01** | **1000** | **0.5** | **−209.8** | **−157.4** |
| [64] | meanstd_grouped | 0.01 | 300 | 1 | −279.8 | −157.4 |
| [64] | meanstd_grouped | 0 | 1000 | 0.5 | −250.7 | −157.4 |
| [128] | meanstd_grouped | 0.01 | 300 | 0.5 | −223.5 | −157.4 |
| [128] | meanstd_cust | 0.01 | 300 | 0.5 | −230.4 | −157.4 |
| **[128]** | **meanstd_grouped** | **0** | **1000** | **0.5** | **−206.3** | **−157.4** |
| [128] | meanstd_grouped | 0 | 1000 | 1 | −238.0 | −157.4 |

### Key Observations

1. **VF clip=1000 consistently outperforms VF clip=300.** Phase 1.5 noted VF clip=300 was 75% active (reducing VF learning by 75%). Relaxing to 1000 gives the critic significantly more learning freedom across all configurations.

2. **vf_loss_coeff=0.5 outperforms 1.0.** [128] with entropy=0, vf_clip=1000: −206.3 (coeff=0.5) vs −238.0 (coeff=1.0). The lower coefficient prevents VF gradients from dominating the total loss.

3. **[128] networks are now competitive with [64]**, reversing Phase 1.5's finding where [128] was 2× worse. Best [128] (−206.3) slightly outperforms best [64] (−209.8). The expanded observation space (timestep + slot-based features) likely benefits from the additional representational capacity.

4. **Entropy coefficient interacts with network size.** For [64], entropy=0.01 is better (−209.8 vs −250.7). For [128], entropy=0 is better (−206.3 vs −223.5). Larger networks may converge more confidently, reducing the need for the entropy exploration bonus.

5. **meanstd_grouped slightly outperforms meanstd_cust** for [128] (−223.5 vs −230.4), supporting adoption of grouped normalization with the updated observation structure.

---

## Phase 1.8: MAPPO — Centralized Critic with Global Observations

### Motivation

IPPO's local critic estimates returns from only the agent's own observation (29 dimensions), creating a VF explained variance ceiling (~0.7–0.85) that limits policy precision. MAPPO uses a centralized critic that observes the global state (all agents' observations), providing a richer information basis for value estimation. This phase tests whether MAPPO can close the remaining gap to the heuristic baseline.

### Implementation

MAPPO is implemented by setting `critic_obs_type="global"` while keeping `actor_obs_type="local"`. The centralized critic receives the full observation `[own_local_obs (29 dim), global_obs (87 dim)]` = 116 dimensions, while the actor continues using only the local 29-dim observation. Parameter sharing remains enabled.

### Bug Discovery: Observation Routing (Fixed)

Initial MAPPO runs produced catastrophically low VF explained variance (~0.05 vs ~0.8 for IPPO) and policy collapse (entropy increasing rather than decreasing).

**Root cause:** When `parameter_sharing=True` and `critic_obs_type="global"`, the shared critic received an identical 87-dim global observation for ALL agents — the concatenation of all agents' local observations WITHOUT the querying agent's own identity. A deterministic shared network given identical inputs for all agents can only predict the same (average) value for every agent.

**Fix:** Modified `setup()`, `_forward_inference()`, `_forward_train()`, and `compute_values()` in `src/algorithms/models/rlmodules/base.py` to route the `full_obs` (116 dim = own_local_obs + global_obs, including the agent's one-hot warehouse ID) when `critic_obs_type="global"`. This enables the shared critic to produce agent-specific value predictions.

### Experimental Rounds (After Fix)

**Round 1 — Network Size Exploration:**

Config: lr=0.001, grad_clip=null, num_minibatches=20, num_epochs=10, vf_loss_coeff=0.5, entropy_coeff=0.01.

| Actor | Critic | Peak Eval Reward | Trend |
|---|---|---|---|
| [64] | [128] | ~−169 | Slight decline after peak |
| [128] | [128] | — | Severe degradation |
| [64] | [128,128] | — | Severe degradation |

VF explained variance improved to ~0.8 (confirming the obs routing fix), but larger networks showed training instability with performance declining over time. **Diagnosis:** lr=0.001 without gradient clipping (`grad_clip=null`) caused unchecked gradient norms and aggressive parameter updates, especially harmful for larger networks.

**Round 2 — Gradient Clipping + LR Schedule:**

Added `grad_clip=0.5` and LR schedule `[[0, 0.001], [4000000, 0]]`.

Partial improvement for [128,128] critic, but performance still degraded over time. **Diagnosis:** `grad_clip=0.5` was 20× too aggressive compared to the MAPPO reference implementation (Yu et al., `max_grad_norm=10.0`). Combined with `num_minibatches=20` (producing 200 noisy gradient updates per iteration on tiny 400-sample minibatches), the effective learning signal was heavily distorted.

**Round 3 — MAPPO-Aligned Hyperparameters:**

Aligned with the reference MAPPO implementation (Yu et al.): `grad_clip=10`, `num_minibatches=1`, `vf_loss_coeff=1`, `lr=0.0005`.

| Actor | Critic | LR | Eval Reward (end) | VF Expl. Var. | Entropy Trend |
|---|---|---|---|---|---|
| [64] | [128,128] | 0.0005 (constant) | ~−350 | ~0.85 | Collapse at step ~260 |
| [64] | [128] | 0.0005 (constant) | ~−350 | ~0.85 | Steady decline |
| [64] | [128] | schedule → 0 | **~−200** (U-shape recovery) | ~0.85 | Slower decline |

### Key Findings

1. **The obs routing bug was the critical implementation error.** Without agent identity in the global observation, the shared centralized critic cannot function. After fixing, VF explained variance immediately jumped from ~0.05 to ~0.8.

2. **Continuous-action MAPPO is highly sensitive to hyperparameters.** The canonical discrete-action MAPPO defaults (from Yu et al.) required significant adaptation for continuous action spaces — particularly around gradient clipping, minibatch structure, and learning rate.

3. **Entropy collapse is the primary remaining failure mode.** All runs showed continuously declining entropy, leading to premature policy narrowing. As the policy becomes more deterministic, the gradient signal for the mean action vanishes (all sampled actions cluster near the mean → flat advantage landscape), while the gradient on log_std continues pushing entropy down. The mean action then drifts due to accumulated noise from stochastic demand.

4. **The LR schedule's U-shaped recovery is the key diagnostic evidence** for the entropy collapse diagnosis: as the LR approaches zero, log_std stops declining, entropy stabilizes, and the small but consistent gradient signal slowly corrects the accumulated mean action drift.

5. **`num_epochs=15` amplifies instability.** With 15 gradient updates on the same batch, later epochs use stale advantages (computed from the behavior policy), acting as noisy perturbations that accelerate both entropy decline and mean action drift.

6. **Decision: systematic hyperparameter search.** After several rounds of manual one-variable-at-a-time tuning, a systematic grid search over `learning_rate`, `num_epochs`, `entropy_coeff`, and `logstd_floor` for both IPPO and MAPPO is needed to find stable configurations for continuous-action multi-agent PPO.

---

## Summary of Identified Problems

| Problem | Phase Identified | Status |
|---|---|---|
| **Evaluation bug** (weight sync in load_checkpoint) | 1.4 | Fixed |
| **Tanh gradient saturation** (output activation traps mean at ±1) | 1.1 | Fixed (output_activation=null) |
| **KL loss accidentally active** (RLlib default not overridden) | 1.1 | Fixed (parameter properly passed) |
| **Entropy collapse** (log_std collapses without floor) | 1.1 | Fixed (log_std init=-1, floor=-2) |
| **Reward scale too large** (VF loss in millions) | 1.2 | Fixed (scale_factor=0.01) |
| **Entropy explosion in multi-agent** (σ increases instead of decreasing) | 1.5 | Partially addressed (PS=True prevents it) |
| **VF explained variance ceiling** (~0.7 with local critic) | 1.5 | Addressed by MAPPO centralized critic (→0.85) |
| **Multi-agent non-stationarity** (warehouse learning trajectories diverge) | 1.5 | Addressed by PS=True |
| **Obs routing bug** (shared global critic received identical input for all agents) | 1.8 | Fixed (route full_obs with agent identity) |
| **Entropy collapse in continuous-action MAPPO** (policy narrows prematurely, mean drifts) | 1.8 | Open — requires entropy_coeff / num_epochs tuning |

---

## Current Best Configuration

### IPPO (Phase 1.7)

Best IPPO configurations on the updated observation space (timestep + slot-based pending orders):

- **Best [128]:** eval reward **−206.3** (31% gap to −157.4 baseline) — meanstd_grouped, entropy=0, vf_clip=1000, vf_loss=0.5
- **Best [64]:** eval reward **−209.8** (33% gap to −157.4 baseline) — meanstd_grouped, entropy=0.01, vf_clip=1000, vf_loss=0.5

```yaml
# Algorithm config (Phase 1.7 best):
algorithm:
  shared:
    num_iterations: 500
    batch_size: 8000
    num_epochs: 10
    num_minibatches: 20
    learning_rate: 0.001
    num_env_runners: 4
    num_envs_per_env_runner: 10

  algorithm_specific:
    use_gae: true
    lam: 0.95
    gamma: 0.99
    use_kl_loss: false
    grad_clip: null
    entropy_coeff: 0       # 0.01 for [64]
    vf_loss_coeff: 0.5
    clip_param: 0.2
    vf_clip_param: 1000
    obs_normalization: "meanstd_grouped"
    parameter_sharing: true
    logstd_init: -1.0
    logstd_floor: -2.0
    networks:
      actor:
        config:
          hidden_sizes: [128]  # or [64]
      critic:
        config:
          hidden_sizes: [128]  # or [64]
```

### MAPPO (Phase 1.8)

Best MAPPO result: ~−200 eval reward (A[64]/C[128] with LR schedule), but training was unstable with performance degradation before recovery. MAPPO requires further hyperparameter tuning — particularly `entropy_coeff`, `num_epochs`, and `logstd_floor` — to achieve stable training with continuous actions.

---

## Next Steps

### Immediate: Systematic Hyperparameter Search (Phase 2.0)

After multiple rounds of manual one-variable-at-a-time tuning for MAPPO (Phase 1.8), a systematic grid search is planned for both IPPO and MAPPO to find the best configuration for each on the simplified symmetric environment.

**Fixed parameters** (based on evidence from Phases 1.1–1.8):

| Parameter | Value | Evidence |
|---|---|---|
| `num_minibatches` | 1 | MAPPO default; 20 caused noisy updates on tiny minibatches |
| `grad_clip` | 10 | MAPPO default; null and 0.5 both problematic |
| `clip_param` | 0.2 | Standard PPO |
| `gamma` | 0.99 | Standard for episode length 100 |
| `lam` | 0.95 | Standard GAE |
| `obs_normalization` | meanstd_grouped | Best from Phase 1.7 |
| `parameter_sharing` | true | Best from Phase 1.5 |
| Actor hidden | [64] | Consistently sufficient; actor obs is only 29 dim |

**Search space:**

| Parameter | Values | Rationale |
|---|---|---|
| `learning_rate` | {0.0003, 0.0005, 0.001} | 0.001 too high for MAPPO; LR schedule recovery at ~0.0003 |
| `num_epochs` | {5, 10, 15} | 15 causes stale advantages; 5 is MAPPO "hard task" setting |
| `entropy_coeff` | {0.01, 0.05, 0.1} | 0.01 insufficient for continuous actions |
| `logstd_floor` | {−2.0, −1.0} | Hard floor to prevent excessive policy narrowing |
| Critic hidden | {[128], [128,128]} | MAPPO critic benefits from larger capacity |

**Total:** 3 × 3 × 3 × 2 × 2 = 108 configs × 3 seeds = 324 runs per algorithm.

**Metric:** Mean eval reward over last 50 iterations (penalizes late-training degradation).

### Medium-term: Increase Complexity (Phase 3)

Once the simplified symmetric environment is solved (or the gap is understood and accepted):

1. **Heterogeneous demand** — different demand parameters across regions.
2. **Geographic imbalance** — asymmetric distances and shipment costs (2EU_1US).
3. **Scaling** — more warehouses, more SKUs.
4. **Team reward** — investigate cooperative reward vs agent reward with coupling.
