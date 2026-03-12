# Experiment Log: Making IPPO Learn in a Simplified Environment

## Overview and Approach

The experimental strategy follows a **simplify-first** principle: start with a highly simplified, symmetric environment where the optimal policy is known analytically (a roughly constant order quantity), make the IPPO algorithm learn this policy, and only then add complexity (heterogeneous demand, geographic imbalance, multi-agent coupling, MAPPO).

The project is currently in **Phase 1** — making IPPO work on the simplified environment. Despite extensive experimentation across three sub-phases (1.1, 1.2, 1.3), the agent has not yet learned the optimal policy. This document describes the experimental journey, diagnosed problems, attempted fixes, and current status.

---

## Simplified Environment

The simplified environment (`env_simplified_symmetric.yaml`) is designed to minimize confounding factors:

- **3 warehouses, 2 SKUs, 3 regions** (symmetric: identical demand, costs, distances across warehouses).
- **Symmetric distances:** Each warehouse is close to its home region (distance 50) and far from others (distance 500), making transshipment rare.
- **Uniform demand:** All warehouses/regions/SKUs have identical Poisson demand parameters (`lambda_orders=4`, `probability_skus=0.667`, `lambda_quantity=5` per SKU), yielding expected demand of ~8 units per SKU per warehouse per timestep.
- **Fixed lead times:** 3 timesteps for all (warehouse, SKU) pairs.
- **Cost structure:** `holding_cost=1.0`, `penalty_cost=5` per SKU, `inbound_variable=1.0`, symmetric outbound costs (`0.05` for home regions and `0.05` for others).
- **Agent-scope rewards:** Each warehouse receives its own cost as reward (no team reward to complicate credit assignment).
- **Episode length:** 100 timesteps.

A single-agent variant with 1 warehouse, 1 SKU, 1 region also exists for isolation testing.

### Baselines (Phase 0)

Three baseline policies were evaluated to establish performance bounds:

| Baseline | Best Config | Mean Episode Reward |
|---|---|---|
| **Random** | Uniform `[-1, 1]` | −343,291 |
| **Constant order** | qty=12 per SKU per timestep | **−15,628** |
| **Heuristic (newsvendor base-stock)** | safety factor z=2.0 | −15,641 |

The constant order policy of 12 units per SKU effectively matches the newsvendor optimal. This confirms the environment works as intended and that the optimal policy is approximately a constant order of ~12 units per warehouse per SKU per timestep. The goal for IPPO is to approach this reward level (~−15,628).

---

## Phase 1.1: Initial IPPO Experiments

### Setup

- **Action space:** `direct` with `max_order_quantities` varied across {20, 30, 40}.
- **Observation normalization:** Varied across {off, ratio, meanstd}.
- **Actor:** MLP [256, 256] with tanh output activation.
- **Log_std:** Free parameter initialized at 0.0 (std = 1.0), no floor.
- **PPO config:** `entropy_coeff=0.01`, `use_kl_loss=false` (intended but not passed to RLlib — see bugs below), `vf_clip_param=1000000`, `clip_param=0.2`.
- **Total runs:** 9 (3 max_qtys × 3 normalizations).

### Results

No configuration approached the baseline reward of −15,628. All runs failed to learn a constant order policy — agents consistently overstocked all SKUs, producing linearly growing inventory.

### Key Observations

1. **Initialization bias:** MaxQty=20 consistently outperformed MaxQty=30 and 40, not because the agent learned better but because the initial policy (action mean ≈ 0 → order ≈ max_qty/2) was closer to the optimal order of 12 when max_qty=20 (initial order ≈ 10) than when max_qty=40 (initial order ≈ 20).

2. **Mean does not move:** The actor's mean output barely moved from its initialization near zero throughout training. The deterministic evaluation policy (which uses only the mean) remained far from optimal for MaxQty=30 and 40.

3. **Entropy collapse:** The free log_std parameter rapidly decreased during training, collapsing the policy's standard deviation to near-zero. This created misleadingly improving training returns (less noisy sampling → fewer costly exploration samples) while the mean action remained unchanged. The deterministic evaluation returns were much worse than training returns, confirming that the "improvement" was due to variance reduction, not mean movement.

4. **Training vs. evaluation gap:** Training rewards improved significantly while evaluation rewards stayed poor — a clear signature of entropy collapse where the stochastic training policy looks better than the deterministic evaluation policy only because it narrowed its sampling window, not because it found better actions.

### Bugs Discovered

- **KL loss not disabled:** The config had `use_kl_loss: false`, but this parameter was not passed to RLlib's `PPOConfig.training()`. RLlib defaulted to `use_kl_loss=True` with `kl_coeff=0.2`, penalizing the policy for deviating from its previous version and severely constraining policy updates. This was confirmed by observing `curr_kl_coeff` increasing over time in WandB.
- **Gradient clipping not passed:** Similarly, `grad_clip: 0.5` from the config was not forwarded to RLlib, so RLlib used its default of `grad_clip=None` (no clipping).

---

## Phase 1.2: Addressing Entropy Collapse and Initialization Bias

### Interventions

Based on the Phase 1.1 diagnosis, the following changes were made:

1. **Std floor:** Initialized `log_std = -1.0` (std ≈ 0.37 instead of 1.0) and clamped it at `min = -2.0` (std ≥ 0.14) in the `_append_log_std` method. This prevents entropy collapse by ensuring a minimum exploration level.

2. **Removed tanh output activation:** Set `output_activation: null` in the actor config. Without tanh, the actor outputs raw values (clipped by RLlib to [-1, 1]), avoiding gradient saturation at the boundaries that makes it hard for the mean to move.

3. **Fixed KL loss bug:** Added `use_kl_loss=ippo_params.use_kl_loss` to the `.training()` call in `ippo.py`. This correctly disables the KL penalty when set to `false`.

4. **Gradient clipping:** Set `grad_clip: null` in the config to maintain the RLlib default of no gradient clipping (matching the behavior of Phase 1.1 runs where it wasn't passed).

### Experimental Design

- **Action space:** `direct`, max_order_quantities ∈ {20, 30, 40}.
- **Learning rate:** ∈ {3e-4, 1e-3}.
- **VF clip param:** ∈ {30000, 100000, 1e10 (off)}.
- **Fixed:** `entropy_coeff=0.01`, ratio normalization, no tanh, std floor, `use_kl_loss=false`.
- **Total runs:** 18 (3 × 2 × 3 factorial).

### Results

| | MaxQty=20 | MaxQty=30 | MaxQty=40 |
|---|---|---|---|
| Best eval (LR=3e-4) | **−16,511** | −78,944 | −153,044 |
| Baseline | −15,628 | −15,628 | −15,628 |

MaxQty=20 improved marginally over Phase 1.1 (~5% gap reduction), confirming the KL fix helped. But MaxQty=30 and 40 remained catastrophically bad — agents still overstocked.

### Key Observations

1. **KL fix helped but was insufficient:** MaxQty=20 improved from ~−17,350 (Phase 1.1 best) to −16,511 (Phase 1.2 best), closing the gap to the baseline by ~50%. The fix was correct but not the only bottleneck.

2. **VF clip doesn't matter much:** No consistent pattern across VF clip values — sometimes 30K was best, sometimes 1e10. Value function clipping is not the primary bottleneck.

3. **Higher LR helps for larger MaxQty:** LR=1e-3 was better for MaxQty=40 (mean needs to travel further), worse for MaxQty=20. But even the best MaxQty=40 result was terrible.

4. **Single-agent also fails:** A 1-warehouse, 1-SKU experiment with MaxQty=30 achieved −51,333 vs. a baseline of −3,796. The same inability to move the mean was present in the simplest possible setting, eliminating multi-agent dynamics, observation complexity, and inter-warehouse coupling as explanations.

### Diagnosed Root Cause: Reward Scale and Value Function Instability

The single-agent failure pointed to a fundamental training issue. With per-step costs around −150 and episode returns around −15,000, the value function starts from predictions near zero and must learn to predict values in the thousands. The initial VF MSE is in the hundreds of millions, creating enormous gradients that cause the VF to overshoot and oscillate. Since the advantage estimates depend on accurate value predictions (`A_t = r_t + γV(s_{t+1}) − V(s_t)`), and the value function is wildly inaccurate, the advantages are dominated by VF noise. The policy gradient, amplified by `1/σ²` (which is ~55x with the std floor), becomes essentially random. The mean wanders aimlessly instead of moving toward the optimal action.

Additionally, a **gradient asymmetry** in the cost structure was identified: the penalty for under-ordering (lost-sales cost = 5 per unit, concentrated in one step) is much sharper than the signal for over-ordering (holding cost = 1 per unit per step, accumulated over many steps). This means the gradient signal to "order more" is 5× stronger and more immediate than the signal to "order less," biasing the agent toward overstocking — especially when starting above the optimal (MaxQty=30 and 40).

---

## Phase 1.3: Reward Scaling, Action Space Designs, and Structural Variations

### Interventions

Phase 1.3 tests multiple independent hypotheses in a screening design:

1. **Reward scaling:** Implemented the `scale_factor` in the reward calculator. With `scale_factor=0.01`, per-step rewards drop from ~−150 to ~−1.5 and episode returns from ~−15,000 to ~−150. This should dramatically reduce VF loss (from ~10⁸ to ~10⁴), enabling the VF to learn accurate predictions and produce clean advantage estimates.

2. **Demand-centered action space:** With `action_space_type: "demand_centered"` and `max_quantity_adjustment` (X), the agent outputs adjustments relative to observed demand. At initialization (action ≈ 0), the agent orders exactly what was demanded — a sensible starting policy that is below the optimal (demand ≈ 8, optimal ≈ 12). This means the agent needs to learn to order MORE, which aligns with the strong penalty gradient signal (stockout avoidance).

3. **Holding cost increase:** Testing `holding_cost` ∈ {1.0, 3.0} to rebalance the gradient asymmetry between holding and penalty costs.

4. **Custom mean-std normalization:** A new observation normalization mode that computes mean/std statistics from a random-policy rollout at the start of training and applies them as fixed normalization throughout.

5. **Mu-sigma head:** State-dependent standard deviation where the actor network outputs both mean and log_std, allowing the agent to modulate exploration based on the current state.

### Experimental Design

Two parallel arrays:

**Direct action space (10 runs):** Varying max_order_quantities ∈ {20, 30, 40}, scale_factor ∈ {1.0, 0.01, 0.001}, holding_cost ∈ {1, 3}, obs_normalization ∈ {ratio, custom_meanstd}, std_type ∈ {free, mu_sigma}.

**Demand-centered action space (10 runs):** Varying max_quantity_adjustment ∈ {10, 15, 20}, scale_factor ∈ {1.0, 0.01, 0.001}, holding_cost ∈ {1, 3}, obs_normalization ∈ {ratio, custom_meanstd}, std_type ∈ {free, mu_sigma}.

All runs use LR=3e-4, entropy_coeff=0.01, vf_clip_param=1e10 (off), use_kl_loss=false, 300 iterations.

### Status

Runs are in progress. Preliminary results suggest that none of the tested configurations have achieved baseline-level performance, though full analysis is pending.

---

## Summary of Identified Problems

| Problem | Phase Identified | Status |
|---|---|---|
| **Entropy collapse** (std collapses to near-zero, hiding the fact that the mean hasn't learned) | 1.1 | Mitigated via std floor (log_std init -1.0, clamp -2.0) |
| **KL loss accidentally active** (RLlib default, not disabled in code) | 1.2 | Fixed (parameter now properly passed) |
| **Initialization bias** (performance depends on max_order_qty, not learning) | 1.1 | Not resolved; demand-centered action space tested in 1.3 |
| **Mean does not move** (actor output stays near initialization) | 1.1 | Not resolved; reward scaling tested in 1.3 |
| **Value function instability** (VF loss in millions, noisy advantages) | 1.2 | Reward scaling (scale_factor=0.01) tested in 1.3 |
| **Gradient asymmetry** (penalty signal 5× stronger than holding signal) | 1.2 | Holding cost increase tested in 1.3; demand-centered alignment tested in 1.3 |
| **Tanh gradient saturation** (output activation squashes gradients at boundaries) | 1.1 | Tested in 1.1 |

---

## PPO Hyperparameter Reference

The current IPPO configuration after all fixes:

```yaml
algorithm:
  shared:
    num_iterations: 300
    batch_size: 8000
    num_epochs: 10
    num_minibatches: 20
    learning_rate: 0.0003
    num_env_runners: 4
    num_envs_per_env_runner: 10
    eval_interval: 10
    num_eval_episodes: 5

  algorithm_specific:
    use_gae: true
    lam: 0.95
    gamma: 0.99
    use_kl_loss: false     # was accidentally True in Phase 1.1
    grad_clip: null         # RLlib default: no gradient clipping
    entropy_coeff: 0.01
    vf_loss_coeff: 0.5
    clip_param: 0.2
    vf_clip_param: 1e10    # effectively disabled
    obs_normalization: "ratio"
    networks:
      use_mu_sigma_head: false
      actor: MLP [256, 256], ReLU, no output activation
      critic: MLP [256, 256], ReLU, no output activation
```

### Log_std Configuration

- Free parameter initialized to -1.0 (std ≈ 0.37).
- Clamped at min = -2.0 (std ≥ 0.14) to prevent entropy collapse.
- When `use_mu_sigma_head: true`, the sigma head replaces the free parameter with state-dependent log_std output clamped to [-4.6, 4.6].

---

## Next Steps

If Phase 1.3 results confirm that reward scaling and/or demand-centered action space solve the learning problem:
- Refine the working configuration.
- Scale to more complex environments (heterogeneous demand, geographic imbalance).
- Introduce MAPPO and compare with IPPO.
- Test information sharing mechanisms.

If Phase 1.3 does not produce a working configuration:
- Investigate discrete action spaces (MultiDiscrete) to bypass continuous-action issues entirely.
- Consider alternative RL algorithms (SAC, which handles entropy differently).
- Examine whether the value function architecture needs fundamental changes (e.g., reward normalization wrappers, separate VF learning rate).
