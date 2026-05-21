# Thesis High-Level Structure (Locked)

This is the agreed high-level skeleton for the thesis. Chapter 2 is locked at section
granularity below; the Methodology and Results structures are the agreed organization and
serve as the brief for the per-section `outline-constructor` runs. Per-paragraph outlines
are produced separately, one section at a time.

Status: locked at the chapter/section level. Hypotheses and falsifiers from
[EXPERIMENTS.md](../../docs/EXPERIMENTS.md) are a private planning record; the thesis
foregrounds only the primary tests (see Methodology pre-registration note).

---

## Top-level chapter skeleton

```
1  Introduction              motivation, RQ1/RQ2, contributions C1-C4, roadmap
2  Background and Related Work
3  Methodology               the apparatus (no numbers, no outcomes)
4  Results                   outcomes, organized by research question
5  Discussion                answers to RQ1/RQ2, threats to validity, limitations
6  Conclusion and Future Work
```

---

## Chapter 2 - Background and Related Work (chapter titels fixed)

Chapter preamble: frames 2.2-2.4 as one methodological toolbox for the problem of 2.1.

### 2.1  Classical Approaches to Inventory Control
- Theory: order policies, newsvendor model, base-stock policies,
  simulation-based optimization.
- Brief grounding paragraph on Bayesian / simulation-based optimization (surrogate +
  acquisition function) - supports the GP-BO baselines used in Methodology.
- `\paragraph{Recent Applications to Inventory Control}` (off-ToC) - applied literature.

### 2.2  Reinforcement Learning
**2.2.1  Single-Agent Reinforcement Learning**
- Theory: MDP, value functions, policy gradients, PPO (with GAE).
- `\paragraph{Recent Applications to Inventory Control}` (off-ToC).

**2.2.2  Multi-Agent Reinforcement Learning**
- Theory: Dec-POMDP, centralized training / decentralized execution (CTDE),
  IPPO and MAPPO (as PPO extensions), parameter sharing, credit assignment.
- `\paragraph{Recent Applications to Inventory Control}` (off-ToC).

### 2.3  Heterogeneity in Multi-Agent Reinforcement Learning
- What agent heterogeneity is; why parameter sharing becomes a binding constraint under it.
- Remedy taxonomy: identity / agent-indication features, per-agent observation
  normalization, specialization vs. sharing, attention- and entity-based architectures.
- Attention/entity architectures appear here only as a remedy CATEGORY (cited, related-work
  level). The self-attention mechanism itself is NOT taught here - it is introduced in
  Methodology where A4 is defined.
- `\paragraph{Recent Applications to Inventory Control}` (off-ToC).

### 2.4  Research Gap and Research Questions
- Funnel close: symmetric / multi-echelon work dominates; intra-echelon heterogeneity under
  decentralized parameter-shared MARL is unaddressed.
- Derives RQ1 (diagnostic) and RQ2 (prescriptive) as the earned conclusion of the gap.
- RQs are also stated briefly in the Introduction (the promise); 2.5 earns them (the
  justification).

---

## Chapter 3 - Methodology (agreed organization, not chapter titels)

Altitude: the apparatus. No numbers, no outcomes.

```
3.1  Problem Formulation
     Formal description of the supply chain model, Dec-POMDP; agents = warehouses; 
     demand_centered action space (+ rationale); observation space; 
     per-agent reward (agent scope), order-level demand model (3-level decomposition) as the benchmark advance;
     how it differs from aggregate per-(region,SKU) rate used in prior MARL work;
3.2  Heterogeneity Axes and the Five Environments
     throughput-invariant device; symmetric + 4 hetero variants; env table;
     per-axis hypothesized failure mode.
3.3  Algorithms
     2x2 grid (IPPO/MAPPO x PS/noPS) + CPPO ceiling; why PS is the focus.
3.4  Reference Baselines and Metrics
     BS-* family + Random/ConstantOrder; normalized score; coupling metric kappa.
3.5  Architectural Interventions A1-A5                   [contribution C3, the toolkit]
     mechanism of each; A4 the architectural centerpiece (self-attention mechanism
     explained here); matched-capacity MLP control.
3.6  Experimental Program and Statistical Protocol
     E0-E4 block matrix; seeds; paired differences; normalized score;
     pre-registration paragraph (full hypothesis/falsifier set deferred to appendix;
     main text foregrounds the primary tests only).
```

Flow: setting -> the benchmark I built -> the stressors I designed -> what learns ->
what I measure against -> what I change -> how I test it.

---

## Chapter 4 - Results (agreed organization, not chapter titels)

Organized by research question, not by experiment ID.

```
4.1  Symmetric baseline (E0)
     everything converges; PS ~ noPS; RL ~ ceilings.
     -> "the setting is sound; symmetry is not where the problem lives" (reference line)
4.2  Heterogeneity breaks parameter-shared MARL (E1)            -> answers RQ1
     degradation across axes, combined worst; re-tuning closes <half the gap
     -> gap is not hyperparameters; localizes failure to PS on SKU+cost axes;
     kappa disarms the "three decoupled problems" objection.
4.3  Recovering performance with interventions (E2)            -> RQ2 part 1
     each A_i on its target axis; A4 the architectural win; matched-MLP control rules
     out raw capacity; A5 best per-env combination.
4.4  Why they work, and why the mapping is non-uniform (E3)
     drop-one attribution; per-SKU mechanism + attention heatmaps; obs-side / reward-side
     sharing; warm-start + curriculum rule out the optimization-trajectory explanation
     -> the "no single fix" finding.
4.5  Scalability and transfer (E4)
     5/7 warehouses; entity-encoder critic; zero-shot real-world-inspired instance.
4.6  A decision rule for practitioners (C4)                    -> RQ2 part 2 / payoff
     heterogeneity profile -> recommended intervention combination.
```

Flow: baseline is fine -> heterogeneity breaks it and it is not the obvious causes ->
here is what fixes it and it is not just capacity -> here is the mechanism and the catch
that no fix is universal -> it scales and transfers -> therefore this decision rule.
