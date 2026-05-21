# Outline: 2.1 Classical Approaches to Inventory Control

**Section type:** standard
**Target length:** ~3 pages (~9 paragraphs). Default "as concise as the substance allows"; this is foundational background, so favor compression over completeness on any single method.

> **LITERATURE.md is empty.** `thesis/literature/LITERATURE.md` contains no paper map. Per the construction rules, every literature-supported beat below is marked `[CITE: <description>]` and **no** `\cite{<key>}` is asserted from `Literature.bib` titles alone. The downstream `cite-claim` skill must resolve each marker before the writer commits a key. Do not infer fit from `.bib` titles.

---

## Narrative arc

The section moves from problem to policy class to optimization method: it first frames the single-warehouse replenishment problem as the choice of an *order policy*, then narrows the space of policies along two axes (when to review, how much to order) to isolate the base-stock family the thesis adopts, gives the base-stock policy its analytical anchor through the newsvendor model, distinguishes static from dynamic policies to mark exactly where learning enters (and thereby hands the baton to the RL chapter), and finally introduces simulation-based / Bayesian optimization as the tool that finds base-stock levels when no closed form exists. The reader starts not knowing what an inventory policy is and ends understanding (a) why base-stock policies are the natural classical reference for this thesis and (b) how the thesis's strongest non-learning baselines (BS-Newsvendor, BS-Optimized, BS-Independent) are constructed. The spine is: *replenishment decision -> policy taxonomy -> base-stock anchor -> static/dynamic boundary -> how base-stock levels are actually computed.*

## Cross-section continuity

This is the **first section of Chapter 2** and follows the chapter preamble that frames Sections 2.2 through 2.4 as one methodological toolbox for the problem introduced here (see `thesis/outline/00_thesis_structure.md`). It therefore opens the Background chapter cold and must establish the inventory-control vocabulary that the RL sections reuse. It sets up two later threads: (i) the static/dynamic distinction (final-but-one paragraph) is the conceptual hinge that the Single-Agent RL section (`\autoref{sec:single-agent-rl}`, Section 2.2.1) picks up when it casts replenishment as a sequential decision problem, and (ii) the base-stock and simulation-optimization material grounds the reference baselines defined in the Methodology chapter (BS-Newsvendor, BS-Adaptive, BS-Optimized, BS-Independent; SCOPE.md §6.2), which the writer should forward-reference via `\autoref{}` rather than re-explain there. The "Recent Applications to Inventory Control" applied-literature paragraph that the locked structure attaches to this section is **out of scope for this outline** (see scope-boundary note at the end) and is authored separately.

---

## 2.1 Classical Approaches to Inventory Control

### Paragraph 1: The single-warehouse replenishment problem reduces to choosing an order policy, a rule that maps the observed inventory situation to a replenishment quantity.

**Connects to previous:** Opens the chapter. No previous paragraph; establishes the foundational object (the order policy) that every subsequent paragraph in the section refines and that the RL chapter will later re-express as a learned policy.

**Beats:**
- Define the classical single-location stochastic inventory problem: a facility holds stock of one or more items, faces stochastic demand each period, and periodically places replenishment orders that arrive after a lead time; the objective is to minimize expected total cost over a horizon, trading holding cost against shortage cost.
  - Evidence: [CITE: standard stochastic inventory-control / inventory-theory textbook treatment of the single-location periodic-review problem]
- Define *order policy* as the decision rule that answers the two questions of replenishment, namely *when* to order and *how much* to order, and frame the rest of the section as a tour of the most common such rules. State that the order quantity (or the parameters of the rule that sets it) is the decision variable while expected total cost is the objective.
  - Evidence: [CITE: textbook framing of an order/replenishment policy as the answer to the "when" and "how much" questions of inventory control]
- Bridge to the thesis's setting in one sentence: this thesis controls a network of such facilities, and the per-warehouse replenishment decision is exactly the object an order policy describes; the multi-warehouse and learning aspects are deferred to later sections via `\autoref{}`.
  - Evidence: explanation

**Length:** medium

**Term-definition responsibilities:** "order policy" / "replenishment policy" (define here, used throughout the section and chapter); "lead time", "holding cost", "shortage/penalty cost" (define briefly; full cost structure is Methodology's job, so keep to one clause each and do not pre-empt the lost-sales-vs-backorder modeling choice — see notation/modeling note below).

---

### Paragraph 2: Order policies are organized along two axes — the review timing and the form of the order-quantity rule — which together generate the standard families used in inventory practice.

**Connects to previous:** Narrows the generic "order policy" of Paragraph 1 into a structured taxonomy; this is the *policy taxonomy* step of the arc.

**Beats:**
- Introduce the parameter vocabulary that distinguishes policies: the reorder point (the inventory threshold that triggers an order), the order-up-to / order-up-to level (the target the order raises inventory to), the order quantity (a fixed lot size), and the review interval (the spacing of review instants).
  - Evidence: [CITE: textbook definition of reorder point, order-up-to level, order quantity, and review interval as the building blocks of order policies]
- Introduce the continuous-review vs. periodic-review distinction: under continuous review inventory is monitored constantly and an order can be triggered at any instant, whereas under periodic review the inventory position is inspected only at fixed instants. Note that continuous review is the idealized textbook case and periodic review is the operationally realistic one, which is also the regime this thesis adopts (one decision per timestep).
  - Evidence: [CITE: textbook treatment of continuous vs. periodic review, with continuous review as the textbook idealization]
- Present the standard policy families as combinations of these choices, covering every policy named in the user's list: the (reorder-point, order-up-to) policy and its periodic variant; the (reorder-point, order-quantity) policy and its periodic variant; the order-up-to / base-stock special case; and the demand-tracking rule that orders a function of observed demand each period. Recommend a compact comparison table (see table note).
  - Evidence: [CITE: comparative overview of (s,S), (s,S,r), (s,q), (s,q,r), (S-1,S), and X+Y-type order policies]
- Close by stating that the list is a sketch of the most prevalent strategies rather than an exhaustive enumeration, and that the right policy depends on the problem structure — which sets up the narrowing in the next paragraph.
  - Evidence: explanation

**Length:** long

**Term-definition responsibilities:** "reorder point", "order-up-to level", "order quantity", "review interval", "continuous review", "periodic review" (all defined here). NOTATION FLAG: the classical literature writes the reorder point as `s` and the order-up-to level as `S`, which collide with the thesis's reserved symbols for state (`s`) and the state set (`\mathcal{S}`) in STYLE.md §5.5. The writer must NOT use bare `s`/`S` for inventory-control quantities. Recommend introducing distinct notation (e.g., a reorder-point symbol and an order-up-to symbol via `\newnotation{}` in `Notation.tex`, or naming the policies in words and a table without overloading `s`/`S`). This is a hard collision and must be resolved before drafting. The table can present the canonical names using clearly-scoped local symbols.

**Table note:** A comparison table of order policies (policy / review type / decision rule) is the natural device here and matches the reference treatment. Mark as `[TABLE PENDING: order-policy overview — columns: policy family, review type, decision rule]` with a `tab:order-policies` label, caption above per STYLE.md §5.7. Reference it in the body before it appears.

---

### Paragraph 3: The base-stock policy is the order-up-to special case in which the reorder point and the order-up-to level coincide, and it is the classical reference policy this thesis builds its non-learning baselines on.

**Connects to previous:** Selects one family from the taxonomy of Paragraph 2 and elevates it; this is the *base-stock anchor* step of the arc and the pivot toward the thesis's own apparatus.

**Beats:**
- Define the base-stock policy: a periodic-review order-up-to rule in which every review the facility orders the quantity that raises its inventory position back to a fixed target level, equivalently the special case where the reorder point equals the order-up-to level. Note the "order-up-to" / one-for-one ("1-1") characterization for integer demand, where each unit of demand triggers a one-unit replenishment once inventory sits at the target.
  - Evidence: [CITE: textbook / reference definition of the base-stock (order-up-to / 1-1) policy as the (s,S) special case with reorder point equal to order-up-to level]
- State why base-stock policies are the natural classical reference: they are simple to specify and widely adopted, they reduce the policy to a single decision variable per item (the target level) rather than a rule shape, and under stationary stochastic demand the order-up-to family is a standard and well-justified choice. This is the property the thesis exploits.
  - Evidence: [CITE: literature establishing base-stock policies as widely used and, under standard assumptions, a strong/optimal policy class for single-location stochastic inventory]
- Connect explicitly forward to the thesis apparatus without pre-empting Methodology: the thesis's order-up-to-style baselines (the analytical, adaptive, and simulation-optimized base-stock references) all instantiate this policy class, and the per-(warehouse, SKU) target levels are their parameters. Forward-reference the Methodology baselines via `\autoref{}` and the demand-centered action space whose neutral action corresponds to demand-tracking (SYSTEM_MODEL.md §4.3).
  - Evidence: explanation (grounding in SCOPE.md §6.2 baseline family and §5.1 action-space rationale)

**Length:** medium

**Term-definition responsibilities:** "base-stock policy", "order-up-to level / target level", "base-stock level" (define here as the single decision variable). MODELING FLAG: classical base-stock exposition (and the reference PDF) assumes backorders; this thesis uses **lost sales** (SCOPE.md, SYSTEM_MODEL.md §4.5). Present the base-stock policy in its standard, shortage-mechanism-neutral form (target level, order-up-to rule) and avoid asserting backorder-specific optimality; defer the lost-sales modeling choice to Methodology. Do not import the reference's backorder notation.

---

### Paragraph 4: The newsvendor model gives the base-stock level its analytical form by setting the target where the marginal cost of overstocking balances the marginal cost of understocking.

**Connects to previous:** Supplies the missing piece left open in Paragraph 3 — *how* a base-stock level is set in closed form — and is the analytical engine behind the thesis's oracle baseline and initial-inventory rule. This is the analytical anchor inside the base-stock step of the arc.

**Beats:**
- Define the single-period newsvendor problem: a decision maker orders a quantity once against a single stochastic demand, incurs an overage cost per unsold unit and an underage (shortage) cost per unmet unit, and chooses the order quantity that minimizes expected cost.
  - Evidence: [CITE: standard newsvendor-model definition with overage and underage costs]
- State the critical-fractile / critical-ratio result: the cost-minimizing order quantity is the demand quantile at the critical ratio underage / (underage + overage), so the optimal service level is determined by the ratio of shortage to (shortage + holding) cost.
  - Evidence: [CITE: newsvendor critical-fractile (critical-ratio) optimality result]
- Connect the newsvendor logic to base-stock levels and to this thesis concretely: the critical ratio fixes the target service level, and under the thesis's penalty-to-holding cost structure this yields an optimal service level of penalty / (penalty + holding). Render the thesis instance inline for verification: `[NUMBER: penalty cost 5, holding cost 1 -> critical ratio 5/6 ~ 0.83 optimal service level]` (SCOPE.md §5.1, holding/penalty rationale). State that the lead-time-demand quantile at this service level, under a Poisson demand model, is what the analytical base-stock baseline and the initial-inventory formula approximate. Forward-reference (do not re-derive) the Methodology initial-inventory rule and the BS-Newsvendor baseline via `\autoref{}`.
  - Evidence: [CITE: newsvendor / base-stock connection — critical ratio as service-level target for the order-up-to level under stochastic lead-time demand]; empirical/grounding (SCOPE.md §5.1 newsvendor base-stock approximation `I_0 = E[D]L + z*sqrt(E[D]L)`, z=3.09; and the c_p/(c_p+c_h) service-level computation)

**Length:** medium

**Term-definition responsibilities:** "newsvendor model", "overage cost", "underage cost", "critical ratio / critical fractile", "service level" (define here). Keep the safety-stock / `z`-factor mechanics light: state that the target adds safety stock above expected lead-time demand to hit the service level, and leave the exact `z`=3.09 / square-root-of-mean Poisson details to Methodology (this paragraph motivates, Methodology specifies). Do not duplicate the full SCOPE.md §5.1 derivation here.

---

### Paragraph 5: Most classical policies, including base-stock, are static: their parameters are fixed once for the whole horizon, whereas dynamic policies reset their decision each period and so can react to changing conditions.

**Connects to previous:** Generalizes from the specific base-stock/newsvendor anchor back up to a structural property that partitions *all* the policies of Paragraph 2 — the static/dynamic distinction — and is the conceptual hinge that motivates the move to reinforcement learning. This is the *static/dynamic boundary* step of the arc.

**Beats:**
- Define the static/dynamic distinction precisely in the sense the user specified: a *static* policy commits its decision variables once at the start of the planning horizon and holds them fixed across all timesteps (e.g., a fixed vector of base-stock levels), whereas a *dynamic* policy re-determines its decision each period from the current situation (e.g., a demand-tracking rule that sets the order online every period).
  - Evidence: [CITE: inventory-literature treatment distinguishing static (open-loop, horizon-fixed) from dynamic (closed-loop, per-period) order policies]
- State the trade-off plainly and without advocacy (STYLE.md §1): static policies are simple and, under stationary demand, can be near-optimal within their class, but they are rigid and cannot adapt to non-stationary or context-dependent conditions; dynamic policies carry an informational advantage because they re-decide each period, at the cost of needing a mechanism to compute that per-period decision.
  - Evidence: [CITE: results / discussion contrasting the rigidity of horizon-fixed policies with the adaptivity of per-period policies]
- Hand the baton to the RL chapter: a learned control policy is the dynamic case taken to its general form, mapping the current observation to an order each period without a fixed analytical rule; name this as the bridge to `\autoref{sec:reinforcement-learning}` (Section 2.2). State that the thesis nonetheless retains strong static base-stock policies as references precisely because they are the competitive classical benchmark a learned dynamic policy must beat.
  - Evidence: explanation (bridge to Section 2.2; grounding in SCOPE.md §6.2 framing of base-stock baselines as the reference scale)

**Length:** medium

**Term-definition responsibilities:** "static policy", "dynamic policy" (define here; these terms recur in the Results framing of learned-vs-base-stock comparisons). This paragraph is the explicit forward-link to the RL sections — keep the bridge to one sentence and let Section 2.2 do the teaching.

---

### Paragraph 6: When no closed form for the optimal base-stock levels exists, they are found by simulation-based optimization, which treats the simulator as a black-box objective to be minimized over the level vector.

**Connects to previous:** Returns from the static/dynamic detour to finish the base-stock thread: Paragraph 4 set levels analytically under idealizing assumptions, but the thesis's network breaks those assumptions, so this paragraph introduces the computational route to base-stock levels. This is the *how base-stock levels are actually computed* step of the arc and the direct grounding for the GP-BO baselines.

**Beats:**
- Motivate why an analytical newsvendor base-stock is insufficient for the thesis's setting: in a multi-warehouse network with cross-region fulfillment, order-level multi-SKU demand, and interacting per-(warehouse, SKU) levels, the cost of a base-stock vector has no tractable closed form and must be estimated by simulating the policy. Frame the level vector as the optimization variable and the (noisy) simulated expected cost as the objective.
  - Evidence: [CITE: simulation-optimization framing for inventory base-stock levels under analytically intractable cost surfaces]
- Define simulation-based optimization at the level needed: an outer optimizer proposes candidate base-stock vectors, each is evaluated by running the stochastic simulator (a noisy, expensive, derivative-free objective), and the optimizer iterates toward the cost-minimizing vector. Name the defining features — black-box, expensive, stochastic evaluations — that make naive grid or gradient search impractical and motivate a sample-efficient method.
  - Evidence: [CITE: simulation-optimization / black-box optimization of expensive stochastic objectives]
- State the thesis usage explicitly so this paragraph visibly earns its place: the thesis's strongest non-learning reference computes base-stock levels this way, and a coordination-free variant optimizes each warehouse's levels independently by iterated best response. Refer to them by descriptive name (the simulation-optimized base-stock reference and the per-warehouse independently-optimized reference) per STYLE.md §4, and forward-reference the Methodology definitions via `\autoref{}`.
  - Evidence: explanation (grounding in SCOPE.md §6.2: BS-Optimized = joint GP-BO over all (n_WH x n_SKU) base-stocks, n_calls=300; BS-Independent = per-warehouse iterated best-response BO)

**Length:** medium

**Term-definition responsibilities:** "simulation-based optimization / simulation optimization" (define here). Introduce "black-box / derivative-free objective" in one clause. Do NOT yet introduce surrogate/acquisition machinery — that is Paragraph 7. Keep the iterated-best-response idea to a single naming sentence; its mechanics belong to Methodology.

---

### Paragraph 7: Bayesian optimization is the sample-efficient instance of simulation-based optimization used for these baselines: it fits a probabilistic surrogate to the observed cost evaluations and uses an acquisition function to choose where to evaluate next.

**Connects to previous:** Specializes the generic black-box optimizer of Paragraph 6 into the concrete method the thesis uses; this is the surrogate-plus-acquisition grounding paragraph the user and the locked structure require, and it is the terminal node of the base-stock thread.

**Beats:**
- Define Bayesian optimization as a surrogate-model-based strategy for optimizing expensive black-box functions: it maintains a probabilistic model (commonly a Gaussian process) of the objective over the parameter space, fit to the evaluations seen so far, and represents both the predicted cost and the uncertainty about it.
  - Evidence: [CITE: Bayesian optimization as surrogate-based optimization of expensive black-box functions; Gaussian-process surrogate]
- Define the acquisition function as the mechanism that turns the surrogate into the next query: it scores candidate points by trading off exploitation (where the surrogate predicts low cost) against exploration (where the surrogate is uncertain), and the optimizer evaluates the simulator at the acquisition maximizer, updates the surrogate, and repeats within a fixed evaluation budget. Name that this sample efficiency is exactly what makes the method appropriate when each simulated evaluation is costly.
  - Evidence: [CITE: acquisition-function role in Bayesian optimization — exploration/exploitation trade-off; common acquisition functions]
- Close the section by connecting the method to the thesis's baseline construction without over-claiming: the simulation-optimized base-stock references locate their per-(warehouse, SKU) target levels with Gaussian-process Bayesian optimization under a fixed evaluation budget, which is why they serve as a strong, near-best stationary reference (the practical heuristic ceiling) against which the learned policies are measured. Keep the claim scoped — it is a strong stationary baseline, not a global optimum — and forward-reference the Methodology baseline and the normalized-score definition via `\autoref{}`.
  - Evidence: explanation (grounding in SCOPE.md §6.2: GP-BO; BS-Optimized as the practical heuristic ceiling and anchor of the normalized score)

**Length:** medium

**Term-definition responsibilities:** "Bayesian optimization", "surrogate model", "Gaussian process" (name, do not derive the GP kernel/posterior), "acquisition function", "exploration/exploitation trade-off" (define here). Scope discipline (STYLE.md §1): describe BO as sample-efficient and as yielding a strong stationary reference; do not call its output "optimal" without qualification. This paragraph is foundational only — the joint-vs-independent BO design, n_calls, and the GP-BO configuration are Methodology's responsibility and must not be specified here.

---

## Required acronyms / glossary entries (for the writer)

The writer must ensure these are defined before first use. `thesis/contents/Abbreviations.tex` currently defines only `acr:alns`, `acr:ecv`, `acr:vrp` — **none of the acronyms this section needs exist yet.** New `\newacronym` entries required (writer adds them per STYLE.md §5.4):

- `acr:sku` — SKU / stock-keeping unit (used pervasively across the section and thesis; **flag as new**).
- `acr:bo` — BO / Bayesian optimization (**flag as new**; used in Paragraphs 6-7 and throughout Methodology).
- `acr:gp` — GP / Gaussian process (**flag as new**; Paragraph 7).
- `acr:rl` — RL / reinforcement learning (**flag as new**; needed for the forward-reference in Paragraph 5; almost certainly required by Section 2.2 as well — define once here or in 2.2, but it is first used in this section's bridge sentence).

Notation entries (`Notation.tex`, per STYLE.md §5.5) the writer must create rather than overload existing symbols:
- A reorder-point symbol and an order-up-to / base-stock-level symbol that do **not** reuse `s` (state) or `\mathcal{S}` (state set). **This is the notation-collision flag from Paragraph 2 — non-optional.**
- Expected demand, lead time, and a safety-factor symbol if the writer chooses to render the newsvendor base-stock expression inline (Paragraph 4); otherwise keep that paragraph prose-only and defer symbols to Methodology.

---

## Scope-boundary decisions (for the user)

1. **"Recent Applications to Inventory Control" excluded.** Per the user's notes, the applied-literature paragraph (which the locked structure attaches to this section as an off-ToC `\paragraph`) is authored separately and is NOT in this outline. This outline covers the *foundational classical methods only*. The writer should leave the off-ToC `\paragraph` slot for that separate pass.
2. **No multi-echelon / serial-system inventory theory.** The reference PDF develops a four-actor serial supply chain (TSCC over echelons). This thesis is single-echelon (SCOPE.md §4, §9), so the classical theory here is presented at the single-location level and the multi-warehouse aspect is named only as a forward-pointer. Multi-echelon classical results are out of scope and excluded.
3. **Backorder vs. lost-sales framing.** The classical literature and the reference PDF assume backorders; the thesis uses lost sales (SYSTEM_MODEL.md §4.5). The order-policy and base-stock material is presented in shortage-mechanism-neutral form, and the lost-sales modeling choice is deferred to Methodology. Flagged in Paragraphs 3-4 so the writer does not import backorder-specific claims.
4. **Newsvendor added beyond the reference's structure.** The reference PDF's order-policy section has no newsvendor paragraph; it is added here (Paragraph 4) because it is on the user's topic list AND it is the analytical engine for both the initial-inventory rule (SCOPE.md §5.1) and the BS-Newsvendor baseline (SCOPE.md §6.2). Marked `[ADDED: required by the arc — analytical anchor for the base-stock baselines]`.
5. **Two-paragraph split of simulation-based optimization.** The user listed one grounding paragraph. It is split into a generic simulation-optimization motivation (Paragraph 6) and a Bayesian-optimization specialization (Paragraph 7) so that the surrogate/acquisition machinery the user explicitly wants is introduced cleanly without conflating "why simulate" with "how Bayesian optimization works." Marked `[ADDED: structural — separates the why from the how]`. If a tighter section is preferred, these can be merged into one long paragraph at some cost to clarity.
6. **All depth on the GP-BO configuration is deferred.** Per the user's "support the GP-BO baselines without over-claiming" instruction, this section motivates and grounds the method only; joint-vs-independent optimization, evaluation budgets, and the normalized-score anchoring live in Methodology and are forward-referenced.
