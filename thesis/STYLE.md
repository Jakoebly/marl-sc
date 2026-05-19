# STYLE.md — Thesis voice, tone, and LaTeX conventions

This file defines the voice, tone, and LaTeX conventions for the thesis. It is loaded by every thesis subagent and **overrides** the writer-agent defaults when they conflict. The writer's hard rules (gap markers, topic-sentence discipline, transition discipline, no em-dash/semicolon clause-joining, no single-sentence paragraphs, no section preambles/summaries) are not restated here — they remain in force.

## 1. Epistemic posture

The thesis is a controlled diagnostic study: identify failure modes of parameter-sharing MARL under heterogeneity, then test targeted interventions. Write as a diagnostician, not an advocate.

- **Match scope language to evidence.** Findings hold in the configurations tested. Words like "in general", "always", "broadly", "robust" misrepresent results and invite reviewer attack. Prefer "in the settings studied", "across the heterogeneity profiles of E2.x", "under the configurations of \autoref{tab:cell-grid}".
- **Report the negative result plainly.** When an intervention does not help, the section says so without softening adverbs. "Reward balancing did not close the gap to BS-Optimized on C-3" beats "reward balancing showed mixed results".
- **Distinguish behavior from cognition.** Agents *select*, *output*, *order*, *map observations to actions*. Agents do not *think*, *decide* (in the cognitive sense), *prefer*, *want*, or *reason about* state.
- **Pre-registered framing stays pre-registered.** Where [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) specifies an "expected outcome" and a "falsifier", the prose preserves that forward-looking phrasing. Do not retroactively reframe falsifiers as "we expected ...".

## 2. Voice

- Third person, formal academic register.
- **"This thesis" over "we"** for thesis-side empirical and contribution statements: "This thesis studies …", "Chapter 4 introduces …", ". The empirical "we propose / we find / we observe" should not be used.
- **Banned hedges and adjectives** (examples; not exhaustive). In addition to those the writer agent already bans: "potential", "promising", "interesting", "novel" (as a self-attribute), "robust" (as an unqualified claim — see §1 on scope language), "significant" (unless meant statistically and tied to a named test), and the construction "results suggest potential benefits".
- **Allowed epistemic hedges**, when tied to specific evidence (examples; not exhaustive): "is consistent with", "is compatible with", "suggests" (followed by *what* is suggested), "indicates", "tracks", "varies with", "appears to under [condition]", "is attributable to [cause] under [condition]", "holds within [condition]". The test for any hedge: it should mark genuine uncertainty about a specific claim, not soften a claim the writer was unwilling to commit to.

### 2.2 Sentence-level patterns to avoid (LLM tells)

LLM-generated prose defaults to a handful of patterns that read as machine-written and weaken the diagnostic tone. The writer agent already prohibits em-dash and semicolon clause-joining. This subsection restates and broadens the prohibition because these patterns are the single most common defection in LLM-drafted academic prose.

- **Em dash joining independent clauses.** Bad: "Reward balancing did not close the gap on C-3 --- the residual gap appears under cost asymmetry." Use a period and start a new sentence. The em dash (`---`) is permitted only for a parenthetical interruption inside a single clause (see §5.9).
- **Hyphen used as an em dash.** A single hyphen (`-`) is never a clause connector. It is reserved for compound modifiers (`base-stock policy`) and label kebab-case. Bad: "MAPPO failed - this is consistent with non-stationarity." Use a period.
- **Semicolon joining clauses.** Bad: "Parameter sharing degraded performance on E2.3; this is attributable to non-stationary credit assignment." Use a period. The semicolon is permitted only inside a list of complex items that themselves contain commas.
- **Trailing participle phrases.** Bad: "MAPPO failed to close the gap, suggesting that …", "Lost sales were under-attributed, leading to …", "The advantages were shared, resulting in …". This construction smuggles a causal claim into a grammatically attached clause and evades the burden of justifying it. Replace with two sentences that name the claim and the evidence: "MAPPO failed to close the gap. This is consistent with …".
- **Triadic stacking.** "X, Y, and Z" used reflexively in every sentence. Vary the rhythm: pair items where two are sufficient, drop "and Z" when Z is redundant.
- **Stock connectives.** "It is worth noting that", "Importantly,", "Notably,", "Interestingly,". Delete the connective. The sentence that follows either earns its place or does not.

## 3. Tense conventions

| Context | Tense | Example |
|---|---|---|
| Definitions, theory, system description | Present | "The environment exposes a flat local observation per warehouse." |
| Thesis-level contribution statements | Present | "Chapter 4 introduces three interventions targeting cost-asymmetric heterogeneity." |
| Experimental procedure | Past | "Each configuration was trained for 200 iterations across five seeds." |
| Numerical results | Past | "MAPPO with parameter sharing failed to close the gap on E2.3." |
| Interpretation of results | Present | "The pattern is consistent with non-stationary credit assignment under shared advantages." |
| Prior literature (what a paper did) | Past | "Foerster et al.\ proposed COMA …" |
| Prior literature (collective state of the field) | Present perfect | "Prior work has established that …" |

Do not shift tenses within a subsection without a logical reason.

## 4. Terminology discipline

The following terms are not interchangeable. Use each in its specific sense throughout.

- **Agent:** the per-warehouse decision entity that consumes observations and emits actions. The environment has `n_warehouses` agents.
- **Policy:** the mapping from observation to action distribution, parameterized by a neural network. Under parameter sharing, all agents *share* one policy but remain distinct agents.
- **Warehouse:** the supply-chain entity that the agent controls. The agent is not the warehouse.
- **Algorithm:** IPPO, MAPPO, or CPPO; the training procedure that produces a policy.
- **Configuration:** one fully-specified env + algorithm pair, the unit at which seed-evaluation operates.
- **Heterogeneity:** configured asymmetry across agents, SKUs, or regions in the senses defined in [docs/SCOPE.md](../docs/SCOPE.md). Do not use the word in a colloquial sense.
- **Failure mode:** a characterizable, repeatable pattern of suboptimal behavior tied to a specific heterogeneity type. Not "a bad result".

## 5. LaTeX conventions

### 5.1 Sectioning and labels

Every chapter, section, subsection, equation, float, algorithm, and named theorem-like environment has a label. Prefixes are required.

| Element | Prefix | Example |
|---|---|---|
| Chapter | `chap:` | `\label{chap:intro}` |
| Section | `sec:` | `\label{sec:problem-formulation}` |
| Subsection | `subsec:` | `\label{subsec:network-topology}` |
| Equation | `eq:` | `\label{eq:bellman}` |
| Figure | `fig:` | `\label{fig:rollout-c3}` |
| Table | `tab:` | `\label{tab:cell-grid}` |
| Algorithm | `alg:` | `\label{alg:mappo-train}` |
| Definition | `def:` | `\label{def:dec-pomdp}` |
| Theorem / lemma | `thm:` / `lem:` | `\label{thm:policy-grad}` |
| Appendix section | `app:` | `\label{app:hyperparams}` |

Labels use kebab-case after the prefix. Do not abbreviate label bodies. Labels survive reorganization and must remain readable.

### 5.2 Cross-references

Two commands are in scope: `\autoref{}` for every label type **except** equations, and `\eqref{}` for equations (the amsmath form, which auto-renders the parentheses and matches the project template).

- **`\autoref{...}` for chapters, sections, subsections, figures, tables, algorithms, definitions, theorems, and appendices.** Never `Section~\ref{...}`, never hand-written "Section 4.2", never `\cref` / `\Cref`. `\autoref` keeps the prefix consistent with the label type.
- **`\eqref{...}` for equations.** Renders as "(3)". Equations are typically cited inline as "see \eqref{eq:bellman}" or "from \eqref{eq:bellman}, …". Never `\autoref` for equations and never the bare `\ref` form.
- **Single ref**: `\autoref{sec:bellman}` renders as "Section 4.1"; `\eqref{eq:bellman}` renders as "(3)".
- **Two refs of the same type**: pair with "and". Example: "\autoref{sec:foo} and \autoref{sec:bar}" renders as "Section 4.1 and Section 4.2". Accept the repeated prefix; do not rewrite as "Sections 4.1 and 4.2" since that bypasses `\autoref`. For equations, "\eqref{eq:a} and \eqref{eq:b}" renders as "(3) and (4)" — no repetition issue.
- **Three or more refs of the same type**: list with commas and a final "and". Example: "\autoref{sec:foo}, \autoref{sec:bar}, and \autoref{sec:baz}" renders as "Section 4.1, Section 4.2, and Section 4.3".
- **Contiguous ranges**: for non-equation refs, `\autoref{sec:foo}--\autoref{sec:baz}` (en dash) renders as "Section 4.1--Section 4.3". The repeated prefix is grammatically awkward in a range; prefer rephrasing to the two-ref or three-ref form above. For equations, `\eqref{eq:a}--\eqref{eq:c}` renders cleanly as "(3)--(5)".
- **Sub-labels**: `\autoref{fig:cell-grid}(b)`.

### 5.3 Citations

The thesis uses `natbib` (loaded without options) with the INFORMS author-year style (`informs2014.bst`). Under this configuration `\cite{}` is the textual form (natbib aliases it to `\citet`), and `\citep{}` is the parenthetical form. Three citation commands are in scope, matching the project template conventions:

- **`\cite{<key>}`** for textual citations where the author name is part of the sentence. Renders as "Foerster et al.\ (2018)". Use when the author is the grammatical subject.
- **`\citep{<key>}`** for parenthetical citations. Renders as "(Foerster et al., 2018)". Use when the citation is supporting evidence and the author is not part of the sentence.
- **`\citep[<prefix>][]{<key>}`** for parenthetical citations with a prefix. Renders as "(see Foerster et al., 2018)". Common prefixes: "see", "cf.", "e.g.".

Conventions:

- **Do not use `\citet{}`, `\citeauthor{}`, `\citeyear{}`, or other natbib variants** beyond `\cite`, `\citep`, and the prefixed `\citep[<prefix>][]`. `\cite` already produces textual output under this config, so `\citet` is redundant. The three forms above cover every needed pattern.
- **Multiple supporting citations**: one command with comma-separated keys (`\citep{a, b, c}` or `\cite{a, b, c}`), not chained `\citep{a}\citep{b}`.
- **Author-as-subject**: "\cite{foerster2018counterfactual} introduced COMA to address …". The rendered form includes the year automatically; do not duplicate the author name in prose.
- **Parenthetical evidence**: "Prior MARL work in inventory control has used hand-crafted state features \citep{a, b, c}."
- **Citation keys come from** [thesis/contents/Literature.bib](../contents/Literature.bib). **Never invent a key.** This is the writer's hard rule, restated here because it is the single most common defection.

### 5.4 Glossary and acronyms

Two glossaries are in use.

- **Acronyms** (`acr:` keys, defined in `Abbreviations.tex` via `\newacronym{acr:key}{SHORT}{long form}`): use `\gls{acr:key}` on first AND subsequent use; the package handles the long→short transition. Plural: `\glspl{acr:key}`. Sentence-initial: `\Gls{...}` / `\Glspl{...}`.
- **Notation** (custom keys defined in `Notation.tex` via `\newnotation{...}`): use `\gls{<key>}` to render the symbol.

Never write a bare acronym (`MARL`, `PPO`, `MDP`, `CTDE`) in prose. If a writer needs an acronym not yet defined, the writer adds the `\newacronym` entry to `Abbreviations.tex` before using it.

### 5.5 Math and notation

- **Sets**: calligraphic ($\mathcal{S}$, $\mathcal{A}$, $\mathcal{O}$).
- **Vectors**: bold italic via `\bm{}` from the `bm` package ($\bm{s}$, $\bm{a}$). Avoid `\mathbf{}` (upright bold) for vectors.
- **Matrices**: bold uppercase italic ($\bm{W}$, $\bm{D}$).
- **Random variables**: italic capital where convention requires it ($S_t$, $A_t$), realizations lowercase.
- **Number sets**: $\mathbb{R}$, $\mathbb{N}$, $\mathbb{R}_{\ge 0}$.
- **Operators**: $\mathbb{E}$, $\Pr$, $\arg\max$, $\arg\min$, $\mathrm{Var}$. Always with the appropriate macro, never bare $E[\cdot]$ or $argmax$.
- **Standard RL/MARL symbols** (do not redefine): state $s$, action $a$, reward $r$, return $G$, policy $\pi$, value $V$, action-value $Q$, advantage $A$, transition kernel $P$, discount $\gamma$, actor parameters $\theta$, critic parameters $\phi$, joint action $\bm{a} = (a_1, \dots, a_N)$, joint observation $\bm{o}$.
- When introducing a symbol for the first time, add a `\newnotation{...}` entry to `Notation.tex` and reference via `\gls{<key>}`. Inline-only symbols (used in a single equation) may skip the glossary entry.

**Display equations and multi-line forms.**

- **Single display equation**: use the `equation` environment with a `\label{eq:...}`. Reference via `\eqref{eq:...}` (see §5.2).
- **Multiple aligned equations** (e.g., a system, or a list of constraints): use the `align` environment with one label per line. Each labeled line is independently referenceable.
- **One equation broken across lines**: use the `split` environment inside `align` (or `equation`). The equation number is centered vertically across the lines, and there is exactly one `\label{eq:...}` per `split`.
- **Inline math** stays in `$ … $`. Long inline expressions that break across lines should be promoted to a display equation.
- **Unnumbered display equations** (rare) use the starred form (`equation*`, `align*`) and carry no label.

### 5.6 Numbers and units

- Use `siunitx` for numbers and units: `\num{1234}`, `\SI{200}{\milli\second}`, `\SI{95}{\percent}`.
- Two decimal places for percentages, returns, and reward-scale statistics, four for p-values when reported, integers for counts.
- Do not mix `\num{}` and bare numerals in the same table.
- For approximate values, use $\approx$, not "around" or "~".

### 5.7 Tables and figures

- `booktabs` only: `\toprule`, `\midrule`, `\bottomrule`. No vertical rules. No `\hline`.
- **Caption position**: above tables, below figures. Standard LaTeX convention; non-negotiable.
- **Placement**: `[t]`, `[b]`, `[tb]`, or `[bt]`. Never `[h]`, `[H]`, or the `!` modifier in body content. Tables and figures must be referenced in the body **before** they appear.
- **Caption form**: `\caption[<short caption>]{<long caption>}`. The short caption (in brackets) appears in the List of Figures / List of Tables and is one phrase. The long caption appears at the float and follows the rules below.
- **Long-caption form**: first sentence states what is shown. Second, if needed, names the takeaway. Captions end with a period.
- **Seed-evaluation plot captions** state the metric, the aggregation across seeds, and the band semantics. Default phrasing: "Lines show the mean across [N] seeds. Bands are the [IQR | 95\% bootstrap CI | $\pm$1 SD]." Any deviation from the default band statistic is named explicitly. The number of seeds is stated even when it is the project default.
- **Result-table captions** state the metric, the aggregation, and the number of seeds. Default phrasing: "Mean $\pm$ standard deviation across [N] seeds." Tables that report a different statistic (median, max-over-seed, etc.) name it explicitly.
- **Source attribution.** When a figure or table is taken from or based on another work, name the source at the end of the long caption with a parenthetical `\citep{}`. Use "data from \citep{...}" for raw data, "adapted from \citep{...}" when the figure has been modified, "reproduced from \citep{...}" when copied verbatim with permission. Example: "Street network of Paris and surroundings -- data from \citep{OpenStreetMap2020}, visualized using QGIS \citep{QGISProject2020}."
- Figure files live under `thesis/images/`. Tables either inline (in `.tex`) or as `.tex` fragments under `thesis/tables/`.

### 5.8 Punctuation and small marks

- **En dash** `--` for numeric ranges (`pp.~10--20`, `5--7 warehouses`).
- **Em dash** `---` allowed for parenthetical interruption inside a single clause. The writer's prohibition on em dashes joining two independent clauses still applies.
- **Quotation marks**: `` ``opening'' ``, never straight `"`.
- **Non-breaking ties**: `\autoref{tab:cell-grid}~(b)`, `\SI{200}{\milli\second}`, `Algorithm~\autoref{alg:mappo-train}` — keep tightly-bound tokens together with `~`.
- **Et al.**: `et al.\ ` (the `\ ` is required so LaTeX does not treat the period as sentence-final).

### 5.9 Hyperparameters and configuration tables

- **Full hyperparameter tables live in the appendix.** One table per algorithm and per configuration combination that needs documenting. Each appendix table has a `tab:` label of the form `tab:hyperparams-<algorithm>-<config>`.
- **Tuning ranges** (when search spaces are reported) are written as closed intervals with the sampling rule named: "$\log$-uniform over $[10^{-5}, 10^{-3}]$", "uniform over $\{1, 2, 4, 8\}$".

## 6. Survey-mode adjustments

These overrides apply only to subsections with `Section type: survey` in the outline.

- Tense: past for what each paper did ("\cite{foerster2018counterfactual} proposed …"); present perfect for the collective state ("The cooperative MARL literature has converged on …").
- No first-person plural, no thesis-side claims ("we propose", "our approach"). The closing synthesis paragraph may name a collective gap descriptively: "The surveyed literature does not address X."
- Each cited paper introduced in the outline must appear with surrounding explanation of what it contributed — citations are not decoration.

## 7. Defer to the writer agent for

- Gap markers and their verbatim format.
- Topic-sentence discipline; transition discipline; banned paragraph openers (`Furthermore`, `Additionally`, `Moreover`, `Also`).
- One main idea per paragraph; no single-sentence paragraphs; no section preambles or summaries.
- The em-dash / semicolon / participle prohibition for joining independent clauses.