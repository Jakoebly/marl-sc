# Thesis Writing Workflow — Session Prompts

Each section below is a template for a new main Claude session (i.e., each step is a separate session). The main session will then dispatch to the appropriate subagent(s) to do the work. This way, I can stay in the main session to review and approve.

**Inside the prompt code blocks:**
- **`<PLACEHOLDER>`**: I have to replace these before sending. 
- **`# comment line`**. A comment or instruction NOT meant to be sent to Claude. Either delete these lines before pasting or leave them (Claude will treat lines starting with `#` as markdown headings/comments).
- Inline parentheticals like `(FOLLOW EXACTLY — do not reorder...)` ARE part of the brief and are directives to the agent.

## 0. General Information

### **Conventions:**

**Folder and file conventions used by all agents:**
- Bibliography in `thesis/contents/Literature.bib`.
- Abbreviations in `thesis/contents/Abbreviations.tex`.
- Images for `\includegraphics` in `thesis/images/`.
- Paper collection in `thesis/literature/LITERATURE.md` (Optional, absence does not break the workflow).
- Notebook for figures and tables in `src/visualization/plots.ipynb`.

**In-text marker conventions used by `thesis-writer` and resolved by `lit-researcher` / `result-fetcher`:**
- `[CITE: <one-line claim description>]` means a paper is needed to support a claim. Resolved in Step 3.
- `[NUMBER PENDING: <description>]` means an experimental value is needed. Resolved in Step 4.
- `[FIGURE PENDING: <description>]` means a paper a plot/image is needed. Resolved in Step 4.
- `[TABLE PENDING: <description>]` means a results table is needed. Resolved in Step 4.
- `% TODO: ...` means a comment to the me that the agent noticed something it could not resolve on its own.

### **Workflow Legend:**
| Step | Agent / skill                | Produces                                                   | Open new session? |
|------|------------------------------|------------------------------------------------------------|-------------------|
| 1    | `outline-constructor`        | `thesis/outline/<SectionName>_outline.md`                  | yes               |
| 2    | `thesis-writer`              | inserts subsection into `thesis/contents/<NN>_<ChapterName>.tex`  | yes               |
| 3    | `/cite-claim` → `lit-researcher` + `bibtex-fetch` | resolved `\cite{key}`, updated `Literature.bib` | yes               |
| 4    | `result-fetcher` (optional `/fetch-result`) | resolved `[NUMBER/FIGURE/TABLE PENDING]` markers | yes               |
| 5A   | `coherence-checker`          | issue list (flow, transitions, bare acronyms)              | yes (same as 5B)  |
| 5B   | `thesis-reviewer`            | issue list (citations, glossary, technical, defense)       | yes (same as 5A)  |
| 6    | `adversarial-reviewer`       | hostile final review with defense questions                | yes               |

For "Recent Applications to Inventory Control" subsections, see the **Survey Subsections** variant at the bottom (steps 1B + 1C + 2B replace Steps 1 and 2, step 3 is usually a no-op, and steps 4–6 are unchanged).

---

## 1. Section Outline

> **What this does:** Invokes the `outline-constructor` subagent to produce a paragraph-level outline for a section. The subagent writes the outline directly to `thesis/outline/<SectionName>_outline.md`. You review the file in your editor and request revisions until satisfied. The outline is the locked input for Step 2.

> **Open in:** new main session.

```
I am starting to work on <SECTION NUMBER AND TITLE> of my Master's thesis on MARL for supply chain inventory management.

Use the `outline-constructor` subagent to produce a detailed paragraph-level outline for this section and write it to @thesis/outline/<SectionName>_outline.md.

Brief the subagent with the following:

---
Task: Produce a paragraph-level outline for <SECTION NUMBER AND TITLE> and write it to @thesis/outline/<SectionName>_outline.md.

Section type: standard

# Target length: (Optional, delete if not used)
Target length: <X> pages (~<Y> paragraphs)

Topic list:
<PASTE TOPIC LIST HERE>

This brainstorming list should be treated as a NON-EXHAUSTIVE MINIMUM. Every item must appear somewhere in the outline, but you are expected to add missing connective topics, definitions, transitions, and any standard topics that a section of this type would normally contain. Use @docs/SCOPE.md and your own domain judgement to decide what is missing.

Notes and constraints:
<PASTE NOTES AND CONSTRAINTS HERE>
---

After the subagent writes the file, show me the summary it returned. I will review the file directly.

If I ask for revisions, dispatch the same subagent again with the revision instructions and the path
to the existing outline file (it should `Edit` the file in place rather than rewriting from scratch
unless I ask for a fresh start).
```

---

## 2. Section Writing

> **What this does:** Invokes the `thesis-writer` subagent to draft the section from the locked outline. Run only after the outline from Step 1 is final. If tone/voice adjustments are needed afterward, re-invoke the subagent with targeted revision instructions.

> **Open in:** new main session.

```
The outline for <SECTION NAME> is finalized at `thesis/outline/<SectionName>_outline.md`.

Use the `thesis-writer` subagent to draft the section and INSERT it into the chapter file.
Do NOT overwrite existing prose unless I explicitly ask for a rewrite.

Brief the subagent with the following:

---
Task: Draft <SECTION NUMBER AND TITLE> from the locked outline and insert it into the chapter file.

Outline: `thesis/outline/<SectionName>_outline.md` (FOLLOW EXACTLY — do not reorder, merge, or split paragraphs; do not introduce beats not in the outline)

Target chapter file: `thesis/contents/<NN>_<ChapterName>.tex`

Insertion point: <end of file | after \subsection{<PreviousSubsectionName>}> 
Before writing, check the target file for an existing \subsection{<SECTION TITLE>}.
If one exists, STOP and ask me — do not create a duplicate.

Voice reference: @thesis/STYLE.md and `thesis/contents/<NN-1>_<PrevChapterName>.tex`. Read both for tone and LaTeX conventions. Match the existing style of the previous chapters and refer to @thesis/STYLE.md if something is unclear.
---

When done, report:
1. The list of [CITE: ...], [NUMBER PENDING: ...], [FIGURE PENDING: ...], and [TABLE PENDING: ...]
   markers left in the draft (with file:line locations).
2. Any new `\newacronym{...}` entries added to @thesis/contents/Abbreviations.tex.
3. Any `% TODO:` comments left for me to resolve.
```

---

## 3. Literature Integration

> **What this does:** Resolves `[CITE: ...]` markers in the drafted section. Uses the `/cite-claim` skill, which drives a **per-marker loop in the main session**: identify markers → dispatch `lit-researcher` → present candidates → user picks / re-searches / skips → on pick, invoke `bibtex-fetch` (to update `Literature.bib`) and `add-literature` (to update `LITERATURE.md`) → replace the marker with `\cite{key}` → move to the next marker.

> **Open in:** new main session.

```
Run the `/cite-claim` skill on \subsection{<SUBSECTION TITLE>} inside `thesis/contents/<NN>_<ChapterName>.tex`.

Scope: only resolve [CITE: ...] markers inside that subsection. Ignore markers elsewhere in the chapter (those are handled in their own pass).

How I want the loop to run:
- List the markers you found before resolving any, in document order.
- Process one marker at a time. For each, dispatch the `lit-researcher` subagent, then present its candidates verbatim and pause for my decision (pick / re-search / skip) — do NOT pre-pick or batch.
- On pick: invoke `bibtex-fetch` if the key is not yet in `Literature.bib`, then invoke `add-literature` (Mode A) so `LITERATURE.md` is updated with cite key, key claims, and the citing section, then replace the marker with `\cite{<key>}`.
- On skip: leave the marker in place.
- End with the summary specified by the skill (resolved / skipped / added to Literature.bib / added to LITERATURE.md).
```

> Alternative prompt for a manual, single-marker literature search without the `/cite-claim` skill loop. Use when you want candidates only and will integrate the chosen paper yourself.

```
Use the `lit-researcher` subagent to find papers for this claim:

"<PASTE THE [CITE: ...] MARKER DESCRIPTION HERE>"

Context: This claim appears in \subsection{<SUBSECTION TITLE>} of `thesis/contents/<NN>_<ChapterName>.tex`, paragraph <N>. The surrounding text is:
"<PASTE 2–3 SENTENCES OF SURROUNDING PROSE>"

# Optional — delete if not used:
Constraints: <venue / time-window / exclusion>

Return up to 3 candidates with verified bibtex (`% source:` comment included), verbatim abstract excerpts, and fit justification. If the adversarial query surfaces contradicting evidence, include the Caveats block.

Do not modify the .tex file, `Literature.bib`, or `LITERATURE.md` — I will integrate the chosen paper manually (typically via `/bibtex-fetch` and `/add-literature`).
```

---

## 4. Section Review

> **What this does:** Two-pass quality check. First the `coherence-checker` reviews flow, transitions, and bare-acronym usage; then the `thesis-reviewer` reviews technical accuracy, citations, glossary correctness, and defense vulnerabilities. Run both, fix issues in the main session (dispatching back to `thesis-writer` for targeted revisions or fixing trivially yourself), then re-run until clean.
>
> **Why two passes (not one):** they have non-overlapping focus. `coherence-checker` reads prose for *flow* (what is hard to follow even when correct). `thesis-reviewer` reads it for *correctness* (what is wrong, unsupported, or vulnerable). Collapsing them produces shallower findings of both kinds. Run Pass A first because fixing flow often dissolves what looked like a technical issue, but not vice versa.

> **Open in:** new main session.

### Pass A — Coherence

```
Use the `coherence-checker` subagent to review the recently-drafted \subsection{<NEW SUBSECTION TITLE>}
inside `thesis/contents/<NN>_<ChapterName>.tex`. Focus the review on that subsection; scan the rest of
the chapter only to verify cross-references and notation consistency.

# omit either bullet if the corresponding adjacent chapter does not exist yet
The subagent should also read adjacent chapters so cross-references can be validated:
- Previous chapter: `thesis/contents/<NN-1>_<PrevChapterName>.tex`
- Next chapter:     `thesis/contents/<NN+1>_<NextChapterName>.tex`

Return the numbered issue list sorted by severity, with the verbatim quote that demonstrates each
issue. Do NOT propose specific rewrites — that's my job, dispatching back to `thesis-writer` if
needed.

After I address issues, I will ask you to re-run the check until the subsection is clean.
```

### Pass B — Technical / Citation Review

```
Use the `thesis-reviewer` subagent to review the recently-drafted \subsection{<NEW SUBSECTION TITLE>}
inside `thesis/contents/<NN>_<ChapterName>.tex`. Focus the review on that subsection; check the rest
of the chapter only for cross-section consistency and bibliography/glossary key existence.

The subagent should check for:
1. Unsupported claims (no citation, no result reference, no derivation in the thesis itself)
2. Wrong or weak citations:
   - Citation key does not exist in `thesis/contents/Literature.bib`
   - The cited paper does not actually support the claim (cross-check against LITERATURE.md if it
     exists, or against the bibtex entry's title/abstract)
   - Citation is merely topic-adjacent rather than evidence for the specific claim
3. Undefined terms and notation inconsistencies (symbol used before defined, conflicting notation
   across sections)
4. Glossary violations:
   - Bare acronyms (e.g., "MARL") that should be `\gls{acr:marl}`
   - `\gls{acr:key}` references whose `\newacronym` entry is missing from
     `thesis/contents/Abbreviations.tex`
   - Acronyms introduced without `\gls{}` on first use
5. Scope drift (claims beyond what `docs/SCOPE.md` defines, or out-of-scope items snuck in)
6. Technical errors about MARL, RL, inventory control, or the specific system in `SYSTEM_MODEL.md`
7. Weak transitions and defense vulnerabilities a committee would attack

Return the numbered issue list sorted by severity, with verbatim evidence. Do NOT propose specific
rewrites. After I fix issues, I will ask you to re-run the check until the section is clean.
```

---

## 6. Final Adversarial Review

> **What this does:** The last review before a section is locked. The `adversarial-reviewer` subagent acts as a hostile defense examiner: it challenges novelty claims against published literature (including web search), identifies experimental design weaknesses, and produces the three most damaging defense questions. Run only after Step 5 is clean.

> **Open in:** new main session.

```
The recently-drafted \subsection{<NEW SUBSECTION TITLE>} inside `thesis/contents/<NN>_<ChapterName>.tex` has passed coherence (Pass A) and technical
(Pass B) review in Step 5.

Run the `adversarial-reviewer` subagent for a final hostile review.

The subagent should:
1. Identify the three most damaging questions an examiner would ask, ranked by damage potential.
2. Challenge every novelty claim against published literature using WebSearch — flag claims that
   are anticipated by a paper not currently cited.
3. Identify experimental design weaknesses a top-venue reviewer would reject (baseline gaps,
   cherry-picked HPs, too few seeds, evaluation metrics that do not match the claim, scope of
   experiments that does not support the generality of conclusions).
4. Flag up to 3 missing related-work papers (2022–2025) that a reviewer would specifically expect
   to see cited in this section.

Return the full adversarial review report. I will decide which issues to address before locking
the section. The agent must NOT propose rewrites and must NOT compliment.
```

---

## Background Section Variant — Survey Subsections ("Recent Applications to Inventory Control")

> **When to use this variant.** The Background and Literature chapter is structured so that each subsection has two parts:
>
> 1. **Foundational Contributions** — methodological exposition (e.g., what MARL is, how it differs from single-agent RL, what CTDE is). Literature here is foundational and theoretical. Use the **standard workflow** (Steps 1–6 above).
> 2. **Recent Applications to Inventory Control** — survey of how the concept has been applied to supply chains. Literature here is application-oriented. Use the **variant below** for Steps 1–3; Steps 4–6 are identical to the standard workflow (Step 4 will usually be a no-op for these subsections since they have no experimental results).
>
> The variant replaces:
> - **Step 1** with **Step 1B + Step 1C** (paper discovery first, then outline)
> - **Step 2** with **Step 2B** (same agent, survey-style brief)
> - **Step 3** is largely unnecessary because most citations come pre-resolved from Step 1B; run it only to fill any residual `[CITE: ...]` markers.
>
> **Why no separate inventory-writer / inventory-outline-constructor agents:** the existing `outline-constructor` and `thesis-writer` both support `section type: survey` natively. Splitting them into dedicated agents would duplicate behavior and create maintenance drift. The variant works by briefing the existing agents differently, not by swapping them.

### Step 1B — Literature Discovery

> **Open in:** new main session.

```
# example PARENT SECTION TITLE: "Section 2.3 — Multi-Agent Reinforcement Learning"
I am writing the "Recent Applications to Inventory Control" part of <PARENT SECTION TITLE>.

Use the `inventory-researcher` subagent to discover the top papers for this survey subsection.

Brief the subagent with the following:

---
Task: Find the top 10–15 papers on <TOPIC> applied to supply chain / inventory control.

# example TOPIC: "Multi-Agent Reinforcement Learning for inventory replenishment and distribution"
Topic: <TOPIC>
# default 2018–2025; adjust if you need a different window
Time window: 2018–2025
Target count: 10–15 papers
# example: "multi-echelon, cooperative agents, comparison with base-stock policies"
Subtopics to emphasize: <SUBTOPICS TO EMPHASIZE>
# example: "routing, scheduling, procurement — out of scope per SCOPE.md"
Subtopics to exclude:  <SUBTOPICS TO EXCLUDE>

Mandatory output:
- Thematic groupings (2–5 clusters), one summary sentence per cluster
- For every paper: title, authors, venue/year, verified bibtex from a canonical source (with
  `% source: <URL>` comment), one-sentence contribution, relevance rating
- Flag any paper already present in `thesis/contents/Literature.bib` (do not re-discover)
- Suggested paragraph structure based on the themes
---

After the subagent returns the landscape, I will review it and pick the subset of papers to actually
cover in the section before moving to Step 1C. Do not pre-prune — give me the full landscape.
```

### Step 1C — Outline for Survey Subsection

> **Open in:** same session as Step 1B (so the inventory-researcher output is still in context).

```
I have reviewed the inventory-researcher output and selected the following papers for inclusion:

<PASTE SELECTED PAPER LIST WITH CITE KEYS, GROUPED BY THEME>

Use the `outline-constructor` subagent to produce a survey-style outline for the subsection
"<SUBSECTION NUMBER AND TITLE>". Write it to `thesis/outline/<SectionName>_outline.md`.

Brief the subagent with the following:

---
Task: Produce a survey-style paragraph-level outline for <SUBSECTION NUMBER AND TITLE> and write it
to `thesis/outline/<SectionName>_outline.md`.

# section type is fixed to "survey" for this variant
Section type: survey
Target length: <X> pages (~<Y> paragraphs)

Papers to cover (selected from inventory-researcher output, grouped by theme):
<PASTE THE THEMATIC GROUPS AND PAPER LIST WITH CITE KEYS>

Required structure:
- One paragraph per theme (or two if a theme is large). Each paragraph must list the cite keys it
  covers, in the order they should appear in prose.
- Every paper I listed above MUST appear in some paragraph by cite key. If a paper does not fit
  any theme, flag it so I can drop or reassign it.
- End with a final SYNTHESIS PARAGRAPH that names the collective gap in the surveyed literature
  that this thesis addresses.

# examples of useful constraints to add:
#   "Do not rank papers — describe contributions neutrally"
#   "Include the comparison with classical policies as a recurring theme"
#   "Aggregate descriptions of similar papers — do not give every paper its own sentence"
My additional notes:
<PASTE NOTES AND CONSTRAINTS HERE>
---

After the file is written, show me the summary. I will review the file directly and ask for revisions.
```

### Step 2B — Writing for Survey Subsection

> Use the **Step 2** prompt as the base. INSERT the block below into the brief (between the existing "Hard rules to honor" block and the closing `---` separator). The outline's `Section type: survey` already tells the agent most of what it needs, but this block keeps the survey-specific rules in front of the agent without it having to re-derive them.

```
# INSERT this into the Step 2 brief (just before the closing "---" of the brief)
Survey-style addendum (because the outline declares `Section type: survey`):

- Paragraphs describe groups of papers and their contributions — they do NOT make first-person
  thesis claims. Write in a descriptive, comparative academic register.
- Every paper named in the outline must appear with `\cite{<key>}`. Cite keys without surrounding
  explanation are forbidden — each \cite must be embedded in prose that says what the paper
  contributed.
- Use `[CITE: ...]` markers only for papers the outline left as `[CITE: ...]` (i.e., papers not yet
  in Literature.bib that the user will add later).
- Do NOT introduce thesis-side claims (e.g., "we propose"). The synthesis paragraph at the end of
  the subsection IS allowed to name the gap the thesis addresses — but framed descriptively
  ("the surveyed literature does not address X"), not prescriptively.
```
