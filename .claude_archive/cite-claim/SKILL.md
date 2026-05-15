---
name: cite-claim
description: Resolves [CITE: ...] markers in a thesis draft by orchestrating the lit-researcher subagent for each marker. Presents candidate papers, asks the user to pick or skip, and (on approval) replaces the marker with \cite{key}. Use when a section has been drafted and the user wants its citation gaps filled in.
---

# cite-claim

## When to use

A thesis draft contains one or more `[CITE: <claim>]` markers and the user wants them resolved with verified citations.

## Workflow

1. **Identify the markers.** Read the target file (or files) and extract every `[CITE: ...]` marker with its location and claim text.
2. **Process one marker at a time.** Batch resolution loses the user's context for picking the right citation; one-by-one is non-negotiable.
3. **For each marker:**
   a. Use the Agent tool with `subagent_type: lit-researcher`. Pass the claim as input plus one paragraph of surrounding context so the researcher understands how the citation will be used.
   b. Present the returned candidates to the user — their fit rating and supporting quote each.
   c. Ask the user to pick one, request a re-search with a tweaked claim, or skip the marker.
   d. **On pick:** if the bibtex entry is not yet in `thesis/resources/references.bib`, invoke the `bibtex-fetch` skill to add it. Then replace the `[CITE: ...]` marker with `\cite{key}` via `Edit`.
   e. **On skip:** leave the marker in place. Do not invent or force-fit a citation.
   f. **On re-search:** loop back to step 3a with the refined claim.
4. **Summarize at the end:** for each marker, list the resolution (cited / skipped / no candidate), the resulting cite key, and whether a new bibtex entry was added.

## Hard rules

- **Never silently swap a marker for a fabricated citation.** A skipped marker stays as a marker.
- **Always present at least one supporting quote** from the abstract before the user picks. If `lit-researcher` returns "no candidates," do not push a weak one.
- **Cross-check duplicates.** If the chosen paper is already in `references.bib`, reuse the existing key instead of adding a second entry.
- **One marker, one decision, one edit.** Do not collapse the loop into batch resolution.

## Output

For each marker, in order:

```
Marker: [CITE: <claim>]  at <file>:<line>
Candidates surfaced: N (or "none")
User decision: cited as \cite{<key>}  |  skipped  |  re-searched
```

End with a one-line summary:

```
Resolved: X of Y markers.  Skipped: Z.  Added to references.bib: K.
```
