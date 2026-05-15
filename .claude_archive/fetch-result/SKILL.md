---
name: fetch-result
description: Resolves [NUMBER PENDING: ...] markers in a thesis draft by orchestrating the result-fetcher subagent for each marker. Replaces resolved markers with LaTeX-formatted numbers; leaves unresolved markers in place with a note. Use when a section has been drafted and its experimental numbers need to be filled in from runs/.
---

# fetch-result

## When to use

A thesis draft contains one or more `[NUMBER PENDING: <description>]` markers and the user wants them resolved with values from `runs/`.

## Workflow

1. **Identify the markers.** Read the target file (or files) and extract every `[NUMBER PENDING: ...]` marker with its location and description.
2. **Group by experiment.** If multiple markers reference the same run, fetch the run once and reuse the values.
3. **For each marker (or group):**
   a. Spawn the `result-fetcher` subagent via the Agent tool, with the description as input.
   b. **On found value:** replace the marker with the LaTeX-formatted number via `Edit`. Append a `% source: <run path>` comment on the same line for traceability.
   c. **On NOT FOUND:** leave the marker in place and append `% NOT FOUND: <searched paths>` after it. Do not estimate or substitute a "close enough" value.
4. **Verify formatting consistency** before finalizing: scan the modified file for plus-minus convention, significant figures, and units. Flag any disagreement with existing tables for the user.
5. **Summarize:** for each marker, list its resolution status and source file.

## Hard rules

- **Never substitute an estimate.** A NOT FOUND marker stays.
- **Preserve traceability.** Every replaced number gets a `% source:` comment pointing at the run path. Reviewers and future-you need to verify.
- **One number, one source.** If a number aggregates over seeds, the source is the aggregation file (e.g., `seed_evaluation_summary.yaml`), not an individual seed file.
- **Match existing formatting** before committing the first replacement. Read one or two existing tables in `thesis/contents/` first. Do not invent a new convention.

## Output

For each marker, in order:

```
Marker: [NUMBER PENDING: <description>]  at <file>:<line>
Status: REPLACED with <value>  |  NOT FOUND
Source: <run path>  |  (searched paths if not found)
```

End with a one-line summary:

```
Resolved: X of Y markers.  Not found: Z.
```
