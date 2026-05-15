---
name: section-coherence
description: Runs a coherence pass over a section (or multiple sections) by orchestrating the coherence-checker subagent. Returns the issue list and, on approval, dispatches targeted fixes to the thesis-writer subagent. Use when the user wants to check flow, transitions, and logical chain in drafted prose.
---

# section-coherence

## When to use

A thesis section — or a group of sections being assembled into a chapter — has been drafted, and the user wants to check flow, transitions, undefined terms, and logical chain. This is distinct from a correctness review (use `thesis-reviewer` for that).

## Workflow

1. **Confirm the scope.** Single section, several sections, or a full chapter? Ask if unclear. Wider scope catches cross-section issues but slows the pass.
2. **Spawn the `coherence-checker` subagent** via the Agent tool with the target file paths. The checker reads SCOPE.md and SYSTEM_MODEL.md automatically; you do not need to pass them.
3. **Receive the issue list.** Present it to the user verbatim — do not summarize it or act on it yet.
4. **Triage with the user:**
   - For each HIGH-severity item: ask whether to fix now, defer, or dismiss.
   - For MED and LOW items: offer batch-approve, batch-defer, or one-by-one.
5. **For approved fixes:** spawn `thesis-writer` via the Agent tool, one issue per invocation. Pass the location, the verbatim evidence from the checker, and the specific issue type. Let the writer decide the rewrite — do not propose specific replacement prose yourself.
6. **Re-run the check** only if the user requests verification after fixes. Otherwise stop.

## Hard rules

- **Do not propose specific rewrites in this skill.** The `coherence-checker` identifies, the `thesis-writer` fixes. Keep the roles separate.
- **One issue per writer invocation.** Asking the writer to fix six things at once distributes its attention and degrades quality.
- **Respect the checker's "highest-leverage fixes" list** when triaging. If the checker flagged three items as top priority, surface those first.
- **Do not silently apply fixes.** Every change is user-approved.

## Output

At the end of the pass:

```
Issues identified: N  (HIGH: a, MED: b, LOW: c)
Fixes applied: X
Deferred: Y
Dismissed: Z
```

If the user re-runs the check after fixes, append the new issue counts so progress is visible.
