---
name: bibtex-fetch
description: Fetches a verified bibtex entry from a canonical source given a paper title, DOI, arXiv ID, or URL. Appends the entry to thesis/resources/references.bib with a `% source:` comment. Use when the user wants to add a paper to the thesis bibliography.
---

# bibtex-fetch

## When to use

The user wants to add a paper to `thesis/resources/references.bib` and has provided one of:

- A title (e.g., *"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"*)
- An arXiv ID (e.g., `2103.01955`)
- A DOI (e.g., `10.1287/opre.2020.2071`)
- A canonical URL (proceedings page, arXiv abs page, ACL Anthology entry)

## What to do

1. **Resolve to a canonical source URL.** Order of preference:
   - arXiv abs page: `https://arxiv.org/abs/<id>`
   - DOI via crossref: `https://api.crossref.org/works/<doi>` — returns structured JSON; convert to bibtex
   - NeurIPS / ICML / ICLR / AAAI proceedings page
   - ACL Anthology entry page
   - Publisher landing page (Springer, Elsevier, IEEE, ACM, INFORMS, etc.)
   - Google Scholar cite export — last resort only

2. **Fetch the bibtex** from the canonical source via `WebFetch`. Most arXiv abs pages expose bibtex via the "Export Bibtex" / `bibtex.cgi` link; crossref returns structured JSON; conference proceedings usually link a `.bib` file.

3. **Validate the entry**:
   - Required fields present: `author`, `title`, `year`, and a venue field (`booktitle` for conferences, `journal` for journals).
   - Author names are correctly LaTeX-escaped (`\"o`, `\'e`, etc.).
   - The cite key follows the convention used elsewhere in `references.bib` (typical: `lastname<year><firstword>`, e.g., `yu2022mappo`).

4. **Check for duplicates.** Read `thesis/resources/references.bib` and search for the same DOI, arXiv ID, or title. If already present, do **not** add a duplicate — report the existing key.

5. **Append the entry** to `references.bib` with `% source: <URL>` as the first line of the entry. Format consistently with existing entries (indentation, field order).

6. **Report back** with the cite key, the source URL, and a one-line confirmation.

## Hard rules

- **Never compose a bibtex entry from training-data memory.** Every entry must come from a fetched, verifiable source. If fetching fails, return `BIBTEX NOT FETCHABLE` and explain what you tried.
- **Never modify existing entries** in `references.bib` without explicit user approval.
- **Always include the `% source: <URL>` comment.** Entries without it will be flagged by the audit.

## Output

When successful:

```
Cite key: <key>
Source: <URL>
Status: ADDED  |  ALREADY PRESENT (existing key: <key>)
Entry:
@inproceedings{...
  ...
}
```

When unfetchable:

```
Status: BIBTEX NOT FETCHABLE
Searched: <list of URLs/sources tried>
Reason: <why each failed>
Suggestion: ask user for a manual paste, or check a different identifier (DOI vs arXiv vs title)
```
