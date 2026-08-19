# Documentation Index

This directory is the project's working record — not polish added after
the fact. If a number or a design choice can't be traced to something in
here (or to `results/runs.csv`), it doesn't go in the paper.

## Files

| File | What it is | What belongs in it |
|---|---|---|
| [`decisions.md`](decisions.md) | The methodology's source of truth | Every non-obvious choice: what was decided, when, why, and what alternatives were rejected. Append-only — superseded entries are marked as superseded, never deleted or edited away. |
| [`decisions_index.md`](decisions_index.md) | Navigation table for `decisions.md` | One row per decision ID: date, one-line title. Update this whenever a new `D-NNN` entry is added to `decisions.md`. |
| [`labbook.md`](labbook.md) | Dated session log | What ran, what happened, what broke, what surprised us. Ugly is fine — this is where "we observed X" claims in the paper come from. One entry per working session. |
| [`INTERFACE.md`](INTERFACE.md) | Frozen contracts | Exact signatures, array shapes, and dtypes for the model interface, the aggregator interface, and the federated loop's call contract. Update only when the frozen contract itself changes (rare, deliberate) — not for every new arm built against it. |
| [`../RUNNING.md`](../RUNNING.md) | Operational reference | For every component: the exact command to run it, the expected output, and known failure signatures. Lives at the repo root (not in `docs/`) since it's the first thing anyone reproducing the pipeline needs. |
| [`reference/`](reference/) | External planning documents | Materials produced outside this repo during planning — dataset brief, quantum primer, literature summaries, Review-1 materials, Review-2 criteria. Added as they're brought in; each gets a line here once present. |

## Commit convention

```
feat: <what was built>
fix: <what was corrected>
docs: <what was recorded>
exp: <what was run, and the headline number>
```

`exp:` commits should reference the relevant decision ID where one
applies (e.g. `exp: Arm 2 alpha sweep, gap to Arm 1 = 0.22pp (D-024)`), so
a result in `results/runs.csv` can be traced back to both the commit that
produced it and the decision that shaped how it was produced.

## Push discipline

At the end of every working session:
1. Append the session entry to `labbook.md`.
2. Append any new decisions to `decisions.md`, and add their rows to
   `decisions_index.md`.
3. Update the Status section of `../README.md` if what's implemented
   changed.
4. Stage, commit with a conventional message (above), and push.
5. If the session produced experimental results, `results/runs.csv` is
   committed in the *same* push as the code that produced them — numbers
   and code stay synchronized, never pushed separately.

## Contributor git setup (per machine, per person)

Grading credits individual contribution via commit history, so this
matters: each collaborator must configure git **locally on their own
machine** so commits are attributed to *their* identity, not shared or
collapsed into one contributor.

On each machine, each person runs (once):

```bash
git config --global user.name  "Your Name"
git config --global user.email "the-email-your-GitHub-account-uses@example.com"
```

The email must match an email address verified on that person's own GitHub
account — that's what GitHub uses to link a commit to a profile and show
it in the contributors graph. If it doesn't match, the commit still has
the right name in `git log`, but GitHub won't attribute it to that
account's profile.

Each person also needs their **own** authentication to this repo (their
own `gh auth login`, or their own SSH key, or their own credential-manager
login) — never share a token or push under someone else's cached
credentials. Two people pushing under one shared login is exactly the
failure mode that collapses attribution into a single contributor.

**Checking authorship after the fact:**

```bash
git log --author="Ayuvi"          # every commit by a given author
git shortlog -sn                  # commit count per author, summarized
```

`git shortlog -sn` is the fastest way to produce a per-author summary on
demand — one line per contributor, commit counts, sorted.

This is a standing convention, not a one-time checklist — a session isn't
finished until this has happened.
