# Demo Runbook

Compiled 2026-08-30, for a live presentation to a federated learning
specialist. Every command below was actually run during compilation, not
assumed — timings, output, and failure modes are what was observed on this
machine, not estimates. Re-verify if presenting from a different machine or
after further changes to the repo.

**Two Python interpreters exist in this repo. Know which one a script needs
before you run it, live, in front of an audience:**

| Interpreter | Has |
|---|---|
| System Python (`python`, or wherever your `python`/`python3` resolves) | numpy, pandas, sklearn, matplotlib — **no pennylane** |
| `.venv\Scripts\python.exe` (repo-local venv) | everything above, **plus pennylane** (0.45.1) |

Anything that imports `models_vqc.py` (directly, or indirectly via another
script) needs the venv. Everything else works with either. Marked per
command below.

---

## Task 1 — Demo path, verified by execution

### 1a. Dirichlet partitioner: client class balance at α=100 vs. α=0.1

```bash
python scripts/partitioner.py
```

- **Interpreter:** system Python (no pennylane needed).
- **Time:** ~9.5s, measured.
- **Expected output:** prints the natural 4-site table first, then one table
  per α (100, 1.0, 0.5, 0.1) showing `n`, `n_class_0`, `n_class_1`,
  `pct_class_1` per client, then two lines confirming the figures were
  saved. The α=100 table is the clean part to point at live — all four
  clients land in a tight 51–60% class-1 band. The α=0.1 table is the
  payoff: clients at **0.0%, 99.8%, 0.0%, 2.4%** class 1 — two clients are
  functionally single-class. This is real output from this run, not a
  cherry-picked example; α is fixed (`SEED = 0` in the script), so it will
  reproduce identically every time.
- **Also regenerates:** `results/figs/partition_alpha_{100,1.0,0.5,0.1}.png`
  and `results/figs/partition_natural_vs_dirichlet.png` — harmless, these
  are already the current committed figures (Task 4 below).
- **What failure looks like:** if run from anywhere other than the repo
  root, `ModuleNotFoundError: No module named 'data_loader'`
  (`sys.path.insert(0, "scripts")` at the top of the script assumes the
  working directory is the repo root). If `data/raw/` is missing or moved,
  a `FileNotFoundError` naming the missing site file. Neither has been
  observed in this session — the script ran cleanly from `D:\qml-ehr`.

### 1b. Short federated run with visibly descending loss, under a minute

**No existing committed script did this for a classical model before this
pass** — `LogisticRegressionModel`/`MLPModel` have no built-in `.loss()`
(only `VQCModel` does, for its own sanity check, see below). Rather than
invent a one-off terminal command, a small script
(`scripts/demo_mlp_loss.py`) was added this session, built entirely from
already-tested pipeline pieces — same `run_federated`, same `MLPModel`,
same `dirichlet_partition` used everywhere else in this repo; only the
per-round loss printing is new, and it's the standard binary cross-entropy
definition, computed directly from `predict_proba`.

```bash
python scripts/demo_mlp_loss.py
```

- **Interpreter:** system Python.
- **Time:** ~4.2s total (verified), training itself under 0.1s — the rest is
  Python/import startup.
- **Expected output:** `client sizes: [178, 170, 174, 214] (guard attempts:
  1)`, then 20 lines `round N/20  loss=X.XXXX  (Ys elapsed)`. Loss goes
  **0.6252 → 0.4866** over 20 rounds (decrease 0.1385). **Round 11 ticks
  back up to 0.5407 before continuing down** — this is real full-batch
  federated-averaging dynamics (a round's aggregated step overshooting
  slightly), not a bug and not smoothed out for the slide. If asked, say so
  plainly rather than let it look like an unexplained artifact — a
  specialist will notice a suspiciously monotonic curve is more suspect
  than a real one with a visible bump.
- **What failure looks like:** same CWD-dependency as 1a
  (`ModuleNotFoundError`). No other external dependency (no venv, no saved
  model files).

### 1c. CCE module working end to end

**Closest clean entry point:** `scripts/shared_test_worst_group.py`. It is
not a "CCE demo" by name (it predates the module and was refactored to
import from it — P-023), but it is a genuine end-to-end exercise of two of
CCE's three exposed functions (`cce_fixed_partition`,
`cce_worst_group_accuracy`) against real, freshly-trained models — not
synthetic numbers.

```bash
.venv\Scripts\python.exe scripts\shared_test_worst_group.py
```

(or `./.venv/Scripts/python.exe scripts/shared_test_worst_group.py` from
Git Bash)

- **Interpreter: must be the venv** — this script imports `VQCModel` for
  its (small) VQC portion.
- **Time:** ~13.4s total, measured (LR ~2.2s, MLP ~5.9s, VQC n=2 ~3.4s more).
- **Expected output:** three blocks. `=== LR ===` →
  `worst-group a100=0.6962 a0.1=0.6968 decline=-0.06pp`. `=== MLP ===` →
  `worst-group a100=0.6990 a0.1=0.6489 decline=5.00pp`. `=== VQC (n=2 only,
  see note) ===` → `decline=-0.81pp, n=2`, plus a per-replicate dump. These
  are the exact published P-007 numbers (`docs/shared_test_validation.md`)
  — reproducing them live is a genuine correctness demonstration, not
  narration.
- **For the arithmetic layer specifically** (the part your guide will
  recognise as "the protocol," i.e. `cce_paired_estimate`), the module's
  own smoke test is faster and simpler if training-then-evaluating is more
  than you want to show:
  ```bash
  python scripts/composition_controlled_eval.py
  ```
  Runs in ~1.7s (system Python, no training at all — pure arithmetic on
  the already-known MLP headline numbers), prints one JSON object:
  `observed_decline_pp: 18.46, ..., decomposition_share_pct: 27.0, ...,
  implied_composition_share_pct: 75.0, interaction_pp: 8.85,
  clears_reporting_floor: true` — the exact numbers in §V-A's table.
- **What failure looks like:** wrong interpreter → `ModuleNotFoundError: No
  module named 'pennylane'`, immediately, no partial output (fails at the
  `import` line before any client-size line prints). Missing
  `results/angle_capture/arm4_{100,0.1}_{0,5}_0.npz` (four files, confirmed
  present this session) → the LR and MLP blocks print successfully, then a
  `FileNotFoundError` partway through the VQC block — if this happens live,
  the honest recovery is "the classical CCE result already printed
  correctly; the VQC portion needs a saved checkpoint that isn't present on
  this machine," not silently rerunning.

**If asked for something more clearly a full pipeline run of CCE against
dataset 2** (the paper's actual headline configuration, K=130): that exists
(`scripts/dataset2_decomposition.py`) but is **not fast** — a single
K=130 configuration takes 3–6 minutes per model (measured this session:
LR K=130 ~178s, MLP K=130 ~327s, for 10 seeds × 5 folds × 3 α conditions
each). Not a live-demo candidate; say so if asked rather than starting it
and stalling.

---

## Task 2 — Figures, in presentation order

At most six, per your ask (five image files; two are shown as a matched
pair). Each verified against its supporting text this session.

| # | File | What it demonstrates |
|---|---|---|
| 1 | `results/figs/partition_natural_vs_dirichlet.png` | Sets up the independent variable: class balance per client, natural 4-site split alongside Dirichlet draws at α=100→0.1 — the natural split is visibly mild (close bar heights, mixed classes); α=0.1 produces near-single-class clients. |
| 2 | `results/figs/worst_client_accuracy_vs_alpha.png` **+** `results/figs/global_accuracy_vs_alpha.png` (show together — they share a y-axis range on purpose) | The motivating anomaly: worst-client accuracy declines visibly as α falls; global accuracy over the same sweep barely moves. Same data, two statistics, one diverges. |
| 3 | `results/figs/composition_decomposition.png` | Dataset 1's decomposition: for LR and VQC, the composition-only bar accounts for nearly all of the observed-decline bar (VQC's residual is negative); for MLP, the residual bar dominates — the one family with a real training effect on this dataset. |
| 4 | `results/figs/client_count_composition_share.png` | The headline confirmed prediction: shared-test-implied composition share rises with client count (α=0.5: 87–91%→96–98%), reaching 97.9% at K=130 for the MLP — the number your abstract now leads with. |
| 5 | `results/figs/alpha_calibration_fine.png` | Methodological rigor: the natural partition's equivalent-α, from a 25-point grid × 30 seeds × two divergence statistics × two aggregations (mean/max), converging to α∈[1.0,1.7] rather than a single guessed value. |

**Cut from the six:** `results/figs/client_divergence_vs_alpha.png`
(FedProx's mechanism figure — real and correct, but FedProx is a secondary
arm; only bring it up if she asks why FedProx doesn't recover more of the
MLP's residual).

**Stale / contradicts text / invites an unanswerable question — flagged
directly, per your ask:**

- **None of the five are stale as of this session** — figures 2, 3, and 5
  were regenerated this session specifically to fix a legend that *was*
  stale (see Task 5). All five were checked against their current caption
  text and match.
- **Figure 4 (`client_count_composition_share.png`) will invite a fair
  question and you have the answer ready, but expect it.** The left panel
  (α=1.0) plots K=4 bars for LR (~77%) and MLP (~75%) with wide error
  bars — these are **below this paper's own 2 percentage-point reporting
  floor** (§III-C) and are *not* cited as percentages anywhere in the
  prose; only the right panel (α=0.5, where both K clear the floor) backs
  the confirmed-prediction claim. The figure caption says this explicitly.
  If she looks at the chart before reading the caption, the natural
  question is "why does the text not use these numbers" — you have a
  one-sentence answer (denominator too small, ratio unstable, §III-C), but
  it's worth having it ready rather than discovering the question live.
- **Figure 2's two-panel comparison uses a shared, un-zoomed y-axis
  (0.0–0.8)**, deliberately — a specialist may notice most of the vertical
  space is empty and ask why the axis isn't tightened. The answer is that
  a zoomed axis would visually exaggerate the global-accuracy line's small
  wobble into something that looks like real movement; the flat/declining
  contrast is the honest picture at true scale. Worth stating before she
  asks, not after.

---

## Task 3 — Code worth opening

Three files, specific ranges — not whole-file scrolling.

1. **`scripts/composition_controlled_eval.py`** — your own instinct was
   right, this is the one to lead with.
   - **Lines 1–59**: the module docstring. Reads as a protocol description
     for an adopter, not internal notes — states the confound, the
     correction, and how to apply it to a new study in one paragraph.
   - **Lines 131–187** (`cce_paired_estimate`): the arithmetic, including
     the docstring justification for the 2pp floor
     (`share = 100 - 100*(TE/observed)`, unbounded as `observed -> 0`) —
     if she asks "why a floor, isn't that arbitrary," this function's
     docstring is the answer, already written down.

2. **`scripts/federated_loop.py`, lines 29–63** (`run_federated`) — this is
   the strongest single piece of code to show a specialist, because it's
   short (35 lines) and *provably* model-agnostic: it calls exactly four
   things on whatever `model_factory()` returns
   (`get_params`/`set_params`/`fit`/`predict_proba`) and one aggregator
   function, and nothing else — no `if model_type == "quantum"` anywhere.
   The module docstring (lines 1–6) states this as a design constraint
   directly. This is the file that answers "how do you actually swap in a
   quantum model without a parallel code path" concretely, in under a
   minute of reading.

3. **`scripts/models.py`** (all 43 lines) — the shortest, cleanest
   implementation of the four-method interface (`LogisticRegressionModel`,
   lines 12–42). Good as the concrete anchor right after showing
   `federated_loop.py`'s abstract call sites — "here is what one of those
   four methods actually does."

**What does not read well, don't open live:**

- **`docs/decisions.md`** (161 KB) and **`docs/labbook.md`** (~109 KB) —
  the project's real audit trail, genuinely valuable, but they are
  append-only session logs, not documents anyone reads start to end. If
  she wants to see the reasoning behind a specific number, search for the
  decision ID in `docs/decisions_index.md` first and jump to that entry —
  do not scroll.
- **`scripts/partition_size_robustness.py`** and
  **`scripts/dataset1_reeval_balanced.py`** — both correct and both
  heavily verification-driven (multiple metrics computed in parallel,
  per-script RNG offsets, dual plain/balanced-accuracy paths for
  traceability). Necessary and already explained in their own docstrings,
  but dense — they read like what they are, forensic re-verification
  scripts, not designed-for-reading demonstration code. If asked to justify
  a specific robustness number, cite the result file
  (`results/partition_size_robustness.json`) and the relevant paper table
  rather than opening the script.

---

## Task 4 — Pre-rendered figure backups

All five figures named in Task 2, plus the two supplementary ones cut from
the six, are current PNGs already committed under `results/figs/` — no
separate backup folder was created, since that would just be a second copy
to keep in sync. Confirmed present and current this session (regenerated
where needed, verified unchanged where not):

| File | Status |
|---|---|
| `results/figs/partition_natural_vs_dirichlet.png` | Regenerated this session (deterministic, `SEED=0` — identical content, refreshed bytes) |
| `results/figs/worst_client_accuracy_vs_alpha.png` | Regenerated this session (legend fix, see Task 5) |
| `results/figs/global_accuracy_vs_alpha.png` | Regenerated this session (same legend fix) |
| `results/figs/composition_decomposition.png` | Regenerated this session (unaffected by the legend bug, refreshed incidentally) |
| `results/figs/client_count_composition_share.png` | Not regenerated — verified its underlying data (`results/dataset2_decomposition_weighted.json`) is unchanged and bit-identical to what the figure was built from |
| `results/figs/alpha_calibration_fine.png` | Not regenerated — unaffected by any change this session, still matches §V-D |
| `results/figs/client_divergence_vs_alpha.png` | Regenerated this session (same legend fix), cut from the six but ready if asked about FedProx |

If a live command fails, open the corresponding PNG directly — every file
above is what that command's own output produces, not a hand-picked nicer
version.

---

## Task 5 — What's fragile (blunt, as asked)

1. **`paper/06_limitations.md`'s first bullet is factually wrong if she
   reads past the header.** It says the feature set was "reduced to 6 via
   PCA." Dataset 1 uses 6 *raw* features with no PCA at all (§IV.A of the
   canonical draft, D-021) — this is the pre-D-021 design, superseded
   months ago in every other document. The file's own header (added this
   session) flags this explicitly and says it wasn't carried into the
   canonical draft — but the header is three lines above a bullet that
   still says the wrong thing, in a file that still exists and is still
   readable. If she opens this specific file and reads only the bullet,
   not the header, she will see something contradicted by every other
   document in the repo, including the paper she's about to be shown.
   **Do not open this file live.**

2. **`paper/02_related_work.md` and `paper/06_limitations.md` still
   contain more than was merged.** Only the genuinely new, still-accurate
   content from these two files was reconciled into the canonical draft
   (P-023). Both files are longer than what made it in. If she reads
   either past the superseded-header, she may find claims (the "two of
   three model families" composition framing in `02_related_work.md`, for
   instance — since revised to "all three" and now further reframed around
   CCE) that directly contradict the current paper. Both are marked
   superseded; neither is edited to remove the contradiction.

3. **Two different VQC timing figures exist in committed text, on
   purpose, unreconciled by design.** The paper says 13,300× throughout
   (standardized this session). `docs/arm4_report.md`, `docs/decisions.md`,
   and `docs/labbook.md` — the actual measurement logs — all say ~13,318×,
   and were deliberately *not* rewritten, since they're this project's
   append-only measured record. If she cross-references the paper against
   `docs/arm4_report.md` she will find a literal number mismatch. The
   explanation (13,300× is an intentional, documented rounding; 13,318×
   is what was actually logged) is written down (P-023, Flag 9) but she
   has to be told, not left to find it and wonder if it's an error.

4. **Flag 3 in the paper's own review-flags section is genuinely open, not
   resolved.** [12] A2G-QFL's description in §II-E covers only half that
   paper's actual contribution (the geometry gain; not the QoS/latency
   gain) — flagged explicitly as "assigned to Ayuvi, not touched this
   pass." If she has read A2G-QFL and asks about its QoS-based gain, the
   honest answer is that this draft doesn't currently address it, and
   knows it doesn't.

5. **The natural-partition α-calibration has a real, still-open internal
   inconsistency, not a stale-figure problem this time — an unreconciled
   pair of genuinely different measurements.** `docs/decisions.md` D-037
   (2026-08-19, frozen, never edited per this project's append-only
   convention) informally estimated the natural partition at "~α 0.5–1.0"
   by matching downstream metrics (worst-client accuracy, parameter
   divergence). The current, rigorous calibration (P-016, §V-D) gives
   α∈[1.0,1.7] by a completely different method (label-distribution
   distance). Both are labeled correctly for what they measure and neither
   has been shown to be wrong — but they have never been reconciled into
   one statement of "how heterogeneous is the natural partition," and the
   paper says so directly (§VII, Flag 1). If asked "so which is it," the
   honest answer is that this is a genuinely different question asked two
   ways, not yet resolved into one number — not "we don't know."

6. **Two RNG seed bases (200,000 and 300,000) exist beyond the one in the
   CCE module's own docstring example (100,000), for a real but
   easy-to-misread reason.** `scripts/composition_controlled_eval.py`'s
   `cce_fixed_partition` takes a `base_seed` parameter specifically because
   three different pre-existing scripts (`shared_test_worst_group.py`,
   `dataset2_decomposition.py`, `dataset1_reeval_balanced.py`) each already
   had their own historical seed offset before the module existed, and
   consolidating them into the module without breaking already-published
   numbers required preserving all three rather than picking one. This is
   documented in the module docstring and in P-023 — but if she opens two
   of the three calling scripts side by side and notices the numbers
   differ without reading the comment explaining why, it looks arbitrary.

7. **Dataset 2's models are a separate, parallel set of classes
   (`models_weighted.py`), not the same `LogisticRegressionModel`/
   `MLPModel` used for dataset 1**, because class-weighted training had to
   be added without silently changing dataset 1's already-published,
   already-cited numbers (P-013). This is the right call and it's
   documented, but "why didn't you just reuse the same model class" is a
   fair, likely question, and the honest answer involves explaining a
   scope decision made mid-project, not just pointing at code.

8. **`docs/README.md`'s evidentiary rule was rewritten this session to
   stop naming `results/runs.csv` as the single traceability file — because
   it wasn't true.** `results/runs.csv` has 31 rows, mostly early timing
   spikes. Every number actually cited in the paper's headline results
   traces to a different, specifically-named file
   (`composition_decomposition_summary.csv`,
   `dataset2_decomposition_weighted.json`, `alpha_calibration_fine.json`,
   `partition_size_robustness.json`, and others). This is now fixed in
   `docs/README.md` itself, but if she has seen an earlier version of this
   repo, or reads `CLAUDE.md`'s own project-instructions file (which still
   states the `results/runs.csv`-only rule verbatim, unedited, since it's
   the standing instructions file, not a report), the discrepancy is real
   and worth naming before she finds it.

9. **The `paper/01`, `03`, `04`, `05`, `07` scaffold files are near-empty
   stubs** (a heading and an HTML comment, nothing else) now carrying a
   superseded-header pointing at the canonical draft. Harmless if opened —
   there's nothing wrong in them, because there's nothing in them — but if
   she's looking for "the introduction" or "the methodology" and clicks
   `01_introduction.md` before finding `paper_draft_v2.md`, she'll see an
   empty file with a redirect notice, which reads as unfinished even
   though the actual content exists elsewhere and always has.

10. **Git history includes a rewritten range** (commits `be78b85` through
    the pre-P-023 tip had an AI co-author trailer stripped from their
    messages, per your explicit instruction, changing their hashes). This
    is invisible in the current tree and in `git log`'s normal output, but
    if she runs `git log --all` or inspects reflogs on a machine that still
    has the old refs cached, the history could look like it was altered —
    which it was, deliberately, on your instruction, not something to be
    caught off guard by if she asks about the commit history's shape.
