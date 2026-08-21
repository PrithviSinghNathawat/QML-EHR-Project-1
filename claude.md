# Project: Quantum vs Classical Federated Learning on Non-IID EHR

## What this is

A simulation-only research study for a university capstone (BCSE497J, VIT Vellore).
The deliverable is a conference paper, not a product. Two undergraduates, ~50 hours each.
Review-1 has passed; we are now in the implementation phase.

We measure how increasing data heterogeneity across simulated hospitals degrades
federated learning, and whether variational quantum classifiers degrade differently
from classical models under identical conditions.

## Working mode — IMPORTANT

I am new to federated learning and quantum machine learning. I must defend this code in
an oral examination and write the methodology section myself.

For every non-trivial component:
1. **Explain first** — plain language, before any code. What does this do and why?
2. **Then minimal code** — simplest thing that works, no premature abstraction.
3. **Then check me** — ask one question about what you just wrote.

Do not write large amounts of code in one go. Stop between components so I can read them.
If I say "just write it," remind me once that I have to defend it.

Boilerplate (CSV logging, argparse, plotting, file IO) — write directly, no explanation needed.

## Documentation duties — do this without being asked

After every component built or choice made, append to:

- **`docs/decisions.md`** — every non-obvious choice: what, date, why, alternatives rejected.
  Append-only; supersede entries rather than deleting. This becomes the paper's Methodology.
- **`docs/labbook.md`** — dated session entries: what ran, what happened, what broke,
  what surprised us. Ugly is fine. This is where "we observed X" claims come from.
- **`results/runs.csv`** — one row per run, from the very first run, including throwaways.
- **`paper/06_limitations.md`** — append every constraint as we hit it, don't wait until the end.

If a number cannot be traced to a row in `runs.csv`, it does not go in the paper.

## Decision-ID convention — per-person prefixes, no exceptions

Two people work in parallel on separate branches. A shared sequential `D-NNN` counter
collided once already (2026-08-20 — both branches independently continued numbering
from the same fork point with different content; resolved by renumbering one side,
logged as P-001). **This must not happen again:**

- All existing `D-*` numbers are frozen — historical, never renumbered, never reused,
  never continued.
- Every new decision entry uses a per-person prefix: **Prithvi's entries are `P-NNN`**,
  **Ayuvi's entries are `A-NNN`**, each their own independent counter starting at 001.
- When starting work, check the highest existing `P-` (or `A-`) number already in
  `docs/decisions.md` for that person and continue from there — do not guess, do not
  reuse a number, do not renumber someone else's already-committed entries.
- Cite decisions by full ID (`D-018`, `P-003`, `A-002`) everywhere — paper prose,
  commit messages, code comments — so a citation stays unambiguous regardless of
  which branch or session produced it.

This applies automatically, without being asked, in every session from both
instances working on this project.

## Locked technical decisions — do not change without asking me

| Item | Decision | Reason |
|---|---|---|
| Dataset | Full UCI Heart Disease, all 4 sites (920 records) | Cleveland alone (303) leaves ~30 rows/client — noise |
| Target | Binarise `num > 0` → 1 | Standard for this dataset |
| Features | PCA to 6 components | One qubit per feature |
| Quantum device | `lightning.qubit` | C++ backend, much faster than `default.qubit` |
| Gradients | `diff_method="adjoint"` | 30–70× faster than parameter-shift. **Non-negotiable** — with parameter-shift this project does not finish |
| Circuit | 6 qubits, ~3 layers, angle encoding, shallow ansatz | Barren plateaus and training time are the binding constraint |
| Partitioning | Dirichlet(α), α ∈ {100, 1.0, 0.5, 0.1} | The independent variable |
| Clients | 4–5 | |
| Seeds | 3–5, fully seeded and reproducible | |
| Compute | CPU. GPU only pays off above ~20 qubits | |

## Architecture — ONE loop, not five scripts

One federated training loop parameterised by `(model_class, aggregator, alpha, seed)`.

- Arm 1: centralized classical (no federation) — upper bound
- Arm 2: classical + FedAvg — naive baseline
- Arm 3: classical + FedProx — classical robustness
- Arm 4: VQC + FedAvg — **headline arm**
- Arm 5: VQC + circular-mean aggregation — ablation, timeboxed

Arm 2→3 is a loss change. 2→4 is a model swap. 4→5 is an aggregator swap.
Five separate scripts means the design is wrong.

## Interface contract — my teammate builds against this

Every model, classical or quantum, implements exactly:

```python
class Model:
    def get_params(self) -> np.ndarray: ...      # flat 1-D float array
    def set_params(self, vec: np.ndarray) -> None: ...
    def fit(self, X, y, epochs: int) -> None: ...
    def predict_proba(self, X) -> np.ndarray: ... # P(class=1), 1-D
```

Every aggregator implements exactly:

```python
def aggregate(param_vectors: list[np.ndarray],
              client_sizes: list[int]) -> np.ndarray: ...
```

FedProx uses the same aggregation as FedAvg — the proximal term lives in the client's
local `fit`, not in the server.

**The federated loop must not know whether the model is classical or quantum.**
If you write `if model_type == "quantum"` anywhere in the loop, stop and tell me.

## Build order — strictly sequential

1. **Timing spike** — nothing else until we have this number
2. Docs + paper scaffold
3. Data loader + preprocessing + PCA
4. Dirichlet partitioner
5. Arm 1 (centralized)
6. Arm 2 (FedAvg — validates the whole loop)
7. Arm 3 (FedProx)
8. Arm 4 (VQC) ← headline
9. Arm 5 (circular mean — timeboxed, may be cut)
10. Plots and analysis

## Known data traps

- **Switzerland site has `chol = 0` for nearly every record** — missing coded as zero,
  not NaN. `ca` and `thal` are heavily missing outside Cleveland. Decide explicitly:
  drop or impute. Log the choice in `decisions.md` — it goes in the paper.
- **Fit the scaler and PCA on the training split only.** Fitting before the split is
  leakage and produces suspiciously high accuracy that looks like success.

## Validation gates — do not trust results until these pass

| Check | Expected | If it fails |
|---|---|---|
| Arm 1 accuracy | ~83–85% | >95% = leakage (check PCA/scaler fit). <60% = target/encoding broken |
| Arm 2 at α=100 | Within ~2% of Arm 1 | FL loop broken — **fix before touching quantum** |
| Arm 2 as α decreases | Monotonic-ish decline | Partitioner or seeding bug |
| Arm 3 vs Arm 2 at low α | Arm 3 ≥ Arm 2 | μ mistuned (published work uses μ=0.05) |
| Same seed twice | Identical output | Something is unseeded |
| Arm 4, 2 clients, α=100 | Loss actually decreases | Flat loss = barren plateau or gradients not flowing |

Print class balance per client at each α so skew is visible, not assumed.

## Guardrails

- This is a **characterization study**. Expected result: classical performs at least as
  well as quantum at every heterogeneity level, with a heavy quantum wall-clock penalty.
  That is the anticipated outcome, not a bug to fix.
- Never tune the quantum arm to try to make it win. Equal tuning effort both sides.
- No scope additions: no differential privacy, no hierarchical FL, no MIMIC-IV, no extra
  datasets, no extra aggregation rules.
- Simple over clever. An examiner reads this, not a production system.
- Never claim a quantum speedup anywhere in code comments or docs.

## Stop and ask me when

- The timing spike number lands (grid size decision)
- A validation gate fails and the cause isn't obvious in 15 minutes
- Anything here would need to change
- You're about to write more than ~80 lines in one go
