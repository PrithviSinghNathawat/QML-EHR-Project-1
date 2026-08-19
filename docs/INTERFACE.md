# Interface Freeze

Frozen 2026-08-18, after Arm 1 + Arm 2 passed the validation gates (see
`docs/decisions.md` and `docs/labbook.md` for the numbers). After this
point, Arms 3, 4, and 5 must be implementable by adding new files, not by
editing `scripts/federated_loop.py` or `scripts/aggregators.py`. If a new
arm seems to require editing either of those, stop and raise it rather than
changing shared infrastructure silently — that's exactly the kind of change
that breaks a teammate's in-progress work on a different arm.

## Model interface

Any model (classical or quantum) passed into the federated loop must
implement exactly these four methods:

```python
class Model:
    def get_params(self) -> np.ndarray: ...
    def set_params(self, vec: np.ndarray) -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
```

**Shapes and dtypes:**

| method | argument | shape | dtype | notes |
|---|---|---|---|---|
| `get_params` | return | `(P,)` | `float64` | `P` is model-specific (classical: 7 = 1 bias + 6 weights; quantum: model-defined, e.g. 18 for 3 layers x 6 qubits). Flat, 1-D, no exceptions. |
| `set_params` | `vec` | `(P,)` | `float64` | Must accept exactly what `get_params` returns for the same model instance. Implementations should copy, not alias, the input array. |
| `fit` | `X` | `(n, 6)` | `float64` | 6 columns = 6 features (`age`, `sex`, `cp`, `restecg`, `thalach`, `exang`), pre-scaled to `[0, pi]` by `scripts/preprocessing.py`. Column order is fixed by `data_loader.FEATURE_COLUMNS` -- a model must not reorder or reinterpret columns. |
| `fit` | `y` | `(n,)` | `int` (0/1) | Binary target, `num > 0`. |
| `fit` | `epochs` | scalar | `int` | Local training steps for this call. The loop decides what "epochs" means per call (centralized: one large `epochs`; federated: `LOCAL_EPOCHS` per round) -- the model doesn't need to know which context it's in. |
| `predict_proba` | `X` | `(n, 6)` | `float64` | Same column contract as `fit`. |
| `predict_proba` | return | `(n,)` | `float64` in `[0, 1]` | `P(class=1)`, not a 2-column `(n, 2)` array. Threshold at 0.5 externally (see `scripts/run_grid.py:evaluate`). |

**Constructor is not part of the frozen interface** -- each model's
`__init__` can take whatever it needs (e.g. `LogisticRegressionModel(n_features, lr, seed)`).
The loop only ever holds a zero-argument `model_factory: Callable[[], Model]`
closure, never calls a model's constructor directly. This is what lets the
loop stay agnostic to what kind of model it's building.

## Aggregator interface

```python
def aggregate(param_vectors: list[np.ndarray], client_sizes: list[int]) -> np.ndarray: ...
```

| argument | shape | dtype | notes |
|---|---|---|---|
| `param_vectors` | list of `(P,)` arrays, all identical `P` | `float64` | One entry per client, in the same order as `client_sizes`. `P` matches whatever the model being aggregated uses -- the aggregator must not assume a specific `P`. |
| `client_sizes` | list of `int`, same length as `param_vectors` | `int` | Row counts, used as aggregation weights. |
| return | `(P,)` | `float64` | Same shape as each input vector. |

`fedavg` (`scripts/aggregators.py`) is the only aggregator implemented so
far: `np.average(stacked_params, axis=0, weights=client_sizes)`. FedProx
(Arm 3) does **not** get a new aggregator -- per `CLAUDE.md`, the proximal
term lives in the client's local `fit`, and FedProx reuses `fedavg`
unchanged.

## Loop call contract

```python
def run_centralized(model_factory: Callable[[], Model], X_train: np.ndarray, y_train: np.ndarray, epochs: int) -> Model: ...

def run_federated(
    model_factory: Callable[[], Model],
    aggregator: Callable[[list[np.ndarray], list[int]], np.ndarray],
    client_data: list[tuple[np.ndarray, np.ndarray]],  # [(X_client, y_client), ...]
    rounds: int,
    local_epochs: int,
    track_divergence: bool = False,  # amendment, D-027, 2026-08-18 -- see below
) -> Model: ...
# or, if track_divergence=True: -> tuple[Model, list[float]]
```

**Amendment history:** `track_divergence` was added 2026-08-18 (D-027) to
support measuring mean pairwise L2 distance between client parameter
vectors per round, without duplicating the training loop into a second
function. Default `False` preserves the exact original return contract
(`-> Model`) -- `scripts/run_grid.py` does not pass this argument and is
unaffected. When `True`, the return type changes to
`tuple[Model, list[float]]` (the model, and one divergence value per
round). This is the first amendment to this frozen interface since the
freeze (D-024) -- additive only, no existing call site's behavior changed.

Both return the trained model itself (not just its params) -- call
`.predict_proba(X_test)` on the return value.

**`scripts/federated_loop.py` contains no `if model_type == "quantum"` (or
equivalent) anywhere, and none should be added.** It only ever calls the
four `Model` methods and the `aggregator` callable. Verified by inspection
at freeze time; re-check this by inspection whenever the loop file changes.

## What's NOT frozen

- Model constructors, hyperparameters, and internals (learning rate,
  weight init, whatever a quantum model needs for its ansatz).
- `scripts/run_grid.py` -- the grid runner is expected to grow (more arms,
  more metrics) without changing the interfaces above.
- `scripts/preprocessing.py` and `scripts/partitioner.py` -- upstream of
  the interface, can change independently as long as they keep producing
  `(n, 6)` float feature arrays and `(n,)` int/float label arrays.

## Validation status at freeze time

Arm 1 (`LogisticRegressionModel`, centralized) and Arm 2 (`LogisticRegressionModel`
+ `fedavg`) both run through this exact interface with no model-specific
branching in the loop. See `docs/decisions.md` for the actual gate numbers.
This is the evidence the interface is sufficient for at least one classical
model; it has not yet been exercised by a quantum model, which is the real
test of whether it's actually implementation-agnostic. Arm 4 should be the
first thing that tries to break this contract.
