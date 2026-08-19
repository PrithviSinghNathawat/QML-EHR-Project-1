"""Timing spike: how long does one federated VQC run cost?

Trains a 6-qubit, 3-layer VQC across 5 synthetic clients (~180 rows x 6
features each, shaped like the real preprocessed EHR data) for 50 federated
rounds, using lightning.qubit + adjoint differentiation. Not meant to
converge to anything meaningful -- purpose is only to measure wall-clock
cost so we can size the real experiment grid.
"""
import time

import pennylane as qml
from pennylane import numpy as pnp

from log_run import append_run

pnp.random.seed(0)

N_QUBITS = 6
N_LAYERS = 3
N_CLIENTS = 5
N_ROWS_PER_CLIENT = 180
N_ROUNDS = 50
LOCAL_STEPS = 5  # local gradient steps per client per round
LR = 0.1

dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev, diff_method="adjoint")
def circuit(x, weights):
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))


def predict(x, weights):
    return (circuit(x, weights) + 1) / 2  # map [-1, 1] -> [0, 1]


def bce_loss(weights, X, y):
    preds = pnp.stack([predict(x, weights) for x in X])
    preds = pnp.clip(preds, 1e-7, 1 - 1e-7)
    return -pnp.mean(y * pnp.log(preds) + (1 - y) * pnp.log(1 - preds))


grad_fn = qml.grad(bce_loss, argnums=0)

# synthetic clients shaped like the real (post-PCA) data: 6 features/row
clients = []
for _ in range(N_CLIENTS):
    X = pnp.random.normal(size=(N_ROWS_PER_CLIENT, N_QUBITS)) * 0.5
    y = pnp.random.randint(0, 2, size=N_ROWS_PER_CLIENT).astype(float)
    clients.append((X, y))

global_weights = pnp.array(0.1 * pnp.random.randn(N_LAYERS, N_QUBITS), requires_grad=True)

round_times = []
start = time.perf_counter()

for r in range(N_ROUNDS):
    r_start = time.perf_counter()
    client_weights = []
    for X, y in clients:
        w = pnp.array(global_weights, requires_grad=True)
        for _ in range(LOCAL_STEPS):
            g = grad_fn(w, X, y)
            w = pnp.array(w - LR * g, requires_grad=True)
        client_weights.append(w)
    global_weights = pnp.mean(pnp.stack(client_weights), axis=0)
    round_times.append(time.perf_counter() - r_start)
    print(f"round {r + 1:2d}/{N_ROUNDS}  {round_times[-1]:.3f}s")

total_time = time.perf_counter() - start
avg_round = sum(round_times) / len(round_times)

print()
print(f"total wall-clock for {N_ROUNDS} rounds, {N_CLIENTS} clients: {total_time:.1f}s")
print(f"avg per-round time: {avg_round:.3f}s")

per_run_sec = total_time  # one run = one (arm, alpha, seed) triple at this round count

for n_seeds in (3, 5):
    n_runs = 2 * 4 * n_seeds  # 2 quantum arms x 4 alpha values x n_seeds
    total_sec = per_run_sec * n_runs
    print(
        f"extrapolated grid (2 quantum arms x 4 alpha x {n_seeds} seeds = {n_runs} runs): "
        f"{total_sec:.0f}s = {total_sec / 60:.1f} min = {total_sec / 3600:.2f} hr"
    )

append_run(
    {
        "arm": f"timing_spike_E{LOCAL_STEPS}",
        "wall_clock_sec": round(total_time, 1),
        "n_qubits": N_QUBITS,
        "circuit_depth": N_LAYERS,
    }
)
