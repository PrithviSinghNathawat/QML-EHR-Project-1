"""One (reduced) federated VQC run, invoked as a subprocess for the
run-level parallelism throughput test. Not a real experimental arm -- just
a representative workload: same circuit, same N_CLIENTS (D-017), same
LOCAL_STEPS (D-005), fewer rounds so the throughput test itself is fast.

Usage: single_run_worker.py <run_id> <out_dir>
Writes <out_dir>/<run_id>.json with {"run_id", "wall_clock_sec"}.
"""
import json
import sys
import time

import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 6
N_LAYERS = 3
N_CLIENTS = 4  # D-017: match the 4 natural sites
N_ROWS_PER_CLIENT = 230  # 4 x 230 = 920, matches real dataset size
N_ROUNDS = 10  # reduced from the real 50 -- this test only needs relative scaling
LOCAL_STEPS = 5  # D-005
LR = 0.1


def main():
    run_id = sys.argv[1]
    out_dir = sys.argv[2]
    seed = int(run_id)

    pnp.random.seed(seed)
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
        return (circuit(x, weights) + 1) / 2

    def bce_loss(weights, X, y):
        preds = pnp.stack([predict(x, weights) for x in X])
        preds = pnp.clip(preds, 1e-7, 1 - 1e-7)
        return -pnp.mean(y * pnp.log(preds) + (1 - y) * pnp.log(1 - preds))

    grad_fn = qml.grad(bce_loss, argnums=0)

    clients = []
    for _ in range(N_CLIENTS):
        X = pnp.random.normal(size=(N_ROWS_PER_CLIENT, N_QUBITS)) * 0.5
        y = pnp.random.randint(0, 2, size=N_ROWS_PER_CLIENT).astype(float)
        clients.append((X, y))

    global_weights = pnp.array(0.1 * pnp.random.randn(N_LAYERS, N_QUBITS), requires_grad=True)

    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        client_weights = []
        for X, y in clients:
            w = pnp.array(global_weights, requires_grad=True)
            for _ in range(LOCAL_STEPS):
                g = grad_fn(w, X, y)
                w = pnp.array(w - LR * g, requires_grad=True)
            client_weights.append(w)
        global_weights = pnp.mean(pnp.stack(client_weights), axis=0)
    wall_clock = time.perf_counter() - start

    with open(f"{out_dir}/{run_id}.json", "w") as f:
        json.dump({"run_id": run_id, "wall_clock_sec": wall_clock}, f)

    print(f"run {run_id}: {wall_clock:.2f}s")


if __name__ == "__main__":
    main()
