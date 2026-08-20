"""Arm 4 sanity check (Task 2, must pass before any full sweep): 2 clients,
alpha=100 (near-IID), a handful of rounds. If the loss doesn't decrease,
every downstream accuracy number is meaningless (barren plateau / gradients
not flowing) -- per the validation gate in CLAUDE.md.
"""
import sys
import time

import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models_vqc import VQCModel  # noqa: E402
from partitioner import dirichlet_partition  # noqa: E402
from preprocessing import get_preprocessed  # noqa: E402

N_ROUNDS = 15
LOCAL_EPOCHS = 5
N_CLIENTS = 2
ALPHA = 100
SEED = 0

if __name__ == "__main__":
    df_train, df_test, X_train, y_train, X_test, y_test = get_preprocessed(SEED)

    client_idx, n_attempts = dirichlet_partition(df_train, ALPHA, n_clients=N_CLIENTS, seed=SEED)
    pos = df_train.index.get_indexer
    client_data = [(X_train[pos(idx)], y_train[pos(idx)]) for idx in client_idx]
    print(f"client sizes: {[len(y) for _, y in client_data]} (guard attempts: {n_attempts})")

    global_params = VQCModel(seed=SEED).get_params()
    losses = []
    start = time.perf_counter()
    for r in range(N_ROUNDS):
        def factory(gp=global_params):
            m = VQCModel(seed=SEED)
            m.set_params(gp)
            return m

        model = run_federated(factory, fedavg, client_data, rounds=1, local_epochs=LOCAL_EPOCHS)
        global_params = model.get_params()
        loss = model.loss(X_train, y_train)
        losses.append(loss)
        print(f"round {r + 1:2d}/{N_ROUNDS}  loss={loss:.4f}  ({time.perf_counter() - start:.1f}s elapsed)")

    total = time.perf_counter() - start
    print(f"\ntotal: {total:.1f}s ({total / N_ROUNDS:.2f}s/round)")
    print(f"loss: round 1 = {losses[0]:.4f}, round {N_ROUNDS} = {losses[-1]:.4f}, "
          f"decrease = {losses[0] - losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, N_ROUNDS + 1), losses, marker="o")
    ax.set_xlabel("federated round")
    ax.set_ylabel("global BCE loss (on pooled training data)")
    ax.set_title(f"Arm 4 sanity check: {N_CLIENTS} clients, alpha={ALPHA}")
    fig.tight_layout()
    fig.savefig("results/figs/arm4_sanity_loss_curve.png", dpi=200)
    plt.close(fig)
    print("saved: results/figs/arm4_sanity_loss_curve.png")
