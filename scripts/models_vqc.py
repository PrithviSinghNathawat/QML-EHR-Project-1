"""Variational quantum classifier, satisfying the exact same interface as
LogisticRegressionModel / MLPModel (get_params/set_params/fit/predict_proba).
Built against the frozen interface (docs/INTERFACE.md) with no changes to
federated_loop.py -- the loop does not know this model is quantum.

Circuit: the locked design (D-002, D-003, D-004) -- 6 qubits, 3 layers,
angle encoding (RY) + RY ansatz + linear CNOT entangling, lightning.qubit,
diff_method="adjoint". 18 trainable parameters (6 qubits x 3 layers x 1
RY/qubit/layer), identical to scripts/single_run_worker.py's circuit and
docs/circuit_diagram.png/.txt -- this is the actual arm, not a spike, so
the circuit definition must match exactly what was documented and timed.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 6
N_LAYERS = 3

_dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(_dev, diff_method="adjoint")
def _circuit(x, weights):
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))


def _predict_one(x, weights):
    return (_circuit(x, weights) + 1) / 2  # map [-1, 1] -> [0, 1]


def _bce_loss(weights, X, y):
    preds = pnp.stack([_predict_one(x, weights) for x in X])
    preds = pnp.clip(preds, 1e-7, 1 - 1e-7)
    return -pnp.mean(y * pnp.log(preds) + (1 - y) * pnp.log(1 - preds))


_grad_fn = qml.grad(_bce_loss, argnums=0)


class VQCModel:
    def __init__(self, n_features: int = N_QUBITS, lr: float = 0.1, seed: int = 0):
        assert n_features == N_QUBITS, "VQC is wired for exactly 6 features -> 6 qubits"
        local_rng = np.random.default_rng(seed)
        self.lr = lr
        self.weights = pnp.array(0.1 * local_rng.standard_normal((N_LAYERS, N_QUBITS)), requires_grad=True)

    def get_params(self) -> np.ndarray:
        return np.asarray(self.weights).ravel().copy()

    def set_params(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=float).reshape(N_LAYERS, N_QUBITS)
        self.weights = pnp.array(vec, requires_grad=True)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        X_p = pnp.array(X, requires_grad=False)
        y_p = pnp.array(y, requires_grad=False)
        w = self.weights
        for _ in range(epochs):
            g = _grad_fn(w, X_p, y_p)
            w = pnp.array(w - self.lr * g, requires_grad=True)
        self.weights = w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_p = pnp.array(X, requires_grad=False)
        return np.array([float(_predict_one(x, self.weights)) for x in X_p])

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Not part of the frozen interface -- used only by the Task 2
        sanity check to plot the training loss curve."""
        return float(_bce_loss(self.weights, pnp.array(X, requires_grad=False), pnp.array(y, requires_grad=False)))


if __name__ == "__main__":
    m = VQCModel()
    print(f"param count: {len(m.get_params())}")
