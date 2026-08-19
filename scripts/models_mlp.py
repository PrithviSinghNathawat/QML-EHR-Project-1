"""Small MLP model, satisfying the same interface as LogisticRegressionModel
(get_params/set_params/fit/predict_proba). Used to test whether the
classical convexity of logistic regression is suppressing the
heterogeneity penalty (Task 4, diagnostic session).

Parameter count: matched to the ACTUAL frozen VQC (18 trainable parameters
-- 6 qubits x 3 layers x 1 RY/qubit/layer, confirmed from
docs/circuit_diagram.txt), not the ~36 mentioned in the diagnostic task
prompt, which does not match the real circuit. Single hidden layer of 2
units gives 17 parameters (12 + 2 + 2 + 1) -- closest clean architecture
to 18 with a standard single-hidden-layer design. See docs/decisions.md.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class MLPModel:
    def __init__(self, n_features: int, hidden: int = 2, lr: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden = hidden
        self.lr = lr
        self.W1 = 0.3 * rng.standard_normal((n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = 0.3 * rng.standard_normal(hidden)
        self.b2 = 0.0

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), [self.b2]])

    def set_params(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=float)
        n, h = self.n_features, self.hidden
        i = 0
        self.W1 = vec[i : i + n * h].reshape(n, h).copy()
        i += n * h
        self.b1 = vec[i : i + h].copy()
        i += h
        self.W2 = vec[i : i + h].copy()
        i += h
        self.b2 = float(vec[i])

    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return A1, _sigmoid(Z2)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        n = len(X)
        for _ in range(epochs):
            A1, p = self._forward(X)
            dZ2 = p - y
            dW2 = A1.T @ dZ2 / n
            db2 = dZ2.mean()
            dA1 = np.outer(dZ2, self.W2)
            dZ1 = dA1 * (1 - A1**2)
            dW1 = X.T @ dZ1 / n
            db1 = dZ1.mean(axis=0)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, p = self._forward(X)
        return p


if __name__ == "__main__":
    m = MLPModel(6)
    print(f"param count: {len(m.get_params())}")
