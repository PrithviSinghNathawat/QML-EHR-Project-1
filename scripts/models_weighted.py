"""Class-weighted variants of LogisticRegressionModel and MLPModel, per P-013.

Reversal of the earlier "evaluation-only, no training change" scope call
(P-011): balanced accuracy fixed the composition-vs-degenerate-classifier
artifact only where an evaluation slice contains both classes. At severe
skew, single-class slices fall back to plain accuracy, and since neither
LR nor MLP ever learned a non-constant decision boundary at 8.8% prevalence
(verified by both models converging to "always predict majority" up to
2,000 epochs), the composition-vs-training decomposition was measuring a
model that never trains at all -- vacuous, not a real result. This module
adds standard inverse-frequency ("balanced") sample weighting to the
gradient, recomputed fresh from each fit() call's own y (so a client's
local imbalance, not a fixed global rate, sets its own local weights,
consistent with a federated setting where each client only sees its own
labels).

Deliberately NOT edited in place: models.py / models_mlp.py, so every
existing dataset-1 result (Arms 1-5, D-001 through D-052, the paper draft)
stays exactly as originally computed, unweighted. These classes exist
alongside them for the re-evaluation this decision requires -- both
weighted and unweighted numbers get reported side by side for dataset 1,
per instruction, not one replacing the other.

Same interface contract as every other model here: get_params/set_params/
fit(X, y, epochs)/predict_proba -- fit() computes the weights internally,
no interface change.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """sklearn's 'balanced' scheme: w_c = n_samples / (n_classes * n_c).
    Falls back to uniform weight 1.0 if only one class is present in this
    particular call's y (weighting is undefined/moot with one class)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.ones_like(y, dtype=float)
    n = len(y)
    class_weight = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weight[label] for label in y], dtype=float)


class WeightedLogisticRegressionModel:
    def __init__(self, n_features: int, lr: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.weights = 0.01 * rng.standard_normal(n_features)
        self.bias = 0.0

    def get_params(self) -> np.ndarray:
        return np.concatenate([[self.bias], self.weights])

    def set_params(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=float)
        self.bias = vec[0]
        self.weights = vec[1:].copy()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        w = _balanced_sample_weight(y)
        w_sum = w.sum()
        for _ in range(epochs):
            p = _sigmoid(X @ self.weights + self.bias)
            grad_z = w * (p - y)
            self.weights -= self.lr * (X.T @ grad_z / w_sum)
            self.bias -= self.lr * (grad_z.sum() / w_sum)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.weights + self.bias)


class WeightedMLPModel:
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
        w = _balanced_sample_weight(y)
        w_sum = w.sum()
        for _ in range(epochs):
            A1, p = self._forward(X)
            dZ2 = w * (p - y)
            dW2 = A1.T @ dZ2 / w_sum
            db2 = dZ2.sum() / w_sum
            dA1 = np.outer(dZ2, self.W2)
            dZ1 = dA1 * (1 - A1**2)
            dW1 = X.T @ dZ1 / w_sum
            db1 = dZ1.sum(axis=0) / w_sum
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, p = self._forward(X)
        return p


if __name__ == "__main__":
    m = WeightedLogisticRegressionModel(6)
    print(f"LR param count: {len(m.get_params())}")
    mlp = WeightedMLPModel(6)
    print(f"MLP param count: {len(mlp.get_params())}")
