"""Model interface implementations. Every model exposes exactly:
get_params() -> np.ndarray, set_params(vec), fit(X, y, epochs), predict_proba(X).
The federated loop only ever calls these four methods (see CLAUDE.md).
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticRegressionModel:
    """Logistic regression trained by full-batch gradient descent. Not
    sklearn's LogisticRegression -- needs a fit(X, y, epochs) signature and
    a flat get_params/set_params vector to satisfy the shared interface
    that the quantum model will also have to satisfy later.
    """

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
        n = len(X)
        for _ in range(epochs):
            p = _sigmoid(X @ self.weights + self.bias)
            grad_z = p - y
            self.weights -= self.lr * (X.T @ grad_z / n)
            self.bias -= self.lr * grad_z.mean()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.weights + self.bias)
