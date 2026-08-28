"""CV/partition protocol for dataset 2, mirroring scripts/cv_protocol.py's
structure but generalized to an arbitrary client count K (P-010) -- dataset 2
has no natural/hospital partition (P-009), so 'mode' here is always a
Dirichlet alpha, never "natural".

Same leakage rule as dataset 1 (D-021): imputer/scaler/one-hot categories
are fit on each fold's TRAINING data only, refit fresh per fold.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "scripts")
from dataset2_preprocessing import get_feature_frame, load_first_encounter  # noqa: E402
from partitioner import MIN_CLIENT_SIZE, _draw_once  # noqa: E402

N_FOLDS = 5


class InfeasiblePartition(Exception):
    """Raised when no draw within max_attempts satisfies MIN_CLIENT_SIZE.
    At K=130, alpha=0.1 is not a rare-unlucky-draw problem -- empirically
    verified (P-010) that 0/1000+ independent draws produce even a single
    non-empty client at every position; no attempt budget fixes this."""


def load_pool():
    """Returns (frame, numeric_cols, has_race) -- frame has raw (unscaled,
    unimputed) numeric/ordinal columns plus 'target' and optionally 'race'."""
    first = load_first_encounter()
    return get_feature_frame(first)


def client_assignment(df: pd.DataFrame, alpha: float, seed: int, n_clients: int, max_attempts: int = 3000):
    """Same MIN_CLIENT_SIZE=15 floor as D-022 (partitioner.py) -- only the
    search budget (max_attempts) is raised above the shared module's
    default 500, since K=130's rarer valid draws (P-010: alpha=0.5 at
    K=130 succeeds ~0.5% of the time) need more attempts to find, not a
    looser floor. Raises InfeasiblePartition if truly unsatisfiable."""
    for attempt in range(max_attempts):
        client_idx = _draw_once(df, float(alpha), n_clients, seed=seed * 1_000_000 + attempt)
        sizes = [len(idx) for idx in client_idx]
        if min(sizes) >= MIN_CLIENT_SIZE:
            assignment = pd.Series(index=df.index, dtype=int)
            for c, idx in enumerate(client_idx):
                assignment.loc[idx] = c
            return assignment, attempt + 1
    raise InfeasiblePartition(
        f"no draw satisfied min_client_size={MIN_CLIENT_SIZE} for alpha={alpha}, "
        f"n_clients={n_clients} after {max_attempts} attempts"
    )


def cv_folds(df: pd.DataFrame, seed: int, n_folds: int = N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = df.index.to_numpy()
    y = df["target"].to_numpy()
    for fold_id, (train_pos, test_pos) in enumerate(skf.split(idx, y)):
        yield fold_id, df.loc[idx[train_pos]], df.loc[idx[test_pos]]


def fit_transform_fold(df_train_fold: pd.DataFrame, df_test_fold: pd.DataFrame, numeric_cols: list, has_race: bool):
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_num = scaler.fit_transform(imputer.fit_transform(df_train_fold[numeric_cols].to_numpy(dtype=float)))
    X_test_num = scaler.transform(imputer.transform(df_test_fold[numeric_cols].to_numpy(dtype=float)))

    if has_race:
        train_dummies = pd.get_dummies(df_train_fold["race"])
        race_cols = list(train_dummies.columns)
        test_dummies = pd.get_dummies(df_test_fold["race"]).reindex(columns=race_cols, fill_value=0)
        X_train = np.concatenate([X_train_num, train_dummies.to_numpy(dtype=float)], axis=1)
        X_test = np.concatenate([X_test_num, test_dummies.to_numpy(dtype=float)], axis=1)
    else:
        X_train, X_test = X_train_num, X_test_num

    y_train = df_train_fold["target"].to_numpy()
    y_test = df_test_fold["target"].to_numpy()
    return X_train, y_train, X_test, y_test


def split_by_client(X, y, assign: pd.Series, df_fold: pd.DataFrame, n_clients: int):
    fold_assign = assign.loc[df_fold.index].to_numpy()
    out = {}
    for c in range(n_clients):
        mask = fold_assign == c
        if mask.any():
            out[c] = (X[mask], y[mask])
    return out


if __name__ == "__main__":
    df, numeric_cols, has_race = load_pool()
    print(f"pool: {len(df)} rows, {len(numeric_cols)} numeric features, race one-hot: {has_race}")
    for K, alpha in [(4, 0.1), (130, 0.5), (130, 1.0), (130, 100)]:
        try:
            assign, n_attempts = client_assignment(df, alpha, seed=0, n_clients=K)
            sizes = assign.value_counts()
            print(f"K={K} alpha={alpha} seed=0: attempts={n_attempts}, min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
        except InfeasiblePartition as e:
            print(f"K={K} alpha={alpha} seed=0: INFEASIBLE -- {e}")
    for fold_id, df_train, df_test in cv_folds(df, seed=0):
        X_train, y_train, X_test, y_test = fit_transform_fold(df_train, df_test, numeric_cols, has_race)
        print(f"fold {fold_id}: X_train {X_train.shape}, X_test {X_test.shape}, pos_rate_train={y_train.mean():.4f}")
