"""5-fold stratified cross-validation protocol for the diagnostic session
(docs/decisions.md, 2026-08-18 "diagnostic session" entries).

Design: client assignment (Dirichlet at a given alpha, or the natural
4-site partition) is drawn ONCE per seed over the FULL 920-row pool --
not re-drawn per fold. This gives each client a stable identity across
all 5 folds of a given seed (client 2's characteristic skew is the same
distribution in every fold), which is what makes "worst-client accuracy"
a meaningful, trackable quantity across folds rather than 5 unrelated
per-fold random groupings.

For each of the 5 CV folds: a client's local TRAINING data for that fold
is (client's rows) intersect (fold's training rows); a client's local
TEST data for that fold is (client's rows) intersect (fold's held-out
rows). Every one of the 920 records is used for global testing exactly
once per seed (across the 5 folds), giving an effective global test size
of 920 -- while every client also gets its own held-out slice every fold,
enabling genuine per-client evaluation (not just a pooled global score).

Imputer/scaler are fit on each fold's TRAINING data only (leakage rule,
D-021) -- refit fresh per fold, not shared across folds.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "scripts")
from data_loader import FEATURE_COLUMNS, SITES, get_feature_frame, load_all_sites  # noqa: E402
from partitioner import dirichlet_partition  # noqa: E402

ANGLE_RANGE = (0.0, np.pi)
N_FOLDS = 5


def load_pool() -> pd.DataFrame:
    df, _ = load_all_sites()
    return get_feature_frame(df)


def client_assignment(df: pd.DataFrame, mode, seed: int) -> pd.Series:
    """mode: a float alpha for Dirichlet, or the string 'natural' for the
    real 4-site partition. Returns a Series of client ids (0..3), indexed
    like df."""
    if mode == "natural":
        site_to_client = {s: i for i, s in enumerate(SITES)}
        return df["site"].map(site_to_client)
    client_idx, _ = dirichlet_partition(df, float(mode), seed=seed)
    assignment = pd.Series(index=df.index, dtype=int)
    for c, idx in enumerate(client_idx):
        assignment.loc[idx] = c
    return assignment


def cv_folds(df: pd.DataFrame, seed: int, n_folds: int = N_FOLDS):
    """Yields (fold_id, df_train_fold, df_test_fold), stratified by target."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = df.index.to_numpy()
    y = df["target"].to_numpy()
    for fold_id, (train_pos, test_pos) in enumerate(skf.split(idx, y)):
        yield fold_id, df.loc[idx[train_pos]], df.loc[idx[test_pos]]


def fit_transform_fold(df_train_fold: pd.DataFrame, df_test_fold: pd.DataFrame):
    """Fit imputer+scaler on this fold's training data only; return
    transformed (X_train, X_test) plus y_train/y_test as numpy arrays."""
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler(feature_range=ANGLE_RANGE)
    X_train = scaler.fit_transform(imputer.fit_transform(df_train_fold[FEATURE_COLUMNS].to_numpy(dtype=float)))
    X_test = scaler.transform(imputer.transform(df_test_fold[FEATURE_COLUMNS].to_numpy(dtype=float)))
    y_train = df_train_fold["target"].to_numpy()
    y_test = df_test_fold["target"].to_numpy()
    return X_train, y_train, X_test, y_test


def split_by_client(X, y, assign: pd.Series, df_fold: pd.DataFrame, n_clients: int = 4):
    """assign: full-pool client assignment. df_fold: the fold's dataframe
    (train or test), whose row order matches X/y. Returns
    {client_id: (X_c, y_c)} for clients that have at least 1 row in this
    fold (a client can have 0 rows in a small held-out fold at extreme
    alpha -- caller must handle missing keys)."""
    fold_assign = assign.loc[df_fold.index].to_numpy()
    out = {}
    for c in range(n_clients):
        mask = fold_assign == c
        if mask.any():
            out[c] = (X[mask], y[mask])
    return out


if __name__ == "__main__":
    df = load_pool()
    print(f"pool: {len(df)} rows")
    assign = client_assignment(df, 0.1, seed=0)
    print("client sizes (full pool, alpha=0.1, seed=0):", assign.value_counts().sort_index().to_dict())
    for fold_id, df_train, df_test in cv_folds(df, seed=0):
        print(f"fold {fold_id}: train={len(df_train)} test={len(df_test)}")
