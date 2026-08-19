"""Preprocessing: stratified train/test split, then median-impute + scale to
the angle-encoding range, fitted on the training split only.

No PCA (per this session's decision) -- each of the 6 retained features
(age, sex, cp, restecg, thalach, exang) maps directly to one qubit's RY
rotation angle. restecg/thalach/exang still have some residual missingness
even after being selected (see docs/decisions.md) -- median-imputed rather
than dropped, since at their site-level rates (<=26.5%) there's still
mostly-real data to impute from, unlike the columns that were excluded
entirely.
"""
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "scripts")
from data_loader import FEATURE_COLUMNS, get_feature_frame, load_all_sites  # noqa: E402

ANGLE_RANGE = (0.0, np.pi)  # RY(0) = |0>, RY(pi) = |1> -- full single-qubit range
TEST_SIZE = 0.2


def load_features() -> "pd.DataFrame":
    df, _ = load_all_sites()
    return get_feature_frame(df)


def split(df, seed: int, test_size: float = TEST_SIZE):
    """Stratified train/test split, preserving the original row index (so
    site identity and Dirichlet partitioning can still find these rows)."""
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=df["target"],
    )
    return df.loc[train_idx], df.loc[test_idx]


def fit_transform_train(df_train):
    """Fit imputer + scaler on the training split only; return the
    transformed feature matrix plus the fitted objects for reuse on test."""
    X_train = df_train[FEATURE_COLUMNS].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=ANGLE_RANGE)
    X_train = scaler.fit_transform(X_train)
    return X_train, imputer, scaler


def transform(df, imputer, scaler):
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    X = imputer.transform(X)
    X = scaler.transform(X)
    return X


def get_preprocessed(seed: int):
    """One call: load, split, fit-on-train, transform both. Returns
    (df_train, df_test, X_train, y_train, X_test, y_test). df_train/df_test
    keep the original index + site column, for client partitioning."""
    df = load_features()
    df_train, df_test = split(df, seed)
    X_train, imputer, scaler = fit_transform_train(df_train)
    X_test = transform(df_test, imputer, scaler)
    y_train = df_train["target"].to_numpy()
    y_test = df_test["target"].to_numpy()
    return df_train, df_test, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    df_train, df_test, X_train, y_train, X_test, y_test = get_preprocessed(seed=0)
    print(f"train: {len(df_train)} rows, test: {len(df_test)} rows")
    print(f"X_train shape: {X_train.shape}, range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"X_test shape: {X_test.shape}, range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"train class balance: {y_train.mean():.3f} positive")
    print(f"test class balance: {y_test.mean():.3f} positive")
