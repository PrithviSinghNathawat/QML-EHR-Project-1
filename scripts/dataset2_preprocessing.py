"""Dataset 2 (Diabetes 130-US Hospitals) loading and feature selection.
Minimal scope, classical arms only (P-010): first-encounter-per-patient
filter for repeat-patient leakage, pooled >=85%-present rule (D-016/D-019's
rule, no site axis exists for this dataset -- see P-009), plus a new
near-zero-variance filter this dataset needed and dataset 1 never did.

High-cardinality ICD9 diagnosis codes (diag_1/2/3) are explicitly EXCLUDED
from the modeling feature set even though they pass both filters --
one-hot encoding hundreds of ICD9 codes is a real design decision outside
"minimal scope" and is flagged here rather than done silently.
"""
import sys

import numpy as np
import pandas as pd

RAW_PATH = "data2/raw/diabetic_data.csv"
MISSINGNESS_FLOOR = 0.85
NZV_FLOOR = 0.99  # drop if one value's share >= this (near-zero-variance)

TARGET_COL = "readmitted"
ID_COLS = ["encounter_id", "patient_nbr"]
EXCLUDED_HIGH_CARDINALITY = ["diag_1", "diag_2", "diag_3"]  # flagged, not encoded -- see module docstring

MEDICATION_ORDINAL = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}


def load_first_encounter() -> pd.DataFrame:
    """One row per patient: the first encounter by encounter_id, following
    Strack et al. 2014's own methodology (subsequent encounters of the same
    patient are not independent of the first)."""
    df = pd.read_csv(RAW_PATH, na_values="?", low_memory=False)
    df = df.sort_values("encounter_id")
    first = df.groupby("patient_nbr", as_index=False).first()
    return first


def select_features(df: pd.DataFrame) -> tuple[list, list]:
    """Returns (retained_numeric_like, dropped) after missingness + NZV
    filters, evaluated on the (already first-encounter-filtered) modeling
    population -- not the raw 101,766-row pool, since that's not what gets
    trained on."""
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]
    present_frac = 1 - df[feature_cols].isna().mean()

    dropped_missing = [c for c in feature_cols if present_frac[c] < MISSINGNESS_FLOOR]
    survives_missing = [c for c in feature_cols if present_frac[c] >= MISSINGNESS_FLOOR]

    dropped_nzv = []
    survives_nzv = []
    for c in survives_missing:
        top_share = df[c].value_counts(normalize=True, dropna=True).iloc[0]
        if top_share >= NZV_FLOOR:
            dropped_nzv.append(c)
        else:
            survives_nzv.append(c)

    retained = [c for c in survives_nzv if c not in EXCLUDED_HIGH_CARDINALITY]
    dropped = dropped_missing + dropped_nzv + [c for c in EXCLUDED_HIGH_CARDINALITY if c in survives_nzv]
    return retained, dropped


def get_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Returns (frame with retained raw feature columns + 'target', encoded
    numerically only for the columns cheap to encode inline (binary/ordinal);
    race is left as a string column for one-hot in fit_transform_fold since
    its categories must be learned on the training fold only -- see
    dataset2_cv_protocol.py), plus (numeric_cols, race_col_present)."""
    retained, dropped = select_features(df)
    out = pd.DataFrame(index=df.index)

    age_order = {f"[{i*10}-{i*10+10})": i * 10 for i in range(10)}

    numeric_cols = []
    for c in retained:
        if c == "race":
            continue  # one-hot fit per-fold, handled downstream
        elif c == "age":
            out["age"] = df["age"].map(age_order)
            numeric_cols.append("age")
        elif c == "gender":
            out["gender"] = df["gender"].map({"Male": 1.0, "Female": 0.0})  # 'Unknown/Invalid' -> NaN, imputed per-fold
            numeric_cols.append("gender")
        elif c in ("change",):
            out["change"] = df["change"].map({"No": 0, "Ch": 1})
            numeric_cols.append("change")
        elif c == "diabetesMed":
            out["diabetesMed"] = df["diabetesMed"].map({"No": 0, "Yes": 1})
            numeric_cols.append("diabetesMed")
        elif c in MEDICATION_ORDINAL or df[c].dropna().isin(MEDICATION_ORDINAL.keys()).all():
            out[c] = df[c].map(MEDICATION_ORDINAL)
            numeric_cols.append(c)
        else:
            out[c] = pd.to_numeric(df[c], errors="raise")
            numeric_cols.append(c)

    if "race" in retained:
        out["race"] = df["race"].fillna("Missing")

    out["target"] = (df[TARGET_COL] == "<30").astype(int)
    return out, numeric_cols, ("race" in retained)


if __name__ == "__main__":
    first = load_first_encounter()
    print(f"encounters after first-encounter-per-patient filter: {len(first)} (from 101,766 raw)")

    retained, dropped = select_features(first)
    print(f"\nfeature retention (missingness >={MISSINGNESS_FLOOR:.0%} + NZV <{NZV_FLOOR:.0%}, "
          f"re-evaluated on the {len(first)}-row filtered population):")
    print(f"  retained (before excluding diag_1/2/3): {len(retained) + sum(c in EXCLUDED_HIGH_CARDINALITY for c in [])}")
    print(f"  retained (modeling feature set, diag_1/2/3 excluded): {len(retained)}")
    print(f"  dropped: {sorted(dropped)}")

    frame, numeric_cols, has_race = get_feature_frame(first)
    print(f"\nmodeling frame: {len(numeric_cols)} numeric/ordinal features"
          f"{' + one-hot race' if has_race else ''}, target positive rate "
          f"{100*frame['target'].mean():.2f}%")
    print("numeric_cols:", numeric_cols)
