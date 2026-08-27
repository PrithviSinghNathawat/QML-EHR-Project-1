"""Characterisation-only pass over the second dataset (Diabetes 130-US
Hospitals). No experiments, no models, no partitioning -- this script only
describes the data, so the numbers in docs/dataset2_characterization.md can
be traced back to something reproducible (same rule as runs.csv: if a number
isn't traceable, it doesn't go in the paper).

Missing values in this dataset are coded as the literal string "?", not
blank/NaN -- read with na_values="?" or every missingness number below is
wrong.
"""
import json

import pandas as pd

RAW_PATH = "data2/raw/diabetic_data.csv"
MISSINGNESS_FLOOR = 0.85  # D-016/D-019's "retain only if >=85% present" rule

TARGET_COL = "readmitted"
ID_COLS = ["encounter_id", "patient_nbr"]


def load() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, na_values="?", low_memory=False)


def characterize() -> dict:
    df = load()

    n_rows = len(df)
    n_patients = df["patient_nbr"].nunique()
    repeat_encounters = n_rows - n_patients

    present_frac = 1 - df.isna().mean()
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]
    missingness = {c: round(100 * (1 - present_frac[c]), 2) for c in feature_cols}

    retained = sorted([c for c in feature_cols if present_frac[c] >= MISSINGNESS_FLOOR])
    dropped = sorted([c for c in feature_cols if present_frac[c] < MISSINGNESS_FLOOR])

    readmit_counts = df[TARGET_COL].value_counts().to_dict()
    readmit_pct = (df[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict()
    binary_positive = (df[TARGET_COL] == "<30")
    binary_pct_positive = round(100 * binary_positive.mean(), 2)

    # candidate site-proxy fields -- NOT real hospital identifiers, checked
    # only to confirm none of them plausibly stand in for one
    proxy_candidates = {}
    for c in ["admission_source_id", "admission_type_id", "discharge_disposition_id",
              "medical_specialty", "payer_code"]:
        proxy_candidates[c] = {
            "nunique": int(df[c].nunique()),
            "missing_pct": round(100 * (1 - present_frac[c]), 2),
        }

    result = {
        "n_rows": n_rows,
        "n_unique_encounters": int(df["encounter_id"].nunique()),
        "n_unique_patients": n_patients,
        "repeat_encounters": repeat_encounters,
        "repeat_encounter_pct": round(100 * repeat_encounters / n_rows, 2),
        "has_hospital_id_column": False,
        "all_columns": list(df.columns),
        "missingness_pct_by_feature": missingness,
        "retention_floor": MISSINGNESS_FLOOR,
        "retained_features_n85": retained,
        "n_retained": len(retained),
        "dropped_features_n85": dropped,
        "readmitted_raw_counts": readmit_counts,
        "readmitted_raw_pct": readmit_pct,
        "readmitted_binary_lt30_positive_pct": binary_pct_positive,
        "site_proxy_candidates": proxy_candidates,
    }
    return result


if __name__ == "__main__":
    result = characterize()
    with open("results/dataset2_characterization.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
