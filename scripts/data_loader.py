"""Load all four UCI Heart Disease processed site files into one DataFrame.

No header row in the raw files; column order is fixed by the UCI docs
(data/raw/heart-disease.names). Missing values are coded as '?'.
"""
import numpy as np
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# The 6 features with the highest worst-site availability (see
# docs/decisions.md, feature-retention entries): age, sex, cp, restecg
# (100/100/100/99.2% worst-site available), thalach, exang (73.5%
# worst-site available, both fail at VA specifically). Everything else
# (trestbps, oldpeak, fbs, slope, chol, ca, thal) drops off a cliff to
# <=72% worst-site availability and is excluded. One qubit per feature,
# no PCA -- 6 features maps directly to 6 qubits.
FEATURE_COLUMNS = [
    "age", "sex", "cp", "restecg", "thalach", "exang",
]

SITES = ["cleveland", "hungarian", "switzerland", "va"]


def load_site(site: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    path = f"{raw_dir}/processed.{site}.data"
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df["site"] = site
    return df


def load_all_sites(raw_dir: str = "data/raw") -> pd.DataFrame:
    df = pd.concat([load_site(s, raw_dir) for s in SITES], ignore_index=True)

    # chol=0 mg/dl is not a real reading -- it's a missing-value code, seen
    # almost exclusively at Switzerland. Treat as missing wherever it appears.
    n_zero_chol = int((df["chol"] == 0).sum())
    df.loc[df["chol"] == 0, "chol"] = np.nan

    df["target"] = (df["num"] > 0).astype(int)
    return df, n_zero_chol


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Modeling-ready columns only: features (ca/thal dropped) + target + site."""
    return df[FEATURE_COLUMNS + ["target", "site"]].copy()


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUMNS if c != "num"]
    rows = {site: df.loc[df["site"] == site, cols].isna().mean() * 100 for site in SITES}
    return pd.DataFrame(rows).T.round(1)


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for site in SITES:
        sub = df.loc[df["site"] == site, "target"]
        n0, n1 = int((sub == 0).sum()), int((sub == 1).sum())
        n = len(sub)
        out.append(
            {
                "site": site,
                "n_total": n,
                "n_class_0": n0,
                "n_class_1": n1,
                "pct_class_1": round(100 * n1 / n, 1),
            }
        )
    return pd.DataFrame(out).set_index("site")


if __name__ == "__main__":
    df, n_zero_chol = load_all_sites()
    print(f"loaded {len(df)} records from {df['site'].nunique()} sites")
    print(f"chol==0 -> NaN conversions: {n_zero_chol} records\n")

    print("=== per-site missingness (%) ===")
    print(missingness_report(df).to_string())
    print()

    print("=== per-site class balance ===")
    print(class_balance_report(df).to_string())
