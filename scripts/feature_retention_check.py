"""Apply the retention rule used to justify dropping ca/thal/chol (>=85%
present at every site) to every remaining candidate column, to check whether
it's being applied consistently. Diagnostic only -- does not change what
data_loader.py actually drops.
"""
import sys

sys.path.insert(0, "scripts")
from data_loader import COLUMNS, load_all_sites, SITES  # noqa: E402

THRESHOLD_PCT_MISSING = 15.0  # ">=85% present" == "<=15% missing"

if __name__ == "__main__":
    df, _ = load_all_sites()
    candidate_cols = [c for c in COLUMNS if c != "num"]

    print(f"rule: retain only if missingness <= {THRESHOLD_PCT_MISSING}% at every site\n")
    header = f"{'column':<10}" + "".join(f"{s:>13}" for s in SITES) + "   verdict"
    print(header)
    print("-" * len(header))

    survivors = []
    for col in candidate_cols:
        pct_missing = {s: round(100 * df.loc[df["site"] == s, col].isna().mean(), 1) for s in SITES}
        fails = any(v > THRESHOLD_PCT_MISSING for v in pct_missing.values())
        verdict = "DROP" if fails else "keep"
        if not fails:
            survivors.append(col)
        row = f"{col:<10}" + "".join(f"{pct_missing[s]:>13}" for s in SITES) + f"   {verdict}"
        print(row)

    print(f"\nsurvivors ({len(survivors)}): {survivors}")
