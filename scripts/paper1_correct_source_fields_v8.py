"""Apply the paper 1 source audit to an author-supplied v7 CSV.

The integrated literature dataset is not included in this repository. The
script changes only two predictor fields; it does not change bond outcomes.
"""
import argparse
import re

import pandas as pd


SUN_DOI = "10.1617/s11527-024-02507-7"
VALIKHANI_DOI = "10.1016/j.conbuildmat.2019.117753"


def correct(source, output):
    df = pd.read_csv(source)
    sun = df["doi"].eq(SUN_DOI)
    valikhani = df["doi"].eq(VALIKHANI_DOI)
    if sun.sum() != 16 or df.loc[sun, "roughness_mm"].notna().sum() != 16:
        raise ValueError("Expected 16 Sun/Zhang records with v7 roughness values")
    if valikhani.sum() != 25 or not df.loc[valikhani, "curing_age_days"].eq(28).all():
        raise ValueError("Expected 25 Valikhani records coded with 28 days in v7")
    df.loc[sun, "roughness_mm"] = float("nan")
    df.loc[valikhani, "curing_age_days"] = 300.0
    for idx in df.index[sun]:
        note = str(df.at[idx, "notes"])
        note = re.sub(r"gamma_mm=([0-9.]+) \(数字图像Ra,mm\)",
                      r"failure_slip_gamma_mm=\1 (post-cracking response, not roughness)", note)
        df.at[idx, "notes"] = note + " | AUDIT v8: roughness_mm set missing."
    for idx in df.index[valikhani]:
        note = str(df.at[idx, "notes"])
        note = re.sub(r"curing_age_days=28 recorded as effective bond age:.*?extra 9 mo = ambient storage\.",
                      "curing_age_days=300 approximately; tests were conducted ten months after UHPC casting (10x30 d).", note)
        df.at[idx, "notes"] = note + " | AUDIT v8: source age corrected from v7's 28 d."
    df.to_csv(output, index=False)
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Author-supplied v7 CSV")
    parser.add_argument("--output", required=True, help="Corrected v8 CSV")
    args = parser.parse_args()
    count = correct(args.input, args.output)
    print(f"Wrote {count} records to {args.output}; corrected 16 gamma and 25 age values")
