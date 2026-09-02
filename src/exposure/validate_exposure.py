#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-exposure-csv", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--expected-sc", type=int, default=None)
    parser.add_argument("--expected-uc", type=int, default=None)
    parser.add_argument("--expected-ua", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.query_exposure_csv)

    required = {"v3_exposure_label", "f_pair", "f_verb", "f_noun"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    counts = df["v3_exposure_label"].value_counts().to_dict()
    failed = 0

    print("=== Ego-ExBind exposure validation ===")

    total = len(df)
    print(f"Total: {total}")

    for label in ["SC", "UC", "UA"]:
        print(f"{label}: {counts.get(label, 0)}")

    if args.expected_total is not None and total != args.expected_total:
        print(f"FAIL expected Total={args.expected_total}, got {total}")
        failed += 1

    expected = {
        "SC": args.expected_sc,
        "UC": args.expected_uc,
        "UA": args.expected_ua,
    }

    for label, value in expected.items():
        if value is not None and counts.get(label, 0) != value:
            print(f"FAIL expected {label}={value}, got {counts.get(label, 0)}")
            failed += 1

    sc_bad = df[(df["v3_exposure_label"] == "SC") & (df["f_pair"] <= 0)]
    uc_bad = df[
        (df["v3_exposure_label"] == "UC")
        & ~((df["f_pair"] == 0) & (df["f_verb"] > 0) & (df["f_noun"] > 0))
    ]
    ua_bad = df[
        (df["v3_exposure_label"] == "UA")
        & ~((df["f_pair"] == 0) & ((df["f_verb"] == 0) | (df["f_noun"] == 0)))
    ]

    print(f"SC rule: {'PASS' if len(sc_bad) == 0 else 'FAIL'}")
    print(f"UC rule: {'PASS' if len(uc_bad) == 0 else 'FAIL'}")
    print(f"UA rule: {'PASS' if len(ua_bad) == 0 else 'FAIL'}")

    failed += int(len(sc_bad) > 0)
    failed += int(len(uc_bad) > 0)
    failed += int(len(ua_bad) > 0)

    print(f"SUMMARY failed={failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
