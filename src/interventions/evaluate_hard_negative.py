#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


EXPECTED = [
    ("Domain-adapted baseline", 31.99, 46.14, 0.841, 0.838, 0.054, 0.078, 0.011, -0.069),
    ("Verb-HN (lambda = 0.25), best", 31.99, 46.16, 0.843, 0.838, 0.055, 0.079, 0.009, -0.072),
    ("Verb-HN (lambda = 0.25), final", 32.89, 46.03, 0.862, 0.844, 0.066, 0.084, 0.020, -0.094),
    ("Verb-HN (lambda = 0.50), best", 31.99, 46.14, 0.843, 0.837, 0.055, 0.080, 0.007, -0.075),
    ("Verb-HN (lambda = 0.50), final", 33.33, 45.18, 0.876, 0.850, 0.079, 0.091, 0.028, -0.108),
]

MODEL_MAP = {
    "M1 adapter": "Domain-adapted baseline",
    "M4 lambda=0.25 best": "Verb-HN (lambda = 0.25), best",
    "M4 lambda=0.25 final": "Verb-HN (lambda = 0.25), final",
    "M4 lambda=0.50 best": "Verb-HN (lambda = 0.50), best",
    "M4 lambda=0.50 final": "Verb-HN (lambda = 0.50), final",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the Ego-ExBind Table 4.3 hard-negative "
            "intervention summary from final evaluation artifacts."
        )
    )
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    required = [
        "Model",
        "mAP AVG",
        "nDCG AVG",
        "NounAcc",
        "VerbAcc",
        "NounMgn",
        "VerbMgn",
        "Noun rho",
        "Verb rho",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing}; have={df.columns.tolist()}")

    df = df[df["Model"].isin(MODEL_MAP)].copy()
    df["Model"] = df["Model"].map(MODEL_MAP)

    order = [x[0] for x in EXPECTED]
    df["Model"] = pd.Categorical(df["Model"], order, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "model": df["Model"].astype(str),
            "mAP_AVG": df["mAP AVG"].astype(float),
            "nDCG_AVG": df["nDCG AVG"].astype(float),
            "NounAcc": df["NounAcc"].astype(float),
            "VerbAcc": df["VerbAcc"].astype(float),
            "NounMgn": df["NounMgn"].astype(float),
            "VerbMgn": df["VerbMgn"].astype(float),
            "Noun_rho": df["Noun rho"].astype(float),
            "Verb_rho": df["Verb rho"].astype(float),
        }
    )

    out_csv = args.output_dir / "table_4_3_hard_negative.csv"
    out.to_csv(out_csv, index=False)

    failed = 0
    print("=== Table 4.3 hard-negative summary ===")
    print(out.to_string(index=False))

    if args.paper_check:
        print("\n=== Table 4.3 validation ===")
        if len(out) != 5:
            print(f"rows: got={len(out)} expected=5 FAIL")
            failed += 1
        else:
            print("rows: got=5 expected=5 PASS")

        for i, exp in enumerate(EXPECTED):
            model, map_avg, ndcg_avg, nacc, vacc, nmgn, vmgn, nrho, vrho = exp
            row = out.iloc[i]
            checks = [
                ("model", row["model"], model, row["model"] == model),
                ("mAP_AVG", round(float(row["mAP_AVG"]), 2), map_avg, round(float(row["mAP_AVG"]), 2) == map_avg),
                ("nDCG_AVG", round(float(row["nDCG_AVG"]), 2), ndcg_avg, round(float(row["nDCG_AVG"]), 2) == ndcg_avg),
                ("NounAcc", float(row["NounAcc"]), nacc, abs(float(row["NounAcc"]) - nacc) <= 0.001),
                ("VerbAcc", float(row["VerbAcc"]), vacc, abs(float(row["VerbAcc"]) - vacc) <= 0.001),
                ("NounMgn", float(row["NounMgn"]), nmgn, abs(float(row["NounMgn"]) - nmgn) <= 0.001),
                ("VerbMgn", float(row["VerbMgn"]), vmgn, abs(float(row["VerbMgn"]) - vmgn) <= 0.001),
                ("Noun_rho", float(row["Noun_rho"]), nrho, abs(float(row["Noun_rho"]) - nrho) <= 0.001),
                ("Verb_rho", float(row["Verb_rho"]), vrho, abs(float(row["Verb_rho"]) - vrho) <= 0.001),
            ]

            row_failed = 0
            print(f"\n{model}")
            for name, got, expected, ok in checks:
                print(f"{name}: got={got} expected={expected} {'PASS' if ok else 'FAIL'}")
                row_failed += int(not ok)
            failed += row_failed

    print("\nOUT_TABLE:", out_csv)
    print(f"SUMMARY failed={failed}")
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
