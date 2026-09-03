#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


EXPECTED = {
    "mAP_AVG": (32.02, 0.18, 2),
    "nDCG_AVG": (46.07, 0.20, 2),
    "NounMgn": (0.055, 0.001, 3),
    "VerbMgn": (0.077, 0.002, 3),
    "Noun_rho": (0.009, 0.008, 3),
    "Verb_rho": (-0.086, 0.014, 3),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the Ego-ExBind Table 4.4 anchor/protect "
            "three-seed intervention summary from final evaluation artifacts."
        )
    )
    parser.add_argument("--per-seed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-check", action="store_true")
    return parser.parse_args()


def find_col(df, names):
    norm = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "_").replace("-", "_")
        if key in norm:
            return norm[key]
    raise KeyError(f"Missing any of {names}; have={df.columns.tolist()}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_seed_csv)

    cols = {
        "Seed": find_col(df, ["Seed", "seed"]),
        "mAP_AVG": find_col(df, ["mAP AVG", "mAP_AVG", "map_avg"]),
        "nDCG_AVG": find_col(df, ["nDCG AVG", "nDCG_AVG", "ndcg_avg"]),
        "NounMgn": find_col(df, ["NounMgn", "noun_margin", "noun margin", "large_noun_margin"]),
        "VerbMgn": find_col(df, ["VerbMgn", "verb_margin", "verb margin", "large_verb_margin"]),
        "Noun_rho": find_col(df, ["Noun rho", "noun_spearman", "noun rho", "large_noun_spearman"]),
        "Verb_rho": find_col(df, ["Verb rho", "verb_spearman", "verb rho", "large_verb_spearman"]),
    }

    out_per = pd.DataFrame({
        "seed": df[cols["Seed"]],
        "mAP_AVG": df[cols["mAP_AVG"]].astype(float),
        "nDCG_AVG": df[cols["nDCG_AVG"]].astype(float),
        "NounMgn": df[cols["NounMgn"]].astype(float),
        "VerbMgn": df[cols["VerbMgn"]].astype(float),
        "Noun_rho": df[cols["Noun_rho"]].astype(float),
        "Verb_rho": df[cols["Verb_rho"]].astype(float),
        "probe_scope": "large_val_SC",
    })

    rows = []
    for metric in ["mAP_AVG", "nDCG_AVG", "NounMgn", "VerbMgn", "Noun_rho", "Verb_rho"]:
        rows.append({
            "metric": metric,
            "mean": out_per[metric].mean(),
            "std": out_per[metric].std(ddof=1),
        })

    out_summary = pd.DataFrame(rows)

    out_per_csv = args.output_dir / "table_4_4_anchor_protect_per_seed.csv"
    out_summary_csv = args.output_dir / "table_4_4_anchor_protect.csv"

    out_per.to_csv(out_per_csv, index=False)
    out_summary.to_csv(out_summary_csv, index=False)

    print("=== Table 4.4 per-seed anchor/protect results ===")
    print(out_per.to_string(index=False))

    print("\n=== Table 4.4 three-seed summary ===")
    print(out_summary.to_string(index=False))

    failed = 0
    if args.paper_check:
        n_seeds = out_per["seed"].nunique()
        ok = n_seeds == 3
        print(f"\nSeeds: got={n_seeds} expected=3 {'PASS' if ok else 'FAIL'}")
        failed += int(not ok)

        scope_ok = out_per["probe_scope"].eq("large_val_SC").all()
        print(f"Probe scope: large val-SC {'PASS' if scope_ok else 'FAIL'}")
        failed += int(not scope_ok)

        print("\n=== Table 4.4 validation ===")
        for metric, (exp_mean, exp_std, digits) in EXPECTED.items():
            row = out_summary[out_summary["metric"].eq(metric)].iloc[0]
            got_mean = round(float(row["mean"]), digits)
            got_std = round(float(row["std"]), digits)
            exp_mean = round(exp_mean, digits)
            exp_std = round(exp_std, digits)
            ok = got_mean == exp_mean and got_std == exp_std
            print(
                f"{metric}: got={got_mean} +/- {got_std} "
                f"expected={exp_mean} +/- {exp_std} {'PASS' if ok else 'FAIL'}"
            )
            failed += int(not ok)

    print("\nOUT_PER_SEED:", out_per_csv)
    print("OUT_TABLE:", out_summary_csv)
    print(f"SUMMARY failed={failed}")
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
