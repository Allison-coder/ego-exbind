#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


EXPECTED = {
    "dec0": {
        "mAP_AVG": (32.02, 0.18, 2),
        "nDCG_AVG": (46.07, 0.20, 2),
        "SC_noun_margin": (0.042, 0.004, 3),
        "UC_noun_margin": (0.102, 0.007, 3),
        "C3_SC_spearman": (0.199, 0.046, 3),
    },
    "dec1": {
        "mAP_AVG": (32.18, 0.07, 2),
        "nDCG_AVG": (45.87, 0.33, 2),
        "SC_noun_margin": (0.043, 0.002, 3),
        "UC_noun_margin": (0.104, 0.006, 3),
        "C3_SC_spearman": (0.245, 0.047, 3),
    },
    "dec50": {
        "mAP_AVG": (31.90, 0.30, 2),
        "nDCG_AVG": (45.51, 0.29, 2),
        "SC_noun_margin": (0.040, 0.002, 3),
        "UC_noun_margin": (0.091, 0.015, 3),
        "C3_SC_spearman": (0.235, 0.044, 3),
    },
    "dec100": {
        "mAP_AVG": (31.92, 0.18, 2),
        "nDCG_AVG": (45.76, 0.29, 2),
        "SC_noun_margin": (0.042, 0.002, 3),
        "UC_noun_margin": (0.101, 0.006, 3),
        "C3_SC_spearman": (0.231, 0.107, 3),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the Ego-ExBind Table 4.5 exposure "
            "decorrelation sweep from controlled C3 probe artifacts."
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
        "config": find_col(df, ["config", "mode", "dec", "lambda_d"]),
        "seed": find_col(df, ["seed", "Seed"]),
        "mAP_AVG": find_col(df, ["mAP_AVG", "mAP AVG", "map_avg"]),
        "nDCG_AVG": find_col(df, ["nDCG_AVG", "nDCG AVG", "ndcg_avg"]),
        "SC_noun_margin": find_col(df, ["SC_noun_margin", "sc_noun_margin", "sc noun margin"]),
        "UC_noun_margin": find_col(df, ["UC_noun_margin", "uc_noun_margin", "uc noun margin"]),
        "C3_SC_spearman": find_col(df, ["C3_SC_spearman", "c3_sc_spearman", "c3 sc spearman", "sc_spearman"]),
    }

    per = pd.DataFrame({
        "config": df[cols["config"]].astype(str),
        "seed": df[cols["seed"]],
        "mAP_AVG": df[cols["mAP_AVG"]].astype(float),
        "nDCG_AVG": df[cols["nDCG_AVG"]].astype(float),
        "SC_noun_margin": df[cols["SC_noun_margin"]].astype(float),
        "UC_noun_margin": df[cols["UC_noun_margin"]].astype(float),
        "C3_SC_spearman": df[cols["C3_SC_spearman"]].astype(float),
        "probe_scope": "controlled_C3",
    })

    order = ["dec0", "dec1", "dec50", "dec100"]
    per["config"] = pd.Categorical(per["config"], order, ordered=True)
    per = per.sort_values(["config", "seed"]).reset_index(drop=True)

    rows = []
    for config in order:
        sub = per[per["config"].astype(str).eq(config)]
        for metric in ["mAP_AVG", "nDCG_AVG", "SC_noun_margin", "UC_noun_margin", "C3_SC_spearman"]:
            rows.append({
                "config": config,
                "metric": metric,
                "mean": sub[metric].mean(),
                "std": sub[metric].std(ddof=1),
            })

    summary = pd.DataFrame(rows)

    out_per = args.output_dir / "table_4_5_decorrelation_per_seed.csv"
    out_table = args.output_dir / "table_4_5_decorrelation.csv"
    per.to_csv(out_per, index=False)
    summary.to_csv(out_table, index=False)

    print("=== Table 4.5 per-seed controlled C3 results ===")
    print(per.to_string(index=False))

    print("\n=== Table 4.5 summary ===")
    print(summary.to_string(index=False))

    failed = 0
    if args.paper_check:
        print("\n=== Table 4.5 three-seed validation ===")
        scope_ok = per["probe_scope"].eq("controlled_C3").all()
        print(f"Probe scope: controlled C3 {'PASS' if scope_ok else 'FAIL'}")
        failed += int(not scope_ok)

        for config, metrics in EXPECTED.items():
            sub = per[per["config"].astype(str).eq(config)]
            n_seed = sub["seed"].nunique()
            ok_seed = n_seed == 3
            print(f"\n{config} seeds: got={n_seed} expected=3 {'PASS' if ok_seed else 'FAIL'}")
            failed += int(not ok_seed)

            for metric, (exp_mean, exp_std, digits) in metrics.items():
                row = summary[
                    summary["config"].eq(config) & summary["metric"].eq(metric)
                ].iloc[0]
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

    print("\nOUT_PER_SEED:", out_per)
    print("OUT_TABLE:", out_table)
    print(f"SUMMARY failed={failed}")
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
