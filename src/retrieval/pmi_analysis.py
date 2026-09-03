#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Ego-ExBind PMI retrieval analysis for Figure 4.2."
    )
    parser.add_argument("--v2t-pair-csv", type=Path, required=True)
    parser.add_argument("--t2v-pair-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-n-units", type=int, default=3)
    parser.add_argument(
        "--paper-check",
        action="store_true",
        help="Validate against the canonical Figure 4.2 analysis setting.",
    )
    return parser.parse_args()


MAIN_MIN_N_UNITS = 3
V2T_PAIR = None
T2V_PAIR = None
OUT_STATS = None
OUT_FIG_PNG = None
OUT_FIG_PDF = None
OUT_INPUTS = None
OUT_README = None
OUT_CAPTION = None
OUT_MAPPING = None

def p_stars(p):
    if not np.isfinite(p):
        return ""
    return "**" if p < 0.05 else " n.s."

def corr_one(pair, direction, metric, y_col):
    d = pair[pair["n_units"] >= MAIN_MIN_N_UNITS].copy()
    d = d[["pair_id", "n_units", "PMI", y_col]].dropna().copy()
    d = d[np.isfinite(d["PMI"]) & np.isfinite(d[y_col])]

    x = d["PMI"].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    if len(d) < 3:
        raise RuntimeError(f"Too few pairs for {direction} {metric}")

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "figure": "pmi",
        "direction": direction,
        "metric": metric,
        "x": "PMI",
        "y": y_col,
        "min_n_units": MAIN_MIN_N_UNITS,
        "n_pairs": int(len(d)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "pearson_sig": p_stars(float(pearson.pvalue)),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "spearman_sig": p_stars(float(spearman.pvalue)),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "ols_p": float(p_value),
        "ols_stderr": float(std_err),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_mean": float(np.mean(y)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }

def make_stats(v2t_pair, t2v_pair):
    rows = []
    rows.append(corr_one(v2t_pair, "V2T", "mAP", "mean_mAP_x100"))
    rows.append(corr_one(v2t_pair, "V2T", "nDCG", "mean_nDCG_x100"))
    rows.append(corr_one(t2v_pair, "T2V", "mAP", "mean_mAP_x100"))
    rows.append(corr_one(t2v_pair, "T2V", "nDCG", "mean_nDCG_x100"))
    return pd.DataFrame(rows)

def add_panel(ax, pair, stat_row, title, y_col, y_label):
    d = pair[pair["n_units"] >= MAIN_MIN_N_UNITS].copy()
    d = d[["PMI", y_col, "n_units"]].dropna().copy()
    d = d[np.isfinite(d["PMI"]) & np.isfinite(d[y_col])]

    x = d["PMI"].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )

    slope = float(stat_row["ols_slope"])
    intercept = float(stat_row["ols_intercept"])
    xx = np.linspace(np.min(x), np.max(x), 200)
    yy = intercept + slope * xx
    ax.plot(xx, yy, linewidth=1.2)

    r = float(stat_row["pearson_r"])
    p = float(stat_row["pearson_p"])
    rho = float(stat_row["spearman_rho"])
    sp = float(stat_row["spearman_p"])
    n = int(stat_row["n_pairs"])

    label = (
        f"r={r:.2f}{p_stars(p)}, ρ={rho:.2f}{p_stars(sp)}\n"
        f"N={n}"
    )

    ax.text(
        0.04,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("PMI(v,n)", fontsize=9.5)
    ax.set_ylabel(y_label, fontsize=9.5)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    ax.grid(alpha=0.20, linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def plot_figure(v2t_pair, t2v_pair, stat_df):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
        facecolor="white",
    )

    panels = [
        ("V2T", "mAP", "(a) V→T mAP", v2t_pair, "mean_mAP_x100", "mAP (%)", axes[0, 0]),
        ("V2T", "nDCG", "(b) V→T nDCG", v2t_pair, "mean_nDCG_x100", "nDCG (%)", axes[0, 1]),
        ("T2V", "mAP", "(c) T→V mAP", t2v_pair, "mean_mAP_x100", "mAP (%)", axes[1, 0]),
        ("T2V", "nDCG", "(d) T→V nDCG", t2v_pair, "mean_nDCG_x100", "nDCG (%)", axes[1, 1]),
    ]

    for direction, metric, title, pair, y_col, y_label, ax in panels:
        ax.set_facecolor("white")
        stat_row = stat_df[
            (stat_df["direction"] == direction)
            & (stat_df["metric"] == metric)
        ].iloc[0]
        add_panel(ax, pair, stat_row, title, y_col, y_label)

    fig.tight_layout(w_pad=1.3, h_pad=1.3)
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight", facecolor="white")

def write_docs():
    OUT_README.write_text(
        f"""# Figure 4.2: SC pair PMI vs retrieval performance

## Research question

Within SC pairs, does verb-noun association strength predict pair-wise EK100-MIR retrieval performance?

Figure 2a tests pair frequency:

- x = log10 f(v,n)

Figure 4.2 tests pair association:

- x = PMI(v,n)

## Main analysis rule

The main Figure 4.2 analysis uses the same rule as Figure 2a:

- `v3_exposure_label == SC`
- pair-level aggregation by `pair_id`
- `n_units >= {MAIN_MIN_N_UNITS}`

This keeps Figure 2a, Figure 4.2, and Table 1 on the same statistical unit.

## Inputs

Figure 4.2 reads the pair-level SC tables generated for Figure 2a:

- `{V2T_PAIR}`
- `{T2V_PAIR}`

## Panels

- (a) V→T mAP vs PMI(v,n)
- (b) V→T nDCG vs PMI(v,n)
- (c) T→V mAP vs PMI(v,n)
- (d) T→V nDCG vs PMI(v,n)

Each point is one SC verb-noun pair with at least three official query units.

The fitted line is an OLS trend line.

Panel annotations report Pearson r, Spearman ρ, and N pairs.
""",
        encoding="utf-8",
    )

    OUT_CAPTION.write_text(
        """# Figure 4.2 caption draft

Pair association strength is compared with retrieval performance within seen compositions. Each point is one SC verb-noun composition with at least three official query units. The x-axis shows PMI(v,n), computed from v3 EgoClip exposure counts. PMI uses the natural logarithm. The y-axis shows pair-wise mean mAP or nDCG over official EK100-MIR query units. Pearson r and Spearman ρ are reported in each panel; ** denotes p<0.05.

Suggested text:

Figure 4.2 tests whether association strength between the verb and noun, rather than raw pair frequency alone, is related to retrieval performance.
""",
        encoding="utf-8",
    )

    OUT_MAPPING.write_text(
        """# Figure 4.2 method mapping to Qu & Xie

Figure 4.2 is the direct analogue of Qu & Xie's key-pair PMI analysis.

| Qu & Xie key-pair PMI analysis | This project |
|---|---|
| concept pair `(c_accessory, c_ImageNet)` | EK100 verb-noun pair `(v,n)` |
| x-axis: PMI of key concept pair | x-axis: PMI(v,n) |
| y-axis: zero-shot top-1 / top-5 accuracy | y-axis: V→T / T→V mAP and nDCG |
| key pair-level PMI, not caption-average PMI | verb-noun pair-level PMI, not averaging over multiple pairs |
| correlation-based evidence | Pearson r + Spearman ρ + OLS slope |

Table 1 is not a direct reproduction of a Qu & Xie table. It is a methodological extension of their PMI-performance correlation: we use nested regression to test whether PMI explains retrieval performance beyond constituent verb and noun frequencies.
""",
        encoding="utf-8",
    )

def main():
    global MAIN_MIN_N_UNITS, V2T_PAIR, T2V_PAIR
    global OUT_STATS, OUT_FIG_PNG, OUT_FIG_PDF, OUT_INPUTS
    global OUT_README, OUT_CAPTION, OUT_MAPPING

    args = parse_args()

    MAIN_MIN_N_UNITS = args.min_n_units
    V2T_PAIR = args.v2t_pair_csv
    T2V_PAIR = args.t2v_pair_csv

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    OUT_STATS = output_dir / "pmi_correlations.csv"
    OUT_FIG_PNG = output_dir / "figure_4_2_pmi.png"
    OUT_FIG_PDF = output_dir / "figure_4_2_pmi.pdf"
    OUT_INPUTS = output_dir / "pmi_analysis_inputs.csv"
    OUT_README = output_dir / "README.md"
    OUT_CAPTION = output_dir / "figure_4_2_caption.md"
    OUT_MAPPING = output_dir / "pmi_method_mapping.md"

    print("V2T_PAIR:", V2T_PAIR, V2T_PAIR.exists())
    print("T2V_PAIR:", T2V_PAIR, T2V_PAIR.exists())
    print("OUT_DIR:", output_dir)
    print("MAIN_MIN_N_UNITS:", MAIN_MIN_N_UNITS)

    v2t_pair = pd.read_csv(V2T_PAIR)
    t2v_pair = pd.read_csv(T2V_PAIR)

    print("\n=== pair rows all ===")
    print("V2T:", len(v2t_pair))
    print("T2V:", len(t2v_pair))

    print("\n=== pair rows main n>=3 ===")
    print("V2T:", int((v2t_pair["n_units"] >= MAIN_MIN_N_UNITS).sum()))
    print("T2V:", int((t2v_pair["n_units"] >= MAIN_MIN_N_UNITS).sum()))

    required = [
        "pair_id",
        "n_units",
        "PMI",
        "mean_mAP_x100",
        "mean_nDCG_x100",
        "log10_f_pair",
        "log10_f_verb",
        "log10_f_noun",
    ]

    for name, df in [("V2T", v2t_pair), ("T2V", t2v_pair)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"{name} missing columns: {missing}")

    v2t_main = v2t_pair[v2t_pair["n_units"] >= MAIN_MIN_N_UNITS].copy()
    t2v_main = t2v_pair[t2v_pair["n_units"] >= MAIN_MIN_N_UNITS].copy()
    v2t_main["direction"] = "V2T"
    t2v_main["direction"] = "T2V"
    pd.concat([v2t_main, t2v_main], ignore_index=True).to_csv(OUT_INPUTS, index=False)

    stat_df = make_stats(v2t_pair, t2v_pair)
    stat_df.to_csv(OUT_STATS, index=False)

    print("\n=== Figure 4.2 stats ===")
    print(stat_df.to_string(index=False))

    plot_figure(v2t_pair, t2v_pair, stat_df)
    write_docs()

    if args.paper_check:
        if MAIN_MIN_N_UNITS != 3:
            raise RuntimeError("--paper-check requires --min-n-units 3.")

        expected = {
            ("V2T", "mAP"): (0.06, 0.07, False),
            ("V2T", "nDCG"): (0.04, 0.06, False),
            ("T2V", "mAP"): (0.21, 0.17, True),
            ("T2V", "nDCG"): (0.25, 0.21, True),
        }

        failed = 0
        for _, row in stat_df.iterrows():
            key = (row["direction"], row["metric"])
            pearson, spearman, significant = expected[key]
            p_ok = abs(float(row["pearson_r"]) - pearson) <= 0.03
            s_ok = abs(float(row["spearman_rho"]) - spearman) <= 0.03
            sig_ok = (row["pearson_sig"].strip() == "**") == significant
            failed += int(not (p_ok and s_ok and sig_ok))

        if failed:
            raise RuntimeError(f"Figure 4.2 paper check failed: {failed}")

    print("\nOUT_STATS:", OUT_STATS)
    print("OUT_FIG_PNG:", OUT_FIG_PNG)
    print("OUT_FIG_PDF:", OUT_FIG_PDF)
    print("OUT_INPUTS:", OUT_INPUTS)
    print("OUT_README:", OUT_README)
    print("OUT_CAPTION:", OUT_CAPTION)
    print("OUT_MAPPING:", OUT_MAPPING)
    print("SUMMARY failed=0")

if __name__ == "__main__":
    main()
