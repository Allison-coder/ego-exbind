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
        description=(
            "Build Ego-ExBind exposure analysis units and compute "
            "pair-frequency retrieval correlations."
        )
    )
    parser.add_argument("--v2t-units-csv", type=Path, required=True)
    parser.add_argument("--t2v-units-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-n-units", type=int, default=3)
    parser.add_argument(
        "--paper-check",
        action="store_true",
        help="Validate against the canonical paper analysis setting.",
    )
    return parser.parse_args()


V2T_CSV = None
T2V_CSV = None
OUT_DIR = None
OUT_V2T_PAIR = None
OUT_T2V_PAIR = None
OUT_STATS = None
OUT_GROUP_SUMMARY = None
OUT_FIG_PNG = None
OUT_FIG_PDF = None
OUT_README = None
OUT_MAPPING = None

EXPECTED = {
    "V2T_mAP_x100": 29.88172021139886,
    "V2T_nDCG_x100": 30.071646777422366,
    "T2V_mAP_x100": 23.448891050955933,
    "T2V_nDCG_x100": 28.12497337856873,
}

# Main Figure 4.1 reliability rule.
# Pair-level means from one or two query units are high-variance estimates.
# Sensitivity tables retain all-pair / n>=2 / n>=3 / n>=5 results.
MAIN_MIN_N_UNITS = 3

def p_stars(p):
    if not np.isfinite(p):
        return ""
    # Match the simple convention used in the reference figure:
    # ** means significant at p < 0.05.
    return "**" if p < 0.05 else "n.s."

def check_required_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")

def check_pair_constant(sc, cols, name):
    bad_rows = []
    for c in cols:
        nunq = sc.groupby("pair_id")[c].nunique(dropna=False)
        bad = nunq[nunq > 1]
        if len(bad):
            bad_rows.append((c, len(bad), bad.head(20).to_dict()))

    if bad_rows:
        print(f"\n=== BAD non-constant pair-level columns in {name} ===")
        for item in bad_rows:
            print(item)
        raise RuntimeError(f"{name}: pair-level frequency/provenance columns are not constant within pair_id.")

def summarize_exposure_groups(v2t, t2v):
    rows = []
    specs = [
        (v2t, "V2T", "V2T_AP", "mAP"),
        (v2t, "V2T", "V2T_nDCG", "nDCG"),
        (t2v, "T2V", "T2V_AP", "mAP"),
        (t2v, "T2V", "T2V_nDCG", "nDCG"),
    ]

    for df, direction, score_col, metric in specs:
        for label in ["SC", "UC", "UA"]:
            d = df[df["v3_exposure_label"] == label]
            vals = pd.to_numeric(d[score_col], errors="coerce").dropna()
            rows.append({
                "direction": direction,
                "metric": metric,
                "exposure_label": label,
                "n_units": int(len(vals)),
                "mean_x100": float(vals.mean() * 100),
                "std_x100": float(vals.std(ddof=1) * 100),
                "sem_x100": float(vals.sem() * 100),
            })

    return pd.DataFrame(rows)


def make_pair_level(df, direction):
    if direction == "V2T":
        metric_ap = "V2T_AP"
        metric_ndcg = "V2T_nDCG"
        unit_id = "query_id" if "query_id" in df.columns else None
    elif direction == "T2V":
        metric_ap = "T2V_AP"
        metric_ndcg = "T2V_nDCG"
        unit_id = "official_text_uid" if "official_text_uid" in df.columns else None
    else:
        raise ValueError(direction)

    required = [
        "pair_id",
        "v3_exposure_label",
        "log10_f_pair",
        "PMI",
        "log10_f_verb",
        "log10_f_noun",
        metric_ap,
        metric_ndcg,
    ]
    check_required_columns(df, required, direction)

    sc = df[df["v3_exposure_label"] == "SC"].copy()

    print(f"\n=== {direction} SC rows ===")
    print("rows:", len(sc))
    print("unique pairs:", sc["pair_id"].nunique())

    # SC should have positive pair frequency and finite log10_f_pair.
    bad_log = sc[~np.isfinite(sc["log10_f_pair"].astype(float))]
    if len(bad_log):
        print(bad_log[["pair_id", "log10_f_pair"]].head(20).to_string(index=False))
        raise RuntimeError(f"{direction}: non-finite log10_f_pair in SC rows.")

    if "f_pair" in sc.columns:
        bad_f = sc[sc["f_pair"].astype(float) <= 0]
        if len(bad_f):
            print(bad_f[["pair_id", "f_pair", "log10_f_pair"]].head(20).to_string(index=False))
            raise RuntimeError(f"{direction}: SC rows with f_pair <= 0.")

    check_pair_constant(
        sc,
        ["log10_f_pair", "PMI", "log10_f_verb", "log10_f_noun"],
        direction,
    )

    agg = {
        metric_ap: "mean",
        metric_ndcg: "mean",
        "log10_f_pair": "first",
        "PMI": "first",
        "log10_f_verb": "first",
        "log10_f_noun": "first",
    }

    if "f_pair" in sc.columns:
        agg["f_pair"] = "first"
    if "f_verb" in sc.columns:
        agg["f_verb"] = "first"
    if "f_noun" in sc.columns:
        agg["f_noun"] = "first"

    pair = sc.groupby("pair_id", as_index=False).agg(agg)

    # Count analysis units per pair.
    if unit_id is not None and unit_id in sc.columns:
        n_units = sc.groupby("pair_id")[unit_id].count().rename("n_units").reset_index()
    else:
        n_units = sc.groupby("pair_id").size().rename("n_units").reset_index()

    pair = pair.merge(n_units, on="pair_id", how="left")
    pair["direction"] = direction

    # Rename metric columns to common names.
    pair = pair.rename(columns={
        metric_ap: "mean_mAP",
        metric_ndcg: "mean_nDCG",
    })

    # Percent scale for plotting and paper tables.
    pair["mean_mAP_x100"] = pair["mean_mAP"] * 100
    pair["mean_nDCG_x100"] = pair["mean_nDCG"] * 100

    # Keep clean column order.
    front = [
        "direction",
        "pair_id",
        "n_units",
        "mean_mAP",
        "mean_nDCG",
        "mean_mAP_x100",
        "mean_nDCG_x100",
        "log10_f_pair",
        "PMI",
        "log10_f_verb",
        "log10_f_noun",
    ]
    rest = [c for c in pair.columns if c not in front]
    pair = pair[front + rest]

    return pair

def corr_one(pair, direction, metric, y_col):
    d = pair[pair["n_units"] >= MAIN_MIN_N_UNITS].copy()
    d = d[["pair_id", "n_units", "log10_f_pair", y_col]].dropna().copy()
    d = d[np.isfinite(d["log10_f_pair"]) & np.isfinite(d[y_col])]

    x = d["log10_f_pair"].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    if len(d) < 3:
        raise RuntimeError(f"Too few pairs for correlation: {direction} {metric}")

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)

    # Simple OLS line for visualization and slope reporting.
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "figure": "pair_frequency",
        "direction": direction,
        "metric": metric,
        "x": "log10_f_pair",
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
    d = d[["log10_f_pair", y_col, "n_units"]].dropna().copy()
    d = d[np.isfinite(d["log10_f_pair"]) & np.isfinite(d[y_col])]

    x = d["log10_f_pair"].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    # Scatter: each point is one EK100 verb-noun pair.
    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )

    # OLS trend line.
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
    ax.set_xlabel(r"$\log_{10} f(v,n)$", fontsize=9.5)
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

def write_readme(v2t_pair, t2v_pair, stat_df):
    text = f"""# Figure 4.1: SC pair frequency vs retrieval performance

This folder contains the SC-only pair-level analysis for Figure 4.1.

## Purpose

Figure 4.1 tests whether pair-supported pretraining frequency predicts EK100-MIR retrieval performance within the SC group. The main figure uses SC pairs supported by at least three official query units.

The analysis is restricted to:

- `v3_exposure_label == "SC"`

because only SC pairs have observed pair frequency:

- `f(v,n) > 0`
- `log10_f_pair` is finite

## Frozen inputs

All analyses read only frozen metric unit tables from:

`--v2t-units-csv` and `--t2v-units-csv`

Input files:

- the V2T metric-unit table
- the T2V metric-unit table

## Unit transformation

Query-level units are collapsed to pair-level units:

`pair_id = verb_class + "_" + noun_class`

For each pair, the script computes:

- mean mAP
- mean nDCG
- first `log10_f_pair`
- first `PMI`
- first `log10_f_verb`
- first `log10_f_noun`
- `n_units`

## Figure panels

- (a) V→T mAP vs `log10_f_pair`
- (b) V→T nDCG vs `log10_f_pair`
- (c) T→V mAP vs `log10_f_pair`
- (d) T→V nDCG vs `log10_f_pair`

Each point is one EK100 verb-noun pair with `n_units >= 3`. The support threshold is stated in the caption rather than repeated in every panel.

The fitted line is an ordinary least squares trend line.

## Statistics

The main statistics table reports correlations after applying `n_units >= 3`. The all-pair and alternative-threshold results are retained in the reliability sensitivity table.

- Pearson r
- Pearson p-value
- Spearman rho
- Spearman p-value
- OLS slope
- N pairs

Significance stars follow the visual convention used in the reference frequency-performance plots:

- `**`: p < 0.05
- `n.s.`: p >= 0.05

## Outputs

- `{OUT_V2T_PAIR.name}`
- `{OUT_T2V_PAIR.name}`
- `{OUT_STATS.name}`
- `{OUT_FIG_PNG.name}`
- `{OUT_FIG_PDF.name}`
- `{OUT_MAPPING.name}`
"""
    OUT_README.write_text(text, encoding="utf-8")

def write_mapping_table():
    text = """# Figure 4.1 method mapping to frequency-performance prior work

This table maps the reference concept-frequency analysis to the current EK100 verb-noun composition setting.

| Reference frequency-performance analysis | This project |
|---|---|
| concept `c` | EK100 verb-noun pair `(v,n)` |
| concept frequency `f(c)` | v3 EgoClip pair-supported frequency `f(v,n)` |
| log-scaled concept frequency | `log10_f_pair = log10 f(v,n)` |
| concept-wise retrieval score | pair-wise mean mAP / nDCG |
| I2T / T2I retrieval | V→T / T→V retrieval |
| Recall@K | EK100-MIR official mAP / nDCG |
| multiple pretraining datasets | not applicable; one EgoClip exposure source |
| multiple model architectures | not applicable; one EgoVLPv2 model |
| Pearson correlation | Pearson r plus Spearman rho and p-values |

Figure 4.1 uses the scatter version as the main statistical evidence: each point is one pair.

A separate binned line plot can be generated later as a visual companion, but the scatter plot should remain the primary evidence because it preserves pair-level observations.
"""
    OUT_MAPPING.write_text(text, encoding="utf-8")

def main():
    global V2T_CSV, T2V_CSV, OUT_DIR
    global OUT_V2T_PAIR, OUT_T2V_PAIR, OUT_STATS, OUT_GROUP_SUMMARY, OUT_FIG_PNG, OUT_FIG_PDF
    global OUT_README, OUT_MAPPING, MAIN_MIN_N_UNITS

    args = parse_args()

    V2T_CSV = args.v2t_units_csv
    T2V_CSV = args.t2v_units_csv
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_V2T_PAIR = OUT_DIR / "sc_pair_units_v2t.csv"
    OUT_T2V_PAIR = OUT_DIR / "sc_pair_units_t2v.csv"
    OUT_STATS = OUT_DIR / "pair_frequency_correlations.csv"
    OUT_GROUP_SUMMARY = OUT_DIR / "exposure_group_summary.csv"
    OUT_FIG_PNG = OUT_DIR / "figure_4_1_pair_frequency.png"
    OUT_FIG_PDF = OUT_DIR / "figure_4_1_pair_frequency.pdf"
    OUT_README = OUT_DIR / "README.md"
    OUT_MAPPING = OUT_DIR / "pair_frequency_method_mapping.md"

    MAIN_MIN_N_UNITS = args.min_n_units

    print("V2T_CSV:", V2T_CSV, V2T_CSV.exists())
    print("T2V_CSV:", T2V_CSV, T2V_CSV.exists())
    print("OUT_DIR:", OUT_DIR)
    print("MAIN_MIN_N_UNITS:", MAIN_MIN_N_UNITS)

    v2t = pd.read_csv(V2T_CSV)
    t2v = pd.read_csv(T2V_CSV)

    print("\n=== input row counts ===")
    print("V2T rows:", len(v2t))
    print("T2V rows:", len(t2v))

    if len(v2t) != 9668:
        raise RuntimeError(f"Expected V2T rows 9668, got {len(v2t)}")
    if len(t2v) != 3842:
        raise RuntimeError(f"Expected T2V rows 3842, got {len(t2v)}")

    print("\n=== official aggregate sanity ===")
    sanity = {
        "V2T_mAP_x100": float(v2t["V2T_AP"].mean() * 100),
        "V2T_nDCG_x100": float(v2t["V2T_nDCG"].mean() * 100),
        "T2V_mAP_x100": float(t2v["T2V_AP"].mean() * 100),
        "T2V_nDCG_x100": float(t2v["T2V_nDCG"].mean() * 100),
    }

    max_diff = 0.0
    for k, v in sanity.items():
        diff = abs(v - EXPECTED[k])
        max_diff = max(max_diff, diff)
        print(f"{k}: {v:.12f} expected={EXPECTED[k]:.12f} diff={diff:.12g}")

    if max_diff > 1e-6:
        raise RuntimeError(f"Official aggregate sanity failed: max_diff={max_diff}")

    group_summary = summarize_exposure_groups(v2t, t2v)

    v2t_pair = make_pair_level(v2t, "V2T")
    t2v_pair = make_pair_level(t2v, "T2V")

    group_summary.to_csv(OUT_GROUP_SUMMARY, index=False)

    v2t_pair.to_csv(OUT_V2T_PAIR, index=False)
    t2v_pair.to_csv(OUT_T2V_PAIR, index=False)

    stat_df = make_stats(v2t_pair, t2v_pair)
    stat_df.to_csv(OUT_STATS, index=False)

    print("\n=== pair-level sizes ===")
    print("V2T pair rows all:", len(v2t_pair))
    print("T2V pair rows all:", len(t2v_pair))
    print("MAIN_MIN_N_UNITS:", MAIN_MIN_N_UNITS)
    v2t_main = int((v2t_pair["n_units"] >= MAIN_MIN_N_UNITS).sum())
    t2v_main = int((t2v_pair["n_units"] >= MAIN_MIN_N_UNITS).sum())

    print("V2T pair rows main:", v2t_main)
    print("T2V pair rows main:", t2v_main)

    if args.paper_check:
        if MAIN_MIN_N_UNITS != 3:
            raise RuntimeError("--paper-check requires --min-n-units 3.")
        if v2t_main != 537:
            raise RuntimeError(f"Expected V2T n>=3 pairs = 537, got {v2t_main}")
        if t2v_main != 374:
            raise RuntimeError(f"Expected T2V n>=3 pairs = 374, got {t2v_main}")

    print("SUMMARY failed=0")

    print("\n=== correlation stats ===")
    print(stat_df.to_string(index=False))

    plot_figure(v2t_pair, t2v_pair, stat_df)
    write_readme(v2t_pair, t2v_pair, stat_df)
    write_mapping_table()

    print("\nOUT_V2T_PAIR:", OUT_V2T_PAIR)
    print("OUT_T2V_PAIR:", OUT_T2V_PAIR)
    print("OUT_STATS:", OUT_STATS)
    print("OUT_GROUP_SUMMARY:", OUT_GROUP_SUMMARY)
    print("OUT_FIG_PNG:", OUT_FIG_PNG)
    print("OUT_FIG_PDF:", OUT_FIG_PDF)
    print("OUT_README:", OUT_README)
    print("OUT_MAPPING:", OUT_MAPPING)
    print("SUMMARY failed=0")

if __name__ == "__main__":
    main()
