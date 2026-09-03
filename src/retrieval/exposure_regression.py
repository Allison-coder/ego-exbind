#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Ego-ExBind atom-, pair-, and PMI-based "
            "retrieval regressions with HC3 robust standard errors."
        )
    )
    parser.add_argument("--v2t-pair-csv", type=Path, required=True)
    parser.add_argument("--t2v-pair-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-n-units",
        type=int,
        default=3,
        help="Minimum number of retrieval units per pair.",
    )
    parser.add_argument(
        "--paper-check",
        action="store_true",
        help="Validate against the canonical Table 4.1 analysis setting.",
    )
    parser.add_argument("--output-long-name", default="table_4_1_regression_long.csv")
    parser.add_argument("--output-wide-name", default="table_4_1_regression_wide.csv")
    parser.add_argument("--output-inputs-name", default="table_4_1_regression_inputs.csv")
    parser.add_argument("--output-diagnostics-name", default="table_4_1_predictor_diagnostics.csv")
    return parser.parse_args()


V2T_PAIR = None
T2V_PAIR = None
OUT_DIR = None
OUT_TABLE_LONG = None
OUT_TABLE_WIDE = None
OUT_INPUTS = None
OUT_DIAG = None
OUT_README = None
OUT_CAPTION = None

MAIN_MIN_N_UNITS = 3

REQUIRED_COLS = [
    "pair_id",
    "n_units",
    "mean_mAP_x100",
    "mean_nDCG_x100",
    "log10_f_verb",
    "log10_f_noun",
    "log10_f_pair",
    "PMI",
]

MODELS = {
    "A_atom_only": {
        "predictors": ["log10_f_verb", "log10_f_noun"],
        "key_predictor": "",
        "description": "score ~ log10_f_verb + log10_f_noun",
    },
    "B_atom_plus_pair_count": {
        "predictors": ["log10_f_verb", "log10_f_noun", "log10_f_pair"],
        "key_predictor": "log10_f_pair",
        "description": "score ~ log10_f_verb + log10_f_noun + log10_f_pair",
    },
    "C_atom_plus_PMI": {
        "predictors": ["log10_f_verb", "log10_f_noun", "PMI"],
        "key_predictor": "PMI",
        "description": "score ~ log10_f_verb + log10_f_noun + PMI",
    },
}

OUTCOMES = [
    ("V2T", "mAP", "mean_mAP_x100"),
    ("V2T", "nDCG", "mean_nDCG_x100"),
    ("T2V", "mAP", "mean_mAP_x100"),
    ("T2V", "nDCG", "mean_nDCG_x100"),
]

def p_fmt(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-99:
        return "<1e-99"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"

def sig_label(p):
    if not np.isfinite(p):
        return ""
    return "**" if p < 0.05 else "n.s."

def fit_ols_hc3(y, X):
    """
    OLS with intercept.

    Returns both classical OLS SE/p-values and HC3 robust SE/p-values.

    HC3 covariance:
        cov_HC3 = (X'X)^(-1) X' diag((e_i / (1 - h_ii))^2) X (X'X)^(-1)

    R2 and beta are unchanged by HC3.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    n = len(y)
    X_design = np.column_stack([np.ones(n), X])
    k = X_design.shape[1]
    df_resid = n - k

    xtx = X_design.T @ X_design
    xtx_inv = np.linalg.pinv(xtx)

    beta = xtx_inv @ X_design.T @ y
    y_hat = X_design @ beta
    resid = y - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan

    # Classical OLS covariance.
    if df_resid > 0:
        sigma2 = ss_res / df_resid
        cov_classic = sigma2 * xtx_inv
        se_classic = np.sqrt(np.diag(cov_classic))
        t_classic = beta / se_classic
        p_classic = 2 * stats.t.sf(np.abs(t_classic), df=df_resid)
    else:
        se_classic = np.full_like(beta, np.nan)
        t_classic = np.full_like(beta, np.nan)
        p_classic = np.full_like(beta, np.nan)

    # HC3 robust covariance.
    # h_ii = diagonal of hat matrix X (X'X)^(-1) X'
    h = np.sum((X_design @ xtx_inv) * X_design, axis=1)
    denom = np.maximum(1.0 - h, 1e-12)
    scaled_resid_sq = (resid / denom) ** 2

    meat = X_design.T @ (X_design * scaled_resid_sq[:, None])
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se_hc3 = np.sqrt(np.diag(cov_hc3))

    if df_resid > 0:
        t_hc3 = beta / se_hc3
        p_hc3 = 2 * stats.t.sf(np.abs(t_hc3), df=df_resid)
    else:
        t_hc3 = np.full_like(beta, np.nan)
        p_hc3 = np.full_like(beta, np.nan)

    return {
        "beta": beta,
        "se_classic": se_classic,
        "t_classic": t_classic,
        "p_classic": p_classic,
        "se_hc3": se_hc3,
        "t_hc3": t_hc3,
        "p_hc3": p_hc3,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "df_resid": df_resid,
    }

def prepare_pair_df(path, direction):
    print(f"{direction}_PAIR:", path, path.exists())
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{direction} missing required columns: {missing}")

    # SC was already enforced in the Figure 2a pair-level tables.
    # Here we enforce the shared reliability rule.
    d = df[df["n_units"] >= MAIN_MIN_N_UNITS].copy()

    for c in REQUIRED_COLS:
        if c != "pair_id":
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[c for c in REQUIRED_COLS if c != "pair_id"])

    for c in [c for c in REQUIRED_COLS if c != "pair_id"]:
        d = d[np.isfinite(d[c])]

    d["direction"] = direction
    return d

def regression_rows(df, direction, metric, score_col):
    rows = []
    y = df[score_col].to_numpy(dtype=float)

    # Fit baseline Model A first for delta R2.
    XA = df[MODELS["A_atom_only"]["predictors"]].to_numpy(dtype=float)
    fitA = fit_ols_hc3(y, XA)
    r2A = fitA["r2"]
    adjA = fitA["adj_r2"]

    for model_name, spec in MODELS.items():
        predictors = spec["predictors"]
        key_predictor = spec["key_predictor"]

        X = df[predictors].to_numpy(dtype=float)
        fit = fit_ols_hc3(y, X)

        beta = fit["beta"]
        pred_to_idx = {pred: i + 1 for i, pred in enumerate(predictors)}

        row = {
            "direction": direction,
            "metric": metric,
            "score_col": score_col,
            "min_n_units": MAIN_MIN_N_UNITS,
            "n_pairs": int(fit["n"]),

            "model": model_name,
            "model_description": spec["description"],
            "predictors": " + ".join(predictors),

            "r2": float(fit["r2"]),
            "adj_r2": float(fit["adj_r2"]),
            "delta_r2_vs_A": float(fit["r2"] - r2A),
            "delta_adj_r2_vs_A": float(fit["adj_r2"] - adjA),

            "intercept": float(beta[0]),
            "df_resid": int(fit["df_resid"]),
        }

        if key_predictor:
            idx = pred_to_idx[key_predictor]
            row["key_predictor"] = key_predictor
            row["key_beta"] = float(beta[idx])

            row["key_se_classic"] = float(fit["se_classic"][idx])
            row["key_t_classic"] = float(fit["t_classic"][idx])
            row["key_p_classic"] = float(fit["p_classic"][idx])
            row["key_p_classic_fmt"] = p_fmt(float(fit["p_classic"][idx]))

            row["key_se_hc3"] = float(fit["se_hc3"][idx])
            row["key_t_hc3"] = float(fit["t_hc3"][idx])
            row["key_p_hc3"] = float(fit["p_hc3"][idx])
            row["key_p_hc3_fmt"] = p_fmt(float(fit["p_hc3"][idx]))
            row["key_sig_hc3"] = sig_label(float(fit["p_hc3"][idx]))
        else:
            row["key_predictor"] = ""
            row["key_beta"] = np.nan
            row["key_se_classic"] = np.nan
            row["key_t_classic"] = np.nan
            row["key_p_classic"] = np.nan
            row["key_p_classic_fmt"] = ""
            row["key_se_hc3"] = np.nan
            row["key_t_hc3"] = np.nan
            row["key_p_hc3"] = np.nan
            row["key_p_hc3_fmt"] = ""
            row["key_sig_hc3"] = ""

        # Store all predictor coefficients and both p-value types.
        for pred in predictors:
            idx = pred_to_idx[pred]
            row[f"beta_{pred}"] = float(beta[idx])

            row[f"se_classic_{pred}"] = float(fit["se_classic"][idx])
            row[f"p_classic_{pred}"] = float(fit["p_classic"][idx])
            row[f"p_classic_{pred}_fmt"] = p_fmt(float(fit["p_classic"][idx]))

            row[f"se_hc3_{pred}"] = float(fit["se_hc3"][idx])
            row[f"p_hc3_{pred}"] = float(fit["p_hc3"][idx])
            row[f"p_hc3_{pred}_fmt"] = p_fmt(float(fit["p_hc3"][idx]))

        rows.append(row)

    return rows

def make_wide_table(long_df):
    rows = []

    for (direction, metric), g in long_df.groupby(["direction", "metric"]):
        A = g[g["model"] == "A_atom_only"].iloc[0]
        B = g[g["model"] == "B_atom_plus_pair_count"].iloc[0]
        C = g[g["model"] == "C_atom_plus_PMI"].iloc[0]

        rows.append({
            "direction": direction,
            "metric": metric,
            "n_pairs": int(A["n_pairs"]),

            "r2_A_atom_only": float(A["r2"]),

            "r2_B_atom_plus_pair_count": float(B["r2"]),
            "delta_r2_B_minus_A": float(B["delta_r2_vs_A"]),
            "beta_log10_f_pair_in_B": float(B["key_beta"]),
            "p_hc3_log10_f_pair_in_B": float(B["key_p_hc3"]),
            "p_hc3_log10_f_pair_in_B_fmt": B["key_p_hc3_fmt"],
            "p_classic_log10_f_pair_in_B": float(B["key_p_classic"]),
            "p_classic_log10_f_pair_in_B_fmt": B["key_p_classic_fmt"],
            "sig_hc3_log10_f_pair_in_B": B["key_sig_hc3"],

            "r2_C_atom_plus_PMI": float(C["r2"]),
            "delta_r2_C_minus_A": float(C["delta_r2_vs_A"]),
            "beta_PMI_in_C": float(C["key_beta"]),
            "p_hc3_PMI_in_C": float(C["key_p_hc3"]),
            "p_hc3_PMI_in_C_fmt": C["key_p_hc3_fmt"],
            "p_classic_PMI_in_C": float(C["key_p_classic"]),
            "p_classic_PMI_in_C_fmt": C["key_p_classic_fmt"],
            "sig_hc3_PMI_in_C": C["key_sig_hc3"],

            "delta_r2_C_minus_B": float(C["delta_r2_vs_A"] - B["delta_r2_vs_A"]),
        })

    out = pd.DataFrame(rows)

    direction_order = {"V2T": 0, "T2V": 1}
    metric_order = {"mAP": 0, "nDCG": 1}

    out["_do"] = out["direction"].map(direction_order)
    out["_mo"] = out["metric"].map(metric_order)
    out = out.sort_values(["_do", "_mo"]).drop(columns=["_do", "_mo"])

    for c in out.columns:
        if c.startswith("r2_") or c.startswith("delta_") or c.startswith("beta_"):
            out[c] = out[c].round(4)

    return out

def fit_r2_for_vif(y, X):
    fit = fit_ols_hc3(y, X)
    return fit["r2"]

def diagnostics_for_direction(df, direction):
    rows = []

    all_predictors = ["log10_f_verb", "log10_f_noun", "log10_f_pair", "PMI"]
    corr = df[all_predictors].corr()

    for i, a in enumerate(all_predictors):
        for b in all_predictors[i + 1:]:
            rows.append({
                "direction": direction,
                "diagnostic": "pairwise_correlation",
                "model": "",
                "var1": a,
                "var2": b,
                "value": float(corr.loc[a, b]),
            })

    # VIF is computed model-by-model.
    for model_name, spec in MODELS.items():
        predictors = spec["predictors"]

        for target in predictors:
            others = [p for p in predictors if p != target]
            if not others:
                continue

            y = df[target].to_numpy(dtype=float)
            X = df[others].to_numpy(dtype=float)
            r2 = fit_r2_for_vif(y, X)
            vif = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)

            rows.append({
                "direction": direction,
                "diagnostic": "vif",
                "model": model_name,
                "var1": target,
                "var2": "other_predictors",
                "value": float(vif),
            })

    return rows

def write_docs():
    OUT_README.write_text(
        f"""# Table 1: atom-only vs pair-count vs PMI regression with HC3 robust SE

This table tests whether composition-level information explains retrieval performance beyond constituent atom frequencies.

## Analysis unit

Same rule as Figure 2a and Figure 2b:

- SC only
- pair-level aggregation by `pair_id`
- `n_units >= {MAIN_MIN_N_UNITS}`

## Models

Model A: atom-only

`score ~ log10_f_verb + log10_f_noun`

Model B: atom + pair count

`score ~ log10_f_verb + log10_f_noun + log10_f_pair`

Model C: atom + PMI

`score ~ log10_f_verb + log10_f_noun + PMI(v,n)`

PMI is computed with the natural logarithm.

## Standard errors

The main p-values reported in the wide table use HC3 heteroskedasticity-robust standard errors.

Classical OLS p-values are retained as diagnostic columns.

## Why Model B and C are separate

Model B tests whether raw pair exposure count adds explanatory power beyond atom frequencies.

Model C tests whether pair association strength adds explanatory power beyond atom frequencies.

Model B and Model C are fit separately because PMI is mathematically related to pair and marginal frequencies. Putting `log10_f_pair` and PMI into the same main regression would create unnecessary collinearity.

## Main table

Use:

`table_4_1_regression_wide.csv`
""",
        encoding="utf-8",
    )

    OUT_CAPTION.write_text(
        """# Table 1 caption draft

Atom-only vs composition-aware regression. We fit pair-level regressions over SC verb-noun compositions with at least three official query units. Model A uses only constituent atom frequencies, log10 f(v) and log10 f(n). Model B additionally includes raw pair exposure, log10 f(v,n). Model C instead includes pair association strength, PMI(v,n). Model B and Model C are fit separately because PMI is derived from pair and marginal frequencies. ΔR² is measured relative to Model A. p-values are computed using HC3 heteroskedasticity-robust standard errors.

Suggested interpretation:

Model B tests whether raw pair count adds explanatory power beyond atom frequency, while Model C tests whether pair association strength adds explanatory power beyond atom frequency.
""",
        encoding="utf-8",
    )

def main():
    global V2T_PAIR, T2V_PAIR, OUT_DIR
    global OUT_TABLE_LONG, OUT_TABLE_WIDE, OUT_INPUTS, OUT_DIAG, OUT_README, OUT_CAPTION
    global MAIN_MIN_N_UNITS

    args = parse_args()

    V2T_PAIR = args.v2t_pair_csv
    T2V_PAIR = args.t2v_pair_csv
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_TABLE_LONG = OUT_DIR / args.output_long_name
    OUT_TABLE_WIDE = OUT_DIR / args.output_wide_name
    OUT_INPUTS = OUT_DIR / args.output_inputs_name
    OUT_DIAG = OUT_DIR / args.output_diagnostics_name
    OUT_README = OUT_DIR / "README.md"
    OUT_CAPTION = OUT_DIR / "table_4_1_caption.md"

    MAIN_MIN_N_UNITS = args.min_n_units

    print("V2T_PAIR:", V2T_PAIR, V2T_PAIR.exists())
    print("T2V_PAIR:", T2V_PAIR, T2V_PAIR.exists())
    print("OUT_DIR:", OUT_DIR)
    print("MAIN_MIN_N_UNITS:", MAIN_MIN_N_UNITS)
    print("SE_TYPE: HC3 robust")

    v2t = prepare_pair_df(V2T_PAIR, "V2T")
    t2v = prepare_pair_df(T2V_PAIR, "T2V")

    print("\n=== regression inputs after filtering ===")
    print("V2T pairs:", len(v2t))
    print("T2V pairs:", len(t2v))

    if args.paper_check:
        if MAIN_MIN_N_UNITS != 3:
            raise RuntimeError("--paper-check requires --min-n-units 3.")
        if len(v2t) != 537:
            raise RuntimeError(f"Expected V2T n>=3 pairs = 537, got {len(v2t)}")
        if len(t2v) != 374:
            raise RuntimeError(f"Expected T2V n>=3 pairs = 374, got {len(t2v)}")

    all_inputs = pd.concat([v2t, t2v], ignore_index=True)
    all_inputs.to_csv(OUT_INPUTS, index=False)

    rows = []

    for direction, df in [("V2T", v2t), ("T2V", t2v)]:
        for d, metric, score_col in OUTCOMES:
            if d != direction:
                continue
            rows.extend(regression_rows(df, direction, metric, score_col))

    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT_TABLE_LONG, index=False)

    wide_df = make_wide_table(long_df)
    wide_df.to_csv(OUT_TABLE_WIDE, index=False)

    diag_rows = []
    diag_rows.extend(diagnostics_for_direction(v2t, "V2T"))
    diag_rows.extend(diagnostics_for_direction(t2v, "T2V"))

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT_DIAG, index=False)

    write_docs()

    print("\n=== Table 1 wide, HC3 p-values are primary ===")
    print(wide_df.to_string(index=False))

    print("\n=== diagnostics ===")
    print(diag.to_string(index=False))

    print("\nOUT_TABLE_LONG:", OUT_TABLE_LONG)
    print("OUT_TABLE_WIDE:", OUT_TABLE_WIDE)
    print("OUT_INPUTS:", OUT_INPUTS)
    print("OUT_DIAG:", OUT_DIAG)
    print("OUT_README:", OUT_README)
    print("OUT_CAPTION:", OUT_CAPTION)
    print("SUMMARY failed=0")

if __name__ == "__main__":
    main()
