#!/usr/bin/env python
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.interventions.adapter import DualResidualAdapters
from src.interventions.exposure_bias import ExposureBias


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run counterfactual exposure reweighting by changing only the "
            "text-side exposure score-bias input while holding representations fixed."
        )
    )
    parser.add_argument("--egovlpv2-root", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--v2t-exposure-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exposure-vectors", type=Path, required=True)
    parser.add_argument("--mode", choices=["original", "zeroed", "pair_zeroed", "median", "shuffled"], default="original")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def safe_corr(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(x[ok]) == 0:
        return np.nan, np.nan, int(ok.sum()), int((~ok).sum())
    return (
        float(pearsonr(y[ok], x[ok])[0]),
        float(spearmanr(y[ok], x[ok])[0]),
        int(ok.sum()),
        int((~ok).sum()),
    )


def apply_counterfactual(raw, mode, seed=2026):
    raw = raw.copy()
    if mode == "original":
        return raw
    if mode in ["zeroed", "pair_zeroed"]:
        return np.zeros_like(raw, dtype="float32")
    if mode == "median":
        for j in range(raw.shape[1]):
            ok = np.isfinite(raw[:, j])
            raw[:, j] = np.median(raw[ok, j])
        return raw
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        return raw[rng.permutation(raw.shape[0])]
    raise ValueError(f"unknown counterfactual mode: {mode}")


def load_m7(path):
    ckpt = torch.load(path, map_location="cpu")
    model = DualResidualAdapters().eval()
    bias = ExposureBias().eval()
    model.load_state_dict(ckpt["model"], strict=False)
    bias.load_state_dict(ckpt["exposure_bias"], strict=True)
    return model, bias, ckpt


def official_v2t_matrix(sim_text_by_video, idx_arr, meta_dir):
    video_id = pd.read_csv(meta_dir / "EPIC_100_retrieval_test.csv").values[:, 0]
    text_id = pd.read_csv(meta_dir / "EPIC_100_retrieval_test_sentence.csv").values[:, 0]
    indexes = [video_id.tolist().index(x) for x in text_id]
    order = [idx_arr.tolist().index(i) for i in range(len(video_id))]
    s = (sim_text_by_video + 1.0) / 2.0
    s = s[order, :][:, order]
    return s.T[:, indexes]


def per_query_ap(sim_v2t, meta_dir):
    rel_path = meta_dir / "relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"
    with open(rel_path, "rb") as f:
        rel = pickle.load(f)
    ranked = (-sim_v2t).argsort()
    ranked_rel = rel[np.arange(rel.shape[0])[:, None], ranked]
    cum = np.cumsum(ranked_rel, axis=1)
    cum[ranked_rel != 1] = 0
    divisor = np.arange(ranked_rel.shape[1]) + 1
    nrel = np.sum(ranked_rel == 1, axis=1)
    return np.sum(cum / divisor, axis=1) / nrel


def main():
    args = parse_args()

    sys.path.insert(0, str(args.egovlpv2_root))
    from model.metric import mir_metrics_vtc

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    cache = torch.load(args.eval_cache, map_location="cpu")
    video = cache["vid_embeds"].float()
    text = cache["text_embeds"].float()
    arr = cache["arr_embeds"].cpu().numpy().astype(int)

    model, exposure_bias, ckpt = load_m7(args.checkpoint)
    model = model.to(device)
    exposure_bias = exposure_bias.to(device)

    with torch.no_grad():
        z_v, z_t = model(video.to(device), text.to(device))
        base = (z_t @ z_v.t()).cpu().numpy().astype("float32")

    exp_df = (
        pd.read_csv(args.v2t_exposure_csv)
        .sort_values("official_eval_row")
        .reset_index(drop=True)
    )

    raw = np.stack(
        [
            np.log10(1.0 + exp_df["f_pair"].astype(float).to_numpy()),
            np.log10(1.0 + exp_df["f_verb"].astype(float).to_numpy()),
            np.log10(1.0 + exp_df["f_noun"].astype(float).to_numpy()),
            exp_df["PMI"].astype(float).to_numpy(),
        ],
        axis=1,
    ).astype("float32")

    norm_obj = torch.load(args.exposure_vectors, map_location="cpu")
    mean = norm_obj["train"]["mean"].numpy()
    std = norm_obj["train"]["std"].numpy()

    raw_for_model = apply_counterfactual(raw, args.mode, args.seed)
    for j in range(raw_for_model.shape[1]):
        bad = ~np.isfinite(raw_for_model[:, j])
        if bad.any():
            raw_for_model[bad, j] = mean[j]

    raw_cache_order = raw_for_model[arr]
    exp_z = torch.from_numpy((raw_cache_order - mean) / np.maximum(std, 1e-8)).float().to(device)

    with torch.no_grad():
        raw_bias = exposure_bias(exp_z).cpu().numpy().astype("float32")

    bias = raw_bias - raw_bias.mean()
    m7 = base + bias[:, None]

    metrics_base = {k: float(v) for k, v in mir_metrics_vtc(base, arr).items()}
    metrics_m7 = {k: float(v) for k, v in mir_metrics_vtc(m7, arr).items()}
    delta = {k: metrics_m7[k] - metrics_base[k] for k in metrics_m7}

    ap_base = per_query_ap(official_v2t_matrix(base, arr, args.meta_dir), args.meta_dir)
    ap_m7 = per_query_ap(official_v2t_matrix(m7, arr, args.meta_dir), args.meta_dir)
    groups = exp_df["v3_exposure_label"].astype(str).to_numpy()

    group_rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        group_rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "V2T_AP_base": float(ap_base[mask].mean() * 100),
                "V2T_AP_m7": float(ap_m7[mask].mean() * 100),
                "V2T_AP_delta": float((ap_m7[mask] - ap_base[mask]).mean() * 100),
            }
        )

    bias_val_order = np.empty_like(bias)
    for cache_i, val_i in enumerate(arr):
        bias_val_order[val_i] = bias[cache_i]

    audit_rows = []
    for col_i, col in enumerate(["log10_1p_f_pair", "log10_1p_f_verb", "log10_1p_f_noun", "PMI"]):
        pr, sr, n_ok, n_drop = safe_corr(bias_val_order, raw[:, col_i])
        audit_rows.append(
            {
                "xcol": col,
                "n": n_ok,
                "dropped_nonfinite": n_drop,
                "pearson": pr,
                "spearman": sr,
            }
        )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "counterfactual_mode": args.mode,
        "counterfactual_seed": int(args.seed),
        "step": int(ckpt["step"]),
        "dev_loss": float(ckpt["dev_loss"]),
        "bias_mean": float(bias.mean()),
        "bias_std": float(bias.std()),
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "metrics_adapter_no_bias": metrics_base,
        "metrics_m7_centered_bias": metrics_m7,
        "delta_m7_minus_no_bias": delta,
    }

    (out / "counterfactual_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    pd.DataFrame(group_rows).to_csv(out / "v2t_sc_uc_ap.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "bias_exposure_correlations.csv", index=False)
    pd.DataFrame(
        {
            "official_eval_row": np.arange(len(exp_df)),
            "narration_id": exp_df["narration_id"],
            "group": groups,
            "bias": bias_val_order,
            "log10_1p_f_pair": raw[:, 0],
            "log10_1p_f_verb": raw[:, 1],
            "log10_1p_f_noun": raw[:, 2],
            "PMI": raw[:, 3],
        }
    ).to_csv(out / "bias_per_eval_item.csv", index=False)

    print("===== Counterfactual exposure reweighting =====")
    print("counterfactual_mode:", args.mode, "seed:", args.seed)
    for k in ["mAP_V2T", "mAP_T2V", "mAP_AVG", "nDCG_V2T", "nDCG_T2V", "nDCG_AVG"]:
        print(k, f"{metrics_base[k]:.6f}", "->", f"{metrics_m7[k]:.6f}", "delta", f"{delta[k]:+.6f}")

    print("\n===== V2T SC/UC AP =====")
    print(pd.DataFrame(group_rows).to_string(index=False))

    print("\n===== bias-exposure corr =====")
    print(pd.DataFrame(audit_rows).to_string(index=False))

    print("OUT_SUMMARY:", out / "counterfactual_summary.json")
    print("SUMMARY failed=0")


if __name__ == "__main__":
    main()
