#!/usr/bin/env python
import argparse
from pathlib import Path
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--egovlpv2-root", type=Path, required=True)
    parser.add_argument("--feature-pt", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--exposure-query-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-v2t-name",
        type=str,
        default="ek100_mir_official_per_query_metrics_v2t.csv",
    )
    parser.add_argument(
        "--output-t2v-name",
        type=str,
        default="ek100_mir_official_per_query_metrics_t2v.csv",
    )
    parser.add_argument(
        "--output-joined-name",
        type=str,
        default="ek100_mir_exposure_with_official_per_query_metrics.csv",
    )
    parser.add_argument(
        "--output-check-name",
        type=str,
        default="official_per_query_metric_check.csv",
    )
    return parser.parse_args()


ARGS = parse_args()
sys.path.insert(0, str(ARGS.egovlpv2_root))

from utils import nDCG, mAP
from model.metric import initialise_jpose_nDCG_values, mir_metrics_vtc

FEATURE_PT = ARGS.feature_pt
META_DIR = ARGS.meta_dir
RELEVANCY_PKL = META_DIR / "relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"

EXPOSURE_QUERY_CSV = ARGS.exposure_query_csv

OUT_DIR = ARGS.output_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_V2T = OUT_DIR / ARGS.output_v2t_name
OUT_T2V = OUT_DIR / ARGS.output_t2v_name
OUT_JOINED = OUT_DIR / ARGS.output_joined_name
OUT_CHECK = OUT_DIR / ARGS.output_check_name

EXPECTED = {
    "nDCG_V2T": 30.071646777422366,
    "nDCG_T2V": 28.12497337856873,
    "nDCG_AVG": 29.09831007799555,
    "mAP_V2T": 29.88172021139886,
    "mAP_T2V": 23.448891050955933,
    "mAP_AVG": 26.665305631177393,
}

def load_relevancy():
    with open(RELEVANCY_PKL, "rb") as f:
        return np.asarray(pickle.load(f))

def official_final_eval_matrix():
    """
    Mirror model.metric.mir_metrics_vtc exactly.

    Raw:
      sim_raw = normalize(text_embeds) @ normalize(vid_embeds).T
      shape = 9668 text rows x 9668 video rows

    mir_metrics_vtc then:
      sim = (sim + 1) / 2
      sim = sim[order, :][:, order]
      sim = sim.T[:, indexes]

    Final:
      sim shape = 9668 video-query rows x 3842 unique-text columns
      rel shape = 9668 x 3842
    """
    obj = torch.load(FEATURE_PT, map_location="cpu")

    vid = F.normalize(obj["vid_embeds"].float(), dim=1)
    txt = F.normalize(obj["text_embeds"].float(), dim=1)
    idx_arr = obj["arr_embeds"].cpu().numpy()

    sim_raw = (txt @ vid.T).cpu().numpy()
    sim = (sim_raw + 1.0) / 2.0

    video_id = pd.read_csv(META_DIR / "EPIC_100_retrieval_test.csv").values[:, 0]
    text_id = pd.read_csv(META_DIR / "EPIC_100_retrieval_test_sentence.csv").values[:, 0]

    video_id_list = video_id.tolist()
    indexes = np.asarray([video_id_list.index(elem) for elem in text_id], dtype=int)

    idx_arr_list = idx_arr.tolist()
    order = np.asarray([idx_arr_list.index(i) for i in range(len(video_id))], dtype=int)

    sim = sim[order, :][:, order]
    sim = sim.T[:, indexes]

    return sim, sim_raw, idx_arr, order, indexes, video_id, text_id

def calculate_AP_per_query_official(sim_mat, relevancy_matrix):
    """
    Exact per-query version of utils/mAP.py::calculate_mAP.

    Important:
    Official mAP treats rel == 1 as relevant for AP.
    We do not change that rule.
    """
    ranked_order = (-sim_mat).argsort()
    ranked_rel_mat = relevancy_matrix[
        np.arange(relevancy_matrix.shape[0])[:, None],
        ranked_order
    ]

    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    cumulative_rel_mat[ranked_rel_mat != 1] = 0

    divisor = np.arange(ranked_rel_mat.shape[1]) + 1
    number_rel_docs = np.sum(ranked_rel_mat == 1, axis=1)

    ap = np.sum(cumulative_rel_mat / divisor, axis=1) / number_rel_docs
    return ap

def eval_full_and_per_query(sim, rel):
    dataset = initialise_jpose_nDCG_values(rel)

    v2t_ndcg_per = nDCG.calculate_nDCG(
        sim,
        rel,
        dataset["action"]["k_values"]["v"],
        IDCG=dataset["action"]["IDCG"]["v"],
        reduction=None,
    )
    t2v_ndcg_per = nDCG.calculate_nDCG(
        sim.T,
        rel.T,
        dataset["action"]["k_values"]["t"],
        IDCG=dataset["action"]["IDCG"]["t"],
        reduction=None,
    )

    v2t_ap_per = calculate_AP_per_query_official(sim, rel)
    t2v_ap_per = calculate_AP_per_query_official(sim.T, rel.T)

    scores = {
        "nDCG_V2T": float(v2t_ndcg_per.mean() * 100),
        "nDCG_T2V": float(t2v_ndcg_per.mean() * 100),
        "nDCG_AVG": float((v2t_ndcg_per.mean() + t2v_ndcg_per.mean()) * 50),
        "mAP_V2T": float(v2t_ap_per.mean() * 100),
        "mAP_T2V": float(t2v_ap_per.mean() * 100),
        "mAP_AVG": float((v2t_ap_per.mean() + t2v_ap_per.mean()) * 50),
    }

    return scores, v2t_ap_per, v2t_ndcg_per, t2v_ap_per, t2v_ndcg_per

def main():
    os.environ["EK100_META_DIR"] = str(META_DIR)

    print("ROOT:", ROOT)
    print("FEATURE_PT:", FEATURE_PT, FEATURE_PT.exists())
    print("META_DIR:", META_DIR, META_DIR.exists())
    print("RELEVANCY_PKL:", RELEVANCY_PKL, RELEVANCY_PKL.exists())
    print("EXPOSURE_QUERY_CSV:", EXPOSURE_QUERY_CSV, EXPOSURE_QUERY_CSV.exists())

    exposure = pd.read_csv(EXPOSURE_QUERY_CSV)
    rel = load_relevancy()
    sim, sim_raw, idx_arr, order, indexes, video_id, text_id = official_final_eval_matrix()

    print("exposure:", exposure.shape)
    print("sim final:", sim.shape)
    print("rel:", rel.shape, rel.dtype, "min/max:", np.nanmin(rel), np.nanmax(rel))
    print("sim_raw:", sim_raw.shape)
    print("idx_arr:", idx_arr.shape, idx_arr.min(), idx_arr.max())
    print("order:", order.shape, order[:10])
    print("indexes:", indexes.shape, indexes[:10])
    print("video_id:", len(video_id))
    print("text_id:", len(text_id))

    assert sim.shape == rel.shape, (sim.shape, rel.shape)
    assert sim.shape[0] == len(exposure), (sim.shape, len(exposure))
    assert len(indexes) == sim.shape[1], (len(indexes), sim.shape)

    scores, v2t_ap, v2t_ndcg, t2v_ap, t2v_ndcg = eval_full_and_per_query(sim, rel)

    black = mir_metrics_vtc(sim_raw, idx_arr)

    rows = []
    for k in ["nDCG_V2T", "nDCG_T2V", "nDCG_AVG", "mAP_V2T", "mAP_T2V", "mAP_AVG"]:
        rows.append({
            "metric": k,
            "from_per_query_mean": scores[k],
            "black_box_mir_metrics_vtc": float(black[k]),
            "expected": EXPECTED[k],
            "abs_diff_vs_black_box": abs(scores[k] - float(black[k])),
            "abs_diff_vs_expected": abs(scores[k] - EXPECTED[k]),
        })

    check = pd.DataFrame(rows)
    check.to_csv(OUT_CHECK, index=False)

    print("\n=== OFFICIAL BASELINE CHECK ===")
    print(check.to_string(index=False))

    max_diff_black = check["abs_diff_vs_black_box"].max()
    max_diff_expected = check["abs_diff_vs_expected"].max()

    # V2T table: official 9668 video-query rows.
    v2t = exposure[[
        "query_id",
        "narration_id",
        "verb_class",
        "noun_class",
        "pair_id",
        "v3_exposure_label",
        "pair_evidence_tier",
        "f_pair",
        "f_verb",
        "f_noun",
        "log10_f_pair",
        "log10_f_verb",
        "log10_f_noun",
        "PMI",
    ]].copy()

    v2t["official_eval_row"] = np.arange(len(v2t), dtype=int)
    v2t["V2T_AP"] = v2t_ap
    v2t["V2T_nDCG"] = v2t_ndcg
    v2t.to_csv(OUT_V2T, index=False)

    # T2V table: official 3842 unique text-query rows.
    t2v_exposure = exposure.iloc[indexes].copy().reset_index(drop=False)
    t2v_exposure = t2v_exposure.rename(columns={"index": "source_exposure_row"})

    t2v = t2v_exposure[[
        "source_exposure_row",
        "query_id",
        "narration_id",
        "verb_class",
        "noun_class",
        "pair_id",
        "v3_exposure_label",
        "pair_evidence_tier",
        "f_pair",
        "f_verb",
        "f_noun",
        "log10_f_pair",
        "log10_f_verb",
        "log10_f_noun",
        "PMI",
    ]].copy()

    t2v["official_text_query_col"] = np.arange(len(t2v), dtype=int)
    t2v["T2V_AP"] = t2v_ap
    t2v["T2V_nDCG"] = t2v_ndcg
    t2v.to_csv(OUT_T2V, index=False)

    # Joined main table: 9668 V2T rows, with T2V only on official text-query representatives.
    joined = v2t.copy()
    t2v_small = t2v[["source_exposure_row", "T2V_AP", "T2V_nDCG", "official_text_query_col"]].copy()
    joined = joined.merge(
        t2v_small,
        left_on="official_eval_row",
        right_on="source_exposure_row",
        how="left",
    )
    joined = joined.drop(columns=["source_exposure_row"])
    joined["has_official_T2V_query_metric"] = joined["official_text_query_col"].notna()
    joined.to_csv(OUT_JOINED, index=False)

    print("\nOUT_CHECK:", OUT_CHECK)
    print("OUT_V2T:", OUT_V2T)
    print("OUT_T2V:", OUT_T2V)
    print("OUT_JOINED:", OUT_JOINED)

    print("\n=== row counts ===")
    print("V2T rows:", len(v2t))
    print("T2V rows:", len(t2v))
    print("joined rows:", len(joined))
    print("joined rows with T2V metric:", int(joined["has_official_T2V_query_metric"].sum()))

    print("\n=== label counts V2T ===")
    print(v2t["v3_exposure_label"].value_counts().to_string())

    print("\n=== label counts T2V official text queries ===")
    print(t2v["v3_exposure_label"].value_counts().to_string())

    failed = (
        max_diff_black > 1e-8
        or max_diff_expected > 1e-6
        or len(v2t) != 9668
        or len(t2v) != sim.shape[1]
        or int(joined["has_official_T2V_query_metric"].sum()) != len(t2v)
    )

    print("\nmax_diff_vs_black_box:", max_diff_black)
    print("max_diff_vs_expected:", max_diff_expected)
    print("SUMMARY failed=0" if not failed else "SUMMARY failed=1")

if __name__ == "__main__":
    main()
