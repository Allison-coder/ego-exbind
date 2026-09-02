#!/usr/bin/env python
from pathlib import Path
import pickle
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.metric import mir_metrics_vtc, initialise_jpose_nDCG_values
from utils import nDCG

OUT_DIR = ROOT / "Data/processed/exposure_metric_analysis_v3_full_20260705"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PT = ROOT / "eval_epic/cached_features/ek100_zs_features_dl0.pt"
META_DIR = Path("/user/work/rr24582/data/EK100/epic-kitchens-100-annotations/retrieval_annotations")
RELEVANCY_PKL = META_DIR / "relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"

EXPOSURE = OUT_DIR / "ek100_mir_v3_full_query_exposure_with_provenance.csv"
V2T = OUT_DIR / "ek100_mir_official_per_query_metrics_v2t_9668_20260705.csv"
T2V = OUT_DIR / "ek100_mir_official_per_query_metrics_t2v_3842_20260705.csv"
CHECK = OUT_DIR / "official_per_query_metric_check_20260705.csv"

OUT_MAP = OUT_DIR / "ek100_mir_official_text_uid_mapping_9668_20260705.csv"
OUT_BROADCAST = OUT_DIR / "ek100_mir_v3_full_metrics_broadcast_9668_20260705.csv"
OUT_V2T_UNITS = OUT_DIR / "ek100_mir_v3_full_metrics_v2t_units_9668_20260705.csv"
OUT_T2V_UNITS = OUT_DIR / "ek100_mir_v3_full_metrics_t2v_units_3842_20260705.csv"
OUT_T2V_CLEAN = OUT_DIR / "ek100_mir_v3_full_metrics_t2v_units_3842_clean_no_collision_20260705.csv"
OUT_COLLISION = OUT_DIR / "ek100_mir_text_uid_collision_report_20260705.csv"
OUT_SUMMARY = OUT_DIR / "text_uid_broadcast_check_20260705.csv"

EXPECTED = {
    "mAP_V2T": 29.88172021139886,
    "nDCG_V2T": 30.071646777422366,
    "mAP_T2V": 23.448891050955933,
    "nDCG_T2V": 28.12497337856873,
}

def norm_text(s):
    # Start strict. Do not lower-case unless strict fails.
    return str(s).strip()

def norm_text_lower(s):
    return str(s).strip().lower()

def load_official_ids_and_indexes():
    obj = torch.load(FEATURE_PT, map_location="cpu")
    idx_arr = obj["arr_embeds"].cpu().numpy()

    video_df = pd.read_csv(META_DIR / "EPIC_100_retrieval_test.csv")
    text_df = pd.read_csv(META_DIR / "EPIC_100_retrieval_test_sentence.csv")

    video_id = video_df.values[:, 0]
    text_id = text_df.values[:, 0]

    video_id_list = video_id.tolist()
    indexes = np.asarray([video_id_list.index(elem) for elem in text_id], dtype=int)

    idx_arr_list = idx_arr.tolist()
    order = np.asarray([idx_arr_list.index(i) for i in range(len(video_id))], dtype=int)

    return video_df, text_df, video_id, text_id, indexes, order, idx_arr

def choose_text_key(video_df, text_df, indexes, exposure):
    """
    Find the exact text key that reproduces the official 3842 unique text queries.

    Priority:
    1. Common text-like columns in official video/text CSVs.
    2. Exposure narration column.

    Acceptance:
    - representative keys from indexes must be unique and length 3842.
    - every 9668 row must map to one representative key.
    """
    candidates = []

    common_cols = [c for c in video_df.columns if c in set(text_df.columns)]
    text_like = [
        c for c in common_cols
        if any(tok in str(c).lower() for tok in ["narration", "caption", "sentence", "text"])
    ]

    for c in text_like:
        candidates.append(("official_common_col_strict", c, video_df[c].astype(str).tolist(), norm_text))
        candidates.append(("official_common_col_lower", c, video_df[c].astype(str).tolist(), norm_text_lower))

    if "narration" in exposure.columns:
        candidates.append(("exposure_narration_strict", "narration", exposure["narration"].astype(str).tolist(), norm_text))
        candidates.append(("exposure_narration_lower", "narration", exposure["narration"].astype(str).tolist(), norm_text_lower))

    reports = []

    for source, col, texts_raw, normalizer in candidates:
        all_keys = [normalizer(x) for x in texts_raw]
        rep_keys = [all_keys[i] for i in indexes]
        unique_rep = len(set(rep_keys))
        key_to_uid = {k: j for j, k in enumerate(rep_keys)}
        mapped = [key_to_uid.get(k, None) for k in all_keys]
        missing = sum(x is None for x in mapped)
        unique_all = len(set(all_keys))

        reports.append({
            "source": source,
            "column": col,
            "normalizer": normalizer.__name__,
            "unique_all_9668": unique_all,
            "unique_rep_3842": unique_rep,
            "missing_rows": missing,
        })

        if unique_rep == len(indexes) and missing == 0:
            return source, col, normalizer.__name__, all_keys, rep_keys, mapped, pd.DataFrame(reports)

    raise RuntimeError("Could not find text key that maps all 9668 rows to 3842 official text_uids. Reports:\n" + pd.DataFrame(reports).to_string(index=False))

def main():
    os.environ["EK100_META_DIR"] = str(META_DIR)

    print("ROOT:", ROOT)
    print("FEATURE_PT:", FEATURE_PT, FEATURE_PT.exists())
    print("META_DIR:", META_DIR, META_DIR.exists())
    print("EXPOSURE:", EXPOSURE, EXPOSURE.exists())
    print("V2T:", V2T, V2T.exists())
    print("T2V:", T2V, T2V.exists())

    exposure = pd.read_csv(EXPOSURE)
    v2t = pd.read_csv(V2T)
    t2v = pd.read_csv(T2V)

    video_df, text_df, video_id, text_id, indexes, order, idx_arr = load_official_ids_and_indexes()

    print("\n=== official axes ===")
    print("video_df:", video_df.shape)
    print("text_df:", text_df.shape)
    print("video_id len:", len(video_id))
    print("text_id len:", len(text_id))
    print("indexes:", indexes.shape, indexes[:10])
    print("order:", order.shape, order[:10])
    print("idx_arr:", idx_arr.shape, int(idx_arr.min()), int(idx_arr.max()))

    # Check exposure query_id against official video_id order.
    if "query_id" in exposure.columns:
        same_qid = exposure["query_id"].astype(str).reset_index(drop=True).equals(pd.Series(video_id).astype(str).reset_index(drop=True))
        print("exposure.query_id equals official video_id order:", same_qid)
        if not same_qid:
            mismatch = pd.DataFrame({
                "row": np.arange(len(exposure)),
                "exposure_query_id": exposure["query_id"].astype(str),
                "official_video_id": pd.Series(video_id).astype(str),
            })
            print("first mismatches:")
            print(mismatch[mismatch["exposure_query_id"] != mismatch["official_video_id"]].head(20).to_string(index=False))
            raise RuntimeError("Exposure row order does not match official video_id order. Stop before mapping.")

    source, col, normalizer, all_keys, rep_keys, mapped, report = choose_text_key(video_df, text_df, indexes, exposure)

    report.to_csv(OUT_DIR / "text_uid_key_candidate_report_20260705.csv", index=False)

    print("\n=== chosen text key ===")
    print("source:", source)
    print("column:", col)
    print("normalizer:", normalizer)
    print("unique all keys:", len(set(all_keys)))
    print("unique representative keys:", len(set(rep_keys)))
    print("mapped rows:", sum(x is not None for x in mapped))
    print("missing rows:", sum(x is None for x in mapped))

    text_uid = np.asarray(mapped, dtype=int)

    # Representative row for each text_uid should match indexes.
    rep_check = pd.DataFrame({
        "text_uid": np.arange(len(indexes), dtype=int),
        "representative_row_from_indexes": indexes,
        "representative_query_id": exposure.iloc[indexes]["query_id"].astype(str).values,
        "representative_pair_id": exposure.iloc[indexes]["pair_id"].astype(str).values,
        "representative_text_key": rep_keys,
    })

    map_df = pd.DataFrame({
        "official_eval_row": np.arange(len(exposure), dtype=int),
        "query_id": exposure["query_id"].astype(str).values,
        "text_uid": text_uid,
        "is_official_text_representative": False,
    })
    map_df.loc[indexes, "is_official_text_representative"] = True

    map_df = map_df.merge(
        rep_check[["text_uid", "representative_row_from_indexes", "representative_query_id", "representative_pair_id"]],
        on="text_uid",
        how="left",
    )

    map_df.to_csv(OUT_MAP, index=False)

    # Attach text_uid to full 9668 V2T rows.
    if len(v2t) != len(exposure):
        raise RuntimeError(f"V2T rows mismatch: {len(v2t)} vs exposure {len(exposure)}")

    v2t_units = v2t.merge(
        map_df[["official_eval_row", "text_uid", "is_official_text_representative", "representative_row_from_indexes", "representative_query_id", "representative_pair_id"]],
        on="official_eval_row",
        how="left",
    )

    if v2t_units["text_uid"].isna().any():
        raise RuntimeError("Some V2T rows missing text_uid")

    # Add text_uid to T2V official 3842 rows.
    # Previous T2V file should have official_text_query_col = 0..3841.
    if "official_text_query_col" not in t2v.columns:
        t2v["official_text_query_col"] = np.arange(len(t2v), dtype=int)

    t2v["text_uid"] = t2v["official_text_query_col"].astype(int)

    # Broadcast T2V metric to all 9668 rows sharing the same text_uid.
    t2v_metric_cols = ["text_uid", "T2V_AP", "T2V_nDCG", "official_text_query_col"]
    broadcast = v2t_units.merge(
        t2v[t2v_metric_cols],
        on="text_uid",
        how="left",
        suffixes=("", "_t2v"),
    )

    print("\n=== broadcast check ===")
    print("broadcast rows:", len(broadcast))
    print("rows with T2V_AP:", int(broadcast["T2V_AP"].notna().sum()))
    print("rows with T2V_nDCG:", int(broadcast["T2V_nDCG"].notna().sum()))

    if len(broadcast) != 9668 or int(broadcast["T2V_AP"].notna().sum()) != 9668:
        raise RuntimeError("Broadcast failed: expected 9668 rows with T2V metrics")

    # Collision check: one official text_uid should not map to multiple exposure pair_ids.
    collision = (
        broadcast.groupby("text_uid")
        .agg(
            n_rows=("query_id", "size"),
            n_pair_ids=("pair_id", "nunique"),
            n_verb_class=("verb_class", "nunique"),
            n_noun_class=("noun_class", "nunique"),
            pair_ids=("pair_id", lambda x: "|".join(sorted(set(map(str, x))))),
            query_ids=("query_id", lambda x: "|".join(list(map(str, x))[:20])),
            narrations=("narration", lambda x: " || ".join(list(map(str, x.drop_duplicates()))[:5]) if "narration" in broadcast.columns else ""),
        )
        .reset_index()
    )
    ambiguous = collision[collision["n_pair_ids"] > 1].copy()
    ambiguous.to_csv(OUT_COLLISION, index=False)

    print("\n=== collision check ===")
    print("text_uid count:", collision.shape[0])
    print("ambiguous text_uid count:", len(ambiguous))
    if len(ambiguous):
        print(ambiguous.head(20).to_string(index=False))

    # V2T unit table: 9668 rows.
    v2t_units.to_csv(OUT_V2T_UNITS, index=False)

    # T2V unit table: exactly 3842 official text queries.
    # Use the official t2v rows, enriched by representative exposure row.
    t2v_units = t2v.merge(
        rep_check[[
            "text_uid",
            "representative_row_from_indexes",
            "representative_query_id",
            "representative_pair_id",
        ]],
        on="text_uid",
        how="left",
    )

    rep_exposure = exposure.reset_index().rename(columns={"index": "representative_row_from_indexes"})
    keep_exp = [
        "representative_row_from_indexes",
        "query_id",
        "narration_id",
        "verb_class",
        "noun_class",
        "pair_id",
        "n_queries_this_pair",
        "v3_exposure_label",
        "pair_evidence_tier",
        "f_pair",
        "f_verb",
        "f_noun",
        "log10_f_pair",
        "log10_f_verb",
        "log10_f_noun",
        "PMI",
        "pair_syntactic_freq",
        "pair_backoff_freq",
        "pair_fallback_freq",
        "pair_marginal_only_freq",
        "has_pair_syntactic_evidence",
        "has_pair_backoff_evidence",
        "has_pair_fallback_evidence",
        "has_pair_marginal_only_evidence",
    ]
    keep_exp = [c for c in keep_exp if c in rep_exposure.columns]

    t2v_units = t2v_units.merge(
        rep_exposure[keep_exp],
        on="representative_row_from_indexes",
        how="left",
        suffixes=("", "_rep"),
    )

    # Clean ambiguous if needed.
    ambiguous_ids = set(ambiguous["text_uid"].astype(int).tolist())
    t2v_units["is_text_uid_ambiguous_pair"] = t2v_units["text_uid"].isin(ambiguous_ids)
    t2v_units_clean = t2v_units[~t2v_units["is_text_uid_ambiguous_pair"]].copy()

    t2v_units.to_csv(OUT_T2V_UNITS, index=False)
    t2v_units_clean.to_csv(OUT_T2V_CLEAN, index=False)

    broadcast.to_csv(OUT_BROADCAST, index=False)

    # Aggregate validation.
    rows = []
    rows.append({
        "check": "V2T_units_mean_AP_x100",
        "value": float(v2t_units["V2T_AP"].mean() * 100),
        "expected": EXPECTED["mAP_V2T"],
        "abs_diff": abs(float(v2t_units["V2T_AP"].mean() * 100) - EXPECTED["mAP_V2T"]),
    })
    rows.append({
        "check": "V2T_units_mean_nDCG_x100",
        "value": float(v2t_units["V2T_nDCG"].mean() * 100),
        "expected": EXPECTED["nDCG_V2T"],
        "abs_diff": abs(float(v2t_units["V2T_nDCG"].mean() * 100) - EXPECTED["nDCG_V2T"]),
    })
    rows.append({
        "check": "T2V_units_mean_AP_x100",
        "value": float(t2v_units["T2V_AP"].mean() * 100),
        "expected": EXPECTED["mAP_T2V"],
        "abs_diff": abs(float(t2v_units["T2V_AP"].mean() * 100) - EXPECTED["mAP_T2V"]),
    })
    rows.append({
        "check": "T2V_units_mean_nDCG_x100",
        "value": float(t2v_units["T2V_nDCG"].mean() * 100),
        "expected": EXPECTED["nDCG_T2V"],
        "abs_diff": abs(float(t2v_units["T2V_nDCG"].mean() * 100) - EXPECTED["nDCG_T2V"]),
    })
    rows.append({
        "check": "broadcast_rows",
        "value": float(len(broadcast)),
        "expected": 9668.0,
        "abs_diff": abs(float(len(broadcast)) - 9668.0),
    })
    rows.append({
        "check": "broadcast_rows_with_T2V",
        "value": float(broadcast["T2V_AP"].notna().sum()),
        "expected": 9668.0,
        "abs_diff": abs(float(broadcast["T2V_AP"].notna().sum()) - 9668.0),
    })
    rows.append({
        "check": "T2V_units_rows",
        "value": float(len(t2v_units)),
        "expected": 3842.0,
        "abs_diff": abs(float(len(t2v_units)) - 3842.0),
    })
    rows.append({
        "check": "ambiguous_text_uid_count",
        "value": float(len(ambiguous)),
        "expected": 0.0,
        "abs_diff": float(len(ambiguous)),
    })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n=== official aggregate validation ===")
    print(summary.to_string(index=False))

    print("\n=== output files ===")
    print("OUT_MAP:", OUT_MAP)
    print("OUT_BROADCAST:", OUT_BROADCAST)
    print("OUT_V2T_UNITS:", OUT_V2T_UNITS)
    print("OUT_T2V_UNITS:", OUT_T2V_UNITS)
    print("OUT_T2V_CLEAN:", OUT_T2V_CLEAN)
    print("OUT_COLLISION:", OUT_COLLISION)
    print("OUT_SUMMARY:", OUT_SUMMARY)

    failed = (
        len(broadcast) != 9668
        or int(broadcast["T2V_AP"].notna().sum()) != 9668
        or len(t2v_units) != 3842
        or summary.loc[summary["check"].isin([
            "V2T_units_mean_AP_x100",
            "V2T_units_mean_nDCG_x100",
            "T2V_units_mean_AP_x100",
            "T2V_units_mean_nDCG_x100",
        ]), "abs_diff"].max() > 1e-6
    )

    # collision is not an execution failure, but must be reported.
    print("\nSUMMARY failed=0" if not failed else "\nSUMMARY failed=1")

if __name__ == "__main__":
    main()
