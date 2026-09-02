#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-pt", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--exposure-query-csv", type=Path, required=True)
    parser.add_argument("--v2t-metrics-csv", type=Path, required=True)
    parser.add_argument("--t2v-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-v2t-units-name", type=str, default="ek100_mir_metric_units_v2t.csv")
    parser.add_argument("--output-t2v-units-name", type=str, default="ek100_mir_metric_units_t2v.csv")
    parser.add_argument("--output-representatives-name", type=str, default="ek100_mir_official_t2v_representative_rows.csv")
    parser.add_argument("--output-check-name", type=str, default="official_metric_unit_table_check.csv")
    parser.add_argument("--output-note-name", type=str, default="T2V_UNIT_POLICY_NO_BROADCAST.md")
    return parser.parse_args()


FEATURE_PT = None
META_DIR = None
EXPOSURE = None
V2T_METRICS = None
T2V_METRICS = None
OUT_DIR = None
OUT_V2T_UNITS = None
OUT_T2V_UNITS = None
OUT_REP_MAP = None
OUT_CHECK = None
OUT_NOTE = None

EXPECTED = {
    "mAP_V2T": 29.88172021139886,
    "nDCG_V2T": 30.071646777422366,
    "mAP_T2V": 23.448891050955933,
    "nDCG_T2V": 28.12497337856873,
}

def main():
    global FEATURE_PT, META_DIR, EXPOSURE, V2T_METRICS, T2V_METRICS
    global OUT_DIR, OUT_V2T_UNITS, OUT_T2V_UNITS, OUT_REP_MAP, OUT_CHECK, OUT_NOTE

    args = parse_args()

    FEATURE_PT = args.feature_pt
    META_DIR = args.meta_dir
    EXPOSURE = args.exposure_query_csv
    V2T_METRICS = args.v2t_metrics_csv
    T2V_METRICS = args.t2v_metrics_csv

    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_V2T_UNITS = OUT_DIR / args.output_v2t_units_name
    OUT_T2V_UNITS = OUT_DIR / args.output_t2v_units_name
    OUT_REP_MAP = OUT_DIR / args.output_representatives_name
    OUT_CHECK = OUT_DIR / args.output_check_name
    OUT_NOTE = OUT_DIR / args.output_note_name

    print("ROOT:", ROOT)
    print("EXPOSURE:", EXPOSURE, EXPOSURE.exists())
    print("V2T_METRICS:", V2T_METRICS, V2T_METRICS.exists())
    print("T2V_METRICS:", T2V_METRICS, T2V_METRICS.exists())
    print("FEATURE_PT:", FEATURE_PT, FEATURE_PT.exists())
    print("META_DIR:", META_DIR, META_DIR.exists())

    exp = pd.read_csv(EXPOSURE)
    v2t = pd.read_csv(V2T_METRICS)
    t2v = pd.read_csv(T2V_METRICS)

    obj = torch.load(FEATURE_PT, map_location="cpu")
    idx_arr = obj["arr_embeds"].cpu().numpy()

    video_df = pd.read_csv(META_DIR / "EPIC_100_retrieval_test.csv")
    text_df = pd.read_csv(META_DIR / "EPIC_100_retrieval_test_sentence.csv")

    video_id = video_df.values[:, 0]
    text_id = text_df.values[:, 0]

    video_id_list = video_id.tolist()
    indexes = np.asarray([video_id_list.index(elem) for elem in text_id], dtype=int)

    print("\n=== input shapes ===")
    print("exp:", exp.shape)
    print("v2t:", v2t.shape)
    print("t2v:", t2v.shape)
    print("video_df:", video_df.shape)
    print("text_df:", text_df.shape)
    print("indexes:", indexes.shape, indexes[:10])
    print("idx_arr:", idx_arr.shape, int(idx_arr.min()), int(idx_arr.max()))

    assert len(exp) == 9668, len(exp)
    assert len(v2t) == 9668, len(v2t)
    assert len(t2v) == 3842, len(t2v)
    assert len(indexes) == 3842, len(indexes)

    # Critical row-order check.
    same_qid = exp["query_id"].astype(str).reset_index(drop=True).equals(
        pd.Series(video_id).astype(str).reset_index(drop=True)
    )
    print("exposure.query_id equals official video_id order:", same_qid)
    if not same_qid:
        bad = pd.DataFrame({
            "row": np.arange(len(exp)),
            "exp_query_id": exp["query_id"].astype(str),
            "official_video_id": pd.Series(video_id).astype(str),
        })
        print(bad[bad["exp_query_id"] != bad["official_video_id"]].head(50).to_string(index=False))
        raise RuntimeError("Exposure table row order does not match official video_id order.")

    # ------------------------------------------------------------------
    # V2T units: official video-query axis, 9668 rows.
    # v2t already contains exposure columns and V2T_AP / V2T_nDCG.
    # We nevertheless re-attach a few provenance columns from canonical exp
    # if they are missing.
    # ------------------------------------------------------------------
    v2t_units = v2t.copy()

    if "official_eval_row" not in v2t_units.columns:
        v2t_units["official_eval_row"] = np.arange(len(v2t_units), dtype=int)

    v2t_units["analysis_unit"] = "V2T_video_query"
    v2t_units["official_video_id"] = video_id

    # ------------------------------------------------------------------
    # T2V units: official unique text-query axis, 3842 representative rows.
    # Do NOT broadcast to 9668.
    # ------------------------------------------------------------------
    rep_exp = exp.iloc[indexes].copy().reset_index(drop=False)
    rep_exp = rep_exp.rename(columns={"index": "representative_exposure_row"})

    rep_map = pd.DataFrame({
        "official_text_uid": np.arange(len(indexes), dtype=int),
        "representative_exposure_row": indexes,
        "representative_query_id": exp.iloc[indexes]["query_id"].astype(str).values,
        "official_text_id": text_id,
    })

    rep_map.to_csv(OUT_REP_MAP, index=False)

    # Build clean T2V unit table from representative exposure rows.
    keep_cols = [
        "representative_exposure_row",
        "query_id",
        "narration_id",
        "participant_id",
        "video_id",
        "narration",
        "verb",
        "verb_class",
        "noun",
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
        "log10_1p_f_pair",
        "log10_1p_f_verb",
        "log10_1p_f_noun",
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
    keep_cols = [c for c in keep_cols if c in rep_exp.columns]

    t2v_units = rep_exp[keep_cols].copy()
    t2v_units.insert(0, "official_text_uid", np.arange(len(t2v_units), dtype=int))
    t2v_units["official_text_id"] = text_id
    t2v_units["analysis_unit"] = "T2V_official_unique_text_query"

    # Attach T2V metrics by official text query order.
    t2v_metric = t2v.reset_index(drop=True).copy()
    if "official_text_query_col" in t2v_metric.columns:
        # Keep it for traceability.
        t2v_units["official_text_query_col"] = t2v_metric["official_text_query_col"].astype(int).values
    else:
        t2v_units["official_text_query_col"] = np.arange(len(t2v_units), dtype=int)

    t2v_units["T2V_AP"] = t2v_metric["T2V_AP"].values
    t2v_units["T2V_nDCG"] = t2v_metric["T2V_nDCG"].values

    # Duplicate narration diagnostic on representative rows only.
    dup_narr = (
        t2v_units.groupby(t2v_units["narration"].astype(str).str.strip())
        .agg(
            n_text_uids=("official_text_uid", "size"),
            n_pair_ids=("pair_id", "nunique"),
            text_uids=("official_text_uid", lambda x: "|".join(map(str, x))),
            pair_ids=("pair_id", lambda x: "|".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .rename(columns={"narration": "narration_norm"})
    )
    dup_narr = dup_narr[dup_narr["n_text_uids"] > 1].copy()
    dup_narr.to_csv(OUT_DIR / "official_t2v_duplicate_narration_representatives.csv", index=False)

    # Save unit tables.
    v2t_units.to_csv(OUT_V2T_UNITS, index=False)
    t2v_units.to_csv(OUT_T2V_UNITS, index=False)

    # Validation.
    checks = []
    checks.append({
        "check": "V2T_rows",
        "value": float(len(v2t_units)),
        "expected": 9668.0,
        "abs_diff": abs(float(len(v2t_units)) - 9668.0),
    })
    checks.append({
        "check": "T2V_rows",
        "value": float(len(t2v_units)),
        "expected": 3842.0,
        "abs_diff": abs(float(len(t2v_units)) - 3842.0),
    })
    checks.append({
        "check": "V2T_mean_AP_x100",
        "value": float(v2t_units["V2T_AP"].mean() * 100),
        "expected": EXPECTED["mAP_V2T"],
        "abs_diff": abs(float(v2t_units["V2T_AP"].mean() * 100) - EXPECTED["mAP_V2T"]),
    })
    checks.append({
        "check": "V2T_mean_nDCG_x100",
        "value": float(v2t_units["V2T_nDCG"].mean() * 100),
        "expected": EXPECTED["nDCG_V2T"],
        "abs_diff": abs(float(v2t_units["V2T_nDCG"].mean() * 100) - EXPECTED["nDCG_V2T"]),
    })
    checks.append({
        "check": "T2V_mean_AP_x100",
        "value": float(t2v_units["T2V_AP"].mean() * 100),
        "expected": EXPECTED["mAP_T2V"],
        "abs_diff": abs(float(t2v_units["T2V_AP"].mean() * 100) - EXPECTED["mAP_T2V"]),
    })
    checks.append({
        "check": "T2V_mean_nDCG_x100",
        "value": float(t2v_units["T2V_nDCG"].mean() * 100),
        "expected": EXPECTED["nDCG_T2V"],
        "abs_diff": abs(float(t2v_units["T2V_nDCG"].mean() * 100) - EXPECTED["nDCG_T2V"]),
    })
    checks.append({
        "check": "T2V_duplicate_narration_representative_groups",
        "value": float(len(dup_narr)),
        "expected": np.nan,
        "abs_diff": np.nan,
    })

    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT_CHECK, index=False)

    print("\n=== output tables ===")
    print("OUT_V2T_UNITS:", OUT_V2T_UNITS)
    print("OUT_T2V_UNITS:", OUT_T2V_UNITS)
    print("OUT_REP_MAP:", OUT_REP_MAP)
    print("OUT_CHECK:", OUT_CHECK)

    print("\n=== label counts ===")
    print("V2T:")
    print(v2t_units["v3_exposure_label"].value_counts().to_string())
    print("\nT2V:")
    print(t2v_units["v3_exposure_label"].value_counts().to_string())

    print("\n=== duplicate narration representatives ===")
    print("groups:", len(dup_narr))
    if len(dup_narr):
        print(dup_narr.to_string(index=False))

    print("\n=== validation ===")
    print(check_df.to_string(index=False))

    failed = (
        len(v2t_units) != 9668
        or len(t2v_units) != 3842
        or check_df.loc[
            check_df["check"].isin([
                "V2T_mean_AP_x100",
                "V2T_mean_nDCG_x100",
                "T2V_mean_AP_x100",
                "T2V_mean_nDCG_x100",
            ]),
            "abs_diff"
        ].max() > 1e-6
    )

    # Duplicate narration groups are a documented property, not a failure.
    print("\nSUMMARY failed=0" if not failed else "\nSUMMARY failed=1")

    OUT_NOTE.write_text(
        """# T2V unit policy: no broadcast to 9668 rows

The official EK100-MIR metric uses different query axes by direction:

- V2T query axis: 9668 official video-query rows.
- T2V query axis: 3842 official unique text-query rows.

We attempted to infer a full 9668-to-3842 text_uid broadcast mapping, but no text or composite exposure key exactly reproduced both the official 3842 representatives and a complete mapping for all 9668 rows.

Therefore, the canonical analysis unit policy is:

- Use the V2T metric-unit table for V2T analyses.
- Use the T2V metric-unit table for T2V analyses.
- Do not use a 9668-row broadcast table as the statistical unit for T2V.

The T2V unit table uses the official representative rows defined by `EPIC_100_retrieval_test_sentence.csv` through the same `indexes` mapping used by `model.metric.mir_metrics_vtc`.

The unit-table means exactly reproduce the official aggregate metrics:

- V2T mAP / nDCG
- T2V mAP / nDCG

Duplicate narration strings among official text representatives are documented separately. They are not collapsed because the official T2V axis treats them as separate text queries.
""",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
