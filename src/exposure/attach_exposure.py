#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd

QUERY_CSV = Path("Data/processed/ek100_mir_test_exposure_labeled.csv")

PAIR_CSV = Path("Data/processed/egoclip_ek100_pair_freq_v3_full.csv")
PAIR_BY_PATH_CSV = Path("Data/processed/egoclip_ek100_pair_freq_by_path_v3_full.csv")
MARGINAL_CSV = Path("Data/processed/egoclip_ek100_marginal_only_v3_full.csv")
VERB_CSV = Path("Data/processed/egoclip_ek100_seen_verbs_v3_full.csv")
NOUN_CSV = Path("Data/processed/egoclip_ek100_seen_nouns_v3_full.csv")

OUT_DIR = Path("Data/processed/exposure_metric_analysis_v3_full_20260705")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_QUERY = OUT_DIR / "ek100_mir_v3_full_query_exposure_with_provenance.csv"
OUT_PAIR = OUT_DIR / "ek100_mir_v3_full_pair_exposure_units_with_provenance.csv"

for p in [QUERY_CSV, PAIR_CSV, PAIR_BY_PATH_CSV, MARGINAL_CSV, VERB_CSV, NOUN_CSV]:
    if not p.exists():
        raise FileNotFoundError(p)

q = pd.read_csv(QUERY_CSV)
pair = pd.read_csv(PAIR_CSV)
pair_path = pd.read_csv(PAIR_BY_PATH_CSV)
marg = pd.read_csv(MARGINAL_CSV)
verb = pd.read_csv(VERB_CSV)
noun = pd.read_csv(NOUN_CSV)

for df in [q, pair, pair_path, marg]:
    df["verb_class"] = df["verb_class"].astype(int)
    df["noun_class"] = df["noun_class"].astype(int)

verb["verb_class"] = verb["verb_class"].astype(int)
noun["noun_class"] = noun["noun_class"].astype(int)

pair_freq = pair.set_index(["verb_class", "noun_class"])["freq"].to_dict()
verb_freq = verb.set_index("verb_class")["freq"].to_dict()
noun_freq = noun.set_index("noun_class")["freq"].to_dict()

pair_set = set(zip(pair["verb_class"], pair["noun_class"]))
seen_verbs = set(verb.loc[verb["freq"] > 0, "verb_class"])
seen_nouns = set(noun.loc[noun["freq"] > 0, "noun_class"])

total_pair = float(pair["freq"].sum())
total_verb = float(verb["freq"].sum())
total_noun = float(noun["freq"].sum())

def label_v3(v, n):
    v = int(v)
    n = int(n)
    if (v, n) in pair_set:
        return "SC"
    if v in seen_verbs and n in seen_nouns:
        return "UC"
    return "UA"

def safe_log10(series):
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = series > 0
    out.loc[mask] = np.log10(series.loc[mask])
    return out

def tier_from_parse_path(path):
    path = str(path)
    if path == "syntactic_object":
        return "core_syntactic_object"
    if path.startswith("pronoun_backoff"):
        return "pronoun_backoff"
    if "noun_after_verb" in path or "noun_anywhere" in path:
        return "fallback_pair_evidence"
    return "other_pair_evidence"

out = q.copy()
out["query_id"] = out["narration_id"].astype(str)
out["pair_id"] = out["verb_class"].astype(str) + "_" + out["noun_class"].astype(str)

# EK100-MIR test-set support size, not EgoClip exposure.
out["n_queries_this_pair"] = (
    out.groupby(["verb_class", "noun_class"])["query_id"]
    .transform("count")
    .astype(int)
)

out["v3_exposure_label"] = [
    label_v3(v, n) for v, n in zip(out["verb_class"], out["noun_class"])
]

out["f_pair"] = [
    int(pair_freq.get((int(v), int(n)), 0))
    for v, n in zip(out["verb_class"], out["noun_class"])
]
out["f_verb"] = [int(verb_freq.get(int(v), 0)) for v in out["verb_class"]]
out["f_noun"] = [int(noun_freq.get(int(n), 0)) for n in out["noun_class"]]

# Safe log10: no divide-by-zero warning.
out["log10_f_pair"] = safe_log10(out["f_pair"])
out["log10_f_verb"] = safe_log10(out["f_verb"])
out["log10_f_noun"] = safe_log10(out["f_noun"])

out["log10_1p_f_pair"] = np.log10(1 + out["f_pair"])
out["log10_1p_f_verb"] = np.log10(1 + out["f_verb"])
out["log10_1p_f_noun"] = np.log10(1 + out["f_noun"])

# PMI only meaningful for seen pairs. UC/UA remain NaN.
p_pair = out["f_pair"] / total_pair
p_verb = out["f_verb"] / total_verb
p_noun = out["f_noun"] / total_noun

out["PMI"] = np.nan
mask = out["f_pair"] > 0
out.loc[mask, "PMI"] = np.log(
    p_pair.loc[mask] / (p_verb.loc[mask] * p_noun.loc[mask])
)

# Aggregate pair-evidence provenance from EgoClip parse paths.
pp = pair_path.copy()
pp["evidence_tier_component"] = pp["parse_path"].map(tier_from_parse_path)

tier_freq = (
    pp.groupby(["verb_class", "noun_class", "evidence_tier_component"], as_index=False)["freq"]
    .sum()
)

tier_wide = (
    tier_freq.pivot_table(
        index=["verb_class", "noun_class"],
        columns="evidence_tier_component",
        values="freq",
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)

for c in [
    "core_syntactic_object",
    "pronoun_backoff",
    "fallback_pair_evidence",
    "other_pair_evidence",
]:
    if c not in tier_wide.columns:
        tier_wide[c] = 0

tier_wide = tier_wide.rename(columns={
    "core_syntactic_object": "pair_syntactic_freq",
    "pronoun_backoff": "pair_backoff_freq",
    "fallback_pair_evidence": "pair_fallback_freq",
    "other_pair_evidence": "pair_other_path_freq",
})

# Dominant pair evidence path.
idx = pp.groupby(["verb_class", "noun_class"])["freq"].idxmax()
dominant = pp.loc[idx, ["verb_class", "noun_class", "parse_path", "freq"]].copy()
dominant = dominant.rename(columns={
    "parse_path": "dominant_pair_parse_path",
    "freq": "dominant_pair_parse_path_freq",
})
dominant["pair_evidence_tier"] = dominant["dominant_pair_parse_path"].map(tier_from_parse_path)

out = out.merge(tier_wide, on=["verb_class", "noun_class"], how="left")
out = out.merge(dominant, on=["verb_class", "noun_class"], how="left")

for c in [
    "pair_syntactic_freq",
    "pair_backoff_freq",
    "pair_fallback_freq",
    "pair_other_path_freq",
    "dominant_pair_parse_path_freq",
]:
    out[c] = out[c].fillna(0).astype(int)

out["dominant_pair_parse_path"] = out["dominant_pair_parse_path"].fillna("none")
out["pair_evidence_tier"] = out["pair_evidence_tier"].fillna("no_pair_evidence")

out["has_pair_syntactic_evidence"] = out["pair_syntactic_freq"] > 0
out["has_pair_backoff_evidence"] = out["pair_backoff_freq"] > 0
out["has_pair_fallback_evidence"] = out["pair_fallback_freq"] > 0
out["has_pair_other_path_evidence"] = out["pair_other_path_freq"] > 0

# Marginal-only is not pair evidence.
marg_atom = (
    marg.groupby(["verb_class", "noun_class"], as_index=False)["freq"]
    .sum()
    .rename(columns={"freq": "pair_marginal_only_freq"})
)
out = out.merge(marg_atom, on=["verb_class", "noun_class"], how="left")
out["pair_marginal_only_freq"] = out["pair_marginal_only_freq"].fillna(0).astype(int)
out["has_pair_marginal_only_evidence"] = out["pair_marginal_only_freq"] > 0

# Safety checks.
sc = out[out["v3_exposure_label"] == "SC"].copy()
ucua = out[out["v3_exposure_label"] != "SC"].copy()

bad_sc = sc[
    (sc["f_pair"] <= 0)
    | sc["log10_f_pair"].isna()
    | np.isinf(sc["log10_f_pair"])
]
bad_ucua = ucua[ucua["f_pair"] > 0]
bad_tier = sc[sc["pair_evidence_tier"] == "no_pair_evidence"]

if len(bad_sc) or len(bad_ucua) or len(bad_tier):
    raise RuntimeError(
        f"Exposure consistency failed: bad_sc={len(bad_sc)} "
        f"bad_ucua={len(bad_ucua)} bad_tier={len(bad_tier)}"
    )

keep_front = [
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
    "exposure_label",
    "exposure_label_fine",
    "v3_exposure_label",
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
    "pair_evidence_tier",
    "dominant_pair_parse_path",
    "dominant_pair_parse_path_freq",
    "pair_syntactic_freq",
    "pair_backoff_freq",
    "pair_fallback_freq",
    "pair_other_path_freq",
    "pair_marginal_only_freq",
    "has_pair_syntactic_evidence",
    "has_pair_backoff_evidence",
    "has_pair_fallback_evidence",
    "has_pair_other_path_evidence",
    "has_pair_marginal_only_evidence",
]

rest = [c for c in out.columns if c not in keep_front]
out = out[keep_front + rest]

out.to_csv(OUT_QUERY, index=False)

pair_level = (
    out.groupby(["verb_class", "noun_class", "pair_id"], as_index=False)
    .agg(
        n_queries=("query_id", "count"),
        v3_exposure_label=("v3_exposure_label", "first"),
        f_pair=("f_pair", "first"),
        f_verb=("f_verb", "first"),
        f_noun=("f_noun", "first"),
        log10_f_pair=("log10_f_pair", "first"),
        log10_f_verb=("log10_f_verb", "first"),
        log10_f_noun=("log10_f_noun", "first"),
        log10_1p_f_pair=("log10_1p_f_pair", "first"),
        log10_1p_f_verb=("log10_1p_f_verb", "first"),
        log10_1p_f_noun=("log10_1p_f_noun", "first"),
        PMI=("PMI", "first"),
        pair_evidence_tier=("pair_evidence_tier", "first"),
        dominant_pair_parse_path=("dominant_pair_parse_path", "first"),
        dominant_pair_parse_path_freq=("dominant_pair_parse_path_freq", "first"),
        pair_syntactic_freq=("pair_syntactic_freq", "first"),
        pair_backoff_freq=("pair_backoff_freq", "first"),
        pair_fallback_freq=("pair_fallback_freq", "first"),
        pair_other_path_freq=("pair_other_path_freq", "first"),
        pair_marginal_only_freq=("pair_marginal_only_freq", "first"),
        has_pair_syntactic_evidence=("has_pair_syntactic_evidence", "first"),
        has_pair_backoff_evidence=("has_pair_backoff_evidence", "first"),
        has_pair_fallback_evidence=("has_pair_fallback_evidence", "first"),
        has_pair_other_path_evidence=("has_pair_other_path_evidence", "first"),
        has_pair_marginal_only_evidence=("has_pair_marginal_only_evidence", "first"),
    )
)
pair_level.to_csv(OUT_PAIR, index=False)

print("OUT_QUERY:", OUT_QUERY)
print("OUT_PAIR:", OUT_PAIR)
print("query rows:", len(out))
print("unique pairs:", len(pair_level))

print("\n=== old coarse labels ===")
print(out["exposure_label"].value_counts().to_string())

print("\n=== v3 coarse labels ===")
print(out["v3_exposure_label"].value_counts().to_string())

print("\n=== query-side n_queries_this_pair describe ===")
print(pair_level["n_queries"].describe(percentiles=[.5,.75,.9,.95,.99]).to_string())

print("\n=== dominant pair_evidence_tier counts, query rows ===")
print(out["pair_evidence_tier"].value_counts().to_string())

print("\n=== dominant pair_evidence_tier counts, pair units ===")
print(pair_level["pair_evidence_tier"].value_counts().to_string())

print("\n=== nonzero pair evidence flags, query rows ===")
for c in [
    "has_pair_syntactic_evidence",
    "has_pair_backoff_evidence",
    "has_pair_fallback_evidence",
    "has_pair_marginal_only_evidence",
]:
    print(f"{c}: {int(out[c].sum())}")

print("\n=== nonzero pair evidence flags, pair units ===")
for c in [
    "has_pair_syntactic_evidence",
    "has_pair_backoff_evidence",
    "has_pair_fallback_evidence",
    "has_pair_marginal_only_evidence",
]:
    print(f"{c}: {int(pair_level[c].sum())}")

print("\n=== pair_backoff_freq ===")
print("query-level repeated sum:", int(out["pair_backoff_freq"].sum()))
print("pair-level unique sum:", int(pair_level["pair_backoff_freq"].sum()))
print("query rows with backoff:", int((out["pair_backoff_freq"] > 0).sum()))
print("pair units with backoff:", int((pair_level["pair_backoff_freq"] > 0).sum()))

print("\n=== SC log10_f_pair check ===")
print("SC rows:", len(sc))
print("SC f_pair <= 0:", int((sc["f_pair"] <= 0).sum()))
print("SC log10_f_pair NaN:", int(sc["log10_f_pair"].isna().sum()))
print("SC log10_f_pair inf:", int(np.isinf(sc["log10_f_pair"]).sum()))

print("\n=== f_pair describe for SC only ===")
print(sc["f_pair"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_string())

print("\n=== PMI describe for SC only ===")
print(sc["PMI"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_string())

print("SUMMARY failed=0")
