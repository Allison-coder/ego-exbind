#!/usr/bin/env python
from pathlib import Path
import pandas as pd

OLD_LABEL_CSV = Path("Data/processed/ek100_mir_test_exposure_labeled.csv")

PAIR_CSV = Path("Data/processed/egoclip_ek100_pair_freq_v3_full.csv")
SEEN_VERBS_CSV = Path("Data/processed/egoclip_ek100_seen_verbs_v3_full.csv")
SEEN_NOUNS_CSV = Path("Data/processed/egoclip_ek100_seen_nouns_v3_full.csv")
MARGINAL_CSV = Path("Data/processed/egoclip_ek100_marginal_only_v3_full.csv")

OUT_DIR = Path("Data/processed/sc_uc_ua_migration_v3_full_20260705")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["SC", "UC", "UA"]

def require(path):
    if not path.exists():
        raise FileNotFoundError(path)

for p in [OLD_LABEL_CSV, PAIR_CSV, SEEN_VERBS_CSV, SEEN_NOUNS_CSV, MARGINAL_CSV]:
    require(p)

old = pd.read_csv(OLD_LABEL_CSV)
pair = pd.read_csv(PAIR_CSV)
seen_v = pd.read_csv(SEEN_VERBS_CSV)
seen_n = pd.read_csv(SEEN_NOUNS_CSV)
marg = pd.read_csv(MARGINAL_CSV)

# This script intentionally uses only coarse SC/UC/UA labels.
# It ignores exposure_label_fine. Rare/freq split is a later, separate step.
required_old_cols = {"verb_class", "noun_class", "exposure_label"}
missing = required_old_cols - set(old.columns)
if missing:
    raise RuntimeError(f"Missing columns in old label CSV: {missing}")

for c in ["verb_class", "noun_class"]:
    old[c] = old[c].astype(int)

for df in [pair, marg]:
    df["verb_class"] = df["verb_class"].astype(int)
    df["noun_class"] = df["noun_class"].astype(int)

seen_v["verb_class"] = seen_v["verb_class"].astype(int)
seen_n["noun_class"] = seen_n["noun_class"].astype(int)

pair_set = set(zip(pair["verb_class"], pair["noun_class"]))
seen_verbs = set(seen_v.loc[seen_v["freq"] > 0, "verb_class"])
seen_nouns = set(seen_n.loc[seen_n["freq"] > 0, "noun_class"])

def label_v3(v, n):
    v = int(v)
    n = int(n)
    if (v, n) in pair_set:
        return "SC"
    if v in seen_verbs and n in seen_nouns:
        return "UC"
    return "UA"

out = old.copy()
out["old_label"] = out["exposure_label"].astype(str).str.strip().str.upper()
out["v3_label"] = [label_v3(v, n) for v, n in zip(out["verb_class"], out["noun_class"])]

bad_old = sorted(set(out["old_label"]) - set(LABEL_ORDER))
if bad_old:
    raise RuntimeError(f"Unexpected old coarse labels: {bad_old}")

matrix = pd.crosstab(out["old_label"], out["v3_label"]).reindex(
    index=LABEL_ORDER,
    columns=LABEL_ORDER,
    fill_value=0,
)

sizes = pd.DataFrame({"label": LABEL_ORDER})
sizes["old_count"] = sizes["label"].map(out["old_label"].value_counts()).fillna(0).astype(int)
sizes["v3_count"] = sizes["label"].map(out["v3_label"].value_counts()).fillna(0).astype(int)

flips = out[out["old_label"] != out["v3_label"]].copy()
flip_summary = (
    flips.groupby(["old_label", "v3_label"], as_index=False)
    .size()
    .sort_values(["old_label", "v3_label"])
)

freq_map = pair.set_index(["verb_class", "noun_class"])["freq"].to_dict()
out["v3_pair_freq"] = [
    int(freq_map.get((int(v), int(n)), 0))
    for v, n in zip(out["verb_class"], out["noun_class"])
]

marg_atom = (
    marg.groupby(["verb_class", "noun_class"], as_index=False)["freq"]
    .sum()
    .rename(columns={"freq": "v3_marginal_only_atom_freq"})
)
out = out.merge(marg_atom, on=["verb_class", "noun_class"], how="left")
out["v3_marginal_only_atom_freq"] = out["v3_marginal_only_atom_freq"].fillna(0).astype(int)

# Global consistency: marginal-only atom evidence must be included in combined seen verb/noun tables.
marg_v = marg.groupby("verb_class", as_index=False)["freq"].sum().rename(columns={"freq": "marginal_freq"})
marg_n = marg.groupby("noun_class", as_index=False)["freq"].sum().rename(columns={"freq": "marginal_freq"})

verb_consistency = marg_v.merge(
    seen_v[["verb_class", "freq"]].rename(columns={"freq": "seen_freq"}),
    on="verb_class",
    how="left",
)
noun_consistency = marg_n.merge(
    seen_n[["noun_class", "freq"]].rename(columns={"freq": "seen_freq"}),
    on="noun_class",
    how="left",
)

verb_consistency["seen_freq"] = verb_consistency["seen_freq"].fillna(0).astype(int)
noun_consistency["seen_freq"] = noun_consistency["seen_freq"].fillna(0).astype(int)

verb_consistency["violation"] = verb_consistency["seen_freq"] < verb_consistency["marginal_freq"]
noun_consistency["violation"] = noun_consistency["seen_freq"] < noun_consistency["marginal_freq"]

verb_bad = verb_consistency[verb_consistency["violation"]].copy()
noun_bad = noun_consistency[noun_consistency["violation"]].copy()

prefix = OUT_DIR / "sc_uc_ua_coarse_migration_v1labels_to_v3_full_20260705"

out.to_csv(str(prefix) + "_per_query.csv", index=False)
matrix.to_csv(str(prefix) + "_matrix.csv")
sizes.to_csv(str(prefix) + "_group_sizes.csv", index=False)
flips.to_csv(str(prefix) + "_flips.csv", index=False)
flip_summary.to_csv(str(prefix) + "_flip_summary.csv", index=False)
verb_consistency.to_csv(str(prefix) + "_verb_marginal_seen_consistency.csv", index=False)
noun_consistency.to_csv(str(prefix) + "_noun_marginal_seen_consistency.csv", index=False)
verb_bad.to_csv(str(prefix) + "_verb_marginal_seen_violations.csv", index=False)
noun_bad.to_csv(str(prefix) + "_noun_marginal_seen_violations.csv", index=False)

with open(str(prefix) + "_README.txt", "w") as f:
    f.write("Coarse SC/UC/UA migration from original exposure_label to v3_full labels.\n")
    f.write("Old coarse labels source: Data/processed/ek100_mir_test_exposure_labeled.csv: exposure_label.\n")
    f.write("This script intentionally ignores exposure_label_fine.\n")
    f.write("Rare/freq split is not computed here and must be handled separately.\n")
    f.write("New labels source: v3_full pair + combined seen verb/noun tables.\n")
    f.write("Rule: SC if pair seen; UC if both verb and noun seen but pair unseen; UA otherwise.\n")
    f.write("Generated before retrieval metric computation.\n")

print("=== INPUTS ===")
print("old_label_csv:", OLD_LABEL_CSV)
print("old label column: exposure_label")
print("fine label column ignored: exposure_label_fine")
print("pair_csv:", PAIR_CSV)
print("seen_verbs_csv:", SEEN_VERBS_CSV)
print("seen_nouns_csv:", SEEN_NOUNS_CSV)
print("marginal_csv:", MARGINAL_CSV)

print("\n=== ROWS ===")
print("queries:", len(out))
print("pair rows:", len(pair))
print("seen verb rows:", len(seen_v))
print("seen noun rows:", len(seen_n))
print("marginal rows:", len(marg))

print("\n=== OLD COARSE LABEL COUNTS ===")
print(out["old_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_string())

print("\n=== V3 COARSE LABEL COUNTS ===")
print(out["v3_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_string())

print("\n=== MIGRATION MATRIX old -> v3 ===")
print(matrix.to_string())

print("\n=== GROUP SIZES ===")
print(sizes.to_string(index=False))

print("\n=== FLIP SUMMARY ===")
if len(flip_summary):
    print(flip_summary.to_string(index=False))
else:
    print("no flips")

print("\n=== MARGINAL / SEEN CONSISTENCY ===")
print("verb violations:", len(verb_bad))
print("noun violations:", len(noun_bad))

if len(verb_bad) or len(noun_bad):
    print("\nBAD VERBS")
    print(verb_bad.head(20).to_string(index=False))
    print("\nBAD NOUNS")
    print(noun_bad.head(20).to_string(index=False))
    raise SystemExit("FAIL: marginal-only counts not reflected in combined seen tables")

print("\nOUTPUT_DIR:", OUT_DIR)
print("SUMMARY failed=0")
