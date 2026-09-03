import os
from pathlib import Path
import numpy as np
import pandas as pd

DAY1 = Path(os.environ["DAY1"])
BIND_ROOT = Path(os.environ["BIND_ROOT"])
OUT = Path(os.environ["LARGE_PROBE_VAL"])

old = pd.read_csv(os.environ["OLD_PROBE"])
val = pd.read_csv("Data/ek100/retrieval_annotations/EPIC_100_retrieval_test.csv")

need = ["narration_id", "verb_class", "noun_class", "verb", "noun", "narration"]
miss = [c for c in need if c not in val.columns]
if miss:
    raise RuntimeError(f"val csv missing columns {miss}; have={val.columns.tolist()}")

# Pair exposure ledger from old probe inventories.
# Prefer pair_inventory because it was part of original C3 construction package.
pair_inv = BIND_ROOT / "pair_inventory.csv"
if not pair_inv.exists():
    raise RuntimeError(f"missing pair inventory: {pair_inv}")

ledger = pd.read_csv(pair_inv)
print("pair_inventory cols:", ledger.columns.tolist())

# Normalize ledger columns.
if "verb_class" not in ledger.columns:
    for c in ["verb_id", "positive_verb_id"]:
        if c in ledger.columns:
            ledger = ledger.rename(columns={c: "verb_class"})
            break
if "noun_class" not in ledger.columns:
    for c in ["noun_id", "positive_noun_id"]:
        if c in ledger.columns:
            ledger = ledger.rename(columns={c: "noun_class"})
            break
if "f_pair" not in ledger.columns:
    for c in ["f_positive_pair", "count", "frequency", "pair_count"]:
        if c in ledger.columns:
            ledger = ledger.rename(columns={c: "f_pair"})
            break

need_ledger = ["verb_class", "noun_class", "f_pair"]
miss = [c for c in need_ledger if c not in ledger.columns]
if miss:
    raise RuntimeError(f"ledger missing {miss}; have={ledger.columns.tolist()}")

ledger = ledger[["verb_class", "noun_class", "f_pair"]].drop_duplicates(["verb_class", "noun_class"]).copy()
ledger["verb_class"] = ledger["verb_class"].astype(int)
ledger["noun_class"] = ledger["noun_class"].astype(int)
ledger["f_pair"] = ledger["f_pair"].astype(float)
ledger["log10_f_pair"] = np.log10(ledger["f_pair"].clip(lower=1e-12))

val = val.copy()
val["verb_class"] = val["verb_class"].astype(int)
val["noun_class"] = val["noun_class"].astype(int)

df = val.merge(ledger, on=["verb_class", "noun_class"], how="left")
df["f_pair"] = df["f_pair"].fillna(0.0)
df["log10_f_pair"] = np.where(df["f_pair"] > 0, np.log10(df["f_pair"]), np.nan)
df["source_label"] = np.where(df["f_pair"] > 0, "SC", "UC")

sc = df[df["source_label"].eq("SC")].copy()
print("val rows:", len(df))
print("val SC rows:", len(sc))
print("val UC rows:", int((df["source_label"] == "UC").sum()))

# Build lookup for text surface by class pair from val itself first.
pair_to_text = {}
for _, r in df.iterrows():
    key = (int(r["verb_class"]), int(r["noun_class"]))
    pair_to_text.setdefault(key, str(r["narration"]))

# Fallback class-name lookup.
verb_name = df.groupby("verb_class")["verb"].first().to_dict()
noun_name = df.groupby("noun_class")["noun"].first().to_dict()

def text_for_pair(v, n):
    key = (int(v), int(n))
    if key in pair_to_text:
        return pair_to_text[key]
    return f"{verb_name.get(int(v), str(v))} {noun_name.get(int(n), str(n))}"

rows = []
pairs = ledger.copy()

for i, r in sc.reset_index(drop=True).iterrows():
    v = int(r["verb_class"])
    n = int(r["noun_class"])
    fpos = float(r["f_pair"])
    logpos = float(r["log10_f_pair"])

    noun_cand = pairs[(pairs["verb_class"] == v) & (pairs["noun_class"] != n)].copy()
    verb_cand = pairs[(pairs["noun_class"] == n) & (pairs["verb_class"] != v)].copy()

    if noun_cand.empty or verb_cand.empty:
        continue

    noun_cand["gap"] = (np.log10(noun_cand["f_pair"].clip(lower=1e-12)) - logpos).abs()
    verb_cand["gap"] = (np.log10(verb_cand["f_pair"].clip(lower=1e-12)) - logpos).abs()

    nr = noun_cand.sort_values(["gap", "f_pair"], ascending=[True, False]).iloc[0]
    vr = verb_cand.sort_values(["gap", "f_pair"], ascending=[True, False]).iloc[0]

    probe_id = f"large_val_sc_{i:05d}"
    rows.append({
        "probe_id": probe_id,
        "row_id": int(r.name) if "row_id" not in r else int(r["row_id"]),
        "narration_id": str(r["narration_id"]),
        "query_id": str(r["narration_id"]),
        "source_label": "SC",
        "positive_verb_id": v,
        "positive_noun_id": n,
        "positive_verb": str(r["verb"]),
        "positive_noun": str(r["noun"]),
        "positive_pair_id": f"{v}-{n}",
        "positive_text": str(r["narration"]),
        "noun_negative_text": text_for_pair(int(nr["verb_class"]), int(nr["noun_class"])),
        "noun_negative_noun_id": int(nr["noun_class"]),
        "noun_negative_pair_id": f"{int(nr['verb_class'])}-{int(nr['noun_class'])}",
        "noun_negative_labels_present": "SC" if float(nr["f_pair"]) > 0 else "UC",
        "verb_negative_text": text_for_pair(int(vr["verb_class"]), int(vr["noun_class"])),
        "verb_negative_verb_id": int(vr["verb_class"]),
        "verb_negative_pair_id": f"{int(vr['verb_class'])}-{int(vr['noun_class'])}",
        "verb_negative_labels_present": "SC" if float(vr["f_pair"]) > 0 else "UC",
        "f_positive_pair": fpos,
        "f_noun_negative_pair": float(nr["f_pair"]),
        "f_verb_negative_pair": float(vr["f_pair"]),
        "log10_f_pair": logpos,
        "log10_1p_f_pair": float(np.log10(1.0 + fpos)),
        "delta_pair_noun": float(np.log10(max(float(nr["f_pair"]), 1e-12)) - logpos),
        "delta_pair_verb": float(np.log10(max(float(vr["f_pair"]), 1e-12)) - logpos),
        "generation_mode": "large_val_sc_same_axis_exposure_matched_v1",
        "selection_seed": 2026,
    })

out = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("wrote:", OUT)
print("large val SC probe shape:", out.shape)
print("narration_id head:", out["narration_id"].head().tolist())
print("delta_pair_noun std:", out["delta_pair_noun"].std())
print("delta_pair_verb std:", out["delta_pair_verb"].std())
print("f_positive_pair min/p50/max:", out["f_positive_pair"].min(), out["f_positive_pair"].median(), out["f_positive_pair"].max())
