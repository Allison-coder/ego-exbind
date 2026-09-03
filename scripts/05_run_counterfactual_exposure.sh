#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
EGOVLPV2_ROOT=""
EVAL_CACHE=""
META_DIR=""
V2T_EXPOSURE_CSV=""
CHECKPOINT_ROOT=""
OUTPUT_DIR=""
DEVICE="cpu"
BINDING_SUMMARY_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --egovlpv2-root) EGOVLPV2_ROOT="$2"; shift 2 ;;
    --eval-cache) EVAL_CACHE="$2"; shift 2 ;;
    --meta-dir) META_DIR="$2"; shift 2 ;;
    --v2t-exposure-csv) V2T_EXPOSURE_CSV="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --binding-summary-csv) BINDING_SUMMARY_CSV="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for v in CONFIG EGOVLPV2_ROOT EVAL_CACHE META_DIR V2T_EXPOSURE_CSV CHECKPOINT_ROOT OUTPUT_DIR; do
  if [[ -z "${!v}" ]]; then
    echo "Missing required argument: $v" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"

SEEDS=$(python - "$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"seeds:\s*\[([^\]]+)\]", text)
if not m:
    raise SystemExit("Could not parse experiment.seeds")
print(" ".join(x.strip() for x in m.group(1).split(",")))
PY
)

MODES=$(python - "$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"modes:\s*\n((?:\s*-\s*\w+\s*\n?)+)", text)
if not m:
    raise SystemExit("Could not parse experiment.modes")
print(" ".join(re.findall(r"-\s*(\w+)", m.group(1))))
PY
)

for SEED in $SEEDS; do
  CKPT="$CHECKPOINT_ROOT/m7_exposure_score_bias_centered_1000_seed${SEED}/m7_score_bias_best.pt"

  EXPOSURE_VECTORS=$(python - "$CKPT" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(ckpt["args"]["exposure_vectors"])
PY
)

  for MODE in $MODES; do
    echo "===== RUN seed=${SEED} mode=${MODE} ====="
    python -m src.interventions.exposure_reweighting \
      --egovlpv2-root "$EGOVLPV2_ROOT" \
      --eval-cache "$EVAL_CACHE" \
      --meta-dir "$META_DIR" \
      --v2t-exposure-csv "$V2T_EXPOSURE_CSV" \
      --checkpoint "$CKPT" \
      --exposure-vectors "$EXPOSURE_VECTORS" \
      --mode "$MODE" \
      --seed "$SEED" \
      --output-dir "$OUTPUT_DIR/seed${SEED}/${MODE}" \
      --device "$DEVICE"
  done
done

python - "$OUTPUT_DIR" "$BINDING_SUMMARY_CSV" <<'PY'
from pathlib import Path
import json
import sys
import pandas as pd

base = Path(sys.argv[1])
binding_csv = sys.argv[2]

rows = []
for d in sorted(base.glob("seed*/original")) + sorted(base.glob("seed*/zeroed")) + sorted(base.glob("seed*/shuffled")):
    seed = int(d.parent.name.replace("seed", ""))
    mode = d.name
    js = json.loads((d / "counterfactual_summary.json").read_text())
    g = pd.read_csv(d / "v2t_sc_uc_ap.csv")

    m = js["metrics_m7_centered_bias"]
    sc = float(g.loc[g["group"].eq("SC"), "V2T_AP_m7"].iloc[0])
    uc = float(g.loc[g["group"].eq("UC"), "V2T_AP_m7"].iloc[0])

    rows.append({
        "seed": seed,
        "mode": mode,
        "mAP_V2T": m["mAP_V2T"],
        "mAP_T2V": m["mAP_T2V"],
        "nDCG_V2T": m["nDCG_V2T"],
        "nDCG_T2V": m["nDCG_T2V"],
        "SC_V2T_AP": sc,
        "UC_V2T_AP": uc,
    })

df = pd.DataFrame(rows)
mean = df.groupby("mode").mean().reset_index()

if binding_csv:
    b = pd.read_csv(binding_csv)
    mean = mean.merge(b, on="mode", how="left")

order = ["original", "zeroed", "shuffled"]
mean["mode"] = pd.Categorical(mean["mode"], order, ordered=True)
mean = mean.sort_values("mode")

out = base / "table_4_6_counterfactual_exposure.csv"
mean.to_csv(out, index=False)

expected = {
    "original": {"mAP_V2T": 33.70, "mAP_T2V": 27.73, "nDCG_V2T": 44.22, "nDCG_T2V": 40.31, "SC_V2T_AP": 34.97, "UC_V2T_AP": 24.51},
    "zeroed": {"mAP_V2T": 32.91, "mAP_T2V": 27.73, "nDCG_V2T": 43.04, "nDCG_T2V": 40.31, "SC_V2T_AP": 33.94, "UC_V2T_AP": 26.14},
    "shuffled": {"mAP_V2T": 32.56, "mAP_T2V": 27.73, "nDCG_V2T": 42.80, "nDCG_T2V": 40.31, "SC_V2T_AP": 33.56, "UC_V2T_AP": 26.15},
}

failed = 0
print("\n=== Table 4.6 validation ===")
for _, r in mean.iterrows():
    mode = str(r["mode"])
    for k, exp in expected[mode].items():
        got = round(float(r[k]), 2)
        ok = got == exp
        print(f"{mode} {k}: got={got:.2f} expected={exp:.2f} {'PASS' if ok else 'FAIL'}")
        failed += int(not ok)

for k in ["mAP_T2V", "nDCG_T2V"]:
    vals = [round(float(mean.loc[mean["mode"].eq(m), k].iloc[0]), 2) for m in order]
    ok = len(set(vals)) == 1
    print(f"{k} invariant: {'PASS' if ok else 'FAIL'} {vals}")
    failed += int(not ok)

if binding_csv:
    bind_expected = {
        "original": {"noun_margin": 0.058, "verb_margin": 0.064, "rho_n": 0.043, "rho_v": -0.053},
        "zeroed": {"noun_margin": 0.058, "verb_margin": 0.065, "rho_n": 0.042, "rho_v": -0.053},
        "shuffled": {"noun_margin": 0.058, "verb_margin": 0.064, "rho_n": 0.042, "rho_v": -0.052},
    }
    for _, r in mean.iterrows():
        mode = str(r["mode"])
        for k, exp in bind_expected[mode].items():
            got = round(float(r[k]), 3)
            ok = got == exp
            print(f"{mode} {k}: got={got:.3f} expected={exp:.3f} {'PASS' if ok else 'FAIL'}")
            failed += int(not ok)

print("\nOUT_TABLE:", out)
print(f"SUMMARY failed={failed}")
raise SystemExit(failed)
PY
