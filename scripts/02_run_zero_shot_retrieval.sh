#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/02_run_zero_shot_retrieval.sh configs/retrieval.yaml"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="$1"

read_yaml() {
  python - "$CONFIG" "$1" <<'PY'
import sys
import yaml

config_path, key = sys.argv[1], sys.argv[2]
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

value = cfg
for part in key.split("."):
    value = value[part]

print(value)
PY
}

EGOVLPV2_ROOT="$(read_yaml data.egovlpv2_root)"
FEATURE_PT="$(read_yaml data.feature_pt)"
META_DIR="$(read_yaml data.meta_dir)"
EXPOSURE_QUERY_CSV="$(read_yaml data.exposure_query_csv)"
OUTPUT_ROOT="$(read_yaml output.root)"

METRICS_DIR="${OUTPUT_ROOT}/query_metrics"
UNITS_DIR="${OUTPUT_ROOT}/metric_units"

mkdir -p "${METRICS_DIR}" "${UNITS_DIR}"

python -m src.retrieval.compute_query_metrics \
  --egovlpv2-root "${EGOVLPV2_ROOT}" \
  --feature-pt "${FEATURE_PT}" \
  --meta-dir "${META_DIR}" \
  --exposure-query-csv "${EXPOSURE_QUERY_CSV}" \
  --output-dir "${METRICS_DIR}"

python -m src.retrieval.build_metric_units \
  --feature-pt "${FEATURE_PT}" \
  --meta-dir "${META_DIR}" \
  --exposure-query-csv "${EXPOSURE_QUERY_CSV}" \
  --v2t-metrics-csv "${METRICS_DIR}/ek100_mir_official_per_query_metrics_v2t.csv" \
  --t2v-metrics-csv "${METRICS_DIR}/ek100_mir_official_per_query_metrics_t2v.csv" \
  --output-dir "${UNITS_DIR}"

python - "$UNITS_DIR" "$(read_yaml validation.expected_v2t_units)" "$(read_yaml validation.expected_t2v_units)" <<'PY'
import sys
from pathlib import Path
import pandas as pd

units_dir = Path(sys.argv[1])
expected_v2t = int(sys.argv[2])
expected_t2v = int(sys.argv[3])

v2t = pd.read_csv(units_dir / "ek100_mir_metric_units_v2t.csv")
t2v = pd.read_csv(units_dir / "ek100_mir_metric_units_t2v.csv")

print("=== Ego-ExBind retrieval validation ===")
print("V2T units:", len(v2t))
print("T2V units:", len(t2v))

failed = int(len(v2t) != expected_v2t or len(t2v) != expected_t2v)
print(f"SUMMARY failed={failed}")
raise SystemExit(failed)
PY

echo "Ego-ExBind zero-shot retrieval pipeline complete."
