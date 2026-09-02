#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/01_build_exposure_ledger.sh configs/exposure.yaml"
  echo ""
  echo "Optional overrides:"
  echo "  LIMIT=100 bash scripts/01_build_exposure_ledger.sh configs/exposure.yaml"
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

EGOCLIP_CSV="$(read_yaml data.egoclip_csv)"
VERB_MAP_CSV="$(read_yaml data.verb_map_csv)"
NOUN_MAP_CSV="$(read_yaml data.noun_map_csv)"
QUERY_CSV="$(read_yaml data.query_csv)"

OUTPUT_ROOT="$(read_yaml output.root)"
RUN_NAME="$(read_yaml ledger.run_name)"
CONFIG_LIMIT="$(read_yaml ledger.limit)"
CHUNKSIZE="$(read_yaml ledger.chunksize)"
BATCH_SIZE="$(read_yaml ledger.batch_size)"
PART_SIZE="$(read_yaml ledger.part_size)"

LIMIT="${LIMIT:-$CONFIG_LIMIT}"

LEDGER_DIR="${OUTPUT_ROOT}/ledger"
QUERY_DIR="${OUTPUT_ROOT}/query_exposure"

mkdir -p "${LEDGER_DIR}" "${QUERY_DIR}"

python -m src.exposure.build_ledger \
  --egoclip-csv "${EGOCLIP_CSV}" \
  --verb-map-csv "${VERB_MAP_CSV}" \
  --noun-map-csv "${NOUN_MAP_CSV}" \
  --output-dir "${LEDGER_DIR}" \
  --run-name "${RUN_NAME}" \
  --limit "${LIMIT}" \
  --chunksize "${CHUNKSIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --part-size "${PART_SIZE}"

python -m src.exposure.attach_exposure \
  --query-csv "${QUERY_CSV}" \
  --pair-csv "${LEDGER_DIR}/egoclip_ek100_pair_freq_${RUN_NAME}.csv" \
  --pair-by-path-csv "${LEDGER_DIR}/egoclip_ek100_pair_freq_by_path_${RUN_NAME}.csv" \
  --marginal-csv "${LEDGER_DIR}/egoclip_ek100_marginal_only_${RUN_NAME}.csv" \
  --verb-csv "${LEDGER_DIR}/egoclip_ek100_seen_verbs_${RUN_NAME}.csv" \
  --noun-csv "${LEDGER_DIR}/egoclip_ek100_seen_nouns_${RUN_NAME}.csv" \
  --output-dir "${QUERY_DIR}"

VALIDATE_ARGS=(
  --query-exposure-csv "${QUERY_DIR}/ek100_mir_query_exposure_with_provenance.csv"
)

if [ "${LIMIT}" = "0" ]; then
  VALIDATE_ARGS+=(
    --expected-total "$(read_yaml validation.expected_total)"
    --expected-sc "$(read_yaml validation.expected_sc)"
    --expected-uc "$(read_yaml validation.expected_uc)"
    --expected-ua "$(read_yaml validation.expected_ua)"
  )
fi

python -m src.exposure.validate_exposure "${VALIDATE_ARGS[@]}"

echo "Ego-ExBind exposure pipeline complete."
