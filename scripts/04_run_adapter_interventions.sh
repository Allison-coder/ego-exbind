#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/04_run_adapter_interventions.sh

Validates the frozen paper-facing adapter intervention summaries:
  Table 4.3: hard negatives
  Table 4.4: anchor/protect over three seeds
  Table 4.5: exposure decorrelation over three seeds
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import csv

root = Path("results/tables")
checks = {
    "Table 4.3": root / "table_4_3.csv",
    "Table 4.4": root / "table_4_4.csv",
    "Table 4.5": root / "table_4_5.csv",
}

failed = 0


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


for name, path in checks.items():
    if not path.exists():
        print(f"{name}: missing {path} FAIL")
        failed += 1
        continue
    rows = read_rows(path)
    if not rows:
        print(f"{name}: empty {path} FAIL")
        failed += 1
    else:
        print(f"{name}: {path} PASS")

table43 = read_rows(checks["Table 4.3"])
table44 = read_rows(checks["Table 4.4"])
table45 = read_rows(checks["Table 4.5"])

expected43 = {
    "Domain-adapted baseline": (31.99, 46.14),
    "Verb-HN (lambda = 0.25), best": (31.99, 46.16),
    "Verb-HN (lambda = 0.25), final": (32.89, 46.03),
    "Verb-HN (lambda = 0.50), best": (31.99, 46.14),
    "Verb-HN (lambda = 0.50), final": (33.33, 45.18),
}

for model, (map_avg, ndcg_avg) in expected43.items():
    row = [r for r in table43 if r["model"] == model]
    ok = (
        len(row) == 1
        and round(float(row[0]["mAP_AVG"]), 2) == map_avg
        and round(float(row[0]["nDCG_AVG"]), 2) == ndcg_avg
    )
    print(f"Table 4.3 {model}: {'PASS' if ok else 'FAIL'}")
    failed += int(not ok)

expected44 = {
    "mAP_AVG": (32.02, 0.18, 2),
    "nDCG_AVG": (46.07, 0.20, 2),
    "NounMgn": (0.055, 0.001, 3),
    "VerbMgn": (0.077, 0.002, 3),
    "Noun_rho": (0.009, 0.008, 3),
    "Verb_rho": (-0.086, 0.014, 3),
}

for metric, (mean, std, digits) in expected44.items():
    row = [r for r in table44 if r["metric"] == metric]
    ok = (
        len(row) == 1
        and round(float(row[0]["mean"]), digits) == round(mean, digits)
        and round(float(row[0]["std"]), digits) == round(std, digits)
    )
    print(f"Table 4.4 {metric}: {'PASS' if ok else 'FAIL'}")
    failed += int(not ok)

expected45 = {
    ("dec0", "mAP_AVG"): (32.02, 0.18, 2),
    ("dec0", "nDCG_AVG"): (46.07, 0.20, 2),
    ("dec0", "SC_noun_margin"): (0.042, 0.004, 3),
    ("dec0", "UC_noun_margin"): (0.102, 0.007, 3),
    ("dec0", "C3_SC_spearman"): (0.199, 0.046, 3),
    ("dec1", "mAP_AVG"): (32.18, 0.07, 2),
    ("dec1", "nDCG_AVG"): (45.87, 0.33, 2),
    ("dec1", "SC_noun_margin"): (0.043, 0.002, 3),
    ("dec1", "UC_noun_margin"): (0.104, 0.006, 3),
    ("dec1", "C3_SC_spearman"): (0.245, 0.047, 3),
    ("dec50", "mAP_AVG"): (31.90, 0.30, 2),
    ("dec50", "nDCG_AVG"): (45.51, 0.29, 2),
    ("dec50", "SC_noun_margin"): (0.040, 0.002, 3),
    ("dec50", "UC_noun_margin"): (0.091, 0.015, 3),
    ("dec50", "C3_SC_spearman"): (0.235, 0.044, 3),
    ("dec100", "mAP_AVG"): (31.92, 0.18, 2),
    ("dec100", "nDCG_AVG"): (45.76, 0.29, 2),
    ("dec100", "SC_noun_margin"): (0.042, 0.002, 3),
    ("dec100", "UC_noun_margin"): (0.101, 0.006, 3),
    ("dec100", "C3_SC_spearman"): (0.231, 0.107, 3),
}

for (config, metric), (mean, std, digits) in expected45.items():
    row = [r for r in table45 if r["config"] == config and r["metric"] == metric]
    ok = (
        len(row) == 1
        and round(float(row[0]["mean"]), digits) == round(mean, digits)
        and round(float(row[0]["std"]), digits) == round(std, digits)
    )
    print(f"Table 4.5 {config} {metric}: {'PASS' if ok else 'FAIL'}")
    failed += int(not ok)

print(f"SUMMARY failed={failed}")
raise SystemExit(failed)
PY

echo "Ego-ExBind adapter intervention evaluation complete."
