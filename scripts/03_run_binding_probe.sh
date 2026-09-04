#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/03_run_binding_probe.sh \
    --large-probe <path_to_large_probe_csv> \
    --eval-cache <path_to_eval_cache> \
    --text-embeds <path_to_text_embeds> \
    --checkpoint <path_to_checkpoint> \
    --meta-dir <path_to_ek100_retrieval_annotations> \
    --out-dir outputs/binding \
    [--device cpu]

Runs the Ego-ExBind verb--noun binding diagnostic.
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

"${PYTHON_BIN}" -m src.binding.evaluate_binding "$@"
