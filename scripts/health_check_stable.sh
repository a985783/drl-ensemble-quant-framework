#!/bin/bash
# One-shot health check for the locked stable MoE runtime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

FAIL=0

check_path() {
    local path="$1"
    local kind="$2"
    if [ "${kind}" = "file" ] && [ ! -f "${path}" ]; then
        echo "[FAIL] Missing file: ${path}"
        FAIL=1
        return
    fi
    if [ "${kind}" = "dir" ] && [ ! -d "${path}" ]; then
        echo "[FAIL] Missing directory: ${path}"
        FAIL=1
        return
    fi
    echo "[OK] ${kind}: ${path}"
}

echo "== Stable MoE Health Check =="
echo "Project: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo ""

check_path "${PROJECT_ROOT}/stable_model_registry.json" file
check_path "${PROJECT_ROOT}/moe_model_registry.json" file
check_path "${PROJECT_ROOT}/crypto_trader/configs/moe_experts.yaml" file
check_path "${PROJECT_ROOT}/checkpoints/moe/stable/experts" dir
check_path "${PROJECT_ROOT}/checkpoints/moe/stable/gate" dir

if ! "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path(".").resolve()
stable = json.loads((root / "stable_model_registry.json").read_text(encoding="utf-8"))
moe = json.loads((root / "moe_model_registry.json").read_text(encoding="utf-8"))

s_run = stable.get("stable_run_id")
m_run = moe.get("stable_run_id")
s_path = stable.get("stable_model_path")
gate_t = moe.get("stable_gate_temperature")
ret = (moe.get("oos20_metrics") or {}).get("total_return")

if s_run != m_run:
    raise SystemExit(f"[FAIL] run_id mismatch: stable={s_run}, moe={m_run}")
if s_path != "checkpoints/moe/stable":
    raise SystemExit(f"[FAIL] unexpected stable_model_path: {s_path}")
if gate_t is None:
    raise SystemExit("[FAIL] missing stable_gate_temperature in moe_model_registry.json")

print(f"[OK] stable_run_id: {s_run}")
print(f"[OK] stable_model_path: {s_path}")
print(f"[OK] stable_gate_temperature: {gate_t}")
print(f"[OK] locked OOS20 total_return: {ret}")
PY
then
    FAIL=1
fi

LEGACY_MATCH="$(rg -n '\$HOME/强化学习|\.crypto_trader_link|/usr/bin/python3' scripts -S --glob '!scripts/health_check_stable.sh' || true)"
if [ -n "${LEGACY_MATCH}" ]; then
    echo "[FAIL] Legacy hardcoded paths still exist:"
    echo "${LEGACY_MATCH}"
    FAIL=1
else
    echo "[OK] No legacy hardcoded runtime paths in scripts/"
fi

if [ "${FAIL}" -ne 0 ]; then
    echo ""
    echo "Stable health check FAILED."
    exit 1
fi

echo ""
echo "Stable health check PASSED."
