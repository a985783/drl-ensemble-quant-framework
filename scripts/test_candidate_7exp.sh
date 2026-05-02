#!/bin/bash
# Candidate testing pipeline for 7-expert MoE (E1 removed).
# This script is fully isolated from live/stable artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-gate_only}"   # gate_only | full_retrain

MANIFEST_7="${MANIFEST_7:-${PROJECT_ROOT}/crypto_trader/configs/moe_experts_7_no_e1.yaml}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_oos20.csv}"
SYMBOL="${SYMBOL:-ETH/USDT:USDT}"
GATE_TEMP="${GATE_TEMP:-0.68}"
GATE_TIMESTEPS="${GATE_TIMESTEPS:-60000}"
STAGE1_TIMESTEPS="${STAGE1_TIMESTEPS:-80000}"

CAND_ROOT="${PROJECT_ROOT}/checkpoints/moe/candidate_7exp"
CAND_EXPERTS_ROOT="${CAND_ROOT}/experts_${RUN_ID}"
CAND_GATE_ROOT="${CAND_ROOT}/gate_${RUN_ID}"

STABLE_MANIFEST="${PROJECT_ROOT}/crypto_trader/configs/moe_experts.yaml"
STABLE_EXPERTS="${PROJECT_ROOT}/checkpoints/moe/stable/experts"
STABLE_GATE="${PROJECT_ROOT}/checkpoints/moe/stable/gate"

ANALYSIS_DIR="${PROJECT_ROOT}/runs/analysis/candidate_7exp/${RUN_ID}"
mkdir -p "${ANALYSIS_DIR}" "${CAND_ROOT}"

echo "=========================================="
echo "  Candidate 7-Expert Test (Isolated)"
echo "=========================================="
echo "Run ID: ${RUN_ID}"
echo "Mode: ${MODE}"
echo "Manifest(7): ${MANIFEST_7}"
echo "Data: ${DATA_PATH}"
echo "Gate temperature: ${GATE_TEMP}"
echo "Analysis dir: ${ANALYSIS_DIR}"
echo ""

if [ ! -f "${MANIFEST_7}" ]; then
  echo "❌ Missing manifest: ${MANIFEST_7}"
  exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
  echo "❌ Missing data: ${DATA_PATH}"
  exit 1
fi

if [ "${MODE}" = "full_retrain" ]; then
  echo "[1/5] Stage1 full retrain (7 experts) -> ${CAND_EXPERTS_ROOT}"
  "${PYTHON_BIN}" -m crypto_trader.train_moe_stage1 \
    --manifest "${MANIFEST_7}" \
    --output-root "${CAND_EXPERTS_ROOT}" \
    --timesteps-override "${STAGE1_TIMESTEPS}" \
    --symbol "${SYMBOL}"
  STAGE1_FOR_GATE="${CAND_EXPERTS_ROOT}"
elif [ "${MODE}" = "gate_only" ]; then
  echo "[1/5] Reuse stable experts for candidate gate training (no live mutation)"
  STAGE1_FOR_GATE="${STABLE_EXPERTS}"
else
  echo "❌ Unsupported MODE=${MODE}. Use gate_only or full_retrain."
  exit 1
fi

echo "[2/5] Train candidate gate -> ${CAND_GATE_ROOT}"
"${PYTHON_BIN}" -m crypto_trader.train_moe_stage2_gate \
  --manifest "${MANIFEST_7}" \
  --stage1-root "${STAGE1_FOR_GATE}" \
  --output-dir "${CAND_GATE_ROOT}" \
  --total-timesteps "${GATE_TIMESTEPS}" \
  --gate-temperature "${GATE_TEMP}" \
  --symbol "${SYMBOL}"

BASELINE_JSON="${ANALYSIS_DIR}/baseline_8exp.json"
CANDIDATE_JSON="${ANALYSIS_DIR}/candidate_7exp.json"
BASELINE_PLOT="${ANALYSIS_DIR}/baseline_8exp_curve.png"
CANDIDATE_PLOT="${ANALYSIS_DIR}/candidate_7exp_curve.png"

echo "[3/5] Backtest baseline stable 8-expert"
M_PATH="${STABLE_MANIFEST}" \
S1_ROOT="${STABLE_EXPERTS}" \
S2_ROOT="${STABLE_GATE}" \
DATA_PATH_IN="${DATA_PATH}" \
GATE_TEMP_IN="${GATE_TEMP}" \
SYMBOL_IN="${SYMBOL}" \
PLOT_PATH_IN="${BASELINE_PLOT}" \
OUT_JSON_IN="${BASELINE_JSON}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
from crypto_trader.backtest_moe import backtest_moe

result = backtest_moe(
    manifest_path=Path(os.environ["M_PATH"]),
    stage1_root=os.environ["S1_ROOT"],
    stage2_root=os.environ["S2_ROOT"],
    data_path=os.environ["DATA_PATH_IN"],
    gate_temperature=float(os.environ["GATE_TEMP_IN"]),
    symbol=os.environ["SYMBOL_IN"],
    plot_path=os.environ["PLOT_PATH_IN"],
)
Path(os.environ["OUT_JSON_IN"]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

echo "[4/5] Backtest candidate 7-expert"
M_PATH="${MANIFEST_7}" \
S1_ROOT="${STAGE1_FOR_GATE}" \
S2_ROOT="${CAND_GATE_ROOT}" \
DATA_PATH_IN="${DATA_PATH}" \
GATE_TEMP_IN="${GATE_TEMP}" \
SYMBOL_IN="${SYMBOL}" \
PLOT_PATH_IN="${CANDIDATE_PLOT}" \
OUT_JSON_IN="${CANDIDATE_JSON}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
from crypto_trader.backtest_moe import backtest_moe

result = backtest_moe(
    manifest_path=Path(os.environ["M_PATH"]),
    stage1_root=os.environ["S1_ROOT"],
    stage2_root=os.environ["S2_ROOT"],
    data_path=os.environ["DATA_PATH_IN"],
    gate_temperature=float(os.environ["GATE_TEMP_IN"]),
    symbol=os.environ["SYMBOL_IN"],
    plot_path=os.environ["PLOT_PATH_IN"],
)
Path(os.environ["OUT_JSON_IN"]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

SUMMARY_JSON="${ANALYSIS_DIR}/summary.json"
echo "[5/5] Build comparison summary -> ${SUMMARY_JSON}"
BASELINE_JSON_IN="${BASELINE_JSON}" \
CANDIDATE_JSON_IN="${CANDIDATE_JSON}" \
RUN_ID_IN="${RUN_ID}" \
MODE_IN="${MODE}" \
MANIFEST_7_IN="${MANIFEST_7}" \
STAGE1_FOR_GATE_IN="${STAGE1_FOR_GATE}" \
CAND_GATE_ROOT_IN="${CAND_GATE_ROOT}" \
STABLE_MANIFEST_IN="${STABLE_MANIFEST}" \
STABLE_EXPERTS_IN="${STABLE_EXPERTS}" \
STABLE_GATE_IN="${STABLE_GATE}" \
DATA_PATH_IN="${DATA_PATH}" \
GATE_TEMP_IN="${GATE_TEMP}" \
SUMMARY_JSON_IN="${SUMMARY_JSON}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

with open(os.environ["BASELINE_JSON_IN"], "r", encoding="utf-8") as f:
    b = json.load(f)
with open(os.environ["CANDIDATE_JSON_IN"], "r", encoding="utf-8") as f:
    c = json.load(f)

summary = {
    "run_id": os.environ["RUN_ID_IN"],
    "mode": os.environ["MODE_IN"],
    "candidate_paths": {
        "manifest_7": os.environ["MANIFEST_7_IN"],
        "stage1_root": os.environ["STAGE1_FOR_GATE_IN"],
        "stage2_root": os.environ["CAND_GATE_ROOT_IN"],
    },
    "baseline_paths": {
        "manifest_8": os.environ["STABLE_MANIFEST_IN"],
        "stage1_root": os.environ["STABLE_EXPERTS_IN"],
        "stage2_root": os.environ["STABLE_GATE_IN"],
    },
    "data_path": os.environ["DATA_PATH_IN"],
    "gate_temperature": float(os.environ["GATE_TEMP_IN"]),
    "baseline": b,
    "candidate": c,
    "delta": {
        "total_return": float(c.get("total_return", 0.0) - b.get("total_return", 0.0)),
        "max_dd": float(c.get("max_dd", 0.0) - b.get("max_dd", 0.0)),
        "alpha": float(c.get("alpha", 0.0) - b.get("alpha", 0.0)),
        "final_net_worth": float(c.get("final_net_worth", 0.0) - b.get("final_net_worth", 0.0)),
    },
}

Path(os.environ["SUMMARY_JSON_IN"]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary["delta"], ensure_ascii=False, indent=2))
PY

# Keep candidate registry isolated from stable registry.
CAND_REGISTRY="${PROJECT_ROOT}/moe_model_registry_candidate.json"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

path = Path("${CAND_REGISTRY}")
obj = {}
if path.exists():
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        obj = {}

obj.update({
    "latest_candidate_run_id": "${RUN_ID}",
    "latest_candidate_mode": "${MODE}",
    "latest_candidate_manifest_path": "${MANIFEST_7}",
    "latest_candidate_stage1_path": "${STAGE1_FOR_GATE}",
    "latest_candidate_stage2_path": "${CAND_GATE_ROOT}",
    "latest_candidate_analysis_dir": "${ANALYSIS_DIR}",
    "latest_candidate_summary_path": "${SUMMARY_JSON}",
})
path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"candidate_registry={path}")
PY

echo ""
echo "✅ Candidate 7-expert test completed."
echo "Summary: ${SUMMARY_JSON}"
echo "Candidate registry: ${CAND_REGISTRY}"
echo ""
echo "⚠️ Live/stable pointers were not modified."
