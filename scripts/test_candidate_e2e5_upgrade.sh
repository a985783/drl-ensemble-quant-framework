#!/bin/bash
# Candidate pipeline: keep 8 experts, upgrade E2/E5 only, then test gate.
# Fully isolated from live/stable pointers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SYMBOL="${SYMBOL:-ETH/USDT:USDT}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/crypto_trader/configs/moe_experts.yaml}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_full.csv}"
BACKTEST_DATA_PATH="${BACKTEST_DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_oos20.csv}"

GATE_TRAIN_TEMP="${GATE_TRAIN_TEMP:-1.6}"
GATE_TIMESTEPS="${GATE_TIMESTEPS:-80000}"
GATE_EVAL_TEMP="${GATE_EVAL_TEMP:-0.68}"

STABLE_EXPERTS="${PROJECT_ROOT}/checkpoints/moe/stable/experts"
STABLE_GATE="${PROJECT_ROOT}/checkpoints/moe/stable/gate"

E2_SRC="${E2_SRC:-${PROJECT_ROOT}/checkpoints/moe/candidate_fix_e3e5e2/20260302_200338/E2_PPO_bear_drawdown/seed_1101/E2_PPO_bear_drawdown}"
E5_SRC="${E5_SRC:-${PROJECT_ROOT}/checkpoints/moe/candidate_fix_e3e5e2/20260302_200338/E5_PPO_lowvol_carry/seed_5505/E5_PPO_lowvol_carry}"

CAND_ROOT="${PROJECT_ROOT}/checkpoints/moe/candidate_e2e5"
CAND_EXPERTS_ROOT="${CAND_ROOT}/experts_${RUN_ID}"
CAND_GATE_ROOT="${CAND_ROOT}/gate_${RUN_ID}"
ANALYSIS_DIR="${PROJECT_ROOT}/runs/analysis/candidate_e2e5/${RUN_ID}"
mkdir -p "${CAND_ROOT}" "${CAND_EXPERTS_ROOT}" "${ANALYSIS_DIR}"

echo "=========================================="
echo " Candidate 8-Expert Test (E2/E5 Upgraded)"
echo "=========================================="
echo "RUN_ID: ${RUN_ID}"
echo "MANIFEST: ${MANIFEST}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "BACKTEST_DATA_PATH: ${BACKTEST_DATA_PATH}"
echo "E2_SRC: ${E2_SRC}"
echo "E5_SRC: ${E5_SRC}"
echo "GATE_TRAIN_TEMP: ${GATE_TRAIN_TEMP}"
echo "GATE_TIMESTEPS: ${GATE_TIMESTEPS}"
echo "GATE_EVAL_TEMP: ${GATE_EVAL_TEMP}"
echo ""

for p in "${MANIFEST}" "${TRAIN_DATA_PATH}" "${BACKTEST_DATA_PATH}" "${E2_SRC}/model.zip" "${E5_SRC}/model.zip"; do
  if [ ! -f "${p}" ]; then
    echo "❌ Missing required file: ${p}"
    exit 1
  fi
done

# Build mixed stage1 root with symlinks: stable for all, replace E2/E5 with upgraded checkpoints.
for eid in E1_PPO_trend_return E2_PPO_bear_drawdown E3_PPO_range_calmar E4_PPO_highvol_risk E5_PPO_lowvol_carry E6_SAC_tail_hedge E7_SAC_fast_adapt E8_A2C_regime_switch; do
  ln -s "${STABLE_EXPERTS}/${eid}" "${CAND_EXPERTS_ROOT}/${eid}"
done
rm "${CAND_EXPERTS_ROOT}/E2_PPO_bear_drawdown"
rm "${CAND_EXPERTS_ROOT}/E5_PPO_lowvol_carry"
ln -s "${E2_SRC}" "${CAND_EXPERTS_ROOT}/E2_PPO_bear_drawdown"
ln -s "${E5_SRC}" "${CAND_EXPERTS_ROOT}/E5_PPO_lowvol_carry"

echo "[1/4] Train candidate gate with mixed experts -> ${CAND_GATE_ROOT}"
"${PYTHON_BIN}" -m crypto_trader.train_moe_stage2_gate \
  --manifest "${MANIFEST}" \
  --stage1-root "${CAND_EXPERTS_ROOT}" \
  --output-dir "${CAND_GATE_ROOT}" \
  --total-timesteps "${GATE_TIMESTEPS}" \
  --gate-temperature "${GATE_TRAIN_TEMP}" \
  --train-data-path "${TRAIN_DATA_PATH}" \
  --symbol "${SYMBOL}"

BASELINE_JSON="${ANALYSIS_DIR}/baseline_8exp.json"
CANDIDATE_JSON="${ANALYSIS_DIR}/candidate_e2e5.json"
SUMMARY_JSON="${ANALYSIS_DIR}/summary.json"
BASELINE_PLOT="${ANALYSIS_DIR}/baseline_8exp_curve.png"
CANDIDATE_PLOT="${ANALYSIS_DIR}/candidate_e2e5_curve.png"

echo "[2/4] Backtest baseline stable 8-expert"
M_PATH="${MANIFEST}" \
S1_ROOT="${STABLE_EXPERTS}" \
S2_ROOT="${STABLE_GATE}" \
DATA_PATH_IN="${BACKTEST_DATA_PATH}" \
GATE_TEMP_IN="${GATE_EVAL_TEMP}" \
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

echo "[3/4] Backtest candidate mixed E2/E5 8-expert"
M_PATH="${MANIFEST}" \
S1_ROOT="${CAND_EXPERTS_ROOT}" \
S2_ROOT="${CAND_GATE_ROOT}" \
DATA_PATH_IN="${BACKTEST_DATA_PATH}" \
GATE_TEMP_IN="${GATE_EVAL_TEMP}" \
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

echo "[4/4] Build summary"
BASELINE_JSON_IN="${BASELINE_JSON}" \
CANDIDATE_JSON_IN="${CANDIDATE_JSON}" \
RUN_ID_IN="${RUN_ID}" \
MANIFEST_IN="${MANIFEST}" \
CAND_EXPERTS_ROOT_IN="${CAND_EXPERTS_ROOT}" \
CAND_GATE_ROOT_IN="${CAND_GATE_ROOT}" \
TRAIN_DATA_PATH_IN="${TRAIN_DATA_PATH}" \
BACKTEST_DATA_PATH_IN="${BACKTEST_DATA_PATH}" \
GATE_TRAIN_TEMP_IN="${GATE_TRAIN_TEMP}" \
GATE_EVAL_TEMP_IN="${GATE_EVAL_TEMP}" \
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
    "manifest": os.environ["MANIFEST_IN"],
    "candidate_stage1_root": os.environ["CAND_EXPERTS_ROOT_IN"],
    "candidate_stage2_root": os.environ["CAND_GATE_ROOT_IN"],
    "train_data_path": os.environ["TRAIN_DATA_PATH_IN"],
    "backtest_data_path": os.environ["BACKTEST_DATA_PATH_IN"],
    "gate_train_temperature": float(os.environ["GATE_TRAIN_TEMP_IN"]),
    "gate_eval_temperature": float(os.environ["GATE_EVAL_TEMP_IN"]),
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

echo ""
echo "✅ Candidate E2/E5 test completed."
echo "Summary: ${SUMMARY_JSON}"
echo "⚠️ Live/stable pointers were not modified."
