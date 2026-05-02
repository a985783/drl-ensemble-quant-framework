#!/bin/bash
# Offline retrain pipeline for prioritized experts, isolated from live/stable pointers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH="${BATCH:-all}"  # batch1 | batch2 | all
SYMBOL="${SYMBOL:-ETH/USDT:USDT}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_full.csv}"
OOS_DATA_PATH="${OOS_DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_oos20.csv}"
MAX_DD="${MAX_DD:-0.30}"

BATCH1_MANIFEST="${PROJECT_ROOT}/crypto_trader/configs/moe_experts_batch1_lowrisk.yaml"
BATCH2_MANIFEST="${PROJECT_ROOT}/crypto_trader/configs/moe_experts_batch2_lowrisk.yaml"

CAND_ROOT="${PROJECT_ROOT}/checkpoints/moe/candidate_priority/${RUN_ID}"
ANALYSIS_ROOT="${PROJECT_ROOT}/runs/analysis/expert_retrain_priority/${RUN_ID}"
mkdir -p "${CAND_ROOT}" "${ANALYSIS_ROOT}"

echo "=========================================="
echo " Offline Expert Retrain (Priority Batches)"
echo "=========================================="
echo "RUN_ID: ${RUN_ID}"
echo "BATCH: ${BATCH}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "OOS_DATA_PATH: ${OOS_DATA_PATH}"
echo "SYMBOL: ${SYMBOL}"
echo "MAX_DD: ${MAX_DD}"
echo "CAND_ROOT: ${CAND_ROOT}"
echo "ANALYSIS_ROOT: ${ANALYSIS_ROOT}"
echo ""

if [ ! -f "${TRAIN_DATA_PATH}" ]; then
  echo "❌ Missing train data: ${TRAIN_DATA_PATH}"
  exit 1
fi
if [ ! -f "${OOS_DATA_PATH}" ]; then
  echo "❌ Missing OOS data: ${OOS_DATA_PATH}"
  exit 1
fi

# Guard: ensure we don't mutate stable/live pointers.
STABLE_REG="${PROJECT_ROOT}/stable_model_registry.json"
MOE_REG="${PROJECT_ROOT}/moe_model_registry.json"

hash_file() {
  local p="$1"
  if [ -f "$p" ]; then
    shasum -a 256 "$p" | awk '{print $1}'
  else
    echo "missing"
  fi
}

STABLE_REG_HASH_BEFORE="$(hash_file "${STABLE_REG}")"
MOE_REG_HASH_BEFORE="$(hash_file "${MOE_REG}")"

run_batch() {
  local batch_name="$1"
  local manifest="$2"
  local expert_ids="$3"

  local out_root="${CAND_ROOT}/${batch_name}/experts"
  local eval_json="${ANALYSIS_ROOT}/${batch_name}_oos_eval.json"

  if [ ! -f "${manifest}" ]; then
    echo "❌ Missing manifest: ${manifest}"
    exit 1
  fi

  echo "[${batch_name}] [1/2] Retrain experts -> ${out_root}"
  "${PYTHON_BIN}" -m crypto_trader.train_moe_stage1 \
    --manifest "${manifest}" \
    --output-root "${out_root}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --symbol "${SYMBOL}"

  echo "[${batch_name}] [2/2] OOS regime evaluation -> ${eval_json}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/crypto_trader/scripts/eval_expert_regime_oos.py" \
    --manifest "${manifest}" \
    --stage1-root "${out_root}" \
    --data-path "${OOS_DATA_PATH}" \
    --output-json "${eval_json}" \
    --expert-ids "${expert_ids}" \
    --min-return-after-cost 0.0 \
    --max-dd "${MAX_DD}" \
    --min-win-rate 0.0 \
    --symbol "${SYMBOL}"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${eval_json}")
obj = json.loads(p.read_text(encoding="utf-8"))
print(f"[${batch_name}] pass={obj.get('pass_count')}/{obj.get('expert_count')}")
for r in obj.get("results", []):
    flag = "PASS" if r.get("pass") else "FAIL"
    print(f"  - {r['expert_id']}: {flag} ret={r['return_after_cost']:.4f} dd={r['max_dd']:.4f} win={r['win_rate']:.3f}")
PY
}

case "${BATCH}" in
  batch1)
    run_batch "batch1" "${BATCH1_MANIFEST}" "E5_PPO_lowvol_carry,E7_SAC_fast_adapt,E3_PPO_range_calmar"
    ;;
  batch2)
    run_batch "batch2" "${BATCH2_MANIFEST}" "E4_PPO_highvol_risk,E2_PPO_bear_drawdown"
    ;;
  all)
    run_batch "batch1" "${BATCH1_MANIFEST}" "E5_PPO_lowvol_carry,E7_SAC_fast_adapt,E3_PPO_range_calmar"
    run_batch "batch2" "${BATCH2_MANIFEST}" "E4_PPO_highvol_risk,E2_PPO_bear_drawdown"
    ;;
  *)
    echo "❌ Unsupported BATCH=${BATCH}. Use batch1 | batch2 | all"
    exit 1
    ;;
esac

STABLE_REG_HASH_AFTER="$(hash_file "${STABLE_REG}")"
MOE_REG_HASH_AFTER="$(hash_file "${MOE_REG}")"

if [ "${STABLE_REG_HASH_BEFORE}" != "${STABLE_REG_HASH_AFTER}" ]; then
  echo "❌ stable_model_registry.json changed unexpectedly."
  exit 2
fi
if [ "${MOE_REG_HASH_BEFORE}" != "${MOE_REG_HASH_AFTER}" ]; then
  echo "❌ moe_model_registry.json changed unexpectedly."
  exit 2
fi

SUMMARY_PATH="${ANALYSIS_ROOT}/run_summary.json"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
out = {
    "run_id": "${RUN_ID}",
    "batch": "${BATCH}",
    "train_data_path": "${TRAIN_DATA_PATH}",
    "oos_data_path": "${OOS_DATA_PATH}",
    "symbol": "${SYMBOL}",
    "max_dd_threshold": float("${MAX_DD}"),
    "candidate_root": "${CAND_ROOT}",
    "analysis_root": "${ANALYSIS_ROOT}",
    "batch1_manifest": "${BATCH1_MANIFEST}",
    "batch2_manifest": "${BATCH2_MANIFEST}",
}
Path("${SUMMARY_PATH}").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

echo ""
echo "✅ Offline prioritized retrain completed."
echo "Summary: ${SUMMARY_PATH}"
echo "No live/stable pointer file changed."
