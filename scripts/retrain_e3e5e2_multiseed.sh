#!/bin/bash
# Retrain E3/E5/E2 with at least 5 seeds each (offline only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-1101,2202,3303,4404,5505}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_full.csv}"
OOS_DATA_PATH="${OOS_DATA_PATH:-${PROJECT_ROOT}/crypto_trader/data_moe_20200101_20260216_oos20.csv}"
SYMBOL="${SYMBOL:-ETH/USDT:USDT}"
TIMESTEPS_OVERRIDE="${TIMESTEPS_OVERRIDE:-80000}"
MAX_DD="${MAX_DD:-0.30}"

BASE_MANIFEST="${PROJECT_ROOT}/crypto_trader/configs/moe_experts.yaml"
CAND_ROOT="${PROJECT_ROOT}/checkpoints/moe/candidate_fix_e3e5e2/${RUN_ID}"
ANALYSIS_ROOT="${PROJECT_ROOT}/runs/analysis/retrain_e3e5e2/${RUN_ID}"
TMP_MANIFEST_DIR="${ANALYSIS_ROOT}/tmp_manifests"
mkdir -p "${CAND_ROOT}" "${ANALYSIS_ROOT}" "${TMP_MANIFEST_DIR}"

if [ ! -f "${BASE_MANIFEST}" ]; then
  echo "❌ Missing base manifest: ${BASE_MANIFEST}"
  exit 1
fi
if [ ! -f "${TRAIN_DATA_PATH}" ]; then
  echo "❌ Missing train data: ${TRAIN_DATA_PATH}"
  exit 1
fi
if [ ! -f "${OOS_DATA_PATH}" ]; then
  echo "❌ Missing OOS data: ${OOS_DATA_PATH}"
  exit 1
fi

echo "=========================================="
echo " Retrain E3/E5/E2 (Multi-seed, Offline)"
echo "=========================================="
echo "RUN_ID: ${RUN_ID}"
echo "SEEDS: ${SEEDS}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "OOS_DATA_PATH: ${OOS_DATA_PATH}"
echo "TIMESTEPS_OVERRIDE: ${TIMESTEPS_OVERRIDE}"
echo "CAND_ROOT: ${CAND_ROOT}"
echo "ANALYSIS_ROOT: ${ANALYSIS_ROOT}"
echo ""

save_manifest() {
  local expert_id="$1"
  local seed="$2"
  local out_manifest="$3"
  "${PYTHON_BIN}" - <<PY
import yaml
from pathlib import Path

base = Path("${BASE_MANIFEST}")
outp = Path("${out_manifest}")
obj = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
experts = obj.get("experts", [])
target = None
for e in experts:
    if e.get("expert_id") == "${expert_id}":
        target = dict(e)
        break
if target is None:
    raise SystemExit(f"expert not found: ${expert_id}")
target["seed"] = int("${seed}")
out = {"experts": [target]}
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(str(outp))
PY
}

run_one() {
  local expert_id="$1"
  local seed="$2"
  local manifest_path="${TMP_MANIFEST_DIR}/${expert_id}_seed${seed}.yaml"
  local out_root="${CAND_ROOT}/${expert_id}/seed_${seed}"
  local eval_json="${ANALYSIS_ROOT}/${expert_id}_seed${seed}_oos.json"

  save_manifest "${expert_id}" "${seed}" "${manifest_path}" >/dev/null

  echo "[train] ${expert_id} seed=${seed}"
  "${PYTHON_BIN}" -m crypto_trader.train_moe_stage1 \
    --manifest "${manifest_path}" \
    --output-root "${out_root}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --symbol "${SYMBOL}" \
    --timesteps-override "${TIMESTEPS_OVERRIDE}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/crypto_trader/scripts/eval_expert_regime_oos.py" \
    --manifest "${manifest_path}" \
    --stage1-root "${out_root}" \
    --data-path "${OOS_DATA_PATH}" \
    --output-json "${eval_json}" \
    --expert-ids "${expert_id}" \
    --min-return-after-cost 0.0 \
    --max-dd "${MAX_DD}" \
    --min-win-rate 0.0 \
    --symbol "${SYMBOL}" >/dev/null
}

for expert_id in E3_PPO_range_calmar E5_PPO_lowvol_carry E2_PPO_bear_drawdown; do
  IFS=',' read -r -a seed_arr <<< "${SEEDS}"
  for s in "${seed_arr[@]}"; do
    run_one "${expert_id}" "${s}"
  done
done

# Build leaderboard by expert.
SUMMARY_JSON="${ANALYSIS_ROOT}/leaderboard.json"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path("${ANALYSIS_ROOT}")
rows = []
for p in sorted(root.glob("*_seed*_oos.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not obj.get("results"):
        continue
    r = obj["results"][0]
    stem = p.stem
    # e.g. E3_xxx_seed1101_oos
    seed = stem.split("seed")[-1].split("_")[0]
    rows.append({
        "file": str(p),
        "expert_id": r["expert_id"],
        "seed": int(seed),
        "pass": bool(r["pass"]),
        "return_after_cost": float(r["return_after_cost"]),
        "max_dd": float(r["max_dd"]),
        "win_rate": float(r["win_rate"]),
        "avg_turnover": float(r["avg_turnover"]),
        "cost_drag_return": float(r["cost_drag_return"]),
        "stage1_root": str(Path("${CAND_ROOT}") / r["expert_id"] / f"seed_{seed}"),
    })

leaders = {}
for eid in sorted({x["expert_id"] for x in rows}):
    cand = [x for x in rows if x["expert_id"] == eid]
    pass_cand = [x for x in cand if x["pass"]]
    if pass_cand:
        best = sorted(pass_cand, key=lambda x: (x["return_after_cost"], -x["max_dd"]), reverse=True)[0]
    else:
        best = sorted(cand, key=lambda x: (x["return_after_cost"], -x["max_dd"]), reverse=True)[0]
    leaders[eid] = best

out = {
    "run_id": "${RUN_ID}",
    "seeds": [int(x) for x in "${SEEDS}".split(",") if x.strip()],
    "timesteps_override": int("${TIMESTEPS_OVERRIDE}"),
    "train_data_path": "${TRAIN_DATA_PATH}",
    "oos_data_path": "${OOS_DATA_PATH}",
    "max_dd_threshold": float("${MAX_DD}"),
    "rows": rows,
    "leaders": leaders,
}
Path("${SUMMARY_JSON}").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

echo ""
echo "✅ Multi-seed retrain finished."
echo "Leaderboard: ${SUMMARY_JSON}"
echo "⚠️ Offline candidate artifacts only; live/stable not touched."
