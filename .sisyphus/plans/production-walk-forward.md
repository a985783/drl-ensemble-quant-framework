# 生产级 Walk-Forward 验证计划

## TL;DR

> **Quick Summary**: 修复 `resolve_execution_frame()` 的特征列不 shift 的 Bug，然后构建 5 折 anchored MoE walk-forward 框架，每折独立重训 XGBoost + 4 专家 + Gate + 温度扫描。通过标准：逐折 alpha > 0%，5 折平均 alpha ≥ 20%。
>
> **Deliverables**:
> - 修复后的 `resolve_execution_frame()`（shift 所有列）
> - 6 个 MoE walk-forward 模块（folding/data_prep/expert_trainer/gate_trainer/backtester/aggregator）
> - 5 折完整训练 + 回测结果
> - 最终裁决报告（PASS/FAIL + 逐折指标）
>
> **Estimated Effort**: Large（修复 0.5h + walk-forward 构建 4-6h + 5 折训练 3-5h）
> **Parallel Execution**: YES - 3 波
> **Critical Path**: 修复 → walk-forward 构建 → 逐折训练（顺序）→ 聚合

---

## Context

### Original Request
用户要求将 MoE 加密量化交易系统推到真正的生产级。当前已通过所有非重训验证（参数扫描、架构消融、bootstrap、子区间、21 场景审计），但缺少 anchored walk-forward——量化圈最高的门槛。

### 已知核心故障
`resolve_execution_frame()` 在 `backtest_moe.py:148-162` 只 shift OHLCV 列，不 shift 特征列。导致训练时特征比交易价格落后 2 根 bar，破坏重训管道。

### 修复方案
将 `for col in ["Open", "High", "Low", "Close", "Volume", "Funding_Rate"]` 改为 `for col in shifted.columns`——shift 所有列，使特征对齐到正确的时点。

### Walk-Forward 架构（Oracle 设计）
5 折 anchored expanding-window，每折独立执行全流程：
1. DataPreparer: 获取数据 → FeatureEngineer → XGBoost 仅 train 上训练 → 预测 Signal_Proba → next-bar shift
2. ExpertTrainer: 4 专家 × regime 切片 + 训练控制 → 独立 Stage1 训练
3. GateTrainer: 训练期内时间验证拆分 → 7 候选温度扫描 → 选最优 → 全量重训
4. FoldBacktester: next-bar 回测 → 全量指标
5. Aggregator: 5 折汇总 → 裁决

---

## Work Objectives

### Core Objective
通过 5 折 anchored walk-forward 验证，证明 MoE 策略在不同市场制度下具有持续正 Alpha（逐折 alpha > 0%，平均 ≥ 20%）。

### Concrete Deliverables
- `crypto_trader/backtest_moe.py` — 修复后的 `resolve_execution_frame()`
- `crypto_trader/walk_forward/moe_config.py` — WalkForwardMoEConfig 定义
- `crypto_trader/walk_forward/folding.py` — FoldingManager
- `crypto_trader/walk_forward/data_prep.py` — DataPreparer（XGBoost per-fold + next-bar shift）
- `crypto_trader/walk_forward/expert_trainer.py` — ExpertTrainer（4 专家 Stage1）
- `crypto_trader/walk_forward/gate_trainer.py` — GateTrainer（温度扫描 + Stage2）
- `crypto_trader/walk_forward/backtester.py` — FoldBacktester
- `crypto_trader/walk_forward/aggregator.py` — Aggregator + 裁决
- `crypto_trader/walk_forward/moe_walk_forward.py` — 主编排器
- `crypto_trader/walk_forward/results/walk_forward_moe/summary/verdict.json` — PASS/FAIL
- 所有已有测试通过（回归验证）

### Must Have
- resolve_execution_frame shift 所有列（修复 Bug）
- 5 折 walk-forward，每折 alpha > 0%
- 5 折平均 alpha ≥ 20%
- 所有训练超参数和 reward profile 在折之间冻结
- 已有测试（22 个）全部通过

### Must NOT Have
- 不修改 stable 注册表
- 不修改旧 stable checkpoint
- 不修改专家 reward profile
- 不修改训练超参数（折之间）
- 不做多资产扩展
- XGBoost 不得在测试数据上训练

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES（22 个测试）
- **Automated tests**: Tests-after
- **Framework**: pytest
- **QA Policy**: 所有任务包含 agent-executed QA scenarios

---

## Execution Strategy

```
Wave 1（并行 - 修复 + 配置 + 折叠器）:
├── Task 1: 修复 resolve_execution_frame [quick]
├── Task 2: 创建 WalkForwardMoEConfig [quick]
└── Task 3: 创建 FoldingManager [quick]

Wave 2（并行 - 核心模块）:
├── Task 4: 创建 DataPreparer [deep]
├── Task 5: 创建 ExpertTrainer [deep]
├── Task 6: 创建 GateTrainer [deep]
└── Task 7: 创建 FoldBacktester + Aggregator [deep]

Wave 3（顺序 - 运行 + 验证）:
├── Task 8: 集成主编排器 [quick]
├── Task 9: 运行 5 折 walk-forward [deep]
└── Task 10: 最终验证 + 裁决报告 [quick]
```

---

## TODOs

- [ ] 1. 修复 resolve_execution_frame——shift 所有列

  **What to do**: 修改 `crypto_trader/backtest_moe.py:148-162`，将仅 shift 6 个 OHLCV 列改为 shift `df.columns` 中所有存在的列。这确保特征列（RSI、MACD、Signal_Proba 等）与价格列同步 shift，消除 2-bar 错位。

  **Must NOT do**: 不改变函数的签名或返回值结构。不改变 `dropped_rows` 和 `execution_mode` 元数据。

  **Recommended Agent Profile**: `quick` | **Skills**: [] | **Parallel**: Wave 1，与 Tasks 2, 3 并行 | **Blocks**: Task 4 | **Blocked By**: None

  **References**: `crypto_trader/backtest_moe.py:148-162`（当前实现），`crypto_trader/tests/test_backtest_moe_overrides.py:36-52`（当前测试）

  **Acceptance Criteria**: `PYTHONPATH=. python -m pytest crypto_trader/tests/test_backtest_moe_overrides.py -v` → passed；Signal_Proba 在 shift 后的 df 中不是 NaN

  **QA Scenarios**:
  ```
  Scenario: 修复后 shift 验证
    Tool: Bash
    Steps:
      1. PYTHONPATH=. python3 -c "
from crypto_trader.backtest_moe import resolve_execution_frame; import pandas as pd
df = pd.DataFrame({'Close':[100,101,102], 'Signal_Proba':[0.5,0.6,0.7], 'RSI':[40,50,60]})
s, m = resolve_execution_frame(df, 'next_bar')
print(s); print(m)
assert s['Signal_Proba'].iloc[0] == 0.6  # shifted
assert s['RSI'].iloc[0] == 50  # shifted
assert len(s) == 2  # one row dropped
"
    Expected Result: 所有列被 shift，Signal_Proba 和 RSI 值正确前移
    Evidence: .sisyphus/evidence/task-1-fix-shift.txt
  ```
  **Commit**: YES | `fix: shift all columns in resolve_execution_frame for next-bar alignment`

- [ ] 2. 创建 WalkForwardMoEConfig

  **What to do**: 新建 `crypto_trader/walk_forward/moe_config.py`，定义 FoldSpec、ExpertSpec、GateSearchSpec、PassCriteria、WalkForwardMoEConfig dataclass。包含 5 折 anchored 扩展窗口、4 专家（E5/E2/E4/E7）的完整配置、冻结的训练超参数、温度候选列表 [0.5, 0.6, 0.68, 0.8, 1.0, 1.5, 2.0]。

  **Must NOT do**: 不修改专家 reward_profile。不修改训练超参数默认值。

  **Recommended Agent Profile**: `quick` | **Skills**: [] | **Parallel**: Wave 1 | **Blocks**: Tasks 4-8 | **Blocked By**: None

  **References**: `crypto_trader/configs/moe_experts.yaml`（专家定义），`crypto_trader/config.py`（超参数），`crypto_trader/walk_forward/train_walk_forward.py:38-53`（现有折定义）

  **Acceptance Criteria**: `python3 -c "from crypto_trader.walk_forward.moe_config import WalkForwardMoEConfig; c = WalkForwardMoEConfig(); assert len(c.folds) == 5; assert len(c.expert_ids) == 4"`

  **QA Scenarios**: 同上

  **Commit**: YES（Wave 1 组提交）

- [ ] 3. 创建 FoldingManager

  **What to do**: 新建 `crypto_trader/walk_forward/folding.py`。从 WalkForwardMoEConfig 生成 5 个 FoldSpec，每个包含 train_start/end、test_start/end、val_split_date（训练期后 20%）。核心保证：各折测试期不得包含在训练期内。

  **Must NOT do**: 不访问实际数据——纯日期计算。

  **Recommended Agent Profile**: `quick` | **Skills**: [] | **Parallel**: Wave 1 | **Blocks**: Task 4 | **Blocked By**: Task 2

  **References**: `crypto_trader/walk_forward/train_walk_forward.py:49-53`（现有 FOLDS 定义）

  **Acceptance Criteria**: `FoldingManager.build_folds()` 返回 5 个 FoldSpec；fold_1.test_start 不早于 fold_1.train_end；val_split_date 在 train_start 和 train_end 之间

  **QA Scenarios**: 同上

  **Commit**: YES（Wave 1 组提交）

---

- [ ] 4. 创建 DataPreparer（各折独立 XGBoost + next-bar）

  **What to do**: 新建 `crypto_trader/walk_forward/data_prep.py`。各折：获取训练窗口原始 OHLCV → `FeatureEngineer.add_technical_indicators()` → `SignalPredictor().train(train_df)`（仅训练数据）→ `.predict_proba()` 训练+测试 → `resolve_execution_frame(next_bar)` 对两者 shift → 保存到折检查点。

  **Must NOT do**: XGBoost 不得在测试数据上 `.fit()`；不修改 FeatureEngineer 或 SignalPredictor 的默认参数。

  **Recommended Agent Profile**: `deep` | **Skills**: [] | **Parallel**: Wave 2，与 Tasks 5, 6, 7 并行 | **Blocks**: Tasks 5, 6, 8 | **Blocked By**: Tasks 1, 3

  **References**: `crypto_trader/data_loader.py`，`crypto_trader/features.py`，`crypto_trader/models/signal_model.py`，`crypto_trader/backtest_moe.py:148-162`（修复后的 resolve_execution_frame）

  **Acceptance Criteria**: 测试折（如简短日期范围）可从头到尾运行 XGBoost 拟合 + 预测 + shift + 保存，无错误；检查点 CSV 存在且包含 Signal_Proba 列

  **QA Scenarios**:
  ```
  Scenario: DataPreparer 单折测试
    Tool: Bash
    Preconditions: 有效 OKX API 密钥或使用 2020-2021 现有缓存数据
    Steps:
      1. PYTHONPATH=. python3 -c "
from crypto_trader.walk_forward.moe_config import WalkForwardMoEConfig, FoldSpec
from crypto_trader.walk_forward.data_prep import DataPreparer
config = WalkForwardMoEConfig()
fold = config.folds[0]
prep = DataPreparer(config, fold)
train_df, test_df = prep.prepare()
assert 'Signal_Proba' in train_df.columns
assert len(train_df) > 0 and len(test_df) > 0
print(f'Train: {len(train_df)} rows, Test: {len(test_df)} rows')
"
    Expected Result: 产生带有 Signal_Proba 的训练和测试 dataframe
    Evidence: .sisyphus/evidence/task-4-data-prep.txt
  ```
  **Commit**: YES（Wave 2 组提交）

- [ ] 5. 创建 ExpertTrainer（4 专家 Stage1）

  **What to do**: 新建 `crypto_trader/walk_forward/expert_trainer.py`。复用 `train_moe_stage1.py` 模式：对 4 个专家，使用 `select_market_slice(train_df, slice_name)` 计算制度掩码（分位数从训练 df 计算），应用 `_training_controls_for_expert()`，创建 TradingEnv + VecNormalize，训练指定步数，保存到 `{fold}/experts/{expert_id}/`。

  **Must NOT do**: 不修改专家 reward_profile；不修改训练超参数；不将制度分位数计算在测试数据上。

  **Recommended Agent Profile**: `deep` | **Skills**: [`stable-baselines3`] | **Parallel**: Wave 2 | **Blocks**: Task 6 | **Blocked By**: Tasks 4

  **References**: `crypto_trader/train_moe_stage1.py:230-307`（`_train_one_expert`），`crypto_trader/moe/regime.py`（`select_market_slice`），`crypto_trader/moe/manifest.py`（`FEATURE_MASKS`）

  **Acceptance Criteria**: 测试折（短日期范围）可训练 4 个专家；model.zip 和 vec_normalize.pkl 存在各专家的检查点目录中；spec.json 包含 regime_coverage

  **Commit**: YES（Wave 2 组提交）

- [ ] 6. 创建 GateTrainer（温度扫描 + Stage2）

  **What to do**: 新建 `crypto_trader/walk_forward/gate_trainer.py`。两阶段：（A）将折训练数据按时间 80/20 拆分；对 7 个候选温度在 gate_train 上训练 Gate PPO，在 gate_val 上评估夏普比率；选择最优温度。（B）使用选定温度在完整训练数据上重训 Gate。保存 gate_model.zip + metadata.json（含 usage_ema、selected_temperature、candidate_results）。

  **Must NOT do**: 不在测试数据上评估候选温度。不修改 load_balance_coef=0.02、diversity_coef=0.01。

  **Recommended Agent Profile**: `deep` | **Skills**: [`stable-baselines3`] | **Parallel**: Wave 2 | **Blocks**: Task 7 | **Blocked By**: Task 5

  **References**: `crypto_trader/train_moe_stage2_gate.py:160-252`（GateRoutingEnv、Gate 训练）

  **Acceptance Criteria**: 温度扫描产生 7 个验证指标；selected_temperature 为最大化验证夏普的值；final gate_model.zip 存在于折检查点中

  **Commit**: YES（Wave 2 组提交）

- [ ] 7. 创建 FoldBacktester + Aggregator

  **What to do**: 新建 `crypto_trader/walk_forward/backtester.py` 和 `crypto_trader/walk_forward/aggregator.py`。Backtester：加载折专家 + Gate，`execution_mode="next_bar"` 调用 `backtest_moe`，返回 total_return/benchmark/alpha/max_dd/sharpe/gate_usage。Aggregator：收集 5 个 FoldResult，计算逐折通过（alpha > 0%）、avg_alpha、整体裁决（all pass AND avg ≥ 20%）。

  **Must NOT do**: 不修改 backtest_moe 的核心逻辑。

  **Recommended Agent Profile**: `deep` | **Skills**: [`stable-baselines3`] | **Parallel**: Wave 2 | **Blocks**: Task 8 | **Blocked By**: Tasks 4, 5, 6

  **References**: `crypto_trader/backtest_moe.py`，`crypto_trader/validation/metrics.py`

  **Acceptance Criteria**: Mock FoldResult 产生正确的 pass/fail 裁决（正 alpha → pass，负 → fail）；aggregator 正确计算平均值

  **Commit**: YES（Wave 2 组提交）

---

- [ ] 8. 集成主编排器 moe_walk_forward.py

  **What to do**: 新建 `crypto_trader/walk_forward/moe_walk_forward.py`。按顺序编排 5 折：`FoldingManager.build_folds()` → 逐折调用 `DataPreparer.prepare()` → `ExpertTrainer.train_all()` → `GateTrainer.select_temperature() + train_final()` → `FoldBacktester.run()`。各折间释放内存。聚合所有 FoldResult → `Aggregator.evaluate()` → 输出裁决。

  **Must NOT do**: 不在各折间共享状态；不在中途失败时继续下一折（保存已完成折，可恢复）。

  **Recommended Agent Profile**: `quick` | **Skills**: [] | **Parallel**: Wave 3 | **Blocks**: Task 9 | **Blocked By**: Tasks 4-7

  **References**: `crypto_trader/walk_forward/train_walk_forward.py`（现有编排器模式）

  **Acceptance Criteria**: 编排器导入所有模块成功；`--dry-run` 打印 5 折计划不执行训练；`--fold 1` 仅运行折 1

  **Commit**: YES（Wave 3 组提交）

- [ ] 9. 运行 5 折 Walk-Forward 训练 + 回测

  **What to do**: 执行 `PYTHONPATH=. python crypto_trader/walk_forward/moe_walk_forward.py`。顺序运行 5 折：每折训练 XGBoost + 4 专家（各 150K/180K 步）+ Gate 温度扫描（7 候选 × 各折）+ 最终 Gate 训练 + 测试期回测。总耗时估计：折 1 较小训练窗口 ~30 分钟，折 5 最大训练窗口 ~90 分钟，总计 ~4-6 小时。各折间保存状态，允许中断后恢复。

  **Must NOT do**: 不跳过任何折；不省去温度扫描步骤；不在测试数据上选择参数。

  **Recommended Agent Profile**: `deep` | **Skills**: [`stable-baselines3`] | **Parallel**: Wave 3（顺序执行）| **Blocks**: Task 10 | **Blocked By**: Task 8

  **References**: `crypto_trader/walk_forward/moe_walk_forward.py`（Task 8），`moe_config.py`（配置）

  **Acceptance Criteria**: 5 折全部完成（可接受个别中途失败后重试）；各折检查点目录包含完整产物；`results/walk_forward_moe/summary/summary.csv` 存在

  **QA Scenarios**:
  ```
  Scenario: 至少 1 折完成训练+回测
    Tool: Bash
    Preconditions: moe_walk_forward.py 可导入，配置有效
    Steps:
      1. PYTHONPATH=. python crypto_trader/walk_forward/moe_walk_forward.py --fold 1
      2. ls crypto_trader/walk_forward/results/walk_forward_moe/fold_1/metrics.json → 存在
      3. python3 -c "import json; m=json.load(open('crypto_trader/walk_forward/results/walk_forward_moe/fold_1/metrics.json')); print(f'Fold1 alpha: {m[\"alpha\"]:.2%}')"
    Expected Result: 折 1 产生有效指标
    Evidence: .sisyphus/evidence/task-9-fold1.txt
  ```
  **Commit**: NO（结果文件太大，仅提交裁决）

- [ ] 10. 最终验证 + 生产级裁决报告

  **What to do**: 运行 Aggregator 生成 `verdict.json` → 如果 PASS，生成 `final_report.md`。内容包括：5 折指标表、avg_alpha 与目标对比、数据泄漏审计结果（F3）、与之前研究结果的对比（参考 revival 报告）。如果任何折失败（alpha ≤ 0），报告失败折和原因。

  **Must NOT do**: 不美化指标。逐折 alpha 必须真实。

  **Recommended Agent Profile**: `quick` | **Skills**: [] | **Parallel**: Wave 3 | **Blocks**: None | **Blocked By**: Task 9

  **References**: `crypto_trader/walk_forward/results/walk_forward_moe/summary/verdict.json`（Task 9 产物），`.sisyphus/reports/final-revival-report.md`（此前报告）

  **Acceptance Criteria**: `verdict.json` 存在，状态为 PASS（所有折 alpha > 0 且 avg ≥ 20%）或 FAIL（有原因的逐折失败）；`final_report.md` 包含完整的 5 折指标表和裁决理由

  **QA Scenarios**:
  ```
  Scenario: 裁决报告生成
    Tool: Bash
    Steps:
      1. PYTHONPATH=. python crypto_trader/walk_forward/aggregator.py
      2. python3 -c "import json; v=json.load(open('crypto_trader/walk_forward/results/walk_forward_moe/summary/verdict.json')); print(json.dumps(v, indent=2))"
      3. ls crypto_trader/walk_forward/results/walk_forward_moe/summary/final_report.md → 存在
    Expected Result: verdict.json 和 final_report.md 均存在
    Evidence: .sisyphus/evidence/task-10-verdict.txt
  ```
  **Commit**: YES | `feat(walk_forward): production-grade verdict + final report`

---

## Final Verification Wave

- [ ] F1. **代码回归检查** — `quick`
  - `PYTHONPATH=. python -m pytest crypto_trader/tests/ -v` → 22 passed
  - 验证修复后的 resolve_execution_frame 不破坏现有测试

- [ ] F2. **Walk-Forward 指标达标检查** — `unspecified-high`
  - 读取 `results/walk_forward_moe/summary/verdict.json`
  - Assert: 每折 alpha > 0%, avg_alpha ≥ 20%
  - Assert: 5 折全部完成，无中途失败

- [ ] F3. **数据泄漏审计** — `oracle`
  - 验证各折 XGBoost 仅在训练数据上 .fit()
  - 验证 Gate 温度仅在训练数据上选择
  - 验证 regime 分位数仅在训练数据上计算
  - 输出: `泄漏检查 [CLEAN/CONTAMINATED]`

- [ ] F4. **最终 21 场景审计（walk-forward 验证后）** — `unspecified-high`
  - 如果 walk-forward 通过，使用最优配置跑完整审计
  - 输出: `verdict [PASS/WARN/FAIL]`

---

## Commit Strategy

- **Wave 1 完成**: `fix: resolve_execution_frame shift all columns for next-bar`
- **Wave 2 完成**: `feat(walk_forward): MoE anchored walk-forward framework`
- **Wave 3 完成**: `feat(walk_forward): 5-fold validation results + verdict`

---

## Success Criteria

### Verification Commands
```bash
# 回归测试
PYTHONPATH=. python -m pytest crypto_trader/tests/ -v

# Walk-forward 运行
PYTHONPATH=. python crypto_trader/walk_forward/moe_walk_forward.py

# 裁决检查
python -c "import json; v=json.load(open('crypto_trader/walk_forward/results/walk_forward_moe/summary/verdict.json')); assert v['status']=='PASS'"
```
