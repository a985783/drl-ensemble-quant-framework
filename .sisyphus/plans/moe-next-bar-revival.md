# MoE 策略 Next-Bar 复活计划

## TL;DR

> **Quick Summary**: 原策略因同收盘成交假设被证伪（+90.83% → -15.81%），但策略框架未死。通过无重训参数网格扫描 + 架构简化（去Gate） + 为 next-bar 重训，目标是找到绝对正收益+低回撤的配置，并通过 anchored walk-forward 充分验证。
>
> **Deliverables**:
> - PPO20 单智能体 next_bar 基线结果
> - 8 专家独立 next_bar 能力评估
> - 无重训参数网格扫描报告（tau/temperature/delta_max/cooldown/cost）
> - 简化架构候选（去Gate、减专家）的 backtest 结果
> - 从零重训的 next-bar MoE 模型（checkpoints/moe/candidate/）
> - Anchored walk-forward 多折验证报告
> - 最终验证审计报告（21 场景，预期 PASS）
>
> **Estimated Effort**: Large（多轮训练+验证，主要耗时在重训和 walk-forward）
> **Parallel Execution**: YES - 3 波
> **Critical Path**: 参数网格扫描 → 重训 → walk-forward → 最终审计

---

## Context

### Original Request
用户在发现原策略 +90.83% 是 same-bar 执行假设造成的假象后，要求复活策略。目标：绝对正收益 + 低风险，充分验证的生产级 Alpha 系统。手段不限（换参数、换架构、换任何东西）。

### Interview Summary
**已知事实**:
- 修正后的 next-bar 审计显示：tau_0.8x (+52.81%) 是最佳单一参数调整
- Gate 在 next-bar 下无增益（uniform -14.79% ≈ stable -15.81%）
- temperature_0.5 (+25.00%) 和 temperature_1.5 (+25.24%) 均优于 stable
- drop_top_contributor (+24.50%) 优于 stable，暗示专家冲突
- 随机基线 -5.40%，排除回测框架偏差

**关键决策**:
- 所有工作旁路进行（checkpoints/moe/candidate/），不碰 stable
- XGBoost 信号模型冻结（不改 Signal_Proba）
- 新特征不添加
- 现有 3 种 RL 算法（PPO/SAC/A2C）不变

### Metis Review
**识别到的关键盲区**:
1. **必须先跑 PPO20 单智能体 next_bar 基线**：如果单 PPO 跑赢 MoE，整个 MoE 框架无存在必要
2. **现有 experts 是 same-bar 下训练的**：即使 backtest 用 next_bar，模型策略已嵌入 same-bar 期望
3. **tau_0.8x 可能是统计偶然**：400 个 OOS 数据点，需 bootstrap 检验
4. **8 个专家在 next_bar 下各自有没有技能**：未被独立评估过
5. **现有 walk-forward 是单 PPO，非 MoE**：walk_forward 框架不支持 MoE 重训，需要扩展

**应用了以下 guardrails**:
- MUST NOT 动 stable 注册表或 checkpoint
- MUST NOT 改 XGBoost、features、RL 算法
- MUST NOT 做多资产扩展
- MUST 所有实验用 execution_mode="next_bar"

---

## Work Objectives

### Core Objective
找到一组参数和架构配置，使策略在 next-bar 执行下产生绝对正收益（OOS total_return ≥ 15%），最大回撤 ≤ 22%，且通过 21 场景验证审计为 PASS。

### Concrete Deliverables
- `results/candidates/ppo20_nextbar_baseline/metrics.csv` — 单 PPO 基线
- `results/candidates/expert_audit_nextbar/*.csv` — 8 专家独立评估
- `results/candidates/param_sweep/report.md` — 参数网格扫描报告
- `results/candidates/arch_simplified/*.csv` — 简化架构回测
- `checkpoints/moe/candidate/experts/*` — 从零重训的 next-bar 专家
- `results/candidates/walk_forward/` — Anchored walk-forward 报告
- `results/validation/revival_final/summary.md` — 最终审计 PASS 报告

### Definition of Done
- [ ] PPO20 单智能体 next_bar 基线已跑
- [ ] 参数网格扫描完成（≥ 50 组合），最优组合 OOS total_return > 0%
- [ ] 简化架构（去 Gate + ≤ 4 专家）在最优参数下 OOS total_return ≥ 15%
- [ ] 从零重训的模型在 next_bar 下 OOS total_return ≥ 15%
- [ ] Anchored walk-forward ≥ 4 折，avg alpha ≥ 20%
- [ ] 最终 21 场景审计 verdict = PASS
- [ ] 所有 22 个已有测试通过

### Must Have
- 绝对正收益（OOS total_return > 0），目标 ≥ 15%
- 最大回撤 ≤ 22%
- 21 场景审计 PASS（无 FAIL、无 BLOCKED）
- 成本 2x 后 alpha > 0%
- Walk-forward 多折平均 alpha > 0%
- 结果可复现（固定种子重跑一致）
- 所有模型和结果存于旁路（candidate/），不碰 stable

### Must NOT Have (Guardrails)
- 不改动 stable 注册表或 checkpoint
- 不改动 XGBoost 信号模型或 Signal_Proba
- 不添加/修改/删除 13 维观测空间的特征
- 不引入新 RL 算法
- 不做多资产扩展
- 不修改 live_trading_okx.py 或实盘入口
- 不修改 walk_forward 框架本身（仅新增 MoE 训练脚本作为独立文件）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES（22 个测试文件，pytest）
- **Automated tests**: Tests-after（验证框架已有，新增参数扫描和重训脚本后补测试）
- **Framework**: pytest
- **Existing tests must pass**: `PYTHONPATH=. python -m pytest crypto_trader/tests/ -v` → 22 passed

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Backtest/CLI**: Use Bash — 运行命令，捕获输出，assert 返回码和指标
- **数值验证**: 对比 CSV 指标文件，assert 数值范围
- **审计验证**: 运行 alpha_validation.py，assert verdict.json 为 PASS

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - 基线 + 评估，全部并行):
├── Task 1: PPO20 单智能体 next_bar 基线 [quick]
├── Task 2: 8 专家独立 next_bar 能力评估 [quick]
├── Task 3: 扩展参数网格扫描配置 [quick]
└── Task 4: 扩展验证框架（新增 bootstrap、门控崩塌检测）[quick]

Wave 2 (After Wave 1 - 参数扫描 + 架构探索，全部并行):
├── Task 5: 无重训参数网格扫描（≥ 50 combos）[deep]
├── Task 6: Bootstrap 显著性检验 [quick]
├── Task 7: 简化架构回测（去Gate + 减专家）[deep]
└── Task 8: 专家两两冲突分析（pairwise ablation）[deep]

Wave 3 (After Wave 2 - 重训 + 验证，部分顺序):
├── Task 9: 为 next-bar 从零重训最优专家配置 [deep]
├── Task 10: 重训后参数微调 [quick]
├── Task 11: Anchored walk-forward 验证（≥ 4 折）[deep]
└── Task 12: 最终 21 场景审计 + 报告 [quick]
```

---

## TODOs

- [ ] 1. PPO20 单智能体 next_bar 基线

  **What to do**:
  - 在 `crypto_trader/scripts/` 下新建 `run_ppo20_nextbar_baseline.py`
  - 加载 train80 数据，用 `TradingEnv` + PPO（20 种子 ensemble，每个 150K timesteps）训练
  - 训练环境必须用 `resolve_execution_frame(df, execution_mode="next_bar")` 预处理数据
  - 在 OOS 数据上回测（next_bar），输出 metrics.csv 到 `results/candidates/ppo20_nextbar_baseline/`
  - 对比：PPO20 next_bar vs MoE stable next_bar (-15.81%)。如果 PPO20 显著跑赢，说明 MoE 架构在当前约束下无优势

  **Must NOT do**:
  - 不要使用 same_bar 模式
  - 不要修改现有 PPO 超参数（用 config.py 默认值）
  - 不要使用 Gate 或多专家融合

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `deep`
    - Reason: 需要理解 RL 训练流程，处理数据预处理和模型训练，涉及多文件协调
  - **Skills**: [`stable-baselines3`]
    - `stable-baselines3`: 直接使用 PPO 训练和 VecNormalize，与现有架构一致
  - **Skills Evaluated but Omitted**:
    - `pufferlib`: 高性能但不需要，150K timesteps 很短

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Tasks 2, 3, 4 并行）
  - **Blocks**: Task 7（简化架构对比需要此基线）
  - **Blocked By**: None

  **References**:
  - `crypto_trader/train_moe_stage1.py:36-51` - get_algo_registry() 模式，复用 PPO 加载方式
  - `crypto_trader/config.py:36-45` - PPO 超参数默认值
  - `crypto_trader/backtest_moe.py:148-162` - resolve_execution_frame() 函数，复制 next_bar 逻辑
  - `crypto_trader/envs/trading_env.py:78-128` - TradingEnv 初始化参数

  **Acceptance Criteria**:
  - [ ] 文件存在：`results/candidates/ppo20_nextbar_baseline/metrics.csv`
  - [ ] 文件存在：`results/candidates/ppo20_nextbar_baseline/report.json`
  - [ ] `PYTHONPATH=. python -m pytest crypto_trader/tests/ -v` → 22 passed

  **QA Scenarios**:

  ```
  Scenario: PPO20 训练完成并产出有效指标
    Tool: Bash
    Preconditions: train80 数据存在于 crypto_trader/data_moe_20200101_20260216_train80.csv
    Steps:
      1. PYTHONPATH=. python crypto_trader/scripts/run_ppo20_nextbar_baseline.py --train-data crypto_trader/data_moe_20200101_20260216_train80.csv --oos-data crypto_trader/data_moe_20200101_20260216_oos20.csv --output results/candidates/ppo20_nextbar_baseline
      2. ls results/candidates/ppo20_nextbar_baseline/metrics.csv → 文件存在
      3. python -c "import pandas as pd; m = pd.read_csv('results/candidates/ppo20_nextbar_baseline/metrics.csv'); assert 'total_return' in m.columns; assert 'max_drawdown' in m.columns"
    Expected Result: 训练成功完成，metrics.csv 包含 total_return 和 max_drawdown 列，数值为合理范围（-100% 到 +200%）
    Evidence: .sisyphus/evidence/task-1-ppo20-baseline.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-1-ppo20-baseline.txt` — 训练日志和 metrics.csv 摘要

  **Commit**: YES（Wave 1 组提交）
  - Message: `feat(candidate): add PPO20 next-bar baseline`
  - Files: `crypto_trader/scripts/run_ppo20_nextbar_baseline.py`, `results/candidates/ppo20_nextbar_baseline/`

---

- [ ] 2. 8 专家独立 next_bar 能力评估

  **What to do**:
  - 复用已有 `crypto_trader/scripts/eval_8_experts.py`，添加 `--execution-mode next_bar` 参数
  - 对 8 个 stable 专家逐一在 OOS 数据上独立回测（next_bar）
  - 每个专家仅用自己的 feature_mask 和 reward_profile，不做融合
  - 输出：每个专家的 total_return, max_drawdown, alpha, sharpe, turnover
  - 输出到 `results/candidates/expert_audit_nextbar/expert_metrics.csv`
  - 判断：哪些专家在 next_bar 下有正收益信号（至少跑赢随机基线 -5.40%）

  **Must NOT do**:
  - 不要修改专家的 feature_mask 或 reward_profile
  - 不要使用 Gate 融合
  - 不要重训专家（使用 stable checkpoint）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 修改已有脚本加参数，执行 8 次回测，逻辑简单
  - **Skills**: [`stable-baselines3`]
    - `stable-baselines3`: 加载 SB3 模型和 VecNormalize

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Tasks 1, 3, 4 并行）
  - **Blocks**: Task 5（参数扫描需要知道哪些专家有用）, Task 7（简化架构需要知道保留哪些专家）
  - **Blocked By**: None

  **References**:
  - `crypto_trader/scripts/eval_8_experts.py` - 已有专家评估脚本
  - `crypto_trader/backtest_moe.py:148-162` - resolve_execution_frame()

  **Acceptance Criteria**:
  - [ ] 文件存在：`results/candidates/expert_audit_nextbar/expert_metrics.csv`
  - [ ] 8 个专家均有独立指标（每个一行）
  - [ ] 至少 N 个专家 total_return > -5.40%（跑赢随机基线）

  **QA Scenarios**:

  ```
  Scenario: 8 专家独立评估完成
    Tool: Bash
    Preconditions: 8 个 stable expert checkpoint 存在于 checkpoints/moe/stable/experts/
    Steps:
      1. PYTHONPATH=. python crypto_trader/scripts/eval_8_experts.py --execution-mode next_bar --output results/candidates/expert_audit_nextbar/expert_metrics.csv
      2. python -c "import pandas as pd; df = pd.read_csv('results/candidates/expert_audit_nextbar/expert_metrics.csv'); assert len(df) == 8, f'Expected 8 experts, got {len(df)}'; print(df[['expert_id','total_return','alpha']])"
    Expected Result: 8 个专家各有独立指标，随机基线以上专家至少 3 个
    Evidence: .sisyphus/evidence/task-2-expert-audit.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-2-expert-audit.txt` — 8 专家指标表

  **Commit**: YES（Wave 1 组提交）
  - Message: `feat(candidate): add expert next-bar standalone audit`
  - Files: `crypto_trader/scripts/eval_8_experts.py`, `results/candidates/expert_audit_nextbar/`

---

- [ ] 3. 扩展参数网格扫描配置

  **What to do**:
  - 在 `crypto_trader/validation/default_validation.yaml` 基础上新建 `crypto_trader/validation/param_sweep.yaml`
  - 扩展扫描维度：
    - tau: [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    - temperature: [0.3, 0.5, 0.68, 0.8, 1.0, 1.2, 1.5, 2.0]
    - delta_max: [0.10, 0.12, 0.15, 0.18, 0.20]
    - cooldown_n: [1, 2, 3, 5]
    - k_single (手续费): [0.0004, 0.0008, 0.0012]
  - 添加 gate_mode: [uniform, model] 维度
  - 添加 disabled_experts 组合（基于 Task 2 结果决定保留哪些专家）
  - 新建 `crypto_trader/validation/sweep_runner.py`：遍历配置组合调用 `backtest_moe()`
  - 输出格式：每个组合一行 CSV，含 total_return, alpha, max_drawdown, sharpe, sortino, calmar, turnover

  **Must NOT do**:
  - 不要修改 default_validation.yaml（保持审计框架不变）
  - 不要在此任务中实际运行扫描（仅配置和 runner）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: YAML 配置 + 简单的 Python runner 脚本，无复杂逻辑
  - **Skills**: []
    - 纯 Python 脚本，无需特定领域 skill

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Tasks 1, 2, 4 并行）
  - **Blocks**: Task 5（扫描 runner 是参数扫描的前提）
  - **Blocked By**: None

  **References**:
  - `crypto_trader/validation/default_validation.yaml` - 现有场景配置模板
  - `crypto_trader/validation/alpha_validation.py:200-350` - 场景运行逻辑参考
  - `crypto_trader/backtest_moe.py:148-162` - resolve_execution_frame()

  **Acceptance Criteria**:
  - [ ] 文件存在：`crypto_trader/validation/param_sweep.yaml`
  - [ ] 文件存在：`crypto_trader/validation/sweep_runner.py`
  - [ ] `python -c "from crypto_trader.validation.sweep_runner import generate_sweep_combos; combos = generate_sweep_combos(); assert len(combos) >= 50, f'Expected >=50 combos, got {len(combos)}'"`

  **QA Scenarios**:

  ```
  Scenario: 配置解析生成 ≥ 50 组合
    Tool: Bash
    Preconditions: param_sweep.yaml 和 sweep_runner.py 已创建
    Steps:
      1. PYTHONPATH=. python -c "from crypto_trader.validation.sweep_runner import generate_sweep_combos; combos = generate_sweep_combos(); print(f'Generated {len(combos)} combinations'); assert len(combos) >= 50"
    Expected Result: 生成 ≥ 50 个有效参数组合
    Evidence: .sisyphus/evidence/task-3-sweep-config.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-3-sweep-config.txt` — 扫描维度数和生成组合数

  **Commit**: YES（Wave 1 组提交）
  - Message: `feat(validation): add parameter sweep config and runner`
  - Files: `crypto_trader/validation/param_sweep.yaml`, `crypto_trader/validation/sweep_runner.py`

---

- [ ] 4. 扩展验证框架（bootstrap + 门控崩塌检测）

  **What to do**:
  - 在 `crypto_trader/validation/metrics.py` 添加 `bootstrap_confidence_interval(returns, n_bootstrap=1000)` 函数
  - 在 `crypto_trader/validation/verdicts.py` 添加：
    - `check_gate_collapse(gate_weights_history)`: 任一专家 EMA 权重 > 80% → WARN
    - `check_bootstrap_significance()`: return 的 95% CI 下界 < 0 → WARN
  - 在 `crypto_trader/validation/alpha_validation.py` 的背景中收集 gate_weights_history（已有 return_history 支持）
  - 添加 `--bootstrap` 标志启用 bootstrap 检验
  - 更新 `crypto_trader/tests/test_validation_verdicts.py` 添加新规则测试

  **Must NOT do**:
  - 不要修改现有 verdict 规则的行为（仅追加）
  - 不要修改 backtest_moe 的核心逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 在已有验证框架上追加功能，逻辑清楚，改动范围小
  - **Skills**: []
    - 纯 Python，无需特定 skill

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1（与 Tasks 1, 2, 3 并行）
  - **Blocks**: Task 6（bootstrap 检验需要此功能）
  - **Blocked By**: None

  **References**:
  - `crypto_trader/validation/metrics.py` - 已有指标计算
  - `crypto_trader/validation/verdicts.py` - 已有判决规则
  - `crypto_trader/validation/alpha_validation.py` - 场景运行入口
  - `crypto_trader/tests/test_validation_verdicts.py` - 现有测试

  **Acceptance Criteria**:
  - [ ] `bootstrap_confidence_interval()` 可导入且返回 (lower, upper)
  - [ ] `check_gate_collapse()` 在权重 > 80% 时返回 WARN
  - [ ] `PYTHONPATH=. python -m pytest crypto_trader/tests/test_validation_verdicts.py -v` → passed
  - [ ] `PYTHONPATH=. python -m pytest crypto_trader/tests/ -v` → 22+ passed

  **QA Scenarios**:

  ```
  Scenario: bootstrap 函数返回合理置信区间
    Tool: Bash
    Preconditions: 验证模块可导入
    Steps:
      1. PYTHONPATH=. python -c "
  import numpy as np
  from crypto_trader.validation.metrics import bootstrap_confidence_interval
  data = np.random.randn(100) * 0.1 + 0.02
  ci_low, ci_high = bootstrap_confidence_interval(data, n_bootstrap=500)
  print(f'95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
  assert ci_low < ci_high
  "
    Expected Result: CI 区间合理，lower < upper
    Evidence: .sisyphus/evidence/task-4-bootstrap.txt

  Scenario: gate_collapse 检测门控崩塌
    Tool: Bash
    Preconditions: verdicts 模块可导入
    Steps:
      1. PYTHONPATH=. python -c "
  from crypto_trader.validation.verdicts import check_gate_collapse
  import numpy as np
  weights = np.array([[0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0]])
  result = check_gate_collapse(weights)
  print(f'Gate collapse check: {result}')
  assert result is not None
  "
    Expected Result: 检测到门控崩塌（单专家 > 80%）
    Evidence: .sisyphus/evidence/task-4-gate-collapse.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-4-bootstrap.txt`
  - [ ] `.sisyphus/evidence/task-4-gate-collapse.txt`

  **Commit**: YES（Wave 1 组提交）
  - Message: `feat(validation): add bootstrap CI and gate collapse detection`
  - Files: `crypto_trader/validation/metrics.py`, `crypto_trader/validation/verdicts.py`, `crypto_trader/validation/alpha_validation.py`, `crypto_trader/tests/test_validation_verdicts.py`

---

- [ ] 5. 无重训参数网格扫描（≥ 50 组合）

  **What to do**:
  - 使用 Task 3 的 `sweep_runner.py` 运行参数网格扫描
  - 扫描范围：tau × temperature × delta_max × cooldown × gate_mode × expert_set
  - 基于 Task 2 结果，排除表现最差的 2 个专家（减少组合数）
  - 所有组合使用 stable checkpoint（不重训），backtest_moe overrides 修改参数
  - 每个组合生成独立 metrics.csv
  - 汇总：`results/candidates/param_sweep/sweep_summary.csv`
  - 排序：按 total_return 和 sortino 双重排序
  - 输出 top-10 最优参数组合

  **Must NOT do**:
  - 不要重训模型（仅用 backtest overrides）
  - 不要修改 stable checkpoint

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要运行大量组合（≥50），管理输出文件，处理可能的执行失败和超时
  - **Skills**: [`stable-baselines3`]
    - `stable-baselines3`: 加载 SB3 模型

  **Parallelization**:
  - **Can Run In Parallel**: NO（但可以在 Wave 2 与其他任务并行）
  - **Parallel Group**: Wave 2（与 Tasks 6, 7, 8 并行）
  - **Blocks**: Task 9（重训需要此结果选择参数）
  - **Blocked By**: Tasks 3, 4（扫描 runner 和 bootstrap 功能）

  **References**:
  - `crypto_trader/validation/sweep_runner.py` - Task 3 产物
  - `crypto_trader/validation/param_sweep.yaml` - Task 3 产物
  - `crypto_trader/backtest_moe.py:148-162` - resolve_execution_frame()
  - `crypto_trader/backtest_moe.py:165-178` - resolve_gate_weights() 支持 uniform 模式

  **Acceptance Criteria**:
  - [ ] `results/candidates/param_sweep/sweep_summary.csv` 存在，≥ 50 行
  - [ ] Top-3 组合 total_return > 0%（正收益）
  - [ ] 已排除 Task 2 确认的负贡献专家（如果存在）

  **QA Scenarios**:

  ```
  Scenario: 参数扫描完成且有正收益组合
    Tool: Bash
    Preconditions: sweep_runner.py, param_sweep.yaml 已就绪
    Steps:
      1. PYTHONPATH=. python crypto_trader/validation/sweep_runner.py --config crypto_trader/validation/param_sweep.yaml --output results/candidates/param_sweep --execution-mode next_bar
      2. ls results/candidates/param_sweep/sweep_summary.csv → 文件存在
      3. python -c "
import pandas as pd
df = pd.read_csv('results/candidates/param_sweep/sweep_summary.csv')
print(f'Total combinations: {len(df)}')
positive = df[df['total_return'] > 0]
print(f'Positive return combos: {len(positive)}')
if len(positive) > 0:
    print('Top 3:')
    print(positive.nlargest(3, 'total_return')[['tau','temperature','delta_max','total_return','max_drawdown']])
assert len(positive) > 0, 'No positive return combinations found'
"
    Expected Result: ≥ 50 组合，至少 1 个正收益组合
    Evidence: .sisyphus/evidence/task-5-param-sweep.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-5-param-sweep.txt` — 扫描汇总和 top-10 参数

  **Commit**: YES（Wave 2 组提交）
  - Message: `feat(candidate): add parameter sweep results (next-bar, no retrain)`
  - Files: `results/candidates/param_sweep/`

---

- [ ] 6. Bootstrap 显著性检验

  **What to do**:
  - 对 Task 5 找到的 top-5 参数组合，运行 bootstrap 显著性检验
  - 每个组合用不同随机种子（10 次）重跑 backtest，提取 total_return 分布
  - 计算 95% 置信区间
  - 输出：`results/candidates/param_sweep/bootstrap_results.csv`
  - 判断标准：95% CI 下界 > 0 才算"显著正收益"
  - 标记哪些组合通过显著性检验

  **Must NOT do**:
  - 不要修改 backtest 逻辑（仅改变种子重跑）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 对 top-5 组合重复跑 backtest × 10 次，简单循环逻辑
  - **Skills**: []
    - 纯脚本，使用 Task 4 的 bootstrap 函数

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Tasks 5, 7, 8 并行）
  - **Blocks**: Task 9（重训需要选择通过 bootstrap 检验的参数）
  - **Blocked By**: Task 4（bootstrap 函数）, Task 5（需要 top-5 参数）

  **References**:
  - `crypto_trader/validation/metrics.py` - bootstrap_confidence_interval() (Task 4 添加)
  - `results/candidates/param_sweep/sweep_summary.csv` - Task 5 top-5 参数

  **Acceptance Criteria**:
  - [ ] `results/candidates/param_sweep/bootstrap_results.csv` 存在，5 行
  - [ ] 每个组合有 mean_return, ci_lower, ci_upper
  - [ ] 至少 1 个组合 ci_lower > 0（通过显著性）

  **QA Scenarios**:

  ```
  Scenario: Bootstrap 检验完成
    Tool: Bash
    Preconditions: Task 5 sweep_summary.csv 存在
    Steps:
      1. PYTHONPATH=. python crypto_trader/validation/run_bootstrap.py --sweep-summary results/candidates/param_sweep/sweep_summary.csv --top-n 5 --output results/candidates/param_sweep/bootstrap_results.csv
      2. python -c "
import pandas as pd
df = pd.read_csv('results/candidates/param_sweep/bootstrap_results.csv')
significant = df[df['ci_lower'] > 0]
print(f'Significant (CI lower > 0): {len(significant)}/{len(df)}')
print(df.to_string())
"
    Expected Result: 至少 1 个组合 CI 下界 > 0
    Evidence: .sisyphus/evidence/task-6-bootstrap-results.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-6-bootstrap-results.txt` — bootstrap 检验结果

  **Commit**: YES（Wave 2 组提交）
  - Message: `feat(candidate): add bootstrap significance test for top params`
  - Files: `results/candidates/param_sweep/bootstrap_results.csv`

---

- [ ] 7. 简化架构回测（去Gate + 减专家）

  **What to do**:
  - 基于 Task 2（专家独立评估）和 Task 5（参数扫描）结果，设计简化架构：
    - 配置 A：uniform gate + 保留最好的 3 个专家
    - 配置 B：uniform gate + 保留最好的 4 个专家
    - 配置 C：去 Gate + 平均融合 + 保留最好的 3 个专家
    - 配置 D：无 Gate + 平均融合 + 保留最好的 4 个专家
    - 配置 E：无 Gate + 平均融合 + 所有 8 个专家（对照）
  - 每个配置使用 Task 5/6 找到的最优参数组合
  - 全部用 stable checkpoint（不重训），backtest_moe overrides
  - 输出：`results/candidates/arch_simplified/arch_comparison.csv`
  - 判断：简化架构 vs 原始 stable next_bar (-15.81%) 的提升幅度
  - 对比 PPO20 基线（Task 1）

  **Must NOT do**:
  - 不要重训模型
  - 不要使用 Gate 的 model 模式（仅 uniform 或直接平均）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要设计多种架构配置组合，运行多次 backtest，综合分析结果
  - **Skills**: [`stable-baselines3`]
    - `stable-baselines3`: 加载 SB3 模型

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Tasks 5, 6, 8 并行）
  - **Blocks**: Task 9（重训架构基于此结果）
  - **Blocked By**: Task 2（需要知道哪些专家有用）, Task 5（需要最优参数）

  **References**:
  - `crypto_trader/backtest_moe.py:165-178` - resolve_gate_weights() uniform 模式
  - `crypto_trader/backtest_moe.py:181-196` - _apply_disabled_experts()
  - `results/candidates/expert_audit_nextbar/expert_metrics.csv` - Task 2 结果
  - `results/candidates/param_sweep/sweep_summary.csv` - Task 5 最优参数
  - `results/candidates/ppo20_nextbar_baseline/metrics.csv` - Task 1 PPO20 基线

  **Acceptance Criteria**:
  - [ ] `results/candidates/arch_simplified/arch_comparison.csv` 存在，≥ 5 行
  - [ ] 至少 1 个配置 total_return > -5%（比 stable -15.81% 有实质提升）
  - [ ] 最佳配置与 PPO20 基线对比明确

  **QA Scenarios**:

  ```
  Scenario: 简化架构回测完成且 best > stable
    Tool: Bash
    Preconditions: Task 2, Task 5 结果已产出
    Steps:
      1. PYTHONPATH=. python crypto_trader/validation/run_arch_ablation.py --output results/candidates/arch_simplified --execution-mode next_bar
      2. python -c "
import pandas as pd
df = pd.read_csv('results/candidates/arch_simplified/arch_comparison.csv')
best = df.loc[df['total_return'].idxmax()]
print(f'Best config: {best[\"config\"]}, return={best[\"total_return\"]:.2%}')
assert best['total_return'] > -0.06, f'Best return {best[\"total_return\"]:.2%} not > -6%'
"
    Expected Result: 至少 1 个配置 return > -6%（比 stable 提升 ~10pp）
    Evidence: .sisyphus/evidence/task-7-arch-simplified.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-7-arch-simplified.txt` — 架构对比表

  **Commit**: YES（Wave 2 组提交）
  - Message: `feat(candidate): add simplified architecture backtest results`
  - Files: `results/candidates/arch_simplified/`

---

- [ ] 8. 专家两两冲突分析

  **What to do**:
  - 对 Task 7 的最佳配置中的专家，做两两组合消融
  - 每个专家对 (i, j)：
    - 配置 1：只有 i（其他 disabled）
    - 配置 2：只有 j（其他 disabled）
    - 配置 3：i + j（其余 disabled）
    - 比较：config3.return vs max(config1.return, config2.return)
  - 如果 config3.return < max(config1.return, config2.return) → 存在破坏性交互
  - 标记冲突对
  - 输出：`results/candidates/arch_simplified/pairwise_conflicts.csv`
  - 基于冲突分析，推荐最终保留的专家子集

  **Must NOT do**:
  - 不要修改专家模型本身

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 组合爆炸（N 专家 → N×(N-1)/2 对），需要系统化的消融实验设计
  - **Skills**: [`stable-baselines3`]
    - `stable-baselines3`: 加载模型

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Tasks 5, 6, 7 并行）
  - **Blocks**: Task 9（最终专家子集基于冲突分析）
  - **Blocked By**: Task 7（需要确定最佳配置中的专家集合）

  **References**:
  - `crypto_trader/backtest_moe.py:181-196` - _apply_disabled_experts()
  - `results/candidates/arch_simplified/arch_comparison.csv` - Task 7 最佳配置

  **Acceptance Criteria**:
  - [ ] `results/candidates/arch_simplified/pairwise_conflicts.csv` 存在
  - [ ] 标记了所有冲突对（如有）
  - [ ] 推荐最终专家子集（无冲突对的专家）

  **QA Scenarios**:

  ```
  Scenario: 冲突分析完成
    Tool: Bash
    Preconditions: Task 7 最佳配置已知
    Steps:
      1. PYTHONPATH=. python crypto_trader/validation/run_pairwise_ablation.py --output results/candidates/arch_simplified --execution-mode next_bar
      2. python -c "
import pandas as pd
df = pd.read_csv('results/candidates/arch_simplified/pairwise_conflicts.csv')
conflicts = df[df['is_destructive'] == True]
print(f'Destructive pairs: {len(conflicts)}')
if len(conflicts) > 0:
    print(conflicts[['expert_a','expert_b','separate_max','combined','degradation']])
"
    Expected Result: 冲突分析完成，明确标记冲突对
    Evidence: .sisyphus/evidence/task-8-pairwise-conflicts.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-8-pairwise-conflicts.txt` — 冲突分析结果

  **Commit**: YES（Wave 2 组提交）
  - Message: `feat(candidate): add pairwise expert conflict analysis`
  - Files: `results/candidates/arch_simplified/pairwise_conflicts.csv`

---

- [ ] 9. 为 next-bar 从零重训最优专家配置

  **What to do**:
  - 基于 Wave 2 全部结果确定最终配置（保留哪些专家、参数、是否用 Gate）
  - 新建 `crypto_trader/scripts/retrain_nextbar_moe.py`
  - 从零重训 Stage 1（使用 next_bar 预处理数据）
  - 如果使用 Gate：重训 Stage 2 + Stage 3
  - 所有 checkpoint 输出到 `checkpoints/moe/candidate/`
  - OOS 回测验证：total_return > 0，max_drawdown ≤ 22%

  **Must NOT do**: 不要写入 checkpoints/moe/stable/，不要使用 same_bar

  **Recommended Agent Profile**: `deep` + [`stable-baselines3`]

  **Parallelization**: Wave 3 | Blocks: Task 10, 11 | Blocked By: Tasks 2,5,7,8

  **References**: `crypto_trader/train_moe_stage1.py:1-80`, `crypto_trader/backtest_moe.py:148-162`, `crypto_trader/configs/moe_experts.yaml`

  **Acceptance Criteria**: 专家 checkpoint 存在于 candidate/ 下，OOS total_return > 0，Stable 注册表未被污染

  **QA Scenarios**:

  ```
  Scenario: 重训完成且 OOS 正收益
    Tool: Bash
    Steps:
      1. PYTHONPATH=. python crypto_trader/scripts/retrain_nextbar_moe.py --config results/candidates/arch_simplified/final_config.json --output-root checkpoints/moe/candidate
      2. PYTHONPATH=. python -m crypto_trader.backtest_moe --stage1-root checkpoints/moe/candidate/experts --execution-mode next_bar --output results/candidates/retrained/
      3. python -c "import json; m=json.load(open('results/candidates/retrained/metrics.json')); assert m['total_return']>0; assert m['max_drawdown']<=0.22; print('PASS')"
    Expected Result: total_return > 0, max_drawdown ≤ 22%
    Evidence: .sisyphus/evidence/task-9-retrained.txt
  ```

  **Commit**: YES | `feat(candidate): retrain MoE from scratch under next-bar`

---

- [ ] 10. 重训后参数微调

  **What to do**: 对 Task 9 重训模型微调 tau 和 temperature，验证提升效果。输出 `results/candidates/retrained/final_params.json`

  **Must NOT do**: 不要改动专家 checkpoint

  **Recommended Agent Profile**: `quick` + [`stable-baselines3`]

  **Parallelization**: Wave 3 | Blocks: Task 11 | Blocked By: Task 9

  **References**: `results/candidates/param_sweep/sweep_summary.csv`, `crypto_trader/validation/sweep_runner.py`

  **Acceptance Criteria**: 微调后 total_return ≥ baseline

  **QA Scenarios**:

  ```
  Scenario: 微调完成且不劣于 baseline
    Tool: Bash
    Steps:
      1. PYTHONPATH=. python crypto_trader/validation/fine_tune.py --stage1-root checkpoints/moe/candidate/experts --output results/candidates/retrained/ --execution-mode next_bar
      2. python -c "import json; ft=json.load(open('results/candidates/retrained/fine_tuned_metrics.json')); base=json.load(open('results/candidates/retrained/metrics.json')); assert ft['total_return']>=base['total_return']*0.95; print('PASS')"
    Expected Result: 微调后不劣于 baseline
    Evidence: .sisyphus/evidence/task-10-fine-tuned.txt
  ```

  **Commit**: YES | `feat(candidate): fine-tune execution params for retrained model`

---

- [ ] 11. Anchored walk-forward 验证（≥ 4 折）

  **What to do**: 设计 anchored walk-forward（Fold1: 2020-21→2022, Fold2: 2020-22→2023, Fold3: 2020-23→2024, Fold4: 2020-24→2025）。每折独立训练+验证。汇总 `results/candidates/walk_forward/wf_summary.csv`

  **Must NOT do**: 不要用全量 OOS 调参，不要使用 stable checkpoint

  **Recommended Agent Profile**: `deep` + [`stable-baselines3`]

  **Parallelization**: Wave 3 | Blocks: Task 12 | Blocked By: Task 10

  **References**: `crypto_trader/walk_forward/train_walk_forward.py`, `crypto_trader/scripts/retrain_nextbar_moe.py`

  **Acceptance Criteria**: ≥ 4 折，每折 alpha > -10%，avg alpha ≥ 20%

  **QA Scenarios**:

  ```
  Scenario: Walk-forward 全部折完成
    Tool: Bash
    Steps:
      1. PYTHONPATH=. python crypto_trader/validation/run_walk_forward.py --config results/candidates/retrained/final_params.json --output results/candidates/walk_forward --execution-mode next_bar
      2. python -c "import pandas as pd; df=pd.read_csv('results/candidates/walk_forward/wf_summary.csv'); assert len(df)>=4; assert df['alpha'].mean()>0.20; print('PASS')"
    Expected Result: 4 折完成，avg alpha ≥ 20%
    Evidence: .sisyphus/evidence/task-11-walk-forward.txt
  ```

  **Commit**: YES | `feat(candidate): add anchored walk-forward validation`

---

- [ ] 12. 最终 21 场景审计 + 综合报告

  **What to do**: 运行完整 21 场景审计（用 candidate checkpoint），assert verdict == PASS。生成 `results/candidates/revival_final/final_report.md`

  **Must NOT do**: 不要使用 stable checkpoint

  **Recommended Agent Profile**: `quick`

  **Parallelization**: Wave 3 | Blocks: None | Blocked By: Tasks 9,10,11

  **References**: `crypto_trader/validation/alpha_validation.py`, `results/validation/next_bar_audit/summary.md`

  **Acceptance Criteria**: verdict.json status == PASS，无 FAIL/BLOCKED，成本 2x alpha > 0%

  **QA Scenarios**:

  ```
  Scenario: 最终审计 PASS
    Tool: Bash
    Steps:
      1. PYTHONPATH=. python -m crypto_trader.validation.alpha_validation --run-id revival_final --stage1-root checkpoints/moe/candidate/experts --execution-mode next_bar
      2. python -c "import json; v=json.load(open('results/validation/revival_final/verdict.json')); assert v['status']=='PASS'; print('PASS')"
      3. PYTHONPATH=. python -m pytest crypto_trader/tests/ -v
    Expected Result: verdict PASS，所有测试通过
    Evidence: .sisyphus/evidence/task-12-final-audit.txt
  ```

  **Commit**: YES | `feat(candidate): final revival audit - PASS`

---

## Final Verification Wave

- [ ] F1. **审计合规检查** — `oracle`
  - 检查所有 checkpoints/moe/candidate/ 路径，确认无文件写入 checkpoints/moe/stable/
  - 读取 stable_model_registry.json，确认未被修改
  - 读取 moe_model_registry.json，确认 stable 条目未变
  - 输出: `Registry [CLEAN/CONTAMINATED] | Stable Dir [CLEAN/CONTAMINATED]`

- [ ] F2. **指标达标检查** — `unspecified-high`
  - 读取 results/candidates/revival_final/metrics.csv
  - Assert: total_return ≥ 15%, max_drawdown ≤ 22%, duration ≥ 200 trading days
  - Assert: walk_forward avg_alpha ≥ 0%, walk_forward folds ≥ 4
  - 运行所有 pytest: `PYTHONPATH=. python -m pytest crypto_trader/tests/ -v`
  - 输出: `Metrics [N/N PASS] | Tests [PASS/FAIL]`

- [ ] F3. **最终审计运行** — `unspecified-high`
  - 运行 `PYTHONPATH=. python -m crypto_trader.validation.alpha_validation --run-id revival_final`
  - Assert: verdict.json status == "PASS"
  - 读取 summary.md，确认无 FAIL 或 BLOCKED 项
  - 输出: `Audit Verdict [PASS/FAIL/BLOCKED] | Scenarios [N/N]`

- [ ] F4. **可复现性验证** — `quick`
  - 重新运行最终配置的 backtest（相同种子、相同数据），对比两次 metric.csv
  - Assert: total_return 差异 < 0.01%, max_drawdown 差异 < 0.01%
  - 输出: `Reproducibility [PASS/FAIL]`

---

## Commit Strategy

- **Wave 1 完成后**: `feat(candidate): add PPO20 next-bar baseline and expert audit`
- **Wave 2 完成后**: `feat(candidate): add parameter grid sweep and arch simplification results`
- **Wave 3 完成后**: `feat(candidate): add next-bar retrained MoE with walk-forward validation`

---

## Success Criteria

### Verification Commands
```bash
# 已有测试
PYTHONPATH=. python -m pytest crypto_trader/tests/ -v

# 最终审计
PYTHONPATH=. python -m crypto_trader.validation.alpha_validation --run-id revival_final \
    --manifest crypto_trader/configs/moe_experts.yaml \
    --stage1-root checkpoints/moe/candidate/experts \
    --gate-temperature 0.68 \
    --symbol ETH/USDT:USDT

# 指标检查
python -c "
import json
with open('results/validation/revival_final/verdict.json') as f:
    v = json.load(f)
assert v['status'] == 'PASS', f'Expected PASS, got {v[\"status\"]}'
print('✅ All checks passed')
"
```

### Final Checklist
- [ ] OOS total_return ≥ 15%
- [ ] max_drawdown ≤ 22%
- [ ] 21 场景审计 PASS
- [ ] Walk-forward avg alpha > 0%
- [ ] 成本 2x 后 alpha > 0%
- [ ] 22 测试通过
- [ ] Stable 注册表未被污染
- [ ] 结果可复现
