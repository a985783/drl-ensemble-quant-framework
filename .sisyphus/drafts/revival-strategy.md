# Draft: 策略复活计划

## Requirements (confirmed)
- 目标：绝对正收益 + 低风险（低回撤）
- 充分验证的生产级 Alpha 系统
- 手段不限：换架构、换参数、换任何东西都可以
- 必须在 next-bar 执行口径下验证
- 不影响当前 stable 策略（旁路进行）

## 已知事实
- same-bar +90.83% 已证伪（同收盘成交假设）
- next-bar 下 stable 配置为 -15.81%（但跑赢 ETH 30%）
- tau_0.8x（τ=0.20）在 next-bar 下 +52.81%，Alpha +98.84%
- temperature_0.5：+25.00%
- temperature_1.5：+25.24%
- drop_top_contributor：+24.50%（说明专家间存在冲突）
- Gate 在 next-bar 下无增益（uniform ≈ stable）
- 随机基线 -5.40%，排除回测框架偏差

## 关键洞察
1. 策略框架没死，死的是为 same-bar 优化的参数
2. τ（迟滞门槛）是最敏感的单一参数
3. Gate 在 next-bar 下是负资产
4. 某些专家组合存在破坏性交互

## Technical Decisions
- 默认 next-bar 执行口径
- 利用已有 validation 框架进行参数探索
- 先简化架构（去 Gate、减专家），再按需加回
- 所有新训练/验证在旁路进行，不碰 stable 模型

## Metis Review Key Findings
1. **必须先跑 PPO20 single-agent next_bar 基线**：如果单 PPO 在 next_bar 下跑赢 MoE，整个 MoE 框架就没必要。这是必须先验证的前提。
2. **tau_0.8x 可能是统计偶然**：400 个 OOS 数据点太少，+52% 可能来自少数几笔交易。需要 bootstrap 检验。
3. **网格搜索不用重训**：τ/temperature/delta_max/cooldown 是执行层参数，可以在现有模型上直接用 backtest_moe overrides 扫。
4. **现有 experts 是在 same-bar 下训练的**：模型内部已经嵌入了 same-bar 的执行期望。最终可能需要从头重训。
5. **缺少单专家独立评估**：需要先确认 8 个专家在 next_bar 下各自有没有技能，再谈融合。

## Scope Boundaries
- INCLUDE: 参数优化（无重训网格扫描 + 重训验证）、架构简化（去Gate、减专家）、单 PPO 基线、anchored walk-forward 重训、bootstrap 统计检验
- EXCLUDE: 多资产扩展、实盘部署、新特征工程、XGBoost 重训、新 RL 算法、walk_forward 框架重构
