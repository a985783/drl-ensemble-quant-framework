# 实盘准入评估（准机构级）

## 结论（当前）
- 结论：**未达“准机构实盘”**，但已满足文档门槛，进入“可执行改进”阶段。
- 原因：证据链与监控闭环仍不完整，且缺少可复现实验与数据版本快照。

## 已满足
- 文档包齐全：research/data/strategy/backtest/robustness/risk/execution/monitoring/governance
- 风控分级：DD 5%/10%/15% 降仓机制已在代码中实现
- SAFE_MODE：API 连续失败自动进入 reduce-only
- 真实对账：本地仓位 vs 交易所仓位不一致将触发 SAFE_MODE

## 关键缺口（阻断实盘）
1. **可复现实验记录不足**
   - 需要将 walk-forward 结果输出为结构化指标（CSV）并写入 `experiment_log.csv`
2. **数据版本不可复现**
   - 需要对回测/训练使用的数据做快照或至少记录哈希与时间范围
3. **监控闭环未验证**
   - 监控计划已写，但尚未验证告警通道与异常演练（部分成交/对账异常）

## 下一步最小任务（不影响模型/回测）
- 生成 walk-forward 指标 CSV 并记录到 `experiment_log.csv`
- 对 `backtest_recent.csv` 与 walk-forward 结果生成统一指标摘要（收益、回撤、胜率、换手等）
- 执行 1 次模拟异常演练（手动修改本地持仓/制造对账偏差）并确认 SAFE_MODE 行为
