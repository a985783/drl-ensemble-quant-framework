# 回测报告
- 回测引擎：`crypto_trader/backtest_ensemble.py` + `TradingEnv`（Gymnasium）
- 回测区间：见下文（含 README 报告区间与最近短期回测）
- 版本：仓库主线（需在 `experiment_log.csv` 记录具体 run_id/参数）

## 1. 模型与假设
- 回测范式：步进式环境回测（RL 环境），特征工程使用 `shift(1)` 避免未来函数
- 成本模型：单边成本 `K_SINGLE=0.0008`，资金费率 `FUNDING_DAILY=0.0003`
- 约束：Hysteresis / Slew Rate / Cooldown 与实盘逻辑一致

## 2. 结果摘要
- README 报告（需复现实验验证）：
  - 2024-2025：年化 +82.8%，最大回撤 14.0%
  - 2018：年化 +148%，最大回撤 20.6%
- 最近短期回测（`results/backtest_recent.csv`）：
  - 区间：2025-12-24 至 2026-02-03
  - 净值变化：+19.16%
  - 最大回撤：6.49%

## 3. 风险与不足
- 成本模型为常数近似，未显式建模真实深度/冲击成本
- 资金费率使用常数近似，实盘可能偏离
- 缺少正式复现实验记录与参数快照（需补全 `experiment_log.csv`）
- 回测结果需与 walk-forward 验证联动评估
