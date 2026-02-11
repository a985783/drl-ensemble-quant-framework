# 监控计划
- 关键指标：权益、回撤、持仓、成交量、滑点、手续费、SAFE_MODE、对账差异、Rollout KPI
- 告警阈值：见下方“触发条件”

## 1. 性能与漂移
- 绩效归因：滚动收益、回撤、波动率、胜率、盈亏比
- 漂移检测：信号翻向频率异常升高、长期低成交率、滑点显著抬升

## 2. 触发条件（建议，可调整）
- SAFE_MODE 被触发（API 连续失败 >= 3）→ 暂停开仓，仅减仓
- 对账异常（`Reconcile_Diff != OK`）连续 >= 2 次 → 暂停开仓，人工核对
- 平均滑点 > 0.5%（`rollout_controller` 默认阈值）→ 降级/暂停
- 成交率 < 90%（`rollout_controller` 默认阈值）→ 降级/暂停
- 回撤进入分级阈值（5%、10%）→ 自动降仓
- 滚动 30D 收益 < 0 且回撤扩大（连续观察期）→ 暂停并复核策略适配性

## 3. 数据来源与落地
- `trade_logs.csv`：成交、滑点、费用、SAFE_MODE、对账结果、rollout_level
- `runs/live_status.json`：阶段状态、权益、持仓、回撤
- `stable_model_registry.json`：rollout 指标与状态
- 报警通道：`ALERT_PROVIDER` + `ALERT_WEBHOOK_URL`（企业微信/钉钉/飞书）
