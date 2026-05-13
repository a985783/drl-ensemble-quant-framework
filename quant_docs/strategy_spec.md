# 策略规范
- 策略名称：Phase B+ Engine（PPO + XGBoost 集成）
- 版本：仓库主线（run_id 由运行时生成）

## 1. 信号与因子
- 主要输入：技术指标特征 + XGBoost 方向概率 + 账户状态
- 技术指标（来自 `features.py`）：RSI、MACD、布林带宽度、ATR、SMA50/200、Dist_SMA_200、Vol_Ratio、Rolling_Vol、收益滞后特征、ROC 等
- 信号模型：XGBoost 预测下一期方向概率（`Signal_Proba`）
- 强化学习：PPO 输出目标仓位意图 `a_raw ∈ [-1, 1]`

## 2. 仓位与组合
- 目标仓位：`[-1, 1]`（多/空/空仓）
- 波动率缩放：`vol_scale = 0.05 / ATR%`，范围 `[0.1, 2.0]`
- 约束机制（Phase B）：
  - Hysteresis：τ=0.25
  - Slew Rate：Δmax=0.15
  - Cooldown：N=3（翻向冷却）
- 风控：分级回撤降仓（见 `risk_plan.md`）

## 3. 成本与执行假设
- 回测成本模型：
  - 单边成本 `K_SINGLE=0.0008`（约 0.05% 手续费 + 0.03% 滑点）
  - 资金费率 `FUNDING_DAILY=0.0003`
- 实盘执行：Limit-then-Market，reduce-only 仅用于减仓/平仓
- 成交约束：若限价超时未成交，转市价补单
