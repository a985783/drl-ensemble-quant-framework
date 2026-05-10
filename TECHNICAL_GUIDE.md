# 强化学习加密货币量化交易系统 — 技术完全手册

> **当前版本：** 4 专家 MoE + SimpleAlpha 简化策略
> **交易所：** OKX 永续合约 `ETH/USDT:USDT`
> **频率：** 日频（每日 UTC 08:05）
> **Python：** 3.9+
> **最后更新：** 2026-05-02

---

## 目录

1. [系统概览](#1-系统概览)
2. [快速开始](#2-快速开始)
3. [目录结构](#3-目录结构)
4. [数据管线](#4-数据管线)
5. [特征工程](#5-特征工程)
6. [XGBoost 信号模型](#6-xgboost-信号模型)
7. [SimpleAlpha 简化策略](#7-simplealpha-简化策略)
8. [MoE 混合专家系统（4 专家版）](#8-moe-混合专家系统4-专家版)
9. [交易环境与执行约束](#9-交易环境与执行约束)
10. [风控与执行安全](#10-风控与执行安全)
11. [Walk-Forward 样本外验证结果](#11-walk-forward-样本外验证结果)
12. [实盘执行链路](#12-实盘执行链路)
13. [调度与运维](#13-调度与运维)
14. [监控与告警](#14-监控与告警)
15. [配置参考](#15-配置参考)
16. [测试体系](#16-测试体系)
17. [技术栈](#17-技术栈)
18. [常见问题排查](#18-常见问题排查)

---

## 1. 系统概览

### 1.1 系统定位

本系统是一套**生产级**加密货币量化交易平台，采用 **4 专家 Mixture-of-Experts（MoE）强化学习** 架构，在 OKX ETH 永续合约上自动执行每日交易。

系统包含两条策略路径：

| 路径 | 文件 | 定位 |
|------|------|------|
| **MoE 主线** | `train_moe_stage1/2.py` + `live_trading_okx.py` | 生产执行策略 |
| **SimpleAlpha** | `strategies/simple_alpha.py` | 轻量验证基准，无需 GPU |

### 1.2 Walk-Forward 验证结果（当前版本）

> 来源：`crypto_trader/walk_forward/results/walk_forward_moe/summary/`，生成于 2026-05-02

#### Fold 5（2026-01-01 ~ 2026-05-02，最新实盘样本外）

| 指标 | 值 |
|------|:---:|
| **总收益** | **+18.23%** |
| **超额收益 Alpha** | **+44.62%** |
| ETH Buy & Hold 基准 | -26.39% |
| 最大回撤 | 9.82% |
| 夏普比率 | 1.36 |
| Sortino 比率 | 2.13 |
| Gate 温度（自动选优） | 0.5 |
| 验证状态 | **PASS** |

**Fold 5 Gate 权重分配：**

| 专家 | 权重 | 行为贡献 |
|------|:----:|:-------:|
| E5_PPO_lowvol_carry | **37.8%** | -8.68% |
| E7_SAC_fast_adapt | 31.5% | +2.88% |
| E2_PPO_bear_drawdown | 20.0% | +5.46% |
| E4_PPO_highvol_risk | 10.7% | +2.83% |

#### 全折汇总（2022 ~ 2026-05）

| Fold | 测试期 | 总收益 | ETH 基准 | Alpha | 回撤 | 夏普 | 温度 | 状态 |
|------|--------|:-----:|:--------:|:-----:|:----:|:----:|:----:|:----:|
| fold_1 | 2022 全年 | -20.14% | -68.23% | **+48.09%** | 21.80% | -1.12 | 2.0 | ✅ |
| fold_2 | 2023 全年 | +15.54% | +87.94% | -72.41% | 7.23% | 0.78 | 0.68 | ❌ |
| fold_3 | 2024 全年 | +25.04% | +50.98% | -25.94% | 16.48% | 0.99 | 0.8 | ❌ |
| fold_4 | 2025 全年 | -10.12% | -17.71% | **+7.59%** | 21.18% | -0.56 | 0.5 | ✅ |
| fold_5 | 2026-01~05 | **+18.23%** | -26.39% | **+44.62%** | 9.82% | 1.36 | 0.5 | ✅ |

> **fold_2/3 FAIL 说明：** 这两年 ETH 分别上涨 88% 和 51%，策略本身有正收益但跑输 Buy & Hold，导致 Alpha 为负。策略在 ETH 下行年份（2022/2025）和当前 2026 年均实现正 Alpha。整体 Walk-Forward 判定：5 折 3 过，平均 Alpha 0.39%（目标 20%），正式状态为 **FAIL**，仍在持续优化中。

### 1.3 架构流程图

```
OKX K线数据
    │
    ▼
DataValidator（数据质量校验）
    │
    ▼
FeatureEngineer（19维特征，全部 shift(1) 防未来函数）
    │
    ▼
SignalPredictor（XGBoost → Signal_Proba 上涨概率）
    │
    ├────────────────────────┐
    ▼                        ▼
SimpleAlpha 策略          MoE 4专家系统
（轻量验证路径）          ┌──────────────┐
                          │ Stage1: 4专家 │
                          │  独立训练     │
                          │ Stage2: Gate  │
                          │  路由训练     │
                          └──────┬───────┘
                                 │
                          live_trading_okx.py
                          ┌──────┼──────┐
                          ▼      ▼      ▼
                       RiskMgr ExecSafe Alerting
                                 │
                          OKX API 下单
```

---

## 2. 快速开始

### 2.1 安装

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2.2 配置 OKX API

```bash
cp .env.example .env
# 编辑 .env 填入：
# OKX_API_KEY / OKX_SECRET_KEY / OKX_PASSPHRASE
# OKX_SANDBOX=false（实盘）或 true（沙盒）
```

### 2.3 构建数据集

```bash
PYTHONPATH=. python crypto_trader/scripts/build_moe_dataset.py \
    --symbol "ETH/USDT:USDT" \
    --start "2020-01-01" --end "2026-05-02" \
    --train-ratio 0.8 \
    --output-prefix "crypto_trader/data_moe_20200101_20260502"
```

生成三个文件：`*_full.csv` / `*_train80.csv` / `*_oos20.csv`

### 2.4 训练 MoE（两阶段）

```bash
# Stage 1：4 专家独立训练（约 12 分钟）
PYTHONPATH=. python -m crypto_trader.train_moe_stage1 \
    --manifest crypto_trader/configs/moe_experts.yaml \
    --output-root checkpoints/moe/stage1 \
    --train-data-path crypto_trader/data_moe_20200101_20260216_train80.csv

# Stage 2：Gate 路由网络训练（约 5 分钟）
PYTHONPATH=. python -m crypto_trader.train_moe_stage2_gate \
    --manifest crypto_trader/configs/moe_experts.yaml \
    --stage1-root checkpoints/moe/stage1 \
    --output-dir checkpoints/moe/stage2 \
    --train-data-path crypto_trader/data_moe_20200101_20260216_train80.csv
```

### 2.5 样本外回测

```bash
PYTHONPATH=. python -m crypto_trader.backtest_moe \
    --manifest crypto_trader/configs/moe_experts.yaml \
    --stage1-root checkpoints/moe/stable/experts \
    --stage2-root checkpoints/moe/stable/gate \
    --data-path crypto_trader/data_moe_20200101_20260216_oos20.csv \
    --gate-temperature 0.5 \
    --symbol ETH/USDT:USDT \
    --plot-path results/eth_stable_oos20.png
```

### 2.6 Walk-Forward 验证

```bash
# MoE Walk-Forward（5 折，约 60 分钟）
PYTHONPATH=. python -m crypto_trader.walk_forward.moe_walk_forward \
    --data-path crypto_trader/data_moe_20200101_20260216_full.csv \
    --output-root crypto_trader/walk_forward/results/walk_forward_moe \
    --checkpoint-root crypto_trader/walk_forward/checkpoints/walk_forward_moe

# SimpleAlpha Walk-Forward（4 折，约 2 分钟）
PYTHONPATH=. python -m crypto_trader.strategies.simple_alpha_walk_forward \
    --data-path crypto_trader/data_moe_20200101_20260216_full.csv \
    --output-dir results/simple_alpha_walk_forward
```

### 2.7 健康检查与启动实盘

```bash
bash scripts/health_check_stable.sh       # 模型完整性检查
PYTHONPATH=. python crypto_trader/run_demo.py   # 模拟盘
PYTHONPATH=. python crypto_trader/run_real.py   # 真实盘
```

---

## 3. 目录结构

```text
强化学习_副本/
├── crypto_trader/
│   ├── config.py                    # 全局配置 Dataclass
│   ├── asset_profile.py             # ETH 资产参数（特征窗口 + 环境约束）
│   ├── data_loader.py               # OKX 历史 K线获取
│   ├── okx_data_loader.py           # OKX 实盘实时数据加载
│   ├── data_validator.py            # 数据质量校验
│   ├── data_versioning.py           # 数据集 SHA256 版本管理
│   ├── features.py                  # 特征工程（全部 shift(1)）
│   ├── logger.py                    # 统一日志
│   ├── seed_utils.py                # 随机种子工具
│   │
│   ├── models/signal_model.py       # XGBoost 上涨概率预测器
│   │
│   ├── envs/trading_env.py          # Gymnasium 交易环境（四重约束 + Reward）
│   │
│   ├── moe/
│   │   ├── manifest.py              # 专家配置解析 & Feature Mask 定义
│   │   └── regime.py                # 市场状态切片（bull/bear/range/high_vol/low_vol）
│   │
│   ├── configs/
│   │   ├── moe_experts.yaml         # ★ 当前 4 专家配置（E2/E4/E5/E7）
│   │   ├── base.yaml                # 基础配置模板
│   │
│   ├── strategies/
│   │   ├── simple_alpha.py          # SimpleAlpha 策略核心
│   │   └── simple_alpha_walk_forward.py  # SimpleAlpha Walk-Forward
│   │
│   ├── train_moe_stage1.py          # Stage1：4 专家独立训练
│   ├── train_moe_stage2_gate.py     # Stage2：Gate 路由网络训练
│   ├── train_moe_stage3_joint.py    # Stage3：交替联合微调（备用）
│   ├── train_ensemble.py            # 多种子集成训练
│   │
│   ├── backtest_moe.py              # MoE 主回测
│   ├── backtest_ensemble.py         # 集成回测（转发到 MoE）
│   ├── backtest_recent.py           # 最近区间快速回测
│   ├── backtest_sanity.py           # 回测健全性检查
│   ├── backtest_test_set.py         # 固定测试集回测
│   │
│   ├── live_trading_okx.py          # 实盘执行主引擎（~1400 行）
│   ├── run_live.py                  # 交互式实盘入口
│   ├── run_demo.py                  # 模拟盘入口（.env.demo）
│   ├── run_real.py                  # 真实盘入口（.env.live）
│   ├── start_simulation.py          # 仿真模式
│   ├── sanity_run.py                # 快速健全性跑通
│   │
│   ├── risk_manager.py              # 分层回撤风险管理
│   ├── execution_safety.py          # 执行安全（幂等/对账/SAFE_MODE）
│   ├── rollout_controller.py        # 渐进放量 + 自动回滚
│   ├── alerting.py                  # 企微/钉钉/飞书 Webhook 告警
│   ├── monitoring_utils.py          # 监控工具函数
│   ├── experiment_log.py            # 实验日志记录器
│   │
│   ├── analytics/performance.py     # 性能指标计算
│   │
│   ├── validation/
│   │   ├── alpha_validation.py      # Alpha 验证主逻辑
│   │   ├── metrics.py               # 验证指标
│   │   ├── verdicts.py              # 验证判定
│   │   ├── report.py                # 报告生成
│   │   ├── sweep_runner.py          # 参数扫描
│   │   ├── default_validation.yaml
│   │   └── param_sweep.yaml
│   │
│   ├── walk_forward/
│   │   ├── moe_walk_forward.py      # MoE Walk-Forward 编排器
│   │   ├── moe_config.py            # Walk-Forward 配置（5折 × 4专家）
│   │   ├── folding.py               # 折叠分割
│   │   ├── data_prep.py             # 折叠数据准备
│   │   ├── expert_trainer.py        # 专家折叠训练
│   │   ├── gate_trainer.py          # Gate 折叠训练（含温度扫描）
│   │   ├── backtester.py            # 折叠回测
│   │   ├── aggregator.py            # 多折汇总
│   │   ├── metrics.py               # Walk-Forward 指标
│   │   ├── backtest_walk_forward.py # 独立回测
│   │   ├── train_walk_forward.py    # 独立训练（单模型）
│   │   ├── checkpoints/             # 折叠模型（fold_1~5 × 4专家 + gate）
│   │   └── results/
│   │       └── walk_forward_moe/
│   │           ├── fold_1~5/metrics.json
│   │           └── summary/
│   │               ├── final_report.md
│   │               ├── summary.csv
│   │               └── verdict.json
│   │
│   ├── scripts/
│   │   ├── build_moe_dataset.py     # 数据集构建
│   │   ├── retrain_nextbar_moe.py   # Next-bar MoE 重训
│   │   ├── eval_moe_experts.py      # 专家独立评估
│   │   ├── eval_expert_regime_oos.py
│   │   ├── eval_xgboost_accuracy.py # XGBoost 诊断
│   │   ├── run_ppo20_nextbar_baseline.py
│   │   ├── investigate_moe_math.py
│   │   ├── debug_moe_math.py
│   │   ├── drill_reconciliation.py
│   │   └── update_experiment_log.py
│   │
│   └── tests/                       # 22 个测试文件（详见第 16 节）
│
├── scripts/                         # 运维 shell 脚本
│   ├── daily_trade.sh               # 日频执行主入口
│   ├── launch_trading.sh            # 守护启动（caffeinate 防休眠）
│   ├── start_background.sh / stop_background.sh
│   ├── check_status.sh              # daemon 状态 + 近期日志
│   ├── health_check_stable.sh       # 模型完整性一键检查
│   ├── setup_cron.sh                # cron 配置
│   ├── setup_trading_launchd.sh     # macOS launchd（推荐）
│   ├── setup_monitoring_launchd.sh
│   ├── migrate_cron_to_launchd.sh
│   ├── monitor_effectiveness_daily.sh
│   ├── monitor_effectiveness_weekly.sh
│   ├── monitor_strategy_effectiveness.py
│   ├── check_daily_execution.py
│   ├── cleanup_old_logs.sh
│   └── common.sh
│
├── checkpoints/signal_model.pkl     # 已训练的 XGBoost 模型
├── results/                         # 回测图表与 JSON 结果
├── logs/                            # 运行日志（daemon / monitor / daily）
├── quant_docs/                      # 量化研究文档（策略规格/风险计划/回测报告等）
├── docs/plans/                      # 设计文档
├── moe_model_registry.json          # 模型注册表
├── stable_model_registry.json       # 稳定版注册表
├── trading_state.json               # 实盘状态持久化
├── trade_logs.csv                   # 历史交易记录
├── requirements.txt
└── .env / .env.example / .env.demo / .env.live
```

---

## 4. 数据管线

### 4.1 数据流

```
OKX REST API → OHLCV 日 K线
    │
    ▼  data_validator.py
    │  ✓ 时间连续性（无缺口）
    │  ✓ 价格跳变 > 50% 标记为异常
    │  ✓ 非正价格过滤
    │  ✓ 重复时间戳去重
    ▼
features.py → 19 维衍生指标，全部 shift(1)
    │
    ▼
signal_model.py → XGBoost fit（仅 train 部分）
    │           → predict_proba 写入 Signal_Proba 列
    ▼
CSV 数据集（full / train80 / oos20）
```

### 4.2 现有数据文件

| 文件 | 时间范围 | 用途 |
|------|---------|------|
| `data_moe_20200101_20260216_full.csv` | 2020-01-01 ~ 2026-02-16 | 主数据集 |
| `data_moe_20200101_20260216_train80.csv` | 2020-07-19 ~ 约 2025-01 | Stage1/2 训练 |
| `data_moe_20200101_20260216_oos20.csv` | 约 2025-01 ~ 2026-02-16 | 样本外测试 |
| `data_moe_20200101_20260222_oos20.csv` | 约 2025-01 ~ 2026-02-22 | 扩展 OOS |
| `data_moe_20251214_20260216_eval.csv` | 2025-12-14 ~ 2026-02-16 | 近 2 月窗口 |

### 4.3 数据版本管理

`data_versioning.py` 对每个数据集文件生成 SHA256 哈希，记录到 `quant_docs/data_versions.csv`，确保实验可复现。

### 4.4 实盘数据加载

`okx_data_loader.py` 每日拉取最新 500 根 K 线，经过相同的特征工程管线，与训练时特征分布保持一致。

---

## 5. 特征工程

**文件：** [crypto_trader/features.py](crypto_trader/features.py)

**核心原则：所有 rolling 指标计算后执行 `shift(1)`，t 时刻只用 t-1 及以前的数据，彻底避免 Look-Ahead Bias。**

### 5.1 完整特征列表

基于 `asset_profile.py` 的 ETH 配置生成：

| 特征名 | 窗口 | 说明 |
|--------|:----:|------|
| RSI | 14 | 相对强弱（昨日收盘） |
| MACD | 12/26 | EMA(12) - EMA(26) |
| MACD_Signal | 9 | MACD 9 日信号线 |
| MACD_Pct | — | MACD / 昨日收盘 |
| BB_Upper / BB_Lower / BB_Width | 20 | 布林带（+2σ / -2σ）及宽度 |
| BB_Width_Pct | — | 布林带宽度百分比 |
| Log_Returns | 1 | ln(Close_t / Close_{t-1}) |
| ATR | 14 | 平均真实波幅 |
| SMA_50 / SMA_200 | 50 / 200 | 简单移动均线 |
| Dist_SMA_200 | — | (Close - SMA_200) / SMA_200 |
| Vol_Ratio | 20 | 成交量 / 20日均量 |
| Rolling_Vol | 20 | 对数收益 20 日滚动标准差 |
| Ret_Lag_1/2/3 | — | 滞后 1~3 日收益 |
| RSI_Lag_1/2/3 | — | 滞后 1~3 日 RSI |
| ROC_3/5/10 | 3/5/10 | 动量（n 日价格变化率） |
| Range_Pct | — | (High - Low) / Close（昨日） |
| Volatility_Regime | 20 | Rolling_Vol / 20 日中位数 |
| **Signal_Proba** | — | XGBoost 预测的下一 K 线上涨概率 |

### 5.2 观测空间（RL 使用 13 维）

```python
obs = [pos, cooldown_remaining, unrealized_pnl_pct, nw_change_pct,
       Signal_Proba, RSI/100, Rolling_Vol, MACD/100, BB_Width/1000,
       Dist_SMA_200, ATR/Close, Vol_Ratio, direction]
# index: 0,   1,               2,                  3,
#        4,            5,        6,           7,          8,
#        9,            10,        11,          12
```

---

## 6. XGBoost 信号模型

**文件：** [crypto_trader/models/signal_model.py](crypto_trader/models/signal_model.py)

### 6.1 定位

XGBoost **不直接产生交易指令**，仅预测「下一 K 线收盘价是否高于当前」的概率，作为 RL 观测空间第 4 维（`Signal_Proba`）。

- XGBoost 擅长：从非线性技术指标中提纯信号（OOS 准确率约 54.55%）
- RL 擅长：多步博弈、动态仓位、风险控制
- 两者互补：XGBoost 提供概率情报，RL 做最终决策

### 6.2 超参数

```python
XGBClassifier(
    n_estimators    = 200,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree = 0.8,
    random_state    = 42,
    eval_metric     = 'logloss',
)
```

### 6.3 训练规则

- **严格时序切分：** 仅在 train80 内部 80% 上 fit，余下 20% 内部验证
- **禁止 Shuffle：** 不随机打乱行，防止时序泄漏
- **OOS 中的 Signal_Proba：** 由仅在训练集学习的模型生成，无数据泄漏
- **输出：** `predict_proba()[:, 1]` → 上涨概率 [0, 1]

### 6.4 单独诊断

```bash
PYTHONPATH=. python crypto_trader/scripts/eval_xgboost_accuracy.py \
    --data-path crypto_trader/data_moe_20200101_20260216_oos20.csv
```

---

## 7. SimpleAlpha 简化策略

**文件：** [crypto_trader/strategies/simple_alpha.py](crypto_trader/strategies/simple_alpha.py)

不依赖 Stable-Baselines3 / GPU 的轻量策略，作为 MoE 的可解释基准。

### 7.1 决策公式

```python
# 1. 信号分数（XGBoost 输出）
signal_score = clip(2.0 * (Signal_Proba - 0.5), -1.0, 1.0)

# 2. 趋势分数
trend_score = tanh(Dist_SMA_200 / 0.12)
macd_score  = tanh((MACD / ATR) / 0.08)
combined_trend = 0.7 × trend_score + 0.3 × macd_score

# 3. 合成得分
raw_score = signal_weight × signal_score + trend_weight × combined_trend

# 4. 阈值过滤（|raw_score| < threshold → 空仓）
# 5. 波动率缩放
vol_scale = clip(target_vol / Rolling_Vol, min_vol_scale, max_vol_scale)
target_pos = clip(raw_score × max_position × vol_scale, -max_pos, +max_pos)
```

### 7.2 默认参数

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `signal_weight` | 0.6 | XGBoost 信号权重 |
| `trend_weight` | 0.4 | 趋势权重 |
| `threshold` | 0.08 | 合成得分最低阈值，低于则空仓 |
| `max_position` | 0.8 | 最大仓位上限 |
| `target_vol` | 0.035 | 波动率目标（3.5%） |
| `min_vol_scale` | 0.2 | 波动率缩放下限 |
| `max_vol_scale` | 1.2 | 波动率缩放上限 |
| `tau` | 0.15 | 迟滞阈值（小于不执行） |
| `delta_max` | 0.25 | 单步最大仓位变化 |
| `fee_rate` | 0.0008 | 单边手续费 0.08% |
| `funding_daily` | 0.0003 | 日化资金费率 0.03% |

### 7.3 参数自动选择

`choose_params()` 在训练集上执行候选参数网格搜索，按训练集夏普最高选参，再在测试集评估，不存在未来函数。

### 7.4 Walk-Forward 运行

```bash
PYTHONPATH=. python -m crypto_trader.strategies.simple_alpha_walk_forward \
    --data-path crypto_trader/data_moe_20200101_20260216_full.csv \
    --output-dir results/simple_alpha_walk_forward
```

4 折结构：每折重新训练 XGBoost → 选参 → 测试集回测，输出 `walk_forward_metrics.csv` 和 `summary.json`。

---

## 8. MoE 混合专家系统（4 专家版）

### 8.1 当前专家配置

**文件：** [crypto_trader/configs/moe_experts.yaml](crypto_trader/configs/moe_experts.yaml)

| 专家 ID | 算法 | 训练切片 | Feature Mask | Reward 重点 | 训练步数 |
|---------|------|---------|:------------:|------------|:-------:|
| **E2_PPO_bear_drawdown** | PPO | bear（动量 < 35 分位） | risk | drawdown 1.60 / sortino 1.20 | 150K |
| **E4_PPO_highvol_risk** | PPO | high_vol（ATR > 70 分位） | risk | drawdown 1.50 / sortino 1.10 | 150K |
| **E5_PPO_lowvol_carry** | PPO | low_vol（ATR < 30 分位） | carry | 均衡（各 1.00） | 150K |
| **E7_SAC_fast_adapt** | SAC | range（波动 < 40 分位） | switch | drawdown 1.10 / sortino 0.90 | 180K |

### 8.2 Feature Mask 定义

**文件：** [crypto_trader/moe/manifest.py](crypto_trader/moe/manifest.py)

每个专家只能看到观测空间的一个子集（按 Index）：

```python
FEATURE_MASKS = {
    "all":    list(range(13)),                        # 全部 13 维
    "trend":  [0, 3, 4, 5, 7, 9, 10, 12],            # 趋势：Signal/RSI/MACD/Dist_SMA200
    "risk":   [0, 1, 3, 6, 8, 10, 11, 12],           # 风险：Cooldown/Vol/BB_Width/ATR
    "carry":  [0, 3, 4, 6, 10, 11, 12],              # 套利：Signal/Vol/ATR/Vol_Ratio
    "switch": [0, 1, 3, 4, 6, 7, 11, 12],            # 切换：Cooldown/Signal/MACD/Vol_Ratio
}
```

当前 4 专家使用的 Mask：E2 → risk，E4 → risk，E5 → carry，E7 → switch

### 8.3 市场状态切片

**文件：** [crypto_trader/moe/regime.py](crypto_trader/moe/regime.py)

Stage1 训练时，用 20 日动量（ROC_20）和 ATR% 对历史数据做状态标注与切片：

| 切片 | 筛选条件 | 说明 |
|------|---------|------|
| `bear` | 动量 < 35 分位 | 下跌行情（E2 专用） |
| `high_vol` | ATR% > 70 分位 | 高波动行情（E4 专用） |
| `low_vol` | ATR% < 30 分位 | 低波动行情（E5 专用） |
| `range` | 波动绝对值 < 40 分位 | 震荡整理（E7 专用） |

切片为空时自动 fallback 到全量数据。

### 8.4 Gate 路由机制

**文件：** [crypto_trader/train_moe_stage2_gate.py](crypto_trader/train_moe_stage2_gate.py)

```
obs (13维)
  → Gate PPO 网络 → logits (4维)
  → softmax(logits / temperature)  → 权重 w = [w_E2, w_E4, w_E5, w_E7]
  → 各专家用 Masked obs 独立推理   → 动作 a_i ∈ [-1, 1]
  → 加权融合: a_mix = Σ(w_i × a_i), clip to [-1, 1]
  → TradingEnv 执行约束 → 落地仓位
```

**Gate 奖励函数：**

```python
gate_reward = reward
            - 0.02 × balance_penalty   # 权重偏离均匀分布的 MSE（防单专家垄断）
            + 0.01 × diversity_bonus   # 专家动作标准差（鼓励多样性）
```

**Gate Temperature：**  
Walk-Forward 对每折独立扫描 `[0.5, 0.6, 0.68, 0.8, 1.0, 1.5, 2.0]` 并选出 val 集最优温度：
- fold_1 → 2.0（权重较分散）
- fold_2 → 0.68
- fold_3 → 0.8
- fold_4 → 0.5
- fold_5 → 0.5（果断选主导专家）

### 8.5 两阶段训练

| 阶段 | 文件 | 训练对象 | 冻结对象 | 说明 |
|------|------|---------|---------|------|
| **Stage 1** | `train_moe_stage1.py` | 4 个专家各自独立 | 无 | 在各自的切片数据上训练 |
| **Stage 2** | `train_moe_stage2_gate.py` | Gate PPO 网络 | 全部专家权重 | 在 train80 全量上学习路由 |

> Stage3 交替联合微调脚本（`train_moe_stage3_joint.py`）保留为可选优化步骤，当前主线验证以两阶段为准。

### 8.6 Walk-Forward 专家配置（moe_config.py）

Walk-Forward 中专家配置与 `moe_experts.yaml` 完全一致（同为 E2/E4/E5/E7），区别仅在于：每折独立训练新的专家和 Gate，不复用 stable 目录的模型。

---

## 9. 交易环境与执行约束

**文件：** [crypto_trader/envs/trading_env.py](crypto_trader/envs/trading_env.py)

### 9.1 观测空间（13 维）

| Index | 特征 | 范围 | 含义 |
|:-----:|------|------|------|
| 0 | `pos` | [-1, 1] | 当前持仓比例 |
| 1 | `cooldown_remaining` | [0, 1] | 冷却期剩余（1=刚反转） |
| 2 | `unrealized_pnl_pct` | 浮点 | 未实现盈亏率 |
| 3 | `nw_change_pct` | 浮点 | 上一步净值变化率 |
| 4 | `Signal_Proba` | [0, 1] | XGBoost 上涨概率 |
| 5 | `RSI / 100` | [0, 1] | 归一化 RSI |
| 6 | `Rolling_Vol` | 浮点 | 20 日滚动波动率 |
| 7 | `MACD / 100` | 浮点 | 归一化 MACD |
| 8 | `BB_Width / 1000` | 浮点 | 归一化布林带宽度 |
| 9 | `Dist_SMA_200` | 浮点 | 距 200 日均线乖离率 |
| 10 | `ATR / Close` | 浮点 | ATR 占收盘价比例 |
| 11 | `Vol_Ratio` | 浮点 | 成交量比率 |
| 12 | `direction` | {-1, 0, 1} | 当前持仓方向 |

### 9.2 动作空间

单一连续值 `[-1, 1]`，表示**目标仓位意图**，不是直接下单量，需经过四重约束后才落地。

### 9.3 四重执行约束

**实现：** `apply_execution_constraints_core()` — 训练和实盘共享同一函数，保证一致性

| 约束 | 参数 | ETH 默认值 | 作用 |
|------|------|:----------:|------|
| **Hysteresis（迟滞）** | `tau` | 0.25 | 变化量 < tau → 维持原仓，过滤噪音 |
| **Slew-rate（变速限制）** | `delta_max` | 0.15 | 单步最大仓位变化，防满仓反手 |
| **Cooldown（冷却期）** | `cooldown_n` | 3 天 | 多空反转后强制归零 3 天 |
| **Clip（硬性边界）** | — | [-1, 1] | 仓位绝对上下界 |

### 9.4 成本建模

| 成本项 | 参数 | 值 |
|--------|------|----|
| 单边手续费 | `k_single` | 0.08% |
| 日化资金费率 | `funding_daily` | 0.03% |
| 波动率目标 | `target_atr_pct` | 5% |
| ATR 地板 | `atr_floor` | 0.5%（防过度加杠杆） |
| 杠杆范围 | `vol_scale_min / max` | 0.1x ~ 2.0x |

### 9.5 Reward 函数

```python
reward = profile.return   × log_return
       + profile.sortino  × sortino_component
       - profile.drawdown × drawdown_penalty
       - profile.turnover × turnover_cost
```

各专家的 `reward_profile` 权重不同，产生不同交易风格（E2/E4 偏重 drawdown 惩罚，E5/E7 均衡）。

---

## 10. 风控与执行安全

### 10.1 RiskManager（三级限仓）

**文件：** [crypto_trader/risk_manager.py](crypto_trader/risk_manager.py)

| 回撤阈值 | 仓位上限 | 模式 |
|:--------:|:-------:|------|
| < 5% | 100% | 正常 |
| ≥ 5% | 80% | Tier 1 轻度收缩 |
| ≥ 10% | 50% | Tier 2 中度收缩 |
| ≥ 15% | 20% | 保命模式 |

> 软性上限：保留方向，不强制平仓；最大回撤阈值默认 10%，可通过配置调整。

### 10.2 ExecutionSafety（执行安全）

**文件：** [crypto_trader/execution_safety.py](crypto_trader/execution_safety.py)

| 能力 | 实现 |
|------|------|
| 幂等下单 | 每笔交易分配唯一 UUID，防止网络重试重复下单 |
| 状态机 | `pending → submitted → filled / failed` 持久化 |
| 持仓对账 | 执行前对比交易所实际持仓与 `trading_state.json` |
| SAFE_MODE | 环境变量触发，禁止开新仓，仅允许减仓 |
| API 健康检测 | 连续失败次数 + 时钟漂移监控 |
| 滑点校验 | 成交价与预期价差超阈值（3%）触发告警 |

### 10.3 RolloutController（渐进放量）

**文件：** [crypto_trader/rollout_controller.py](crypto_trader/rollout_controller.py)

候选模型上线按仓位乘数分阶段放量，不直接全量切换：

```
0.25x → 0.5x → 1.0x
（每级至少 3 天 / 6 笔交易，KPI 不达标自动降级）
```

KPI 指标：滑点 < 0.5%，订单成交率 > 90%，对账错误 < 2 次。

### 10.4 告警（多渠道）

**文件：** [crypto_trader/alerting.py](crypto_trader/alerting.py)

支持企业微信 / 钉钉 / 飞书 Webhook，告警级别：

| 级别 | 触发条件 |
|------|---------|
| INFO | 每日执行摘要、仓位变更 |
| WARNING | 回撤触及 5% / 10%，API 异常 |
| CRITICAL | 回撤 ≥ 15% 保命模式，权益异常波动 > 5% |

---

## 11. Walk-Forward 样本外验证结果

### 11.1 MoE Walk-Forward（5 折，当前版本）

**目录：** `crypto_trader/walk_forward/results/walk_forward_moe/`

5 折锚定式（Expanding Window），训练集起点固定为 2020-01-01：

| Fold | 训练截止 | 测试期 | Gate 温度 |
|------|---------|--------|:---------:|
| fold_1 | 2021-12-31 | 2022 全年 | 2.0 |
| fold_2 | 2022-12-31 | 2023 全年 | 0.68 |
| fold_3 | 2023-12-31 | 2024 全年 | 0.8 |
| fold_4 | 2024-12-31 | 2025 全年 | 0.5 |
| fold_5 | 2025-12-31 | 2026-01~05 | 0.5 |

各折 Gate 权重（反映市场切换规律）：

| Fold | E2 熊市 | E4 高波 | E5 低波 | E7 切换 |
|------|:-------:|:-------:|:-------:|:-------:|
| fold_1（2022 熊市） | 10.5% | **29.9%** | **42.6%** | 17.1% |
| fold_2（2023 牛市） | 5.9% | 9.1% | **48.5%** | **36.5%** |
| fold_3（2024 牛市） | 18.4% | 26.6% | 21.0% | **34.0%** |
| fold_4（2025 震荡） | **42.4%** | 6.9% | 26.8% | 23.8% |
| fold_5（2026 下跌） | 20.0% | 10.7% | **37.8%** | 31.5% |

### 11.2 SimpleAlpha Walk-Forward（4 折）

**文件：** `crypto_trader/strategies/simple_alpha_walk_forward.py`

| Fold | 训练截止 | 测试期 |
|------|---------|--------|
| fold1_test2022 | 2021-12-31 | 2022 全年 |
| fold2_test2023 | 2022-12-31 | 2023 全年 |
| fold3_test2024 | 2023-12-31 | 2024 全年 |
| fold4_test2025 | 2024-12-31 | 2025 全年 |

每折：在训练集上训练 XGBoost + 参数网格选优 → 在测试集上评估，输出 `walk_forward_metrics.csv` + `summary.json`。

---

## 12. 实盘执行链路

### 12.1 入口文件

| 文件 | 用途 |
|------|------|
| `run_live.py` | 交互式启动（含二次确认） |
| `run_demo.py` | 模拟盘（加载 `.env.demo`） |
| `run_real.py` | 真实盘（加载 `.env.live`） |
| `live_trading_okx.py` | 主执行引擎（~1400 行） |

### 12.2 每日执行流程（UTC 08:05）

```
① 防重复执行（lock file 检查）
② 数据就绪检查（最新 K 线已收盘）
③ 风控巡检（账户权益、持仓对账）
④ 拉取最新 500 天 K 线（OKX REST）
⑤ 特征工程（与训练管线完全一致）
⑥ SignalPredictor 推理 → Signal_Proba
⑦ 构造 13 维观测
⑧ Gate → 4 专家 Masked 推理 → 加权融合动作
⑨ 波动率缩放（ATR% 目标制）
⑩ RiskManager 分层限仓
⑪ 四重执行约束（Hysteresis → SlewRate → Cooldown → Clip）
⑫ 计算目标合约数量
⑬ OKX API 下单（限价单 + IOC）
⑭ 滑点校验 → 持仓对账 → 状态写入 trading_state.json
⑮ 日志记录 → Webhook 告警
```

### 12.3 状态持久化

`trading_state.json` 记录每日执行后的关键状态：

```json
{
  "current_position": 0.5,
  "last_trade_date": "2026-05-01",
  "cooldown_remaining": 0,
  "peak_net_worth": 12543.21,
  "current_net_worth": 11832.10,
  "current_drawdown": 0.0567
}
```

---

## 13. 调度与运维

### 13.1 macOS launchd（推荐）

```bash
bash scripts/setup_trading_launchd.sh    # 配置交易 daemon
bash scripts/setup_monitoring_launchd.sh # 配置监控 daemon
bash scripts/migrate_cron_to_launchd.sh  # 从 cron 迁移
```

### 13.2 cron 方式

```bash
bash scripts/setup_cron.sh               # 写入 cron（UTC 08:05 每日）
bash scripts/setup_monitoring_cron.sh    # 写入监控 cron
```

### 13.3 本地手动操作

```bash
bash scripts/start_background.sh         # 后台启动
bash scripts/check_status.sh             # 查看状态 + 近期日志
bash scripts/stop_background.sh          # 停止
bash scripts/health_check_stable.sh      # 模型完整性一键检查
```

---

## 14. 监控与告警

```bash
# 日监控（近 7 日执行与收益）
bash scripts/monitor_effectiveness_daily.sh

# 周报告（近 30 日整体表现）
bash scripts/monitor_effectiveness_weekly.sh

# 检查今日是否已正常执行
PYTHONPATH=. python scripts/check_daily_execution.py
```

日志文件：

| 文件 | 内容 |
|------|------|
| `logs/daemon.log` | 主守护进程日志 |
| `logs/monitor.log` | 监控日志 |
| `logs/daily_YYYYMMDD.log` | 每日执行详情 |
| `logs/effectiveness_daily.log` | 日策略有效性 |
| `logs/effectiveness_weekly.log` | 周策略有效性 |

---

## 15. 配置参考

### 15.1 ETH 资产配置（asset_profile.py）

**特征窗口：**

| 参数 | 值 |
|------|----|
| RSI 窗口 | 14 |
| MACD | 12/26/9 |
| 布林带 | 20 日，2σ |
| ATR 窗口 | 14 |
| SMA | 50 / 200 |
| 成交量均线 | 20 |
| 滚动波动率 | 20 |

**环境约束：**

| 参数 | 值 |
|------|----|
| `tau`（迟滞） | 0.25 |
| `delta_max`（变速限制） | 0.15 |
| `cooldown_n`（冷却期） | 3 天 |
| `k_single`（手续费） | 0.08% |
| `funding_daily`（资金费） | 0.03% |
| `target_atr_pct` | 5% |
| `atr_floor` | 0.5% |
| `vol_scale_min / max` | 0.1x / 2.0x |

### 15.2 PPO 超参数（config.py）

| 参数 | 值 |
|------|----|
| `learning_rate` | 3e-4 |
| `gamma` | 0.995 |
| `n_steps` | 2048 |
| `batch_size` | 256 |
| `ent_coef` | 0.005 |
| `clip_range` | 0.2 |
| `gae_lambda` | 0.95 |

SAC（E7）使用相同的 `learning_rate` / `gamma` / `batch_size`，每步训练一次（`train_freq=(1,"step")`）。

### 15.3 风控阈值（config.py RiskConfig）

| 参数 | 值 |
|------|----|
| `tier1_drawdown / limit` | 5% / 80% |
| `tier2_drawdown / limit` | 10% / 50% |
| `survival_drawdown / limit` | 15% / 20% |
| `max_drawdown_limit` | 10%（总熔断阈值） |
| `max_slippage_risk_on` | 3% |

### 15.4 环境变量（.env）

```bash
OKX_API_KEY=...
OKX_SECRET_KEY=...
OKX_PASSPHRASE=...
OKX_SANDBOX=false          # true=沙盒
SAFE_MODE=false            # true=禁止开新仓
WECOM_WEBHOOK_URL=...      # 企业微信（三选一）
DINGTALK_WEBHOOK_URL=...   # 钉钉
FEISHU_WEBHOOK_URL=...     # 飞书
```

---

## 16. 测试体系

```bash
# 运行全部测试
PYTHONPATH=. pytest crypto_trader/tests/ -v

# 带覆盖率
PYTHONPATH=. pytest crypto_trader/tests/ --cov=crypto_trader --cov-report=term
```

| 测试文件 | 覆盖模块 |
|---------|---------|
| `test_moe_stage1_spec.py` | Stage1 训练规格 |
| `test_moe_stage2_stage3_spec.py` | Stage2/3 门控规格 |
| `test_moe_manifest.py` | 专家配置解析 & Feature Mask |
| `test_moe_regime.py` | 市场切片逻辑 |
| `test_moe_walk_forward_runtime.py` | MoE Walk-Forward 运行时 |
| `test_backtest_moe_overrides.py` | 回测参数覆盖 |
| `test_backtest_moe_utils.py` | 回测工具函数 |
| `test_execution_safety_local_position.py` | 本地仓位状态机 |
| `test_execution_safety_recovery.py` | 执行安全恢复 |
| `test_live_daily_bar_alignment.py` | 实盘日 K 线对齐 |
| `test_reconcile_drill.py` | 持仓对账钻取 |
| `test_runtime_parity.py` | 训练/推理时序一致性 |
| `test_signal_model_runtime_alignment.py` | 信号模型运行时对齐 |
| `test_simple_alpha_strategy.py` | SimpleAlpha 策略逻辑 |
| `test_simple_alpha_walk_forward.py` | SimpleAlpha Walk-Forward |
| `test_data_versioning.py` | 数据版本管理 |
| `test_experiment_log_writer.py` | 实验日志写入 |
| `test_trading_env_diversity_hooks.py` | 交易环境多样性钩子 |
| `test_validation_metrics.py` | 验证指标计算 |
| `test_validation_runner_config.py` | 验证配置 |
| `test_validation_verdicts.py` | 验证判定逻辑 |
| `test_walk_forward_metrics.py` | Walk-Forward 指标 |

---

## 17. 技术栈

| 类别 | 库 | 版本要求 |
|------|-----|:-------:|
| RL 框架 | stable-baselines3 | ≥ 2.0.0 |
| RL 环境 | gymnasium | ≥ 0.29.0 |
| 深度学习 | PyTorch | ≥ 2.0.0 |
| 信号预测 | xgboost | ≥ 2.0.0 |
| 预处理 | scikit-learn | ≥ 1.3.0 |
| 数据处理 | pandas / numpy | ≥ 2.0.0 / ≥ 1.24.0 |
| 交易所 API | ccxt | ≥ 4.0.0 |
| 配置管理 | python-dotenv | ≥ 1.0.0 |
| 可视化 | matplotlib | ≥ 3.7.0 |
| 进度条 | tqdm | ≥ 4.65.0 |

**模型文件格式：**

| 扩展名 | 内容 |
|--------|------|
| `*.zip` | SB3 模型权重（PPO / SAC） |
| `*.pkl`（walk_forward/checkpoints） | VecNormalize 归一化状态 |
| `*.pkl`（checkpoints/） | XGBoost 信号模型 |
| `*.json` | 配置元数据 / 实验结果 |
| `*.yaml` | 专家配置清单 |

---

## 18. 常见问题排查

### Q1：实盘执行后仓位未变化

1. 查 `logs/daemon.log` 是否有 `Hysteresis filtered` → 信号变化量 < tau=0.25，正常过滤
2. 检查 `SAFE_MODE=true` 是否被意外设置
3. 查 `trading_state.json` 中 `current_drawdown` 是否触发 RiskManager 限仓
4. 检查 `cooldown_remaining` 是否处于冷却期

### Q2：XGBoost 准确率异常

```bash
PYTHONPATH=. python crypto_trader/scripts/eval_xgboost_accuracy.py \
    --data-path crypto_trader/data_moe_20200101_20260216_oos20.csv
```

- 52%~56% 为正常区间（加密市场高噪声）
- 低于 50% → 检查 `shift(1)` 是否生效，排查时序泄漏

### Q3：回测与实盘结果差异大

1. 数据更新后 Signal_Proba 分布漂移 → 重新运行 `build_moe_dataset.py`
2. 实盘/回测数据 padding 方式不一致 → 检查 `okx_data_loader.py` 的填充逻辑
3. Gate temperature 不一致 → 确认实盘与 Walk-Forward 使用相同温度（当前 0.5）

### Q4：Gate 权重长期集中在单一专家

- 当前 E5 权重在多折中偏高（37~48%），属于低波动环境下 Gate 的合理选择
- 如需更均衡：调高 `gate_load_balance_coef`（默认 0.02）或升高温度

### Q5：Walk-Forward 训练 OOM

```bash
# 减少并行进程
python -m crypto_trader.walk_forward.moe_walk_forward --n-jobs 2

# 减少训练步数
python -m crypto_trader.walk_forward.moe_walk_forward --expert-timesteps 50000

# Smoke 模式快速验证
python -m crypto_trader.walk_forward.moe_walk_forward --smoke-test
```

### Q6：OKX API 连接失败

1. 检查 `.env` API Key 配置
2. 确认 OKX IP 白名单
3. 查看 `execution_safety.py` 中 API 健康检测报告
4. 网络需代理时，在 CCXT 初始化中配置 `proxies`
