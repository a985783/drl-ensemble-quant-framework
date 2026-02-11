# ETH 智能量化交易系统

基于深度强化学习的加密货币永续合约自动化交易系统，专为 OKX 交易所设计。

## 核心特性

- **20 模型集成投票**: 使用 PPO 强化学习算法训练 20 个独立模型，通过平均投票消除单一模型偏差
- **双层信号架构**: XGBoost 趋势预测 + PPO 仓位决策
- **四重执行约束**: 滞回阈值、变速限制、冷却期、风控分级
- **机构级风控**: 三档回撤管理 (5%/10%/15%)、SAFE_MODE 安全模式、实时对账
- **智能订单执行**: Limit-then-Market 策略，优先限价单降低滑点，超时自动转市价

## 技术栈

| 组件 | 技术 |
|------|------|
| 强化学习 | PPO (Stable-Baselines3) |
| 信号模型 | XGBoost |
| 交易接口 | CCXT (OKX) |
| 数据处理 | Pandas, NumPy |
| 特征工程 | RSI, MACD, Bollinger, ATR, Volume Ratio |

## 项目结构

```
crypto_trader/
├── envs/trading_env.py          # 交易环境 (Gymnasium)
├── models/signal_model.py       # XGBoost 信号模型
├── features.py                  # 特征工程 (防未来函数)
├── risk_manager.py              # 三档风控管理器
├── execution_safety.py          # 执行安全层 (幂等性/对账/SAFE_MODE)
├── rollout_controller.py        # 渐进式部署控制器
├── alerting.py                  # 告警系统 (钉钉/企微/飞书)
├── data_loader.py               # OKX 数据获取
├── data_validator.py            # 数据质量校验
├── analytics/performance.py     # 盈亏归因分析
├── train_ensemble.py            # 集成模型训练
├── backtest_ensemble.py         # 集成回测
├── backtest_recent.py           # 最新时段回测
├── live_trading_okx.py          # 实盘交易引擎
└── walk_forward/                # Walk-Forward 滚动验证
    ├── train_walk_forward.py
    └── backtest_walk_forward.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# OKX API 配置
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase
OKX_DEMO_MODE=True  # 模拟盘模式

# 告警配置 (可选)
ALERT_PROVIDER=dingtalk  # 或 wecom / feishu
ALERT_WEBHOOK_URL=https://oapi.dingtalk.com/robot/send?access_token=xxx
```

### 3. 训练模型

```bash
cd crypto_trader
python train_ensemble.py
```

训练完成后，模型保存在 `checkpoints/ensemble/`。

### 4. 运行回测

```bash
python backtest_ensemble.py
```

### 5. 启动实盘 (模拟盘)

```bash
python live_trading_okx.py --auto
```

## 交易策略详解

### 状态空间 (13维)

| 特征 | 说明 |
|------|------|
| Position | 当前持仓比例 (-1 ~ 1) |
| Cooldown Remaining | 冷却期剩余比例 |
| Unrealized PnL | 未实现盈亏 |
| Net Worth Change | 净值变化率 |
| Signal Probability | XGBoost 涨跌预测概率 |
| RSI | 相对强弱指数 (归一化) |
| Rolling Volatility | 滚动波动率 |
| MACD | 移动平均收敛散度 |
| Bollinger Width | 布林带宽度 |
| Distance to SMA200 | 距200日均线距离 |
| ATR | 平均真实波幅 |
| Volume Ratio | 成交量比率 |
| Position Direction | 当前持仓方向 |

### 动作空间

连续值 `[-1, +1]`:
- `-1` = 全仓做空
- `0` = 空仓观望
- `+1` = 全仓做多

### 执行约束

1. **滞回阈值 (Hysteresis)**: 目标仓位与当前仓位差值 < 0.25 时不触发交易
2. **变速限制 (Slew Rate)**: 单次最大仓位变化 ±0.15
3. **冷却期 (Cooldown)**: 反向交易后 3 天内禁止再次反向
4. **波动率缩放**: 根据 ATR 动态调整目标仓位

### 风险管理

| 回撤水平 | 仓位上限 | 说明 |
|----------|----------|------|
| < 5% | 100% | 正常运行 |
| 5% - 10% | 80% | 轻度防御 |
| 10% - 15% | 50% | 中度防御 |
| > 15% | 20% | 灾难保险模式 |

## 回测验证

### Walk-Forward 滚动验证

```bash
cd crypto_trader/walk_forward
python train_walk_forward.py      # 训练
python backtest_walk_forward.py   # 回测
```

时间分割:
- Fold 1: 训练 2020-2021 → 测试 2022
- Fold 2: 训练 2020-2022 → 测试 2023
- Fold 3: 训练 2020-2023 → 测试 2024
- Fold 4: 训练 2020-2024 → 测试 2025

### 最新时段回测

```bash
python backtest_recent.py
```

从测试集结束后 (2025-12-24) 至今的回测，用于验证模型在最新市场的表现。

## 实盘部署

### 执行安全机制

- **幂等性**: 相同操作 5 分钟内不会重复执行
- **对账系统**: 实时比对本地状态与交易所持仓
- **SAFE_MODE**: 异常时自动进入只减仓模式
- **时钟校验**: 检测本地与交易所时间漂移

### 渐进式部署 (Rollout)

新模型部署采用渐进式策略:
1. **Level 0.25**: 25% 仓位试运行 (至少 3 天)
2. **Level 0.5**: 50% 仓位验证 (至少 3 天)
3. **Level 1.0**: 全量部署或回滚

### 监控指标

| 指标 | 阈值 | 动作 |
|------|------|------|
| API 连续失败 | >= 3 次 | 进入 SAFE_MODE |
| 时钟漂移 | > 30 秒 | 进入 SAFE_MODE |
| 持仓对账差异 | > 5% | 进入 SAFE_MODE |
| 平均滑点 | > 0.5% | 降级部署 |
| 成交率 | < 90% | 降级部署 |

### 定时运行

使用 `cron` 设置每日自动执行:

```bash
# 编辑 crontab
crontab -e

# 每天 UTC 00:00 执行
0 0 * * * cd /path/to/project && /usr/bin/python3 crypto_trader/live_trading_okx.py --auto >> logs/trading.log 2>&1
```

## 性能指标

系统内置机构级归因分析:

```bash
python backtest_recent.py  # 自动生成归因报告
```

输出指标:
- **Alpha**: 策略超额收益 (年化)
- **Beta**: 市场敏感度
- **Sharpe Ratio**: 夏普比率
- **Information Ratio**: 信息比率
- **Max Drawdown**: 最大回撤
- **Calmar Ratio**: 收益/回撤比

## 安全警告

⚠️ **实盘交易前必须完成以下检查**:

1. 在 `.env.live` 中设置 `CONFIRM_REAL_MONEY=True`
2. 在模拟盘完整运行至少 7 天
3. 配置并测试告警系统
4. 检查 API Key 权限 (仅需交易权限，禁用提现)
5. 理解三档风控逻辑
6. 确保能承受最大 20% 回撤

## 日志与监控

```
runs/
├── live_status.json          # 实时状态快照
├── live_heartbeat.txt        # 心跳时间戳
└── trade_logs.csv            # 历史交易明细

trade_logs.csv 字段:
- Timestamp: 交易时间
- Raw_Signal: 模型原始输出
- Constraint_Reason: 约束触发原因
- Slippage: 实际滑点
- Safe_Mode: 安全模式状态
- Reconcile_Diff: 对账差异
```

## 配置说明

配置文件位于 `crypto_trader/configs/base.yaml`，支持以下参数:

```yaml
# 数据配置
data:
  symbol: "ETH/USDT:USDT"
  interval: "1d"
  train_start: "2020-01-01"
  train_end: "2025-12-15"

# 模型超参数
model:
  learning_rate: 3e-4
  gamma: 0.995
  n_steps: 2048
  batch_size: 256
  ent_coef: 0.005
  total_timesteps: 150000

# 风控参数
risk:
  max_drawdown_limit: 0.15
  tier1_drawdown: 0.05
  tier1_limit: 0.8
  tier2_drawdown: 0.10
  tier2_limit: 0.5
```

## 许可证

MIT License

---

**免责声明**: 本系统仅供学习和研究使用，不构成投资建议。加密货币交易具有高风险，可能导致全部本金损失。使用本系统进行实盘交易前，请充分了解相关风险。
