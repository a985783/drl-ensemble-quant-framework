# 🚀 ETH Intelligent Trading Bot | ETH 智能交易机器人

A production-grade cryptocurrency quantitative trading system based on Deep Reinforcement Learning (DRL) + XGBoost ensemble.

基于深度强化学习 (DRL) + XGBoost 集成的机构级加密货币量化交易系统。

---

## ⚠️ Disclaimer & Alpha Protection | 免责声明与 Alpha 保护

> [!IMPORTANT]
> **This repository provides the complete engineering framework and training pipeline.**
> Due to the temporal nature of quantitative strategies, pre-trained model weights and live trading parameters are NOT included in this open-source release. Users can train their own models using the provided scripts.
>
> **本仓库提供完整的工程架构与训练管线。**
> 由于量化策略的时效性，预训练模型权重与实盘参数不包含在开源库中。用户可使用提供的训练脚本自行训练。

> [!CAUTION]
> **Risk Warning**: Cryptocurrency trading involves substantial risk of loss. This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own due diligence before trading.
>
> **风险警告**：加密货币交易涉及重大亏损风险。本软件仅供教育和研究目的。过往业绩不保证未来收益。交易前请务必自行尽职调查。

---

## 📖 Overview | 项目简介

This project implements a full-stack quantitative trading system, codenamed **"Phase B+ Engine"**, designed for perpetual futures markets.

本项目实现了一个全栈量化交易系统，代号 **"Phase B+ Engine"**，专为永续合约市场设计。

### Key Features | 核心特性

| Feature | Description |
|---------|-------------|
| **20-Model Ensemble** | Wisdom of crowds eliminates single-model bias | 20模型投票消除单一模型偏见 |
| **Smart Execution** | Limit-then-Market order strategy reduces slippage | 限价优先市价补单，减少滑点 |
| **Tiered Risk Control** | Gradual position reduction based on drawdown | 基于回撤的分级降仓 |
| **Full Audit Trail** | Every decision logged with rationale | 每笔交易决策路径完整记录 |

---

## 🏗️ System Architecture | 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer | 数据层                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   OKX API    │───▸│  DataLoader  │───▸│  OHLCV Data  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Feature Layer | 特征层                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FeatureEngineer: RSI, MACD, BB, ATR, SMA, Vol...       │  │
│  │  (All with shift(1) to prevent look-ahead bias)         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Signal Layer | 信号层                           │
│  ┌──────────────┐         ┌─────────────────────────────────┐  │
│  │   XGBoost    │────┬───▸│      20x PPO Ensemble          │  │
│  │ SignalModel  │    │    │  (Different seeds, same data)  │  │
│  └──────────────┘    │    └─────────────────────────────────┘  │
│         │            │                    │                     │
│         ▼            ▼                    ▼                     │
│    [Probability]  [Features]    [Actions: -1 to +1]            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Execution Layer | 执行层                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TradingEnv with 4-Piece Constraints:                   │  │
│  │  1. Hysteresis (τ): Small changes filtered               │  │
│  │  2. Slew-Rate (δ): Max change per step                   │  │
│  │  3. Cooldown (N): Min interval between flips              │  │
│  │  4. Cost Model: Fee + Slippage + Funding                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ RiskManager  │───▸│   OKX API    │───▸│  Execution   │      │
│  │ (Tiered DD)  │    │ (Limit/Mkt)  │    │   & Logs     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure | 项目结构

```
├── config_template.yaml      # Configuration template (edit to customize)
├── config.yaml               # Your local config (gitignored)
├── .env.example              # API credential template
├── .env                      # Your credentials (gitignored)
├── requirements.txt          # Python dependencies
│
├── crypto_trader/
│   ├── config.py             # Configuration loader with fallback
│   ├── data_loader.py        # OKX/Yahoo historical data fetcher
│   ├── features.py           # Technical indicator engineering
│   ├── risk_manager.py       # Tiered drawdown-based limits
│   │
│   ├── envs/
│   │   └── trading_env.py    # RL environment with constraints
│   │
│   ├── models/
│   │   └── signal_model.py   # XGBoost direction predictor
│   │
│   ├── train_ensemble.py     # PPO ensemble training script
│   ├── backtest_ensemble.py  # Backtesting framework
│   └── live_trading_okx.py   # Live trading main script
│
└── checkpoints/              # Trained model storage (gitignored)
    └── ensemble/
        ├── ppo_seed_*.zip
        └── vec_norm_seed_*.pkl
```

---

## 🚀 Quick Start | 快速开始

### 1. Installation | 安装

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/crypto-trader.git
cd crypto-trader

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration | 配置

```bash
# Copy template files
cp config_template.yaml config.yaml
cp .env.example .env

# Edit config.yaml with your parameters
# Edit .env with your OKX API credentials
```

### 3. Train Models | 训练模型

```bash
cd crypto_trader
python train_ensemble.py
```

### 4. Backtest | 回测

```bash
python backtest_ensemble.py
```

### 5. Live Trading (Paper) | 模拟交易

```bash
# Ensure OKX_DEMO_MODE=True in .env
python live_trading_okx.py --auto
```

---

## 🧠 DRL State Space | 强化学习状态空间

The PPO agents receive a 13-dimensional observation:

| Dim | Feature | Range | Description |
|-----|---------|-------|-------------|
| 0 | `pos` | [-1, 1] | Current normalized position |
| 1 | `cooldown` | [0, 1] | Remaining cooldown ratio |
| 2 | `unrealized_pnl` | (-∞, ∞) | Unrealized profit/loss |
| 3 | `nw_change` | (-∞, ∞) | Net worth change ratio |
| 4 | `signal_proba` | [0, 1] | XGBoost UP probability |
| 5 | `rsi` | [0, 1] | RSI / 100 |
| 6 | `rolling_vol` | (0, ∞) | 20-day volatility |
| 7 | `macd` | (-∞, ∞) | MACD / 100 |
| 8 | `bb_width` | (0, ∞) | Bollinger Band width / 1000 |
| 9 | `dist_sma` | (-∞, ∞) | Distance to 200 SMA |
| 10 | `atr` | (0, ∞) | ATR / Close |
| 11 | `vol_ratio` | (0, ∞) | Volume / SMA(Volume) |
| 12 | `direction` | {-1, 0, 1} | Position direction |

---

## 🎛️ Configuration Reference | 配置参考

See `config_template.yaml` for all configurable parameters:

- **features**: Technical indicator periods (RSI, MACD, BB, ATR, SMA)
- **training**: PPO hyperparameters (learning_rate, gamma, n_steps, etc.)
- **environment**: Execution constraints (tau, delta_max, cooldown_n, costs)
- **risk**: Tiered position limits based on drawdown
- **signal_model**: XGBoost hyperparameters

---

## 📜 License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing | 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact | 联系方式

For questions and support, please open an issue on GitHub.

---

*Made with ❤️ for the Quant Community*
