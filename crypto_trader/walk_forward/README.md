# Walk-Forward MoE 滚动训练与回测

这是当前项目的样本外验证模块，主线为 **5 折 anchored walk-forward + 4 专家 MoE**。它用于验证 `crypto_trader/configs/moe_experts.yaml` 中的稳定专家组合，不影响 `checkpoints/moe/stable/` 下的生产模型。

## 当前专家

| 专家 | 角色 |
|------|------|
| `E2_PPO_bear_drawdown` | 熊市与回撤控制 |
| `E4_PPO_highvol_risk` | 高波动风险控制 |
| `E5_PPO_lowvol_carry` | 低波动 carry |
| `E7_SAC_fast_adapt` | 快速适应 |

## 时间分割

| Fold | 训练期 | 测试期 |
|------|--------|--------|
| `fold_1` | 2020-01-01 至 2021-12-31 | 2022-01-01 至 2022-12-31 |
| `fold_2` | 2020-01-01 至 2022-12-31 | 2023-01-01 至 2023-12-31 |
| `fold_3` | 2020-01-01 至 2023-12-31 | 2024-01-01 至 2024-12-31 |
| `fold_4` | 2020-01-01 至 2024-12-31 | 2025-01-01 至 2025-12-31 |
| `fold_5` | 2020-01-01 至 2025-12-31 | 2026-01-01 至 2026-12-31 |

## 使用方法

从项目根目录运行：

```bash
PYTHONPATH=. python -m crypto_trader.walk_forward.moe_walk_forward
```

回测单折已训练结果时，优先使用项目根目录的 MoE 回测入口。公开仓库默认不提交
`checkpoints/` 和 `results/` 下的 fold 级运行产物，需要先运行上面的 walk-forward
命令重新生成：

```bash
PYTHONPATH=. python -m crypto_trader.backtest_moe \
  --manifest crypto_trader/configs/moe_experts.yaml \
  --stage1-root crypto_trader/walk_forward/checkpoints/walk_forward_moe/fold_5/experts \
  --stage2-root crypto_trader/walk_forward/checkpoints/walk_forward_moe/fold_5/gate \
  --data-path crypto_trader/walk_forward/results/walk_forward_moe/fold_5/data/test_prepared.csv \
  --execution-mode next_bar
```

## 当前结果目录

```text
crypto_trader/walk_forward/
├── moe_config.py                         # 5 折 x 4 专家配置
├── moe_walk_forward.py                   # MoE Walk-Forward 编排器
├── expert_trainer.py                     # 专家折叠训练
├── gate_trainer.py                       # Gate 折叠训练
├── checkpoints/                          # 本地生成；不进入 Git
└── results/                              # 本地生成；摘要见 docs/WALK_FORWARD_SUMMARY.md
```

旧的单 PPO walk-forward 文件仍保留为兼容/历史基线，不再作为项目主入口。
