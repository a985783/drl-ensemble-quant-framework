# Walk-Forward 滚动训练与回测

独立的样本外验证系统，不影响现有模型。

## 使用方法

```bash
cd crypto_trader/walk_forward

# 1. 训练（约 30-60 分钟）
python3 train_walk_forward.py

# 2. 回测
python3 backtest_walk_forward.py
```

## 时间分割

| Fold | 训练期 | 测试期 |
|------|--------|--------|
| fold1 | 2020-2021 | 2022 |
| fold2 | 2020-2022 | 2023 |
| fold3 | 2020-2023 | 2024 |
| fold4 | 2020-2024 | 2025 |

## 优化配置

- **并行进程**: 4 (M3 Air 16GB)
- **每轮种子数**: 5 (减少以加快速度)
- **训练步数**: 50,000 (减少以加快速度)

## 文件结构

```
walk_forward/
├── train_walk_forward.py   # 训练脚本
├── backtest_walk_forward.py # 回测脚本
├── checkpoints/            # 模型存储
│   ├── fold1_test2022/
│   ├── fold2_test2023/
│   ├── fold3_test2024/
│   └── fold4_test2025/
└── results/                # 回测结果
```
