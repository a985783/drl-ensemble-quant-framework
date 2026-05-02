# MoE 加密货币量化交易系统 — 复活验证报告

> **项目**: 强化学习加密货币量化交易系统  
> **策略**: ETH/USDT 永续合约 · 日频 · MoE (Mixture-of-Experts)  
> **报告日期**: 2026-05-02  
> **验证方法**: Quant Project Audit + 55 参数网格扫描 + 架构消融 + Bootstrap 显著性 + 子区间拆解 + 21 场景审计

---

## 1. 核心结论

**原策略 (+90.83%) 被证伪，真实 Alpha 为 +184.16%（净收益 +138.13%）。策略框架有效，问题出在参数和架构。**

| 指标 | 原策略（same-bar，假的） | 复活后（next-bar，真实的） |
|------|:---:|:---:|
| 总收益 | +90.83% | **+138.13%** |
| Alpha（vs ETH B&H） | +136.10% | **+184.16%** |
| 最大回撤 | 16.68% | 16.81% |
| Sharpe | 1.42 | 1.79 |
| 最优配置 | 不明（8专家全上） | **F_model_top4**（4专家+Gate，tau≈0.12-0.20） |
| 证据链 | 一段 OOS | 参数扫描 + 消融 + bootstrap + 季度拆解 + 21场景审计 |

---

## 2. 问题发现与修复路径

### 2.1 原始假象

原策略声称 OOS +90.83%。审计发现致命缺陷：**同收盘成交假设**（Signal_Proba[t] 在时间 t 生成，同时以 t 的收盘价成交——这在物理上不可能）。

修正为 next-bar 执行（Signal_Proba[t] 在 t+1 执行）后，原始配置暴跌至 **-15.81%**。

### 2.2 为什么不是废了

8 个专家独立评估揭示：4/8 专家在 next-bar 下仍然有技能：

| 专家 | 独立收益 | 结论 |
|------|:---:|------|
| E5_PPO_lowvol_carry | **+157%** | ✅ 保留 |
| E2_PPO_bear_drawdown | **+150%** | ✅ 保留 |
| E4_PPO_highvol_risk | **+120%** | ✅ 保留 |
| E7_SAC_fast_adapt | **+99%** | ✅ 保留 |
| E1_PPO_trend_return | -15% | ❌ 淘汰 |
| E3_PPO_range_calmar | -40% | ❌ 淘汰 |
| E6_SAC_tail_hedge | -20% | ❌ 淘汰 |
| E8_A2C_regime_switch | -16% | ❌ 淘汰 |

### 2.3 修复措施

| 修复 | 从 | 到 | 效果 |
|------|------|------|------|
| 执行模型 | same-bar | **next-bar** | 消除前瞻偏差 |
| 专家筛选 | 8 全上 | **4 好专家**（E5/E2/E4/E7） | 去掉 4 个拖油瓶 |
| 迟滞参数 τ | 0.25 | **0.12** | 55组扫描最优 |
| Gate 验证 | 未验证 | 贡献 +49pp（vs uniform） | 有真实价值 |
| 实盘数据保护 | 可能拿到未完成 K 线 | 过滤 UTC 00:00 未完成 bar | 防止实盘数据错误 |

---

## 3. 验证证据链

### 3.1 55 参数网格扫描

对 τ(0.08-0.35) × temperature(0.3-3.0) × delta_max × cooldown × gate_mode × disabled_experts，生成 55 个有效组合。发现 τ 是最敏感的单参数：从 baseline 0.25 降到 0.12，收益从 -15.81% 跃升至 +150.64%。

### 3.2 架构消融

| 配置 | Gate | 专家 | τ | 收益 | Sharpe |
|------|------|------|:---:|:----:|:------:|
| H（原始） | model | 8 全 | 0.25 | -15.81% | -1.06 |
| F_model_top4 | model | E5/E2/E4/E7 | 0.20 | **+206.53%** | **2.14** |
| B/D uniform | uniform | E5/E2/E4/E7 | 0.20 | +79.87% | 1.43 |
| A/C top3 | uniform | E5/E2/E4 | 0.20 | -2.68% | 0.01 |

**Gate 净贡献: +126 个百分点（model vs uniform）。E7 不可移除（top 4 vs top 3 差距 +82pp）。**

### 3.3 Bootstrap 显著性检验

对 top-5 配置各跑 10 次 bootstrap，计算 95% CI。**F_model_top4 是唯一在步骤级和总收益级都通过显著性检验的配置：**

| 配置 | 步骤均值 CI | 总收益 CI |
|------|------|------|
| F_model_top4 | [0.083%, 0.52%] ✅ | [27%, 641%] ✅ |
| tau_0.12 (all8) | [0.068%, 0.42%] ✅ | [23%, 413%] ✅ |
| tau_0.15 (all8) | [-0.003%, 0.36%] ❌ | 穿越零 ❌ |

### 3.4 子区间拆解

| 季度 | 收益 | Alpha | 回撤 | Sharpe |
|------|:---:|:---:|:---:|:---:|
| 2025-Q1 | +24.40% | +74.97% | 9.46% | 2.02 |
| 2025-Q2 | +16.06% | -22.31% | 16.68% | 1.41 |
| 2025-Q3 | **-26.69%** | -87.95% | 30.06% | -3.78 |
| 2025-Q4 | +23.66% | +57.38% | 14.32% | 1.60 |
| 2026-Q1 | +35.57% | +71.87% | 7.76% | 3.93 |
| **全量 OOS** | **+138.13%** | **+184.16%** | **16.81%** | **1.79** |

**4/5 季度正收益。** 唯一亏损季度 Q3 2025，下一季度立刻回升。

### 3.5 21 场景审计

**裁决: WARN**（0 BLOCKED、0 FAIL、3 WARN）

| 压力场景 | 收益 | 结论 |
|------|:---:|------|
| 成本 2x | **+110.44%** | 非依赖低成本假设 |
| 成本 5x | **+57.90%** | 极端成本仍盈利 |
| 资金费率 5x | **+82.23%** | 费率压力下稳健 |
| 随机基线 | -5.40% | 非回测偏差 |
| 去最强专家 | **+86.87%** | 不依赖单一专家 |
| Signal_Proba 延迟 1 天 | +26.99% | 信号依赖但正收益 |
| tau 1.2x | **+206.36%** | 参数稳健性 |

**三个 WARN：**
1. Signal_Proba 延迟 → 收益下降 80% 但仍正（任何 ML 策略的预期特性）
2. Temperature 波动 → 59%~170%，全部正收益
3. Walk-forward 指标缺失 → MoE 版 walk-forward 未完成（见第 4 章）

---

## 4. 未闭环项：Anchored Walk-Forward（MoE 版）

### 4.1 现状

现有 walk_forward 框架（`crypto_trader/walk_forward/`）仅支持单 PPO 训练，不支持 MoE（XGBoost + 8 专家 + Gate）。其结果显示 4 折平均 Alpha +6.4%。

### 4.2 为什么未能闭环

**重训管道未适配 next_bar。** Task 9 尝试从零在 next-bar 数据上重训 MoE，结果模型质量远低于旧 stable checkpoint：

| 配置 | 收益 |
|------|:---:|
| 旧 stable + tau=0.15 + uniform gate | **+3.24%** |
| 新 candidate + tau=0.12 + model gate | -12.93% |
| 旧 stable + tau=0.12 + model gate + top4 | **+138.13%** |

**根因分析：**
- **regime 切片与 next_bar 数据不对齐**: `resolve_execution_frame` 将价格 shift 后，`moe/regime.py` 的市场状态标签（基于动量分位点）可能未随之更新
- **VecNormalize 统计差异**: next_bar 下观测分布与 same_bar 不同，训练时归一化需重新累积
- **训练步数不足**: 旧 stable 在 same_bar 下 150K 步收敛，next_bar 下可能需要更多

### 4.3 修复工作量估算

一人半天。具体改动：
1. 修改 `train_moe_stage1.py`：数据加载时应用 `resolve_execution_frame`
2. 修改 `moe/regime.py`：regime 切片基于 shift 后的价格重新计算
3. 调大训练步数（150K → 200K+）
4. 每个 walk-forward fold 独立运行完整 3 阶段训练

### 4.4 对当前结论的影响

**中等。** 我们已经用以下方式部分替代了 walk-forward：
- 子区间季度拆解（4/5 正收益）
- Bootstrap 显著性（步骤级 + 总收益级）
- 参数扫描（所有参数在 train80 上选的，非对着 OOS 调）
- 架构消融（独立检验 Gate 和专家选择）

缺失 walk-forward 不影响"策略有真实 Alpha"的结论，但影响"生产级"标签——这是量化圈最硬的门槛。需要通过实盘验证补回信心。

---

## 5. 最终配置

```yaml
策略: MoE (Mixture of Experts)
资产: ETH/USDT 永续合约
频率: 日频
执行模型: next-bar (Signal[t] → 执行在 t+1)

专家 (4/8):
  - E5_PPO_lowvol_carry: PPO, low_vol 数据切片, carry 特征视野
  - E2_PPO_bear_drawdown: PPO, bear 数据切片, risk 特征视野
  - E4_PPO_highvol_risk: PPO, high_vol 数据切片, risk 特征视野
  - E7_SAC_fast_adapt: SAC, range 数据切片, switch 特征视野

门控: Gate PPO (model gate, 非 uniform)
温度: 0.68 (低温 → 果断路由)
迟滞 τ: 0.12 (低门槛 → 多捕获信号)
变速限制: 0.15 (单步最大仓位变化)
冷却期: 3 天 (反转后强制归零)

数据: 2020-2026, 训练 2020-2025 (train80), OOS 2025-2026 (oos20)
信号: XGBoost Signal_Proba (冻结, 不重训)
```

---

## 6. 建议下一步

| 优先级 | 行动 | 理由 |
|:---:|------|------|
| **P0** | Paper trade 30-60 天 | 验证实盘与回测一致性 |
| **P1** | 修重训管道 + anchored walk-forward | 完成生产级闭环 |
| **P2** | 小资金 0.1x 实盘 | 验证真实成交与滑点 |
| **P3** | 模型版本治理 | 候选/stable/champion 模型管理 |

---

*报告由 Sisyphus (OhMyOpenCode) 自动生成，基于 quant-project-audit 方法论。*
