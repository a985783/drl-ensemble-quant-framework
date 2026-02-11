#!/usr/bin/env python3
"""
最新时间段回测脚本
从测试集结束后的时间段开始回测 (2025-12-24 ~ 今天)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
from datetime import datetime

sys.path.insert(0, 'crypto_trader')
from envs.trading_env import TradingEnv
from risk_manager import RiskManager
from data_loader import DataLoader
from features import FeatureEngineer
from models.signal_model import SignalPredictor
from data_versioning import record_data_version
try:
    from crypto_trader.analytics.performance import PerformanceAttribution
except ImportError:
    from analytics.performance import PerformanceAttribution

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def backtest_recent():
    """
    从测试集结束后的时间段进行回测
    """
    seeds = [
        42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
        6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
    ]
    
    # 时间范围：从测试集结束后到今天
    start_date = "2025-12-24"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"   📊 最新时段回测 (测试集后)")
    print(f"{'='*60}")
    print(f"   📅 回测期间: {start_date} ~ {end_date}")
    
    # 获取最新数据
    print("\n【数据】正在获取最新 ETH 数据...")
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    # 获取足够的历史数据用于信号模型训练和特征计算
    extended_start = "2024-01-01"  # 提前获取更多数据用于模型训练
    raw_df = loader.fetch_data(extended_start, end_date, "ETH/USDT:USDT", interval="1d")

    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    record_data_version(
        raw_df,
        symbol="ETH/USDT:USDT",
        interval="1d",
        requested_start=extended_start,
        requested_end=end_date,
        source="okx",
        output_path="quant_docs/data_versions.csv",
        note="backtest_recent raw data (training + recent slice)",
    )
    
    print(f"【数据】获取了 {len(raw_df)} 条原始数据")
    
    # 添加技术指标
    processed_df = engineer.add_technical_indicators(raw_df)
    
    # Signal Model
    print("【模型】训练信号模型...")
    predictor = SignalPredictor()
    predictor.train(processed_df)
    probs = predictor.predict_proba(processed_df)
    processed_df['Signal_Proba'] = probs
    
    # 只保留目标时段的数据
    recent_df = processed_df.loc[start_date:]
    print(f"【数据】回测数据: {len(recent_df)} 条 ({recent_df.index.min()} ~ {recent_df.index.max()})")
    
    if len(recent_df) < 2:
        print("❌ 数据不足，无法回测")
        return
    
    # 加载模型和VecNormalize归一化器
    models = []
    vec_norms = []
    
    # 获取空间定义用于 custom_objects
    from gymnasium import spaces
    import pickle
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
    act_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    
    print("\n【模型】加载集成模型...")
    for seed in seeds:
        model_path = f"checkpoints/ensemble/ppo_seed_{seed}.zip"
        vec_path = f"checkpoints/ensemble/vec_norm_seed_{seed}.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            continue
        
        try:
            # 创建临时环境
            rm_temp = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
            def make_temp_env(): return TradingEnv(recent_df.copy(), risk_manager=rm_temp)
            temp_vec = DummyVecEnv([make_temp_env])
            
            # 尝试加载 VecNormalize
            try:
                vec_norm = VecNormalize.load(vec_path, temp_vec)
            except Exception:
                # 如果加载失败，使用 pickle 直接加载归一化参数
                with open(vec_path, 'rb') as f:
                    saved_data = pickle.load(f)
                # 创建新的 VecNormalize 并手动设置参数
                vec_norm = VecNormalize(temp_vec, norm_obs=True, norm_reward=False)
                if hasattr(saved_data, 'obs_rms'):
                    vec_norm.obs_rms = saved_data.obs_rms
                elif isinstance(saved_data, dict) and 'obs_rms' in saved_data:
                    vec_norm.obs_rms = saved_data['obs_rms']
            
            vec_norm.training = False
            vec_norm.norm_reward = False
            
            # 加载模型
            model = PPO.load(
                model_path, 
                env=vec_norm,
                custom_objects={
                    "observation_space": obs_space,
                    "action_space": act_space
                }
            )
            models.append(model)
            vec_norms.append(vec_norm)
            print(f"    ✅ 加载 seed {seed}")
            
        except Exception as e:
            print(f"    ⚠️ 加载 seed {seed} 失败: {e}")
            continue
    
    print(f"【模型】成功加载 {len(models)} 个模型")
    
    if len(models) == 0:
        print("❌ 没有找到任何模型")
        return
    
    # 准备回测环境
    rm_main = RiskManager(max_drawdown_limit=1.0, freeze_period_steps=0)
    def make_main_env(): return TradingEnv(recent_df, risk_manager=rm_main)
    main_env = DummyVecEnv([make_main_env])
    
    obs_main = main_env.reset()
    net_worths = [10000]
    positions = [0]
    actions_taken = []
    
    print("\n【回测】开始执行...")
    for i in range(len(recent_df) - 1):
        # 收集所有模型的动作
        actions = []
        for model, vec_norm in zip(models, vec_norms):
            # 使用对应的VecNormalize归一化观察值
            norm_obs = vec_norm.normalize_obs(obs_main)
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0][0])
        
        # 集成决策: 平均
        avg = np.mean(actions)
        final_action = np.clip(avg, -1.0, 1.0)
        actions_taken.append(final_action)
        
        # 执行
        step_action = np.array([[final_action]])
        obs_main, _, _, info = main_env.step(step_action)
        net_worths.append(info[0]['net_worth'])
        positions.append(info[0].get('position', 0))
    
    # 计算指标
    net_worths = np.array(net_worths)
    peak = np.maximum.accumulate(net_worths)
    drawdown = (peak - net_worths) / peak
    max_dd = drawdown.max()
    
    # Benchmark (Buy & Hold)
    initial_price = recent_df['Close'].iloc[0]
    benchmark = (recent_df['Close'] / initial_price) * 10000
    
    tr_ret = (net_worths[-1] - 10000) / 10000
    bh_ret = (benchmark.iloc[-1] - 10000) / 10000
    
    # 计算夏普比率
    daily_returns = np.diff(net_worths) / net_worths[:-1]
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = np.mean(daily_returns) / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"   📈 回测结果 ({start_date} ~ {end_date})")
    print(f"{'='*60}")
    print(f"   📊 策略收益: {tr_ret * 100:+.2f}%")
    print(f"   📊 基准收益: {bh_ret * 100:+.2f}%")
    print(f"   🎯 超额收益 (Alpha): {(tr_ret - bh_ret) * 100:+.2f}%")
    print(f"   📉 最大回撤: {max_dd * 100:.2f}%")
    print(f"   📐 夏普比率: {sharpe:.2f}")
    print(f"   💰 最终净值: ${net_worths[-1]:,.2f}")
    print(f"{'='*60}")
    
    # === Institutional Performance Attribution ===
    print("\n【机构级归因分析】生成中...")
    try:
        # Construct Series with Date index
        equity_series = pd.Series(net_worths, index=recent_df.index[:len(net_worths)])
        bench_series = recent_df['Close'].iloc[:len(net_worths)]
        
        attribution = PerformanceAttribution(equity_series, bench_series)
        metrics = attribution.calculate_metrics()
        
        print("-" * 40)
        print(f"Alpha (年化): {metrics['Alpha (Ann.)']:.2%}")
        print(f"Beta (市场敏感度): {metrics['Beta']:.2f}")
        print(f"Information Ratio: {metrics['Info Ratio']:.2f}")
        print("-" * 40)
        
        attribution.generate_report("results/institutional_report.png")
    except Exception as e:
        print(f"⚠️ 归因分析生成失败: {e}")
    # ============================================
    
    # 每日详情
    print(f"\n{'='*60}")
    print("   📅 每日交易详情")
    print(f"{'='*60}")
    print(f"{'日期':<12} {'价格':>10} {'动作':>8} {'仓位':>8} {'净值':>12}")
    print("-" * 60)
    
    for i, date in enumerate(recent_df.index[:len(net_worths)]):
        price = recent_df['Close'].iloc[i]
        action = actions_taken[i-1] if i > 0 and i-1 < len(actions_taken) else 0
        pos = positions[i] if i < len(positions) else 0
        nw = net_worths[i]
        
        action_str = f"{action:+.2f}" if action != 0 else "0.00"
        pos_str = "多" if pos > 0.3 else ("空" if pos < -0.3 else "平")
        
        print(f"{date.strftime('%Y-%m-%d'):<12} ${price:>9,.2f} {action_str:>8} {pos_str:>8} ${nw:>11,.2f}")
    
    # ============ 可视化 ============
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 子图1: 净值曲线
    ax1 = axes[0]
    dates = recent_df.index[:len(net_worths)]
    ax1.plot(dates, net_worths, color='#FFD700', linewidth=2.5, label=f'策略 ({tr_ret*100:+.1f}%)')
    ax1.plot(recent_df.index[:len(benchmark)], benchmark[:len(dates)], color='gray', linestyle='--', linewidth=1.5, label=f'基准 ({bh_ret*100:+.1f}%)')
    ax1.fill_between(dates, net_worths, 10000, where=(net_worths >= 10000), color='green', alpha=0.2)
    ax1.fill_between(dates, net_worths, 10000, where=(net_worths < 10000), color='red', alpha=0.2)
    ax1.axhline(y=10000, color='white', linestyle=':', alpha=0.5)
    ax1.set_ylabel('净值 (USD)', fontsize=12)
    ax1.set_title(f'最新时段回测 ({start_date} ~ {end_date})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1a1a2e')
    
    # 子图2: 仓位
    ax2 = axes[1]
    ax2.fill_between(dates, positions[:len(dates)], 0, where=(np.array(positions[:len(dates)]) > 0), color='green', alpha=0.5, label='多头')
    ax2.fill_between(dates, positions[:len(dates)], 0, where=(np.array(positions[:len(dates)]) < 0), color='red', alpha=0.5, label='空头')
    ax2.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    ax2.set_ylabel('仓位', fontsize=12)
    ax2.set_ylim(-1.2, 1.2)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#1a1a2e')
    
    # 子图3: 回撤
    ax3 = axes[2]
    dd_series = drawdown[:len(dates)]
    ax3.fill_between(dates, dd_series * 100, 0, color='red', alpha=0.6)
    ax3.set_ylabel('回撤 (%)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_ylim(max(dd_series * 100) * 1.2 if max(dd_series) > 0 else -1, 0)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#1a1a2e')
    
    # 整体样式
    fig.patch.set_facecolor('#0f0f23')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('gray')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('results', exist_ok=True)
    output_path = 'results/backtest_recent.png'
    plt.savefig(output_path, dpi=150, facecolor='#0f0f23', edgecolor='none', bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存: {output_path}")
    
    # 也保存CSV结果
    result_df = pd.DataFrame({
        'Date': dates,
        'Price': recent_df['Close'].iloc[:len(dates)].values,
        'Action': [0] + actions_taken[:len(dates)-1],
        'Position': positions[:len(dates)],
        'NetWorth': net_worths,
        'Drawdown': dd_series
    })
    csv_path = 'results/backtest_recent.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"✅ 详细数据已保存: {csv_path}")
    
    return {
        'total_return': tr_ret,
        'benchmark_return': bh_ret,
        'alpha': tr_ret - bh_ret,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'final_value': net_worths[-1]
    }


if __name__ == "__main__":
    backtest_recent()
