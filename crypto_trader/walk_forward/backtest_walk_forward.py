"""
Walk-Forward 滚动回测
使用每个 fold 训练的模型测试对应的样本外时期
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from data_loader import DataLoader
from features import FeatureEngineer
from models.signal_model import SignalPredictor
from envs.trading_env import TradingEnv
from risk_manager import RiskManager
from data_versioning import record_data_version
from walk_forward.metrics import build_metrics_rows, write_metrics_csv

CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
# 与训练一致的 20 个种子
SEEDS = [
    42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
]

FOLDS = [
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31", "name": "fold1_test2022"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "name": "fold2_test2023"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "name": "fold3_test2024"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "name": "fold4_test2025"},
]


def fetch_test_data(fold_config):
    """获取测试数据（信号模型只在训练期数据上训练）"""
    loader = DataLoader()
    engineer = FeatureEngineer()
    fold_name = fold_config["name"]
    
    # 获取训练期数据用于训练信号模型
    train_start = "2020-01-01"
    train_end = fold_config["train_end"]
    train_df = loader.fetch_data(train_start, train_end, "ETH/USDT:USDT", interval="1d")
    if isinstance(train_df.columns, pd.MultiIndex):
        train_df.columns = train_df.columns.get_level_values(0)
    record_data_version(
        train_df,
        symbol="ETH/USDT:USDT",
        interval="1d",
        requested_start=train_start,
        requested_end=train_end,
        source="okx",
        output_path="quant_docs/data_versions.csv",
        note=f"walk_forward {fold_name} train raw data",
    )
    train_df = engineer.add_technical_indicators(train_df)
    
    # 训练信号模型（只用训练期数据）
    predictor = SignalPredictor()
    predictor.train(train_df)
    
    # 获取测试期数据
    test_start = fold_config["test_start"]
    test_end = fold_config["test_end"]
    test_df = loader.fetch_data(test_start, test_end, "ETH/USDT:USDT", interval="1d")
    if isinstance(test_df.columns, pd.MultiIndex):
        test_df.columns = test_df.columns.get_level_values(0)
    record_data_version(
        test_df,
        symbol="ETH/USDT:USDT",
        interval="1d",
        requested_start=test_start,
        requested_end=test_end,
        source="okx",
        output_path="quant_docs/data_versions.csv",
        note=f"walk_forward {fold_name} test raw data",
    )
    test_df = engineer.add_technical_indicators(test_df)
    
    # 用训练好的信号模型预测测试期
    probs = predictor.predict_proba(test_df)
    test_df['Signal_Proba'] = probs
    
    return test_df


def backtest_fold(fold_config):
    """回测单个 fold"""
    fold_name = fold_config["name"]
    models_dir = os.path.join(CHECKPOINTS_DIR, fold_name)
    
    print(f"\n{'='*50}")
    print(f"回测 {fold_name}")
    print(f"测试期: {fold_config['test_start']} ~ {fold_config['test_end']}")
    print(f"{'='*50}")
    
    if not os.path.exists(models_dir):
        print(f"  ✗ 模型目录不存在: {models_dir}")
        return None
    
    # 获取测试数据
    print("获取测试数据...")
    test_df = fetch_test_data(fold_config)
    print(f"测试数据: {len(test_df)} 条")
    
    # 加载模型
    models = []
    envs = []
    
    for seed in SEEDS:
        model_path = os.path.join(models_dir, f"ppo_seed_{seed}.zip")
        vec_path = os.path.join(models_dir, f"vec_norm_seed_{seed}.pkl")
        
        if not os.path.exists(model_path):
            continue
        
        model = PPO.load(model_path)
        rm = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
        
        def make_env():
            return TradingEnv(test_df, risk_manager=rm)
        
        temp_vec = DummyVecEnv([make_env])
        vec_norm = VecNormalize.load(vec_path, temp_vec)
        vec_norm.training = False
        vec_norm.norm_reward = False
        
        models.append(model)
        envs.append(vec_norm)
    
    print(f"加载 {len(models)} 个模型")
    
    if len(models) == 0:
        print("  ✗ 没有可用模型")
        return None
    
    # 回测
    rm_main = RiskManager(max_drawdown_limit=1.0, freeze_period_steps=0)
    
    def make_main_env():
        return TradingEnv(test_df, risk_manager=rm_main)
    
    main_env = DummyVecEnv([make_main_env])
    obs = main_env.reset()
    net_worths = [10000]
    
    for i in range(len(test_df) - 1):
        actions = []
        for model, env_norm in zip(models, envs):
            norm_obs = env_norm.normalize_obs(obs)
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0][0])
        
        final_action = np.clip(np.mean(actions), -1.0, 1.0)
        obs, _, _, info = main_env.step(np.array([[final_action]]))
        net_worths.append(info[0]['net_worth'])
    
    # 计算指标
    net_worths = np.array(net_worths)
    peak = np.maximum.accumulate(net_worths)
    max_dd = ((peak - net_worths) / peak).max()
    
    total_return = (net_worths[-1] - 10000) / 10000
    benchmark = (test_df['Close'].iloc[-1] / test_df['Close'].iloc[0]) - 1
    
    result = {
        "fold": fold_name,
        "test_period": f"{fold_config['test_start']} ~ {fold_config['test_end']}",
        "total_return": total_return,
        "benchmark": benchmark,
        "alpha": total_return - benchmark,
        "max_dd": max_dd,
        "net_worths": net_worths,
        "dates": test_df.index[:len(net_worths)],
        "benchmark_curve": (test_df['Close'] / test_df['Close'].iloc[0]) * 10000
    }
    
    print(f"  策略收益: {total_return*100:.2f}%")
    print(f"  基准收益: {benchmark*100:.2f}%")
    print(f"  Alpha: {(total_return-benchmark)*100:.2f}%")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    
    return result


def run_all_backtests():
    """运行所有 fold 的回测"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("Walk-Forward 滚动回测")
    print("="*60)
    
    results = []
    for fold in FOLDS:
        result = backtest_fold(fold)
        if result:
            results.append(result)
    
    if not results:
        print("\n没有完成任何回测，请先运行 train_walk_forward.py")
        return
    
    # 汇总结果
    print("\n" + "="*60)
    print("汇总结果 (真正的样本外测试)")
    print("="*60)
    print(f"{'Fold':<20} {'测试期':<25} {'策略':>10} {'基准':>10} {'Alpha':>10} {'回撤':>10}")
    print("-"*90)
    
    total_equity = 10000
    for r in results:
        print(f"{r['fold']:<20} {r['test_period']:<25} {r['total_return']*100:>9.2f}% {r['benchmark']*100:>9.2f}% {r['alpha']*100:>9.2f}% {r['max_dd']*100:>9.2f}%")
        total_equity *= (1 + r['total_return'])
    
    print("-"*90)
    cumulative_return = (total_equity / 10000 - 1) * 100
    print(f"累计收益 (2022-2025): {cumulative_return:.2f}%")

    # Export structured metrics
    metrics_rows = build_metrics_rows(results)
    metrics_path = os.path.join(RESULTS_DIR, "walk_forward_metrics.csv")
    write_metrics_csv(metrics_rows, metrics_path)
    print(f"指标已保存: {metrics_path}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    for i, r in enumerate(results):
        # 归一化到上一个 fold 的结束值
        if i == 0:
            scale = 1.0
        else:
            scale = results[i-1]['net_worths'][-1] / 10000
        
        nw = r['net_worths'] * scale
        ax.plot(r['dates'], nw, color=colors[i], label=f"{r['fold']} ({r['total_return']*100:.1f}%)", linewidth=2)
    
    ax.set_title('Walk-Forward Backtest: True Out-of-Sample Performance')
    ax.set_ylabel('Net Worth ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "walk_forward_results.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n图表已保存: {output_path}")


if __name__ == "__main__":
    run_all_backtests()
