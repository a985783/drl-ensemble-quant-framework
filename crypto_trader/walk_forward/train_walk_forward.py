"""
Walk-Forward 滚动训练系统
独立于现有模型，使用并行训练

时间分割：
- 轮次1: 训练 2020-2021 → 测试 2022
- 轮次2: 训练 2020-2022 → 测试 2023  
- 轮次3: 训练 2020-2023 → 测试 2024
- 轮次4: 训练 2020-2024 → 测试 2025

Mac M3 Air 16GB 优化：
- 使用 4 个并行进程（每轮训练 5 个种子并行）
- 减少 timesteps 加快训练
"""
import os
import sys
import json
import multiprocessing as mp
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np
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

# 配置
WALK_FORWARD_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
# 与原始 train_ensemble.py 完全一致的 20 个种子
SEEDS = [
    42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
]
N_PARALLEL = 4  # M3 Air 并行进程数

# 时间分割配置
FOLDS = [
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31", "name": "fold1_test2022"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "name": "fold2_test2023"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "name": "fold3_test2024"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "name": "fold4_test2025"},
]


def fetch_and_prepare_data(start_date, end_date):
    """获取并处理数据"""
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    raw_df = loader.fetch_data(start_date, end_date, "ETH/USDT:USDT", interval="1d")
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    processed_df = engineer.add_technical_indicators(raw_df)
    
    # 信号模型只在训练数据上训练（避免泄漏）
    predictor = SignalPredictor()
    predictor.train(processed_df)
    probs = predictor.predict_proba(processed_df)
    processed_df['Signal_Proba'] = probs
    
    return processed_df


def train_single_model(args):
    """训练单个模型（用于并行）"""
    seed, train_df, fold_name, output_dir = args
    
    model_path = os.path.join(output_dir, f"ppo_seed_{seed}.zip")
    vec_path = os.path.join(output_dir, f"vec_norm_seed_{seed}.pkl")
    
    if os.path.exists(model_path):
        print(f"  [跳过] {fold_name} seed {seed} 已存在")
        return seed, True
    
    try:
        np.random.seed(seed)
        rm = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
        
        def make_env():
            return TradingEnv(train_df, risk_manager=rm)
        
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
        
        # 与原始 train_ensemble.py 完全一致的参数
        model = PPO(
            "MlpPolicy", vec_env, verbose=0,
            learning_rate=3e-4,
            gamma=0.995,           # Higher gamma for longer horizon
            n_steps=2048,          # Larger buffer
            batch_size=256,        # Larger batch
            ent_coef=0.005,        # Lower entropy for less exploration noise
            clip_range=0.2,
            gae_lambda=0.95,
            seed=seed
        )
        
        # 与原始一致：150k timesteps
        model.learn(total_timesteps=150000)
        
        model.save(model_path)
        vec_env.save(vec_path)
        
        print(f"  ✓ {fold_name} seed {seed} 完成")
        return seed, True
        
    except Exception as e:
        print(f"  ✗ {fold_name} seed {seed} 失败: {e}")
        return seed, False


def train_fold(fold_config):
    """训练一个 fold 的所有模型"""
    fold_name = fold_config["name"]
    train_end = fold_config["train_end"]
    
    print(f"\n{'='*50}")
    print(f"训练 {fold_name} (训练截止: {train_end})")
    print(f"{'='*50}")
    
    # 创建目录
    output_dir = os.path.join(WALK_FORWARD_DIR, fold_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取训练数据（只到 train_end）
    print("获取训练数据...")
    train_df = fetch_and_prepare_data("2020-01-01", train_end)
    print(f"训练数据: {len(train_df)} 条 ({train_df.index[0]} ~ {train_df.index[-1]})")
    
    # 并行训练
    args_list = [(seed, train_df, fold_name, output_dir) for seed in SEEDS]
    
    print(f"开始并行训练 {len(SEEDS)} 个模型 ({N_PARALLEL} 进程)...")
    
    with mp.Pool(N_PARALLEL) as pool:
        results = pool.map(train_single_model, args_list)
    
    success_count = sum(1 for _, success in results if success)
    print(f"完成: {success_count}/{len(SEEDS)} 个模型成功")
    
    return output_dir


def run_all_training():
    """运行所有 fold 的训练"""
    os.makedirs(WALK_FORWARD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("Walk-Forward 滚动训练")
    print(f"并行进程: {N_PARALLEL}")
    print(f"每轮种子数: {len(SEEDS)}")
    print("="*60)
    
    start_time = datetime.now()
    
    for fold in FOLDS:
        train_fold(fold)
    
    elapsed = datetime.now() - start_time
    print(f"\n总耗时: {elapsed}")
    print("训练完成！运行 backtest_walk_forward.py 进行回测")


if __name__ == "__main__":
    # macOS 需要这个
    mp.set_start_method('spawn', force=True)
    run_all_training()
