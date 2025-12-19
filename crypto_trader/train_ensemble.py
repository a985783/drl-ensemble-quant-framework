"""
Ensemble Training Script | 集成训练脚本

Trains multiple PPO models with different random seeds for ensemble learning.
Each model learns to trade within execution constraints (Phase B architecture).

使用不同随机种子训练多个 PPO 模型用于集成学习。
每个模型学习在执行约束下进行交易（Phase B 架构）。

Usage | 使用方法:
    python train_ensemble.py
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import DataLoader
from features import FeatureEngineer
from models.signal_model import SignalPredictor
from envs.trading_env import TradingEnv
from risk_manager import RiskManager

try:
    from config import training_config, risk_config
except ImportError:
    # Fallback if config not found - use defaults
    def training_config():
        return {}
    def risk_config():
        return {}


def train_ensemble():
    """
    Train an ensemble of PPO models with diverse random seeds.
    使用不同随机种子训练 PPO 模型集成。
    
    The ensemble approach reduces variance and improves robustness:
    - Each model is trained with identical data but different initialization
    - Final predictions average across all models
    - Diversity in seeds leads to diversity in learned policies
    
    集成方法减少方差并提高稳健性：
    - 每个模型使用相同数据但不同初始化训练
    - 最终预测是所有模型的平均值
    - 种子多样性带来策略多样性
    """
    # Load configuration | 加载配置
    train_cfg = training_config()
    risk_cfg = risk_config()
    
    # 1. Prepare Data | 准备数据
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    print("Fetching ETH Data for Ensemble Training...")
    print("正在获取 ETH 数据用于集成训练...")
    raw_df = loader.fetch_data("2020-01-01", "2025-12-15", "ETH-USD", interval="1d")
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    processed_df = engineer.add_technical_indicators(raw_df)
    
    # Train Signal Model (shared across ensemble)
    # 训练信号模型（在集成模型间共享）
    predictor = SignalPredictor()
    predictor.train(processed_df)
    probs = predictor.predict_proba(processed_df)
    processed_df['Signal_Proba'] = probs
    
    # Split data | 划分数据
    split_idx = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_idx]
    test_df = processed_df.iloc[split_idx:]
    
    test_df.to_csv("crypto_trader/test_data_ETH_ensemble.csv")
    
    # Get ensemble configuration | 获取集成配置
    seeds = train_cfg.get('seeds', [42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
                                     6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555])
    n_models = train_cfg.get('n_models', 20)
    seeds = seeds[:n_models]  # Use only n_models seeds
    
    os.makedirs("checkpoints/ensemble", exist_ok=True)
    
    for seed in seeds:
        model_path = f"checkpoints/ensemble/ppo_seed_{seed}.zip"
        if os.path.exists(model_path):
            print(f"Model Seed {seed} already exists. Skipping.")
            continue
            
        print(f"\n=== Training Seed {seed} ===")
        np.random.seed(seed)
        
        # Risk Manager with configurable parameters
        # 使用可配置参数的风险管理器
        rm = RiskManager(
            max_drawdown_limit=risk_cfg.get('max_drawdown_limit', 0.15),
            freeze_period_steps=risk_cfg.get('freeze_period_steps', 1)
        )
        
        def make_env():
            return TradingEnv(train_df, risk_manager=rm)
            
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
        
        # Train with configurable hyperparameters | 使用可配置超参数训练
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=0,
            learning_rate=train_cfg.get('learning_rate', 3e-4),
            gamma=train_cfg.get('gamma', 0.99),
            n_steps=train_cfg.get('n_steps', 2048),
            batch_size=train_cfg.get('batch_size', 64),
            ent_coef=train_cfg.get('ent_coef', 0.01),
            clip_range=train_cfg.get('clip_range', 0.2),
            gae_lambda=train_cfg.get('gae_lambda', 0.95),
            seed=seed
        )
        
        total_timesteps = train_cfg.get('total_timesteps', 100000)
        model.learn(total_timesteps=total_timesteps)
        
        # Save model and normalization stats | 保存模型和归一化统计
        model.save(f"checkpoints/ensemble/ppo_seed_{seed}.zip")
        vec_env.save(f"checkpoints/ensemble/vec_norm_seed_{seed}.pkl")
        print(f"Saved checkpoints/ensemble/ppo_seed_{seed}.zip")
        
    print("\nEnsemble Training Complete! | 集成训练完成！")


if __name__ == "__main__":
    train_ensemble()

