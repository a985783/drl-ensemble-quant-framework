
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data_loader import DataLoader
from features import FeatureEngineer
from models.signal_model import SignalPredictor
from envs.trading_env import TradingEnv
from risk_manager import RiskManager
from config import load_config, get_default_config

def train_ensemble():
    # Load config (defaults)
    config = get_default_config()
    
    # 1. Prepare Data
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    symbol = config.data.symbol
    start_date = config.data.train_start
    end_date = config.data.train_end
    
    print(f"Fetching {symbol} Data for Ensemble Training ({start_date} ~ {end_date})...")
    raw_df = loader.fetch_data(start_date, end_date, symbol, interval=config.data.interval)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    processed_df = engineer.add_technical_indicators(raw_df)

    # Split first to prevent data leakage
    split_idx = int(len(processed_df) * config.data.train_split_ratio)
    train_df_full = processed_df.iloc[:split_idx]
    test_df_full = processed_df.iloc[split_idx:]

    # Train Signal Model only on training data to prevent leakage
    print("Training Signal Model (only on training data)...")
    predictor = SignalPredictor()
    predictor.train(train_df_full)

    # Apply prediction to both train and test
    train_df_full = train_df_full.copy()
    test_df_full = test_df_full.copy()
    train_df_full['Signal_Proba'] = predictor.predict_proba(train_df_full)
    test_df_full['Signal_Proba'] = predictor.predict_proba(test_df_full)

    # Combine for full dataset with predictions
    processed_df = pd.concat([train_df_full, test_df_full])
    train_df = train_df_full
    test_df = test_df_full
    
    # Save test data for potential manual inspection
    test_data_path = f"crypto_trader/test_data_ensemble.csv"
    test_df.to_csv(test_data_path)
    print(f"Test data saved to {test_data_path}")
    
    seeds = config.seed.ensemble_seeds
    checkpoint_dir = config.paths.checkpoints_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for seed in seeds:
        model_path = f"{checkpoint_dir}/ppo_seed_{seed}.zip"
        vec_path = f"{checkpoint_dir}/vec_norm_seed_{seed}.pkl"
        
        if os.path.exists(model_path):
            print(f"Model Seed {seed} already exists. Skipping.")
            continue
            
        print(f"\n=== Training Seed {seed} ===")
        np.random.seed(seed)
        
        # Risk Manager settings from config
        rm = RiskManager(
            max_drawdown_limit=config.risk.max_drawdown_limit, 
            freeze_period_steps=config.risk.freeze_period_steps
        )
        
        def make_env():
            return TradingEnv(
                train_df,
                risk_manager=rm,
                atr_floor=config.risk.atr_floor,
                vol_scale_min=config.risk.vol_scale_min,
                vol_scale_max=config.risk.vol_scale_max
            )
            
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
        
        # Train with config hyperparams
        model = PPO("MlpPolicy", vec_env, verbose=0, 
                    learning_rate=config.model.learning_rate,
                    gamma=config.model.gamma,
                    n_steps=config.model.n_steps,
                    batch_size=config.model.batch_size,
                    ent_coef=config.model.ent_coef,
                    clip_range=config.model.clip_range,
                    gae_lambda=config.model.gae_lambda,
                    seed=seed)
        
        model.learn(total_timesteps=config.model.total_timesteps)
        
        # Save
        model.save(model_path)
        vec_env.save(vec_path)
        print(f"Saved {model_path}")
        
    print("\nEnsemble Training Complete!")

if __name__ == "__main__":
    train_ensemble()
