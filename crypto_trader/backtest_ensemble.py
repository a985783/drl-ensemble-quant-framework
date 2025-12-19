
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
sys.path.insert(0, 'crypto_trader')
from envs.trading_env import TradingEnv
from risk_manager import RiskManager

from data_loader import DataLoader
from features import FeatureEngineer
from models.signal_model import SignalPredictor

def backtest_ensemble():
    seeds = [
        42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
        6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
    ]
    data_path = "crypto_trader/test_data_ETH_ensemble.csv"
    
    if os.path.exists(data_path):
        test_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print("Data file not found. Regenerating...")
        loader = DataLoader()
        engineer = FeatureEngineer()
        
        # Exact same parameters as train_ensemble.py
        raw_df = loader.fetch_data("2020-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"), "ETH-USD", interval="1d")
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
            
        processed_df = engineer.add_technical_indicators(raw_df)
        
        # Signal Model
        predictor = SignalPredictor()
        predictor.train(processed_df)
        probs = predictor.predict_proba(processed_df)
        processed_df['Signal_Proba'] = probs
        
        # Split (80/20)
        split_idx = int(len(processed_df) * 0.8)
        test_df = processed_df.iloc[split_idx:]
        
        # Save for next time
        test_df.to_csv(data_path)
        print(f"Data regenerated and saved to {data_path}")
    
    # Load Models and Envs
    models = []
    envs = []
    
    print("Loading Ensemble Models...")
    for seed in seeds:
        # Load Model
        model_path = f"checkpoints/ensemble/ppo_seed_{seed}.zip"
        vec_path = f"checkpoints/ensemble/vec_norm_seed_{seed}.pkl"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model {seed} not found. Skipping.")
            continue
            
        model = PPO.load(model_path)
        
        # Load VecNormalize stats (Crucial!)
        # We need a dummy env to load it into
        rm = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
        def make_env(): return TradingEnv(test_df, risk_manager=rm) # Empty df is fine for structure
        temp_vec = DummyVecEnv([make_env])
        vec_norm = VecNormalize.load(vec_path, temp_vec)
        vec_norm.training = False
        vec_norm.norm_reward = False
        
        models.append(model)
        envs.append(vec_norm)
        
    print(f"Loaded {len(models)} models.")
    
    # Prepare Main Test Env (Real execution environment)
    # Unifying to Tiered Risk Only (No Hard Stop)
    rm_main = RiskManager(max_drawdown_limit=1.0, freeze_period_steps=0) 
    def make_main_env(): return TradingEnv(test_df, risk_manager=rm_main)
    main_env = DummyVecEnv([make_main_env])
    # We don't normalize the main env's OBSERVATIONS for the agent, 
    # instead we normalize observations FOR EACH AGENT individually using their own stats.
    
    obs_main = main_env.reset()
    net_worths = [10000]
    avg_actions = []
    
    for i in range(len(test_df) - 1):
        # Collect votes
        actions = []
        for model, env_norm in zip(models, envs):
            # Normalize observation for this specific agent
            # Note: manually normalizing is tricky without the VecEnv wrapper doing it.
            # Workaround: Set the main_env obs into the env_norm and normalize.
            # But VecNormalize expects 'step' or 'reset'.
            
            # Correct approach:
            # The 'obs_main' is raw. We need to normalize it using 'env_norm.normalize_obs(obs_main)'
            # Stable Baselines VecNormalize has a method 'normalize_obs'.
            
            norm_obs = env_norm.normalize_obs(obs_main)
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0][0])
            
        # Decision: Average
        avg = np.mean(actions)
        # Phase B Update: Use simple mean (no boost) because constraints are trained in.
        final_action = np.clip(avg, -1.0, 1.0)
            
        # Execute
        step_action = np.array([[final_action]])
        obs_main, _, _, info = main_env.step(step_action)
        net_worths.append(info[0]['net_worth'])
        
    # Metrics
    net_worths = np.array(net_worths)
    peak = np.maximum.accumulate(net_worths)
    drawdown = (peak - net_worths) / peak
    max_dd = drawdown.max()

    initial_price = test_df['Close'].iloc[0]
    benchmark = (test_df['Close'] / initial_price) * 10000

    tr_ret = (net_worths[-1] - 10000) / 10000
    bh_ret = (benchmark.iloc[-1] - 10000) / 10000

    print('\n=== Ensemble (10 Models Averaging) Result ===')
    print(f'Total Return: {tr_ret * 100:.2f}%')
    print(f'Benchmark Return: {bh_ret * 100:.2f}%')
    print(f'Alpha: {(tr_ret - bh_ret) * 100:.2f}%')
    print(f'Max Drawdown: {max_dd * 100:.2f}%')

    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(test_df.index[:len(net_worths)], net_worths, color='gold', label=f'Ensemble (20 Models) (+{tr_ret*100:.1f}%)')
    plt.plot(test_df.index, benchmark, color='gray', linestyle='--', label='Benchmark')
    plt.title('Ensemble Strategy (Low Frequency)')
    plt.ylabel('Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/net_worth_ETH_ensemble.png')
    print('Saved plot to results/net_worth_ETH_ensemble.png')

if __name__ == "__main__":
    backtest_ensemble()
