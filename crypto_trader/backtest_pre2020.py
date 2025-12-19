
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

def backtest_pre2020():
    seeds = [
        42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
        6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
    ]
    data_path = "crypto_trader/test_data_ETH_pre2020.csv"
    
    # 1. Fetch Pre-2020 Data (e.g. 2018-01-01 to 2019-12-31)
    # We use 2018 because 2017 data availability/quality might be spotty on some feeds, 
    # and 2018 Bear Market is a great stress test.
    start_date = "2017-10-01"
    end_date = "2019-12-31"
    
    if os.path.exists(data_path):
        test_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded existing data from {data_path}")
    else:
        print(f"Fetching data from {start_date} to {end_date} using YFinance...")
        import yfinance as yf
        engineer = FeatureEngineer()
        
        try:
            # Download from Yahoo Finance
            raw_df = yf.download("ETH-USD", start=start_date, end=end_date, interval="1d")
            
            # YFoundation sometimes returns MultiIndex columns, flatten if needed
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
                
            # Ensure standard columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            # If 'Adj Close' exists, maybe prefer it? No, keep standard Close for consistency with OKX.
            if len(raw_df) == 0:
                print("Error: YFinance returned no data.")
                return
            
            raw_df = raw_df[required_cols]

        except Exception as e:
            print(f"Error fetching data: {e}")
            return
            
        processed_df = engineer.add_technical_indicators(raw_df)
        
        # Signal Model
        predictor = SignalPredictor()
        predictor.train(processed_df)
        probs = predictor.predict_proba(processed_df)
        processed_df['Signal_Proba'] = probs
        
        # Use the whole period as test set
        test_df = processed_df
        test_df.to_csv(data_path)
        print(f"Data saved to {data_path}")
    
    print(f"Test Data Points: {len(test_df)}")

    # 2. Load Models (PPO Agents trained on 2020-2025)
    models = []
    envs = []
    
    print("Loading Ensemble Models...")
    for seed in seeds:
        model_path = f"checkpoints/ensemble/ppo_seed_{seed}.zip"
        vec_path = f"checkpoints/ensemble/vec_norm_seed_{seed}.pkl"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model {seed} not found. Skipping.")
            continue
            
        model = PPO.load(model_path)
        
        # Load VecNormalize stats from Training
        rm = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
        def make_env(): return TradingEnv(test_df, risk_manager=rm)
        temp_vec = DummyVecEnv([make_env])
        vec_norm = VecNormalize.load(vec_path, temp_vec)
        vec_norm.training = False
        vec_norm.norm_reward = False
        
        models.append(model)
        envs.append(vec_norm)
        
    print(f"Loaded {len(models)} models.")
    
    # 3. Execution Env (TIERED RISK ONLY: Cap at 0.8x/0.5x, but no hard kill until 100%)
    rm_main = RiskManager(max_drawdown_limit=1.0, freeze_period_steps=0) 
    def make_main_env(): return TradingEnv(test_df, risk_manager=rm_main, enable_kill_switch=False)
    main_env = DummyVecEnv([make_main_env])
    
    obs_main = main_env.reset()
    net_worths = [10000]
    positions = [0.0]
    
    # 4. Backtest Loop
    for i in range(len(test_df) - 1):
        actions = []
        for model, env_norm in zip(models, envs):
            norm_obs = env_norm.normalize_obs(obs_main)
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0][0])
            
        avg = np.mean(actions)
        final_action = np.clip(avg, -1.0, 1.0)
            
        step_action = np.array([[final_action]])
        obs_main, _, _, info = main_env.step(step_action)
        net_worths.append(info[0]['net_worth'])
        positions.append(info[0]['position'])
        
    # 5. Metrics
    net_worths = np.array(net_worths)
    positions = np.array(positions)
    peak = np.maximum.accumulate(net_worths)
    drawdown = (peak - net_worths) / peak
    max_dd = drawdown.max()

    initial_price = test_df['Close'].iloc[0]
    benchmark = (test_df['Close'] / initial_price) * 10000

    tr_ret = (net_worths[-1] - 10000) / 10000
    bh_ret = (benchmark.iloc[-1] - 10000) / 10000
    
    # Simple Annualized
    days = len(test_df)
    years = days / 365.0
    annual_ret = ((1 + tr_ret) ** (1/years)) - 1

    print('\n=== Pre-2020 OOS Test (2018 Bear - 2019 Recovery) ===')
    print(f'Period: {start_date} to {end_date}')
    print(f'Total Return: {tr_ret * 100:.2f}%')
    print(f'Annualized: {annual_ret * 100:.2f}%')
    print(f'Benchmark (ETH): {bh_ret * 100:.2f}%')
    print(f'Alpha: {(tr_ret - bh_ret) * 100:.2f}%')
    print(f'Max Drawdown: {max_dd * 100:.2f}%')

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    ax1.plot(test_df.index[:len(net_worths)], net_worths, color='purple', label=f'Strategy (20 Models) (+{tr_ret*100:.1f}%)')
    ax1.plot(test_df.index, benchmark, color='gray', linestyle='--', label='ETH Buy&Hold')
    ax1.set_title('Net Worth: 2018-2019 (20 Models + Tiered Risk)')
    ax1.set_ylabel('Value (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # Position
    ax2.plot(test_df.index[:len(positions)], positions, color='blue', label='Position Leverage')
    ax2.set_title('Position Leverage')
    ax2.set_ylabel('Leverage (-1 to 1)')
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig('results/net_worth_pre2020_20models.png')
    print('Saved plot to results/net_worth_pre2020_20models.png')

if __name__ == "__main__":
    backtest_pre2020()
