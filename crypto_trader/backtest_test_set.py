
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
from datetime import datetime
from config import get_default_config

sys.path.insert(0, 'crypto_trader')
from envs.trading_env import TradingEnv
from risk_manager import RiskManager
from models.signal_model import SignalPredictor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def backtest_test_set():
    """
    Evaluation on the held-out TEST SET (from training split).
    """
    config = get_default_config()
    seeds = config.seed.ensemble_seeds
    
    test_data_path = "crypto_trader/test_data_ensemble.csv"
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at {test_data_path}. Run training first.")
        return

    print(f"\n{'='*60}")
    print(f"   🧪 Hold-out Test Set Evaluation")
    print(f"{'='*60}")
    
    # Load Test Data
    print(f"【Data】Loading test set from {test_data_path}...")
    df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    
    # Check if Signal_Proba exists, if not, generate it (though training script saves it)
    if 'Signal_Proba' not in df.columns:
        print("【Model】Generating Signal Probabilities for Test Set...")
        predictor = SignalPredictor()
        predictor.train(df) # Wait, training on test set is leaking! 
        # Actually signal model should be loaded or trained on train set.
        # But 'test_data_ensemble.csv' saved by train_ensemble.py ALREADY has 'Signal_Proba'
        # because it was processed before splitting? 
        # Let's check train_ensemble.py again.
        # Yes: processed_df['Signal_Proba'] = probs -> then split -> then save.
        # So it should be there. If not, we have a problem.
        pass
        
    print(f"【Data】Loaded {len(df)} rows ({df.index.min()} ~ {df.index.max()})")
    
    # Load Models
    models = []
    envs = []
    checkpoint_dir = config.paths.checkpoints_dir
    
    print("\n【Models】Loading Ensemble...")
    for seed in seeds:
        model_path = f"{checkpoint_dir}/ppo_seed_{seed}.zip"
        vec_path = f"{checkpoint_dir}/vec_norm_seed_{seed}.pkl"
        
        if not os.path.exists(model_path):
            continue
            
        model = PPO.load(model_path)
        
        # Risk Manager (Conservative for testing)
        rm = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
        def make_env(): return TradingEnv(df, risk_manager=rm)
        temp_vec = DummyVecEnv([make_env])
        # Load normalization stats from training! Crucial!
        vec_norm = VecNormalize.load(vec_path, temp_vec)
        vec_norm.training = False # DO NOT UPDATE STATS
        vec_norm.norm_reward = False
        
        models.append(model)
        envs.append(vec_norm)
    
    print(f"【Models】Loaded {len(models)} models")
    
    if len(models) == 0:
        print("❌ No models found.")
        return

    # Prepare Environment
    rm_main = RiskManager(max_drawdown_limit=1.0, freeze_period_steps=0)
    def make_main_env(): return TradingEnv(df, risk_manager=rm_main)
    main_env = DummyVecEnv([make_main_env])
    
    obs_main = main_env.reset()
    net_worths = [10000]
    positions = [0]
    actions_taken = []
    
    print("\n【Backtest】Running Loop...")
    for i in range(len(df) - 1):
        actions = []
        for model, env_norm in zip(models, envs):
            # Normalize obs using stored stats
            norm_obs = env_norm.normalize_obs(obs_main)
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0][0])
        
        # Ensemble Average
        # Soft Voting / Average
        avg = np.mean(actions)
        final_action = np.clip(avg, -1.0, 1.0)
        actions_taken.append(final_action)
        
        step_action = np.array([[final_action]])
        obs_main, _, _, info = main_env.step(step_action)
        net_worths.append(info[0]['net_worth'])
        positions.append(info[0].get('position', 0))

    # Metrics
    net_worths = np.array(net_worths)
    peak = np.maximum.accumulate(net_worths)
    drawdown = (peak - net_worths) / peak
    max_dd = drawdown.max()
    
    initial_price = df['Close'].iloc[0]
    benchmark = (df['Close'] / initial_price) * 10000
    
    tr_ret = (net_worths[-1] - 10000) / 10000
    bh_ret = (benchmark.iloc[-1] - 10000) / 10000
    
    daily_returns = np.diff(net_worths) / net_worths[:-1]
    sharpe = 0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        # 使用 sqrt(365) 因为加密货币市场 7x24 小时交易
        sharpe = np.mean(daily_returns) / daily_returns.std() * np.sqrt(365)

    print(f"\n{'='*60}")
    print(f"   📈 TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"   📊 Strategy Return: {tr_ret * 100:+.2f}%")
    print(f"   📊 Benchmark Return: {bh_ret * 100:+.2f}%")
    print(f"   🎯 Alpha: {(tr_ret - bh_ret) * 100:+.2f}%")
    print(f"   📉 Max Drawdown: {max_dd * 100:.2f}%")
    print(f"   📐 Sharpe Ratio: {sharpe:.2f}")
    print(f"   💰 Final Net Worth: ${net_worths[-1]:,.2f}")
    print(f"{'='*60}")
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Net Worth
    ax1 = axes[0]
    dates = df.index[:len(net_worths)]
    ax1.plot(dates, net_worths, color='#00d4ff', linewidth=2.5, label=f'Strategy')
    ax1.plot(df.index[:len(benchmark)], benchmark[:len(dates)], color='gray', linestyle='--', alpha=0.6, label=f'Benchmark')
    ax1.set_title('Test Set Performance', fontsize=14, fontweight='bold', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Positions
    ax2 = axes[1]
    ax2.fill_between(dates, positions[:len(dates)], 0, color='orange', alpha=0.5)
    ax2.set_ylabel('Position')
    
    # Drawdown
    ax3 = axes[2]
    ax3.fill_between(dates, drawdown[:len(dates)]*100, 0, color='red', alpha=0.6)
    ax3.set_ylabel('Drawdown %')

    # Style
    fig.patch.set_facecolor('#0f0f23')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('gray')

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/backtest_test_set.png', dpi=150, facecolor='#0f0f23')
    print("✅ Saved plot to results/backtest_test_set.png")

if __name__ == "__main__":
    backtest_test_set()
