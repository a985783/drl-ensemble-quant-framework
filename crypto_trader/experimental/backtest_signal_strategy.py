
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Add parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from crypto_trader.data_loader import DataLoader
from crypto_trader.experimental.tune_v2 import FeatureEngineerV2

def backtest():
    print("Loading Data...")
    loader = DataLoader()
    # Fetch including recent data for OOS test
    df = loader.fetch_data("2020-01-01", "2025-01-01", "ETH-USD", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    print("Generating Features...")
    eng = FeatureEngineerV2()
    df_feat = eng.add_technical_indicators(df)
    
    # Label for training
    df_feat['Target'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
    df_feat = df_feat.dropna()
    
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']
    feats = [c for c in df_feat.columns if c not in exclude]
    
    X = df_feat[feats]
    y = df_feat['Target']
    close_price = df_feat['Close']
    
    # Split Train/Test (Last 1 year as Test)
    test_size = 365
    train_idx = range(0, len(df_feat) - test_size)
    test_idx = range(len(df_feat) - test_size, len(df_feat))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    prices_test = close_price.iloc[test_idx]
    dates_test = df_feat.index[test_idx]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples (Last 365 days).")
    
    # Train with Best Params
    best_params = {
        'subsample': 0.6, 
        'n_estimators': 200, 
        'min_child_weight': 1, 
        'max_depth': 5, 
        'learning_rate': 0.01, 
        'gamma': 0.1, 
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Predict
    probs = model.predict_proba(X_test)[:, 1]
    
    # Strategy Logic
    # Thresholds
    LONG_THRESH = 0.55
    SHORT_THRESH = 0.45
    
    position = np.zeros(len(probs))
    
    # Vectorized logic
    position = np.where(probs > LONG_THRESH, 1, np.where(probs < SHORT_THRESH, -1, 0))
    
    # Shift position by 1 because prediction is for "Next Day", 
    # but we take action at Close of Today (or Open of Tomorrow).
    # If target is (Close_t+1 > Close_t), then prediction at t guides action at t.
    # So PnL at t+1 = Position_t * Return_t+1
    
    # Calculate Returns
    # Return at t is (Close_t - Close_t-1) / Close_t-1
    market_returns = prices_test.pct_change()
    
    # Strategy Return
    # Position chosen at t-1 (based on data up to t-1) applies to return at t.
    # Our 'position' array aligns with X_test (dates).
    # X_test[0] is data at t=0. Pred[0] is for return t=0->t=1.
    # So Strategy Return at t=1 is Position[0] * MarketReturn[1].
    
    # Align arrays
    pos_series = pd.Series(position, index=dates_test)
    ret_series = market_returns # Aligned by index
    
    # Shift position to align with future return
    # Position at date D drives PnL on date D+1
    strat_ret = pos_series.shift(1) * ret_series
    
    # Cumulative
    strat_cum = (1 + strat_ret).cumprod()
    market_cum = (1 + ret_series).cumprod()
    
    final_strat = strat_cum.iloc[-1]
    final_market = market_cum.iloc[-1]
    
    print("\n--- Backtest Results (Simple Signal Strategy) ---")
    print(f"Period: {dates_test[0].date()} to {dates_test[-1].date()}")
    print(f"Strategy Return: {(final_strat - 1)*100:.2f}%")
    print(f"Market Return:   {(final_market - 1)*100:.2f}%")
    
    # Stats
    wins = strat_ret[strat_ret > 0]
    losses = strat_ret[strat_ret < 0]
    win_rate = len(wins) / (len(wins) + len(losses))
    print(f"Win Rate: {win_rate*100:.2f}%")
    
    # Feature Importance
    print("\nFeature Importance:")
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print(imp.head(10))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(market_cum, label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(strat_cum, label='XGBoost V2 Strat', color='blue')
    plt.title('V2 Signal Model Strategy vs Market (Out-of-Sample)')
    plt.legend()
    plt.savefig('results/v2_signal_backtest.png')
    print("Plot saved to results/v2_signal_backtest.png")

if __name__ == "__main__":
    backtest()
