
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import warnings

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from crypto_trader.data_loader import DataLoader

# Embed FeatureEngineerV2 directly to avoid import ghosts
class FeatureEngineerV2:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Shift inputs
        close = close.shift(1)
        high = high.shift(1)
        low = low.shift(1)
        volume = volume.shift(1)
        
        # V1 Features
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        df['MACD'] = macd
        df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()
        
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_Width'] = (std20 * 4) / sma20 
        df['BB_Pos'] = (close - sma20) / (std20 * 2) 
        
        df['Log_Ret'] = np.log(close / close.shift(1))
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # V2 Advanced
        up = high - high.shift(1)
        down = low.shift(1) - low
        up_val = up.values
        down_val = down.values
        pos_dm = np.where((up_val > down_val) & (up_val > 0), up_val, 0.0)
        neg_dm = np.where((down_val > up_val) & (down_val > 0), down_val, 0.0)
        df['_pos_dm'] = pos_dm
        df['_neg_dm'] = neg_dm
        
        tr_s = tr.rolling(14).mean().replace(0, np.nan)
        pos_di = 100 * (df['_pos_dm'].rolling(14).mean() / tr_s)
        neg_di = 100 * (df['_neg_dm'].rolling(14).mean() / tr_s)
        denom = (pos_di + neg_di).replace(0, np.nan)
        dx = 100 * (abs(pos_di - neg_di) / denom)
        df['ADX'] = dx.rolling(14).mean()
        df['DI_Plus'] = pos_di
        df['DI_Minus'] = neg_di
        df.drop(columns=['_pos_dm', '_neg_dm'], inplace=True)
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp)
        
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        tp_curr = typical_price.values
        tp_prev = typical_price.shift(1).values
        rmf_val = raw_money_flow.values
        pos_flow = np.where(tp_curr > tp_prev, rmf_val, 0.0)
        neg_flow = np.where(tp_curr < tp_prev, rmf_val, 0.0)
        df['_pos_flow'] = pos_flow
        df['_neg_flow'] = neg_flow
        pos_mf = df['_pos_flow'].rolling(14).sum()
        neg_mf = df['_neg_flow'].rolling(14).sum().replace(0, np.nan)
        mfi_ratio = pos_mf / neg_mf
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df.drop(columns=['_pos_flow', '_neg_flow'], inplace=True)
        
        period_stats = 20
        df['Skew'] = df['Log_Ret'].rolling(period_stats).skew()
        df['Kurtosis'] = df['Log_Ret'].rolling(period_stats).kurt()
        
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
            df[f'Vol_Lag_{lag}'] = df['Vol_20'].shift(lag)
            
        return df.replace([np.inf, -np.inf], np.nan).dropna()

def tune():
    print("Loading Data...")
    loader = DataLoader()
    df = loader.fetch_data("2020-01-01", "2024-01-01", "ETH-USD", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    print("Generating V2 Features...")
    eng = FeatureEngineerV2()
    df_feat = eng.add_technical_indicators(df)
    
    # Prepare X, y
    df_feat['Target'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
    df_feat = df_feat.dropna()
    
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']
    feats = [c for c in df_feat.columns if c not in exclude]
    
    X = df_feat[feats]
    y = df_feat['Target']
    
    # Check balance
    print(f"Class Balance: {y.value_counts(normalize=True).to_dict()}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    print("Starting RandomizedSearchCV (20 iter)...")
    
    # Manual TimeSeries Split for Search
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss', 
        use_label_encoder=False,
        n_jobs=-1
    )
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='neg_log_loss', # Optimize for probability
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_scaled, y)
    
    print("\n--- Tuning Results ---")
    print(f"Best LogLoss: {-search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")
    
    # Validation with Best Params
    best_model = search.best_estimator_
    
    # Calculate Accuracy with explicit predict
    # Need to manual split to be sure? 
    # Use the best_index_ to invoke CV results or just trust CV score
    
    # Let's print the Mean Test Accuracy for the best model too (if we had multitarget scoring)
    # Instead, let's run a quick TimeSeries CV with best params to get metrics
    print("\nVerifying Best Params on CV...")
    
    acc_scores = []
    
    for train_idx, test_idx in tscv.split(X_scaled):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        best_model.fit(X_tr, y_tr, verbose=False)
        preds = best_model.predict(X_te)
        acc_scores.append(accuracy_score(y_te, preds))
        
    print(f"Best Model CV Accuracy: {np.mean(acc_scores):.4f}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    tune()
