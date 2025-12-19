
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Add parent dir to path to import legacy modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crypto_trader.data_loader import DataLoader
from crypto_trader.features import FeatureEngineer as FeatureEngineerV1
# from crypto_trader.experimental.experimental_features import FeatureEngineerV2

class FeatureEngineerV2:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        print("DEBUG: FeatureEngineerV2.add_technical_indicators STARTED (Embedded)")
        df = df.copy()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Shift all inputs by 1
        close = close.shift(1)
        high = high.shift(1)
        low = low.shift(1)
        volume = volume.shift(1)
        
        # --- 1. Basic V1 Features ---
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
        
        # --- 2. Advanced Features ---
        # ADX
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        # Use DataFrame/Series for alignment
        # vectors for comparison
        up_val = up.values
        down_val = down.values
        
        pos_dm = np.where((up_val > down_val) & (up_val > 0), up_val, 0.0)
        neg_dm = np.where((down_val > up_val) & (down_val > 0), down_val, 0.0)
        
        # Assign to DF to keep index
        df['_pos_dm'] = pos_dm
        df['_neg_dm'] = neg_dm
        
        tr_s = tr.rolling(14).mean()
        
        # Avoid division by zero
        tr_s = tr_s.replace(0, np.nan)
        
        pos_di = 100 * (df['_pos_dm'].rolling(14).mean() / tr_s)
        neg_di = 100 * (df['_neg_dm'].rolling(14).mean() / tr_s)
        
        denom = pos_di + neg_di
        denom = denom.replace(0, np.nan)
        
        dx = 100 * (abs(pos_di - neg_di) / denom)
        df['ADX'] = dx.rolling(14).mean()
        df['DI_Plus'] = pos_di
        df['DI_Minus'] = neg_di
        
        # Cleanup temp
        df.drop(columns=['_pos_dm', '_neg_dm'], inplace=True)
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp)
        
        # MFI
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        # Use numpy for conditional logic but assign back to Series with Index
        tp_curr = typical_price.values
        tp_prev = typical_price.shift(1).values
        rmf_val = raw_money_flow.values
        
        pos_flow = np.where(tp_curr > tp_prev, rmf_val, 0.0)
        neg_flow = np.where(tp_curr < tp_prev, rmf_val, 0.0)
        
        df['_pos_flow'] = pos_flow
        df['_neg_flow'] = neg_flow
        
        pos_mf = df['_pos_flow'].rolling(14).sum()
        neg_mf = df['_neg_flow'].rolling(14).sum()
        
        neg_mf = neg_mf.replace(0, np.nan) # Avoid div by zero
        mfi_ratio = pos_mf / neg_mf
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        df.drop(columns=['_pos_flow', '_neg_flow'], inplace=True)
        
        # Stats
        period_stats = 20
        df['Skew'] = df['Log_Ret'].rolling(period_stats).skew()
        df['Kurtosis'] = df['Log_Ret'].rolling(period_stats).kurt()
        
        # Lags
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
            df[f'Vol_Lag_{lag}'] = df['Vol_20'].shift(lag)
            
        print(f"Debug Embedded: Shape before dropna: {df.shape}")
        
        # Check nulls per col
        pd.set_option('display.max_rows', 500)
        print("Nulls per col:\n", df.isnull().sum())
        
        df = df.dropna()
        print(f"Debug Embedded: Shape after dropna: {df.shape}")
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Debug Embedded: Shape after inf drop: {df.shape}")
        
        return df

def train_and_eval(X, y, name):
    print(f"\n--- Training {name} ---")
    print(f"Features: {X.shape[1]}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    acc_scores = []
    log_losses = []
    f1_scores = []
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        print(f"Fold {fold}: Acc={acc:.4f}, LogLoss={ll:.4f}, F1={f1:.4f}")
        
        acc_scores.append(acc)
        log_losses.append(ll)
        f1_scores.append(f1)
        fold += 1
        
    avg_acc = np.mean(acc_scores)
    avg_ll = np.mean(log_losses)
    avg_f1 = np.mean(f1_scores)
    
    print(f"[{name}] Average: Acc={avg_acc:.4f}, LogLoss={avg_ll:.4f}, F1={avg_f1:.4f}")
    return avg_acc, avg_ll, avg_f1

def main():
    loader = DataLoader()
    print("Fetching Data...")
    # Use long history for better comparison
    df = loader.fetch_data("2020-01-01", "2024-01-01", "ETH-USD", interval="1d")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # V1
    print("Generating V1 Features...")
    eng_v1 = FeatureEngineerV1()
    df_v1 = eng_v1.add_technical_indicators(df.copy())
    
    # V2
    print("Generating V2 Features...")
    eng_v2 = FeatureEngineerV2()
    df_v2 = eng_v2.add_technical_indicators(df.copy())
    
    # Prepare Labels (Same for both)
    # Target: Close > Close.shift(1)? No, Close.shift(-1) > Close
    # We must align indices carefully as dropping NaNs changes length.
    
    def prepare_xy(data):
        d = data.copy()
        d['Target'] = (d['Close'].shift(-1) > d['Close']).astype(int)
        d = d.dropna()
        # Drop non-feature cols
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']
        feats = [c for c in d.columns if c not in exclude]
        return d[feats], d['Target']
    
    X1, y1 = prepare_xy(df_v1)
    X2, y2 = prepare_xy(df_v2)
    
    # Truncate to same length for fair comparison if needed?
    # Actually TimeSeriesSplit handles it, but let's just run.
    
    res1 = train_and_eval(X1, y1, "Baseline (V1)")
    res2 = train_and_eval(X2, y2, "Experimental (V2)")
    
    print("\n--- Final Comparison ---")
    print(f"Accuracy Diff: {(res2[0] - res1[0])*100:.2f}%")
    print(f"LogLoss Diff:  {res2[1] - res1[1]:.4f} (Lower is better)")

if __name__ == "__main__":
    main()
