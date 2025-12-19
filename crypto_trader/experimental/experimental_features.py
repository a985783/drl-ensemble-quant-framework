
"""
features_v2.py - Advanced Feature Engineering (Experimental)
Implementing ADX, CCI, Stochastic, MFI, Skew, Kurtosis manually to avoid talib dependency issues.
"""
import pandas as pd
import numpy as np

class FeatureEngineerV2:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        print("DEBUG: FeatureEngineerV2.add_technical_indicators STARTED")
        # raise Exception("I AM RUNNING") # Uncomment to force crash
        df = df.copy()
        
        # Base Data from t-1 to avoid leakage
        # All calculations use shifted raw data or shift the final result
        # To be safe and consistent with v1, we calculate on raw, then shift the RESULT
        # EXCEPT for things that essentially rely on today's close spread, which we shift at the end.
        
        # V1 Logic: 
        #   close_prev = close.shift(1)
        #   calculate_indicator(close_prev)
        # This is safe. We will follow this pattern.
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Shift all inputs by 1 to represent "Yesterday's Data" available at "Today's Open"
        close = close.shift(1)
        high = high.shift(1)
        low = low.shift(1)
        volume = volume.shift(1)
        
        # --- 1. Basic V1 Features (Optimized) ---
        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        df['MACD'] = macd
        df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_Width'] = (std20 * 4) / sma20 # Normalized width
        df['BB_Pos'] = (close - sma20) / (std20 * 2) # Z-score roughly
        
        # Log Returns
        df['Log_Ret'] = np.log(close / close.shift(1))
        
        # ATR (14)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # --- 2. Trend Strength (Advanced) ---
        # ADX (14)
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)
        
        # Smoothed DM/TR
        # Using simple rolling mean for approximation instead of Wilder's smoothing for speed/simplicity
        tr_s = tr.rolling(14).mean()
        pos_di = 100 * (pd.Series(pos_dm).rolling(14).mean() / tr_s)
        neg_di = 100 * (pd.Series(neg_dm).rolling(14).mean() / tr_s)
        
        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
        df['ADX'] = dx.rolling(14).mean()
        df['DI_Plus'] = pos_di
        df['DI_Minus'] = neg_di
        
        # CCI (20)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp)
        
        # --- 3. Oscillators ---
        # Stochastic (14, 3)
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        df['Kdj_K'] = k_percent.rolling(3).mean()
        df['Kdj_D'] = df['Kdj_K'].rolling(3).mean()
        
        # Williams %R (14)
        df['WillR'] = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # --- 4. Volume & Flow ---
        # MFI (14) - Money Flow Index (RSI of Volume-weighted prices)
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        # Check if typical price rose or fell
        tp_shift = typical_price.shift(1)
        pos_flow = np.where(typical_price > tp_shift, raw_money_flow, 0)
        neg_flow = np.where(typical_price < tp_shift, raw_money_flow, 0)
        
        pos_mf = pd.Series(pos_flow).rolling(14).sum()
        neg_mf = pd.Series(neg_flow).rolling(14).sum()
        mfi_ratio = pos_mf / neg_mf
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # OBV Slope (5 days)
        # Simplified: Volume if close > close_prev, -Volume if close < close_prev
        obv_sign = np.sign(close.diff())
        obv = (obv_sign * volume).cumsum()
        df['OBV_Slope'] = obv.diff(5) / 5 # Rate of change
        
        # --- 5. Advanced Stats ---
        period_stats = 20
        df['Skew'] = df['Log_Ret'].rolling(period_stats).skew()
        df['Kurtosis'] = df['Log_Ret'].rolling(period_stats).kurt()
        
        # --- 6. Extended Lags ---
        # Fibonacci Lags on Log Returns and Volatility
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
            df[f'Vol_Lag_{lag}'] = df['Vol_20'].shift(lag)
            
        print(f"Debug V2: Shape before dropna: {df.shape}")
        # Check nulls per col
        # print("Nulls per col:", df.isnull().sum().to_dict())
        
        # Drop NaNs
        df = df.dropna()
        print(f"Debug V2: Shape after dropna: {df.shape}")
        
        # Check for Infinity
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Debug V2: Shape after inf drop: {df.shape}")
        
        return df
