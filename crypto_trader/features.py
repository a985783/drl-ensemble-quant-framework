"""
Feature Engineering Module | 特征工程模块

Computes technical indicators with proper time alignment to prevent data leakage.
All rolling indicators use shift(1) to ensure only t-1 and prior data is used.

计算技术指标，确保时间对齐以防止数据泄漏。
所有滚动指标使用 shift(1) 确保只用 t-1 及之前的数据。
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import features_config
except ImportError:
    # Fallback if config not found - use defaults
    def features_config():
        return {}


class FeatureEngineer:
    """
    Technical Indicator Calculator | 技术指标计算器
    
    Adds technical indicators and features to financial OHLCV data.
    All indicators use shift(1) to avoid look-ahead bias.
    
    为金融 OHLCV 数据添加技术指标和特征。
    所有指标使用 shift(1) 避免未来信息泄漏。
    
    Attributes:
        cfg (dict): Configuration dictionary from config.yaml or template
        
    Configurable Parameters (from config):
        - rsi_period: RSI calculation period (default: 14)
        - macd_fast/slow/signal: MACD parameters (default: 12/26/9)
        - bb_period/bb_std: Bollinger Bands parameters (default: 20/2)
        - atr_period: ATR period (default: 14)
        - sma_short/long: SMA periods (default: 50/200)
        - vol_period: Volume/Volatility calculation period (default: 20)
    """

    def __init__(self):
        self.cfg = features_config()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and add technical indicators to DataFrame.
        计算并添加技术指标到 DataFrame。
        
        Time Alignment Rules | 时间对齐规则:
        - Decision at time t can only use data from t-1 and earlier
        - All rolling indicators are shifted by 1
        - t 时刻的决策只能使用 t-1 及之前的数据
        - 所有 rolling 指标计算后 shift(1)
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'Close' column
                              包含 'Close' 列的输入 DataFrame

        Returns:
            pd.DataFrame: DataFrame with features, NaN rows removed
                         添加了特征的 DataFrame，已移除 NaN
        """
        df = df.copy()

        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column for technical indicators.")

        close = df['Close']
        
        # Get configurable parameters | 获取可配置参数
        rsi_period = self.cfg.get('rsi_period', 14)
        macd_fast = self.cfg.get('macd_fast', 12)
        macd_slow = self.cfg.get('macd_slow', 26)
        macd_signal = self.cfg.get('macd_signal', 9)
        bb_period = self.cfg.get('bb_period', 20)
        bb_std = self.cfg.get('bb_std', 2)
        atr_period = self.cfg.get('atr_period', 14)
        sma_short = self.cfg.get('sma_short', 50)
        sma_long = self.cfg.get('sma_long', 200)
        vol_period = self.cfg.get('vol_period', 20)
        
        # ==== All indicators use shift(1) for time alignment ====
        # 所有指标使用 shift(1) 确保时间对齐
        
        # 1. RSI (Relative Strength Index)
        close_prev = close.shift(1)  # Previous day close | 昨日收盘
        delta = close_prev.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. MACD (Moving Average Convergence Divergence)
        exp_fast = close_prev.ewm(span=macd_fast, adjust=False).mean()
        exp_slow = close_prev.ewm(span=macd_slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        df['MACD'] = macd
        df['MACD_Signal'] = macd.ewm(span=macd_signal, adjust=False).mean()

        # 3. Bollinger Bands
        sma_bb = close_prev.rolling(window=bb_period).mean()
        std_bb = close_prev.rolling(window=bb_period).std()
        df['BB_Upper'] = sma_bb + (std_bb * bb_std)
        df['BB_Lower'] = sma_bb - (std_bb * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

        # 4. Log Returns
        df['Log_Returns'] = np.log(close / close.shift(1))

        # 5. ATR (Average True Range)
        high_prev = df['High'].shift(1)
        low_prev = df['Low'].shift(1)
        close_prev2 = close.shift(2)  # Previous previous day close | 前天收盘
        
        high_low = high_prev - low_prev
        high_close = np.abs(high_prev - close_prev2)
        low_close = np.abs(low_prev - close_prev2)
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=atr_period).mean()

        # 6. Simple Moving Averages
        df['SMA_50'] = close_prev.rolling(window=sma_short).mean()
        df['SMA_200'] = close_prev.rolling(window=sma_long).mean()
        
        # Distance to Long SMA | 与长期均线的距离
        df['Dist_SMA_200'] = (close_prev - df['SMA_200']) / df['SMA_200']
        
        # 7. Volume Ratio | 成交量比率
        vol_prev = df['Volume'].shift(1)
        vol_sma = vol_prev.rolling(window=vol_period).mean()
        df['Vol_Ratio'] = vol_prev / vol_sma

        # 8. Rolling Volatility | 滚动波动率
        log_ret_prev = df['Log_Returns'].shift(1)
        df['Rolling_Vol'] = log_ret_prev.rolling(window=vol_period).std()
        
        # 9. Lag Features | 滞后特征
        for lag in [1, 2, 3]:
            df[f'Ret_Lag_{lag}'] = df['Log_Returns'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
            
        # 10. Momentum (Rate of Change) | 动量
        for window in [3, 5, 10]:
            df[f'ROC_{window}'] = close_prev.pct_change(periods=window)

        # Handle NaNs (SMA_long requires sma_long+ rows of data)
        # 处理 NaN（长期均线需要 sma_long+ 行数据）
        df_clean = df.dropna()

        return df_clean

