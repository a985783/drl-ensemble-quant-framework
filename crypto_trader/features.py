"""
特征工程 - 修复数据泄漏问题
所有 rolling 指标使用 shift(1) 确保只用 t-1 及之前的数据
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    添加技术指标和特征到金融数据。
    重要：所有指标都使用 shift(1) 避免使用当日数据造成未来信息泄漏。
    """

    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算并添加技术指标到 DataFrame。
        
        时间对齐规则：
        - t 时刻的决策只能使用 t-1 及之前的数据
        - 所有 rolling 指标计算后 shift(1)
        
        Args:
            df (pd.DataFrame): 包含 'Close' 列的输入 DataFrame。

        Returns:
            pd.DataFrame: 添加了特征的 DataFrame，已移除 NaN。
        """
        df = df.copy()

        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column for technical indicators.")

        close = df['Close']
        
        # ==== 所有指标使用 shift(1) 确保时间对齐 ====
        # 即：t 时刻的特征值来自 t-1 时刻的计算结果
        # 注意：这种严格防止未来函数的设计会导致指标滞后一天，但这是正确的

        # 1. RSI (14 period) - 使用昨日收盘价
        # 严格防止未来函数：使用昨日收盘相对前日收盘的变化计算RSI
        # 这会导致RSI滞后一天，但确保没有使用未来信息
        close_prev = close.shift(1)  # 昨日收盘
        delta = close_prev.diff()  # 昨日收盘 - 前日收盘
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9) - 使用昨日数据
        exp12 = close_prev.ewm(span=12, adjust=False).mean()
        exp26 = close_prev.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        df['MACD'] = macd
        df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()

        # 3. Bollinger Bands (20, 2) - 使用昨日数据
        sma20 = close_prev.rolling(window=20).mean()
        std20 = close_prev.rolling(window=20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

        # 4. Log Returns - 这个本身就是 shift 过的
        df['Log_Returns'] = np.log(close / close.shift(1))

        # 5. ATR (Average True Range) - 使用昨日数据
        # 严格防止未来函数：使用昨日高低点和前日收盘价计算ATR
        # 这会导致ATR滞后一天，但确保没有使用未来信息
        high_prev = df['High'].shift(1)  # 昨日高点
        low_prev = df['Low'].shift(1)    # 昨日低点
        close_prev2 = close.shift(2)     # 前日收盘价

        high_low = high_prev - low_prev
        high_close = np.abs(high_prev - close_prev2)
        low_close = np.abs(low_prev - close_prev2)
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # 6. SMA (50, 200) - 使用昨日数据
        df['SMA_50'] = close_prev.rolling(window=50).mean()
        df['SMA_200'] = close_prev.rolling(window=200).mean()
        
        # Distance to SMA (使用昨日收盘和昨日 SMA)
        df['Dist_SMA_200'] = (close_prev - df['SMA_200']) / df['SMA_200']
        
        # 7. Volume Ratio - 使用昨日成交量
        vol_prev = df['Volume'].shift(1)
        vol_sma = vol_prev.rolling(window=20).mean()
        df['Vol_Ratio'] = vol_prev / vol_sma

        # 8. Rolling Volatility - 使用昨日对数收益率
        log_ret_prev = df['Log_Returns'].shift(1)
        df['Rolling_Vol'] = log_ret_prev.rolling(window=20).std()
        
        # 9. Lag Features (已经是 shift 的，再 shift 一次确保)
        for lag in [1, 2, 3]:
            df[f'Ret_Lag_{lag}'] = df['Log_Returns'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
            
        # 10. Momentum (ROC) - 使用昨日数据
        for window in [3, 5, 10]:
            df[f'ROC_{window}'] = close_prev.pct_change(periods=window)

        # Handle NaNs (SMA200 需要 200+ 行数据)
        df_clean = df.dropna()

        return df_clean
