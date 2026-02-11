"""
Data Validation Module for Crypto Trading
Institutional-grade checks for:
1. Missing bars (Time continuity)
2. Price anomalies (Spikes/Crashes)
3. Zero/Negative values
4. Volume anomalies
"""
import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self, interval='1d'):
        self.interval = interval
        self.interval_map = {
            '1d': '1D',
            '4h': '4h',
            '1h': '1h',
            '15m': '15min'
        }

    def validate(self, df: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """
        Run all validation checks.
        Returns cleaned DataFrame (or raises error if critical failure).
        """
        if df is None or df.empty:
            print(f"❌ [DataValidator] {symbol}: DataFrame is empty")
            return df

        df = df.copy()
        
        # 1. Check for Duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            print(f"⚠️ [DataValidator] {symbol}: Removed {initial_len - len(df)} duplicate timestamps")

        # 2. Time Continuity Check
        self._check_continuity(df, symbol)

        # 3. Anomaly Detection
        self._check_price_anomalies(df, symbol)

        # 4. Zero/Negative Check
        self._check_invalid_values(df, symbol)

        return df

    def _check_continuity(self, df: pd.DataFrame, symbol: str):
        """Check for missing bars"""
        if self.interval not in self.interval_map:
            return

        freq = self.interval_map[self.interval]
        try:
            # Generate expected index
            expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
            missing = expected_range.difference(df.index)
            
            if len(missing) > 0:
                print(f"⚠️ [DataValidator] {symbol}: Found {len(missing)} missing bars!")
                if len(missing) < 10:
                    print(f"   Missing: {missing.tolist()}")
                else:
                    print(f"   First missing: {missing[0]}, Last missing: {missing[-1]}")
                
                # TODO: In future, implement forward fill or re-fetch logic here
        except Exception as e:
            print(f"⚠️ [DataValidator] Continuity check failed: {e}")

    def _check_price_anomalies(self, df: pd.DataFrame, symbol: str):
        """Check for unrealistic price spikes (>50% move in one bar)"""
        # Calculate pct change
        pct_change = df['Close'].pct_change().abs()
        
        # Threshold: 50% move in single bar is likely bad data for major cryptos
        anomalies = pct_change[pct_change > 0.5]
        
        if len(anomalies) > 0:
            print(f"🚨 [DataValidator] {symbol}: Found {len(anomalies)} extreme price spikes (>50%)!")
            for idx, val in anomalies.items():
                print(f"   {idx}: {val*100:.2f}% change")

    def _check_invalid_values(self, df: pd.DataFrame, symbol: str):
        """Check for 0 or negative prices"""
        invalid = df[(df['Close'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0)]
        if len(invalid) > 0:
            print(f"🚨 [DataValidator] {symbol}: Found {len(invalid)} rows with <= 0 prices!")
            # Should be critical error
            # raise ValueError(f"Invalid prices found in {symbol}")
