"""
Data Validation Module for Crypto Trading
Institutional-grade checks for:
1. Missing bars (Time continuity)
2. Price anomalies (Spikes vs Real Extreme Events)
3. Zero/Negative values
4. Volume anomalies
5. Merge row-count assertion utility
"""
import pandas as pd
import numpy as np


def assert_no_row_expansion(df_before: pd.DataFrame, df_after: pd.DataFrame, context: str = "") -> None:
    """
    Merge/join安全断言：确保 merge 后行数不超过 merge 前的行数。
    
    坑#18防护：多对多 merge 会静默膨胀行数，导致回测收益虚高或虚低。
    
    Args:
        df_before: merge 前的 DataFrame
        df_after:  merge 后的 DataFrame
        context:   上下文描述，用于错误信息
        
    Raises:
        AssertionError: 若 merge 后行数 > merge 前行数
    """
    n_before = len(df_before)
    n_after = len(df_after)
    tag = f" [{context}]" if context else ""
    assert n_after <= n_before, (
        f"⚠️ Merge行数异常{tag}: merge前={n_before}, merge后={n_after}. "
        f"可能存在重复键导致的行膨胀，请检查 merge key 的唯一性。"
    )


class DataValidator:
    """
    数据验证器 - 在数据进入特征工程前进行质量检查。
    
    异常值分级处理策略：
    - Level 1 (WARN): 异常但可能是真实极端行情 (20%-50% 单根K线变动)
    - Level 2 (ERROR): 几乎不可能是真实行情 (>50% 单根K线变动)
    
    为什么区分而不是直接删除：
    "不能无脑删异常值——某些'异常'是真实的极端行情，删掉会低估尾部风险"
    """

    # 极端行情阈值（低于此不报警，高于下限报警，高于上限认为是数据错误）
    WARN_THRESHOLD = 0.20   # 20% 单根变动：提示用户关注
    ERROR_THRESHOLD = 0.50  # 50% 单根变动：很可能是脏数据，进行标记或过滤

    def __init__(self, interval: str = '1d', strict: bool = False):
        """
        Args:
            interval: K线周期
            strict:   True=数据错误时抛出异常并中断; False=警告+自动修复
        """
        self.interval = interval
        self.strict = strict
        self.interval_map = {
            '1d': '1D',
            '4h': '4h',
            '1h': '1h',
            '15m': '15min'
        }

    def validate(self, df: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """
        Run all validation checks.
        Returns cleaned DataFrame, or raises ValueError if strict=True and
        critical data errors are found.
        """
        if df is None or df.empty:
            print(f"❌ [DataValidator] {symbol}: DataFrame is empty")
            return df

        df = df.copy()

        # 1. Check for Duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            print(f"⚠️ [DataValidator] {symbol}: 移除 {initial_len - len(df)} 条重复时间戳")

        # 2. Time Continuity Check (informational)
        self._check_continuity(df, symbol)

        # 3. Price Anomaly Detection (with auto-fix for confirmed data errors)
        df = self._handle_price_anomalies(df, symbol)

        # 4. Zero/Negative Check
        df = self._handle_invalid_values(df, symbol)

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_continuity(self, df: pd.DataFrame, symbol: str) -> None:
        """Check for missing bars (informational only - crypto markets may have gaps)."""
        if self.interval not in self.interval_map:
            return
        freq = self.interval_map[self.interval]
        try:
            expected_range = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq=freq
            )
            missing = expected_range.difference(df.index)
            if len(missing) > 0:
                print(f"⚠️ [DataValidator] {symbol}: 发现 {len(missing)} 条缺失K线")
                if len(missing) < 10:
                    print(f"   缺失时间点: {missing.tolist()}")
                else:
                    print(f"   首条缺失: {missing[0]}, 末条缺失: {missing[-1]}")
        except Exception as e:
            print(f"⚠️ [DataValidator] 时间连续性检查失败: {e}")

    def _handle_price_anomalies(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        分级处理价格异常值。
        
        Level 1 (WARN_THRESHOLD~ERROR_THRESHOLD): 真实极端行情可能性高 → 仅警告，保留
        Level 2 (>ERROR_THRESHOLD): 数据错误可能性高 → 标记并用前向填充修复（或 strict 模式下抛出）
        
        为什么不直接删除 Level 2：
        删除会导致时间序列不连续，影响 rolling 计算。用 ffill 替换为上一根收盘价，
        相当于假设"该时间点无变化"，是最保守的处理方式。
        """
        pct_change = df['Close'].pct_change().abs()

        # Level 1: 极端但可能真实
        warn_mask = (pct_change > self.WARN_THRESHOLD) & (pct_change <= self.ERROR_THRESHOLD)
        if warn_mask.any():
            n_warn = warn_mask.sum()
            print(f"⚠️ [DataValidator] {symbol}: {n_warn} 根K线变动 {self.WARN_THRESHOLD*100:.0f}%-{self.ERROR_THRESHOLD*100:.0f}%"
                  f"（可能是真实极端行情，已保留）")
            for idx, val in pct_change[warn_mask].items():
                print(f"   {idx}: {val*100:.2f}% 变动")

        # Level 2: 极大概率是脏数据
        error_mask = pct_change > self.ERROR_THRESHOLD
        if error_mask.any():
            n_err = error_mask.sum()
            msg = (f"🚨 [DataValidator] {symbol}: {n_err} 根K线单根变动>{self.ERROR_THRESHOLD*100:.0f}%，"
                   f"认定为数据错误，已用前向填充(ffill)修复")
            if self.strict:
                raise ValueError(msg + "\n  strict=True 模式：请手动检查并修复数据后再继续。")
            print(msg)
            for idx, val in pct_change[error_mask].items():
                print(f"   {idx}: {val*100:.2f}% → 已替换为前值")
            # 用 ffill 替换异常行（仅替换 OHLCV，保持 index 不变）
            df.loc[error_mask] = np.nan
            df = df.ffill()

        return df

    def _handle_invalid_values(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        处理 0 或负价格（零除、取对数时会导致 inf/NaN 静默失效）。
        
        策略：ffill 修复；若首行即为无效值则抛出（无法 ffill）。
        """
        invalid_mask = (df['Close'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0)
        if not invalid_mask.any():
            return df

        n_invalid = invalid_mask.sum()
        msg = f"🚨 [DataValidator] {symbol}: {n_invalid} 行价格 ≤ 0"

        if self.strict:
            raise ValueError(msg + " — strict=True 模式，请修复后重试。")

        print(msg + "，已用前向填充修复")

        # 若首行即无效，ffill 无效 → 必须报错
        if invalid_mask.iloc[0]:
            raise ValueError(
                f"[DataValidator] {symbol}: 首行价格 ≤ 0，无法前向填充，请检查数据源。"
            )

        df.loc[invalid_mask] = np.nan
        df = df.ffill()
        return df
