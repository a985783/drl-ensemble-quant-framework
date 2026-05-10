"""
数据加载器 - 从 OKX 获取永续合约历史数据
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
try:
    from crypto_trader.data_validator import DataValidator
    from crypto_trader.ccxt_utils import apply_ccxt_proxy_config, resolve_ccxt_proxies
except ImportError:
    from data_validator import DataValidator
    from ccxt_utils import apply_ccxt_proxy_config, resolve_ccxt_proxies

class DataLoader:
    """
    从 OKX 获取永续合约历史数据
    （与实盘同口径）
    """

    def __init__(self):
        load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))
        # 使用主网获取历史数据（模拟盘历史数据有限）
        exchange_cfg = {
            'enableRateLimit': True,
        }
        exchange_cfg = apply_ccxt_proxy_config(exchange_cfg)
        proxies = resolve_ccxt_proxies()
        if proxies:
            print(f"【网络】DataLoader 使用代理: {proxies}")

        self.exchange = ccxt.okx(exchange_cfg)
        # 注意：获取历史K线不需要API Key，公开数据
        print("【DataLoader】初始化: 强制使用 OKX 实盘 API (Mainnet) 获取市场数据")

        
        self._markets_loaded = False

    def _ensure_markets(self):
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def _resolve_symbol(self, ticker: str) -> str:
        """
        Resolve input ticker to OKX perpetual CCXT symbol.

        Accepted formats:
        - CCXT perp: "ETH/USDT:USDT"
        - CCXT spot: "ETH/USDT" (mapped to perp)
        - Legacy: "ETH-USD", "ETH-USDT", "ETH-USDT-SWAP"
        """
        if "/" in ticker:
            if ":" in ticker:
                return ticker
            parts = ticker.split("/")
            if len(parts) == 2 and parts[1].upper() == "USDT":
                return f"{parts[0]}/USDT:USDT"
            return ticker

        upper = ticker.upper()
        if upper.endswith("-SWAP"):
            base = upper.split("-")[0]
            return f"{base}/USDT:USDT"

        base = ticker.split("-")[0]
        return f"{base}/USDT:USDT"

    def fetch_funding_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取 OKX 永续合约历史 Funding Rate。
        
        OKX 每 8 小时结算一次 (UTC 00:00, 08:00, 16:00)，即每日 3 次。
        返回按日聚合的 Funding_Rate（日绝对成本 = 3 次结算费率绝对值之和）。
        
        Args:
            symbol: CCXT 永续合约格式，如 'ETH/USDT:USDT'
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            
        Returns:
            DataFrame indexed by date with column 'Funding_Rate' (daily absolute cost)
        """
        print(f"【OKX】正在获取 {symbol} 历史 Funding Rate...")
        
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_funding = []
        current_since = since
        
        while current_since < until:
            try:
                # OKX: fetch_funding_rate_history via CCXT
                funding = self.exchange.fetch_funding_rate_history(
                    symbol, since=current_since, limit=100
                )
                if not funding:
                    break
                all_funding.extend(funding)
                # Move past the last fetched timestamp
                current_since = funding[-1]['timestamp'] + 1
            except Exception as e:
                print(f"【警告】获取 Funding Rate 出错: {e}")
                break
        
        if not all_funding:
            print("【警告】未获取到 Funding Rate 数据，将使用固定值")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for f in all_funding:
            ts = pd.Timestamp(f['timestamp'], unit='ms', tz='UTC')
            rate = float(f.get('fundingRate', 0.0))
            records.append({'Timestamp': ts, 'funding_rate_8h': rate})
        
        fr_df = pd.DataFrame(records)
        fr_df['date'] = fr_df['Timestamp'].dt.normalize()
        
        # 按日聚合：取每日所有 8 小时费率的绝对值之和作为日成本
        # 因为无论多/空，持仓成本是费率的绝对值
        # 注意：实际上多头在正费率时付出、空头在正费率时收取，但我们用绝对值
        # 作为保守估计（最差情况成本），方向在 TradingEnv 中已由 abs(pos) 处理
        daily_funding = fr_df.groupby('date')['funding_rate_8h'].agg(
            Funding_Rate=lambda x: x.abs().sum()
        ).reset_index()
        daily_funding = daily_funding.set_index('date')
        daily_funding.index = pd.DatetimeIndex(daily_funding.index, tz='UTC')
        
        print(f"【OKX】获取 {len(daily_funding)} 天的 Funding Rate 数据 "
              f"(均值: {daily_funding['Funding_Rate'].mean()*100:.4f}%/天, "
              f"最大: {daily_funding['Funding_Rate'].max()*100:.4f}%/天)")
        
        return daily_funding

    def fetch_data(self, start_date: str, end_date: str, ticker: str, 
                   interval: str = "1d", include_funding: bool = True) -> pd.DataFrame:
        """
        获取永续合约历史K线数据
        
        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            ticker: 交易对 (推荐 'ETH/USDT:USDT'；旧格式会自动映射到永续合约)
            interval: K线周期 ('1d', '4h', '1h')
            include_funding: 是否获取并附加历史 Funding Rate (默认 True)
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume [, Funding_Rate]
        """
        # 转换 ticker 格式 (统一到 OKX 永续合约)
        symbol = self._resolve_symbol(ticker)
        if symbol != ticker:
            print(f"【OKX】Symbol 映射: {ticker} -> {symbol}")
        
        print(f"【OKX】正在获取 {symbol} 永续合约数据 ({start_date} 到 {end_date})...")
        
        self._ensure_markets()
        
        # 转换时间间隔格式
        timeframe_map = {
            '1d': '1d',
            '1h': '1h',
            '4h': '4h',
            '15m': '15m',
        }
        timeframe = timeframe_map.get(interval, '1d')
        
        # 转换日期为时间戳
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_ohlcv = []
        current_since = since
        
        while current_since < until:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=300)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
            except Exception as e:
                print(f"【警告】获取数据出错: {e}")
                break
        
        if not all_ohlcv:
            print("【警告】未获取到任何数据")
            return pd.DataFrame()
        
        # 转为 DataFrame
        # 坑#4 注释: OKX 返回的 Timestamp 是 Bar 的【开始时间】(bar-start)，非结束时间。
        # 例如日线 2024-01-15 00:00:00 UTC 代表该日 bar 从此刻开始。
        # features.py 中使用 shift(1) 确保 t 时刻只用 t-1 bar 结束后的数据，时序正确。
        df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
        df.set_index('Timestamp', inplace=True)
        
        # 去重并过滤到指定范围
        df = df[~df.index.duplicated(keep='first')]
        df = df[df.index <= end_date]
        
        print(f"【OKX】共获取 {len(df)} 条 {timeframe} 数据")
        
        # === Data Validation ===
        validator = DataValidator(interval=interval)
        df = validator.validate(df, symbol=symbol)
        # =======================

        # === 附加历史 Funding Rate ===
        out_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if include_funding and interval == '1d':
            try:
                fr_df = self.fetch_funding_history(symbol, start_date, end_date)
                if not fr_df.empty:
                    # 按日期 merge (left join，缺失日用固定值填充)
                    n_before = len(df)
                    df = df.join(fr_df[['Funding_Rate']], how='left')
                    assert len(df) == n_before, (
                        f"Funding Rate merge 导致行数变化: {n_before} -> {len(df)}"
                    )
                    # 缺失日用固定 fallback 值填充（ffill 是安全的，不含未来信息）
                    df['Funding_Rate'] = df['Funding_Rate'].fillna(0.0003)
                    out_cols.append('Funding_Rate')
                    print(f"【OKX】Funding Rate 已附加到数据 ({(df['Funding_Rate'] != 0.0003).sum()} 天有真实数据)")
            except Exception as e:
                print(f"【警告】Funding Rate 获取失败，使用固定值: {e}")

        return df[[c for c in out_cols if c in df.columns]]


# 测试
if __name__ == "__main__":
    loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = loader.fetch_data(start_date, end_date, 'ETH/USDT:USDT', interval='1d')
    print("\n永续合约 K 线数据:")
    print(df.tail(10))
