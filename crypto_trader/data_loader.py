"""
数据加载器 - 从 OKX 获取永续合约历史数据
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

class DataLoader:
    """
    从 OKX 获取永续合约历史数据
    （与实盘同口径）
    """

    def __init__(self):
        load_dotenv()
        # 使用主网获取历史数据（模拟盘历史数据有限）
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
        })
        # 注意：获取历史K线不需要API Key，公开数据
        
        self._markets_loaded = False

    def _ensure_markets(self):
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def fetch_data(self, start_date: str, end_date: str, ticker: str, interval: str = "1d") -> pd.DataFrame:
        """
        获取永续合约历史K线数据
        
        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            ticker: 交易对 (如 'ETH-USD' 会自动转换为 'ETH/USDT:USDT')
            interval: K线周期 ('1d', '4h', '1h')
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        # 转换 ticker 格式
        # Yahoo 格式: ETH-USD, BTC-USD
        # CCXT 格式: ETH/USDT:USDT (永续合约)
        base = ticker.split('-')[0]  # ETH, BTC, SOL
        symbol = f"{base}/USDT:USDT"
        
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
        df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        
        # 去重并过滤到指定范围
        df = df[~df.index.duplicated(keep='first')]
        df = df[df.index <= end_date]
        
        print(f"【OKX】共获取 {len(df)} 条 {timeframe} 数据")
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# 测试
if __name__ == "__main__":
    loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = loader.fetch_data(start_date, end_date, 'ETH-USD', interval='1d')
    print("\n永续合约 K 线数据:")
    print(df.tail(10))
