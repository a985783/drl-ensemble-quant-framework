"""
OKX 永续合约数据加载器
- 获取永续合约历史K线（与实盘同口径）
- 获取资金费率历史
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

class OKXDataLoader:
    def __init__(self):
        load_dotenv()
        self.exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET_KEY'),
            'password': os.getenv('OKX_PASSPHRASE'),
            'enableRateLimit': True,
        })
        # 使用模拟盘或主网取决于配置
        if os.getenv('OKX_DEMO_MODE') == 'True':
            self.exchange.set_sandbox_mode(True)
        
    def fetch_perpetual_ohlcv(self, symbol='ETH/USDT:USDT', timeframe='1d', 
                               start_date='2020-01-01', end_date=None):
        """
        获取永续合约历史K线数据
        
        Args:
            symbol: CCXT 格式的永续合约符号
            timeframe: K线周期 ('1d', '4h', '1h' 等)
            start_date: 开始日期
            end_date: 结束日期
        """
        print(f"【OKX】正在获取 {symbol} 永续合约历史数据...")
        
        self.exchange.load_markets()
        
        # 转换日期为时间戳
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date or datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d').timestamp() * 1000)
        
        all_ohlcv = []
        current_since = since
        
        while current_since < until:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=300)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1  # 下一个时间点
                print(f"  已获取 {len(all_ohlcv)} 条数据...")
            except Exception as e:
                print(f"  获取出错: {e}")
                break
        
        # 转为 DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        
        # 去重
        df = df[~df.index.duplicated(keep='first')]
        
        print(f"【OKX】共获取 {len(df)} 条 {timeframe} 数据")
        return df
    
    def fetch_funding_rate_history(self, symbol='ETH-USDT-SWAP', start_date='2020-01-01', end_date=None):
        """
        获取资金费率历史
        OKX 资金费率每 8 小时结算一次
        
        Returns:
            DataFrame with columns: [timestamp, fundingRate]
        """
        print(f"【OKX】正在获取 {symbol} 资金费率历史...")
        
        # OKX 的资金费率 API
        # 注意：CCXT 标准化接口可能不完全支持，可能需要用 private API
        
        all_rates = []
        
        try:
            # 使用 CCXT 的 fetch_funding_rate_history (如果支持)
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            until = int(datetime.strptime(end_date or datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d').timestamp() * 1000)
            
            current_since = since
            while current_since < until:
                try:
                    # 尝试标准 CCXT 接口
                    rates = self.exchange.fetch_funding_rate_history('ETH/USDT:USDT', since=current_since, limit=100)
                    if not rates:
                        break
                    all_rates.extend(rates)
                    current_since = rates[-1]['timestamp'] + 1
                    print(f"  已获取 {len(all_rates)} 条资金费率...")
                except Exception as e:
                    print(f"  资金费率获取出错: {e}")
                    break
                    
        except Exception as e:
            print(f"【警告】无法获取资金费率历史: {e}")
            print("【提示】将使用估算值 (0.01% / 8h = 0.03% / day)")
            return None
        
        if not all_rates:
            return None
            
        df = pd.DataFrame(all_rates)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        print(f"【OKX】共获取 {len(df)} 条资金费率数据")
        return df
    
    def aggregate_daily_funding(self, funding_df):
        """
        将 8 小时资金费率聚合为日级别
        
        永续合约特点：
        - 每 8 小时结算一次 (00:00, 08:00, 16:00 UTC)
        - 持有多头时，如果费率为正，需要支付给空头
        - 持有空头时，如果费率为正，收到多头支付
        """
        if funding_df is None:
            return None
            
        daily = funding_df.resample('D')['fundingRate'].sum()
        return daily
    
    def get_combined_data(self, start_date='2020-01-01', end_date=None):
        """
        获取 OHLCV + 资金费率的组合数据
        """
        # 1. 获取 K线
        ohlcv_df = self.fetch_perpetual_ohlcv(start_date=start_date, end_date=end_date)
        
        # 2. 获取资金费率
        funding_df = self.fetch_funding_rate_history(start_date=start_date, end_date=end_date)
        daily_funding = self.aggregate_daily_funding(funding_df)
        
        # 3. 合并
        if daily_funding is not None:
            ohlcv_df['FundingRate'] = daily_funding
            ohlcv_df['FundingRate'].fillna(0.0003, inplace=True)  # 默认 0.03%/天
        else:
            # 使用经验估算值
            ohlcv_df['FundingRate'] = 0.0003  # 默认 0.03%/天 (牛市通常正费率)
        
        return ohlcv_df


def test_loader():
    """测试数据加载器"""
    loader = OKXDataLoader()
    
    # 获取最近 30 天数据作为测试
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = loader.fetch_perpetual_ohlcv(start_date=start_date, end_date=end_date)
    print("\n永续合约 K 线数据:")
    print(df.tail())
    
    funding = loader.fetch_funding_rate_history(start_date=start_date, end_date=end_date)
    if funding is not None:
        print("\n资金费率数据:")
        print(funding.tail())


if __name__ == "__main__":
    test_loader()
