"""
Performance Attribution Module for Institutional Quant Trading
Calculates Alpha, Beta, and other key metrics against a benchmark.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class PerformanceAttribution:
    def __init__(self, strategy_net_worth: pd.Series, benchmark_prices: pd.Series, risk_free_rate: float = 0.0):
        """
        Args:
            strategy_net_worth: Time series of portfolio value (e.g. daily close).
            benchmark_prices: Time series of benchmark price (e.g. ETH close).
            risk_free_rate: Annualized risk-free rate (decimal, e.g. 0.02 for 2%).
        """
        self.equity = strategy_net_worth
        self.benchmark = benchmark_prices
        self.rf = risk_free_rate
        
        # Align data
        self.df = pd.DataFrame({'Equity': self.equity, 'Benchmark': self.benchmark}).dropna()
        
        # Calculate daily returns
        self.df['Ret_Strat'] = self.df['Equity'].pct_change().fillna(0)
        self.df['Ret_Bench'] = self.df['Benchmark'].pct_change().fillna(0)
        
    def calculate_metrics(self) -> dict:
        """
        Calculate institutional performance metrics.
        """
        strat_ret = self.df['Ret_Strat'].values
        bench_ret = self.df['Ret_Bench'].values
        
        # 1. Total Return
        total_ret = (self.df['Equity'].iloc[-1] / self.df['Equity'].iloc[0]) - 1
        bench_total_ret = (self.df['Benchmark'].iloc[-1] / self.df['Benchmark'].iloc[0]) - 1
        
        # 2. Beta & Alpha (CAPM)
        # Linear Regression: R_strat = alpha + beta * R_bench + epsilon
        slope, intercept, r_value, p_value, std_err = stats.linregress(bench_ret, strat_ret)
        beta = slope
        # Annualized Alpha (assuming 365 crypto days)
        alpha_daily = intercept
        alpha_annual = (1 + alpha_daily) ** 365 - 1
        
        # 3. Sharpe Ratio
        excess_ret = strat_ret - (self.rf / 365)
        std_dev = np.std(excess_ret, ddof=1)
        if std_dev == 0:
            sharpe = 0
        else:
            sharpe = (np.mean(excess_ret) / std_dev) * np.sqrt(365)
            
        # 4. Information Ratio (Active Return / Tracking Error)
        active_ret = strat_ret - bench_ret
        tracking_error = np.std(active_ret, ddof=1)
        if tracking_error == 0:
            info_ratio = 0
        else:
            info_ratio = (np.mean(active_ret) / tracking_error) * np.sqrt(365)
            
        # 5. Max Drawdown
        roll_max = self.df['Equity'].cummax()
        drawdown = (self.df['Equity'] - roll_max) / roll_max
        max_dd = drawdown.min()
        
        # 6. Calmar Ratio
        calmar = abs(total_ret / max_dd) if max_dd != 0 else 0
        
        return {
            "Total Return": total_ret,
            "Benchmark Return": bench_total_ret,
            "Alpha (Ann.)": alpha_annual,
            "Beta": beta,
            "Sharpe Ratio": sharpe,
            "Info Ratio": info_ratio,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "R-Squared": r_value ** 2
        }

    def generate_report(self, save_path: str = "performance_report.png"):
        """
        Generate a visual performance report.
        """
        metrics = self.calculate_metrics()
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Cumulative Returns (Top Full Width)
        ax1 = fig.add_subplot(gs[0, :])
        # Rebase to 1.0
        norm_strat = self.df['Equity'] / self.df['Equity'].iloc[0]
        norm_bench = self.df['Benchmark'] / self.df['Benchmark'].iloc[0]
        
        ax1.plot(norm_strat.index, norm_strat, label='Strategy', color='#FFD700', linewidth=2)
        ax1.plot(norm_bench.index, norm_bench, label='Benchmark', color='gray', linestyle='--', alpha=0.7)
        ax1.fill_between(norm_strat.index, norm_strat, norm_bench, where=(norm_strat >= norm_bench), color='green', alpha=0.1)
        ax1.fill_between(norm_strat.index, norm_strat, norm_bench, where=(norm_strat < norm_bench), color='red', alpha=0.1)
        ax1.set_title(f"Cumulative Performance (Total: {metrics['Total Return']:.1%})", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown (Middle Left)
        ax2 = fig.add_subplot(gs[1, 0])
        roll_max = self.df['Equity'].cummax()
        drawdown = (self.df['Equity'] - roll_max) / roll_max
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax2.set_title(f"Underwater Plot (Max DD: {metrics['Max Drawdown']:.1%})")
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Beta (Middle Right) - 30 Day
        ax3 = fig.add_subplot(gs[1, 1])
        if len(self.df) > 30:
            rolling_beta = self.df['Ret_Strat'].rolling(30).cov(self.df['Ret_Bench']) / self.df['Ret_Bench'].rolling(30).var()
            ax3.plot(rolling_beta.index, rolling_beta, color='blue', alpha=0.8)
            ax3.axhline(1, color='gray', linestyle='--', alpha=0.5)
            ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title("Rolling 30D Beta (Market Sensitivity)")
        else:
            ax3.text(0.5, 0.5, "Not enough data for Rolling Beta", ha='center')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap (Bottom Left) -> Text Metrics Table instead for simplicity here
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create Metrics Table
        table_data = [
            ["Metric", "Value", "Description"],
            ["Total Return", f"{metrics['Total Return']:.2%}", "Absolute profit"],
            ["Alpha (Annual)", f"{metrics['Alpha (Ann.)']:.2%}", "Excess return vs Beta"],
            ["Beta", f"{metrics['Beta']:.2f}", "Correlation to market (1.0 = same)"],
            ["Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", "Risk-adjusted return (>1 is good)"],
            ["Information Ratio", f"{metrics['Info Ratio']:.2f}", "Consistency of outperformance"],
            ["Max Drawdown", f"{metrics['Max Drawdown']:.2%}", "Worst peak-to-valley loss"],
            ["Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}", "Return / Max Drawdown"]
        ]
        
        table = ax4.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Coloring
        for i in range(len(table_data)):
            for j in range(3):
                if i == 0:
                    table[(i, j)].set_facecolor('#404040')
                    table[(i, j)].get_text().set_color('white')
                    table[(i, j)].get_text().set_weight('bold')
                else:
                    table[(i, j)].set_facecolor('#f5f5f5' if i % 2 else 'white')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ [Attribution] Report saved to {save_path}")
        # plt.close() # Keep open if needed or close to save memory
