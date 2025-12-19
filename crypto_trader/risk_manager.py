"""
Risk Manager Module | 风险管理模块

Implements tiered position limits based on portfolio drawdown.
Protects capital during adverse market conditions while preserving upside.

实现基于投资组合回撤的分级仓位限制。
在不利市场条件下保护本金，同时保留上涨潜力。
"""
from typing import Optional
import sys
from pathlib import Path

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import risk_config
except ImportError:
    # Fallback if config not found - use defaults
    def risk_config():
        return {}


class RiskManager:
    """
    Tiered Risk Management for Daily Trading | 日度交易分级风险管理
    
    Implements a multi-tier drawdown-based position limiting system:
    实现多层级基于回撤的仓位限制系统：
    
    Tiers | 层级:
        - Tier 1 (5% DD): Reduce max position to 80%
        - Tier 2 (10% DD): Reduce max position to 50%
        - Survival Mode (>max_drawdown_limit): Minimal position (20%)
        
        - 第一层（5% 回撤）：最大仓位降至 80%
        - 第二层（10% 回撤）：最大仓位降至 50%
        - 生存模式（>max_drawdown_limit）：最小仓位（20%）
    
    Attributes:
        max_drawdown_limit (float): Threshold for survival mode (default from config)
        freeze_period_steps (int): Steps to freeze trading after hard stop
        
    Note:
        Unlike hard stop-loss systems, this approach gradually reduces exposure
        rather than crystallizing losses at the worst moment.
        与硬止损系统不同，此方法逐渐减少敞口，而不是在最差时刻锁定损失。
    """

    def __init__(
        self, 
        max_drawdown_limit: Optional[float] = None, 
        freeze_period_steps: Optional[int] = None
    ):
        """
        Initialize Risk Manager with configurable parameters.
        使用可配置参数初始化风险管理器。
        
        Args:
            max_drawdown_limit: Maximum drawdown before survival mode.
                               If None, reads from config.
            freeze_period_steps: Steps to freeze after hard stop.
                                If None, reads from config.
        """
        cfg = risk_config()
        
        self.max_drawdown_limit = max_drawdown_limit or cfg.get('max_drawdown_limit', 0.10)
        self.freeze_period_steps = freeze_period_steps or cfg.get('freeze_period_steps', 3)
        
        # Tier thresholds from config | 层级阈值从配置读取
        self.tier1_threshold = cfg.get('tier1_threshold', 0.05)
        self.tier1_cap = cfg.get('tier1_cap', 0.8)
        self.tier2_threshold = cfg.get('tier2_threshold', 0.10)
        self.tier2_cap = cfg.get('tier2_cap', 0.5)
        
        self.freeze_counter = 0

    def check_risk(self, current_drawdown: float, proposed_action: float) -> Optional[float]:
        """
        Enforce tiered risk limits based on current drawdown.
        根据当前回撤执行分级风险限制。
        
        Args:
            current_drawdown (float): Current portfolio drawdown [0, 1]
                                     当前投资组合回撤 [0, 1]
            proposed_action (float): Agent's proposed position [-1, 1]
                                    代理建议的仓位 [-1, 1]
                                    
        Returns:
            Optional[float]: Constrained position if risk limits triggered,
                           None if no override needed.
                           如果触发风险限制则返回约束后的仓位，
                           无需覆盖则返回 None。
        """
        # Rule 1: Freeze Period active | 规则1：冻结期生效
        if self.freeze_counter > 0:
            self.freeze_counter -= 1
            return 0.0 

        # Rule 2: Tiered Drawdown Control | 规则2：分级回撤控制
        cap = 1.0
        
        if current_drawdown > self.tier1_threshold:
            cap = self.tier1_cap
            
        if current_drawdown > self.tier2_threshold:
            cap = self.tier2_cap
            
        # Hard Limit -> Survival Mode | 硬限制 -> 生存模式
        if current_drawdown > self.max_drawdown_limit:
            cap = 0.2  # Enough to recover, small enough to survive
            
        # Apply Cap | 应用上限
        if abs(proposed_action) > cap:
            return cap * (1 if proposed_action > 0 else -1)

        return None  # No override needed | 无需覆盖

    def reset(self):
        """
        Reset risk state for new episode.
        为新周期重置风险状态。
        """
        self.freeze_counter = 0

