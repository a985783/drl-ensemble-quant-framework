
class RiskManager:
    """
    Tiered Risk Management for Daily Trading.
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.10,
        freeze_period_steps: int = 3,
        tier1_drawdown: float = 0.05,
        tier1_limit: float = 0.8,
        tier2_drawdown: float = 0.10,
        tier2_limit: float = 0.5,
        survival_drawdown: float = 0.15,
        survival_limit: float = 0.2,
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.freeze_period_steps = freeze_period_steps
        self.tier1_drawdown = tier1_drawdown
        self.tier1_limit = tier1_limit
        self.tier2_drawdown = tier2_drawdown
        self.tier2_limit = tier2_limit
        self.survival_drawdown = survival_drawdown
        self.survival_limit = survival_limit
        self.freeze_counter = 0

    def check_risk(self, current_drawdown: float, proposed_action: float):
        """
        Enforces tiered risk limits.
        
        Tiers (soft caps, no forced liquidation):
        - DD > 5%: Max Leverage 0.8x
        - DD > 10%: Max Leverage 0.5x
        - DD > max_limit: Max Leverage 0.2x (survival mode)
        """
        # Rule 1: Freeze Period active
        if self.freeze_counter > 0:
            self.freeze_counter -= 1
            return 0.0 

        # Rule 2: Tiered Drawdown Control
        cap = 1.0
        if current_drawdown > self.tier1_drawdown:
            cap = self.tier1_limit
        if current_drawdown > self.tier2_drawdown:
            cap = self.tier2_limit

        survival_th = self.survival_drawdown if self.survival_drawdown is not None else self.max_drawdown_limit
        if current_drawdown > survival_th:
            cap = self.survival_limit

        # Apply Cap (direction preserved)
        if abs(proposed_action) > cap:
            return cap * (1 if proposed_action > 0 else -1)

        return None  # No override needed if within limits

    def reset(self):
        """Resets risk state (e.g. for new episode)."""
        self.freeze_counter = 0


def build_risk_manager_from_config(config):
    """
    Build a RiskManager from unified config object.
    Keeps live/backtest risk thresholds on a single source of truth.
    """
    return RiskManager(
        max_drawdown_limit=config.risk.max_drawdown_limit,
        freeze_period_steps=config.risk.freeze_period_steps,
        tier1_drawdown=config.risk.tier1_drawdown,
        tier1_limit=config.risk.tier1_limit,
        tier2_drawdown=config.risk.tier2_drawdown,
        tier2_limit=config.risk.tier2_limit,
        survival_drawdown=config.risk.survival_drawdown,
        survival_limit=config.risk.survival_limit,
    )
