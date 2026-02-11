
class RiskManager:
    """
    Tiered Risk Management for Daily Trading.
    """

    def __init__(self, max_drawdown_limit: float = 0.10, freeze_period_steps: int = 3):
        self.max_drawdown_limit = max_drawdown_limit
        self.freeze_period_steps = freeze_period_steps
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
        if current_drawdown > 0.05:
            cap = 0.8
        if current_drawdown > 0.10:
            cap = 0.5
            
        # Survival Mode: Cap at 0.2x if drawdown exceeds 15%
        if current_drawdown > 0.15:
            cap = 0.2

        # Apply Cap (direction preserved)
        if abs(proposed_action) > cap:
            return cap * (1 if proposed_action > 0 else -1)

        return None  # No override needed if within limits

    def reset(self):
        """Resets risk state (e.g. for new episode)."""
        self.freeze_counter = 0
