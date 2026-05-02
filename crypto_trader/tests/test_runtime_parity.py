from __future__ import annotations

from types import SimpleNamespace


def test_shared_constraint_core_cooldown_flip() -> None:
    from crypto_trader.envs import trading_env as te

    exec_pos, new_flip, reason = te.apply_execution_constraints_core(
        target_pos=-0.8,
        current_pos=0.1,
        last_flip_marker=10.0,
        current_marker=12.0,
        tau=0.25,
        delta_max=0.15,
        cooldown_window=3.0,
    )

    assert exec_pos == 0.0
    assert new_flip == 10.0
    assert reason == "Cooldown"


def test_build_risk_manager_from_config_uses_tier_values() -> None:
    from crypto_trader.risk_manager import build_risk_manager_from_config

    cfg = SimpleNamespace(
        risk=SimpleNamespace(
            max_drawdown_limit=0.22,
            freeze_period_steps=5,
            tier1_drawdown=0.07,
            tier1_limit=0.77,
            tier2_drawdown=0.11,
            tier2_limit=0.44,
            survival_drawdown=0.19,
            survival_limit=0.18,
        )
    )

    rm = build_risk_manager_from_config(cfg)

    assert rm.max_drawdown_limit == 0.22
    assert rm.freeze_period_steps == 5
    assert rm.tier1_drawdown == 0.07
    assert rm.tier1_limit == 0.77
    assert rm.tier2_drawdown == 0.11
    assert rm.tier2_limit == 0.44
    assert rm.survival_drawdown == 0.19
    assert rm.survival_limit == 0.18
