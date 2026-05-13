from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FeatureProfile:
    rsi_window: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_window: int
    bb_std: float
    atr_window: int
    sma_fast: int
    sma_slow: int
    volume_window: int
    rolling_vol_window: int


@dataclass(frozen=True)
class EnvProfile:
    atr_floor: float
    vol_scale_min: float
    vol_scale_max: float
    target_atr_pct: float
    tau: float
    delta_max: float
    cooldown_n: int
    k_single: float
    funding_daily: float


@dataclass(frozen=True)
class AssetProfile:
    key: str
    feature: FeatureProfile
    env: EnvProfile


DEFAULT_PROFILE = AssetProfile(
    key="ETH",
    feature=FeatureProfile(
        rsi_window=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_window=20,
        bb_std=2.0,
        atr_window=14,
        sma_fast=50,
        sma_slow=200,
        volume_window=20,
        rolling_vol_window=20,
    ),
    env=EnvProfile(
        atr_floor=0.005,
        vol_scale_min=0.1,
        vol_scale_max=2.0,
        target_atr_pct=0.05,
        tau=0.25,
        delta_max=0.15,
        cooldown_n=3,
        k_single=0.0008,
        funding_daily=0.0003,
    ),
)


PROFILE_MAP: Dict[str, AssetProfile] = {
    "ETH": DEFAULT_PROFILE,
}


def infer_asset_key(symbol: Optional[str], interval: Optional[str] = None) -> str:
    _ = interval
    if symbol is None:
        return DEFAULT_PROFILE.key

    token = str(symbol).upper()
    if "ETH" in token:
        return "ETH"
    return DEFAULT_PROFILE.key


def get_asset_profile(symbol: Optional[str], interval: Optional[str] = None) -> AssetProfile:
    _ = interval
    key = infer_asset_key(symbol)
    return PROFILE_MAP.get(key, DEFAULT_PROFILE)
