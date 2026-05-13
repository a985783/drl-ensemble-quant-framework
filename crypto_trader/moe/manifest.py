from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import yaml

SUPPORTED_ALGOS = {"ppo", "a2c", "sac"}

# Observation slots in TradingEnv:
# [0:pos,1:cooldown,2:unreal_pnl,3:nw_chg,4:signal,5:rsi,6:roll_vol,7:macd,
#  8:bb_width,9:dist_sma200,10:atr_pct,11:vol_ratio,12:direction]
FEATURE_MASKS: Dict[str, List[int]] = {
    "all": list(range(13)),
    "trend": [0, 3, 4, 5, 7, 9, 10, 12],
    "risk": [0, 1, 3, 6, 8, 10, 11, 12],
    "carry": [0, 3, 4, 6, 10, 11, 12],
    "switch": [0, 1, 3, 4, 6, 7, 11, 12],
}


@dataclass
class ExpertSpec:
    expert_id: str
    algorithm: str
    seed: int
    data_slice: str = "full"
    feature_mask: Optional[List[int]] = None
    reward_profile: Optional[Dict[str, float]] = None
    timesteps: int = 100_000


@dataclass
class ExpertManifest:
    experts: List[ExpertSpec]


def resolve_feature_mask(mask: Optional[Union[str, Sequence[int]]]) -> Optional[List[int]]:
    if mask is None:
        return None
    if isinstance(mask, str):
        key = mask.strip().lower()
        if key not in FEATURE_MASKS:
            raise ValueError(f"Unknown feature mask: {mask}")
        return FEATURE_MASKS[key][:]

    resolved = sorted({int(i) for i in mask})
    for i in resolved:
        if i < 0 or i >= 13:
            raise ValueError(f"Feature mask index out of range: {i}")
    return resolved


def _validate_unique_ids(experts: List[ExpertSpec]) -> None:
    ids = [e.expert_id for e in experts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate expert_id found in manifest")


def load_expert_manifest(path: Union[str, Path]) -> ExpertManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expert manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw_experts = raw.get("experts", [])
    experts: List[ExpertSpec] = []

    for item in raw_experts:
        algo = str(item.get("algorithm", "")).lower().strip()
        if algo not in SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm '{algo}' for expert {item.get('expert_id')}")

        spec = ExpertSpec(
            expert_id=str(item["expert_id"]),
            algorithm=algo,
            seed=int(item.get("seed", 42)),
            data_slice=str(item.get("data_slice", "full")).strip().lower(),
            feature_mask=resolve_feature_mask(item.get("feature_mask")),
            reward_profile=item.get("reward_profile") or {},
            timesteps=int(item.get("timesteps", 100_000)),
        )
        experts.append(spec)

    _validate_unique_ids(experts)
    return ExpertManifest(experts=experts)
