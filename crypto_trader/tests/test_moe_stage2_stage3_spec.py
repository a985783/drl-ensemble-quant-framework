from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_moe_stage2_gate import (
    build_gate_artifacts,
    softmax_weights,
    validate_stage1_artifacts,
)
from train_moe_stage3_joint import allocate_expert_timesteps


def test_softmax_weights_properties() -> None:
    logits = np.array([0.2, -0.1, 0.7], dtype=np.float32)
    weights = softmax_weights(logits, temperature=0.8)

    assert weights.shape == (3,)
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert np.all(weights > 0)


def test_gate_artifact_builder_resolves_four_experts() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "moe_experts.yaml"
    artifacts = build_gate_artifacts(manifest_path, stage1_root="checkpoints/moe/stage1")

    assert len(artifacts) == 4
    assert all(a.model_path.name == "model.zip" for a in artifacts)
    assert all(a.vecnorm_path.name == "vec_normalize.pkl" for a in artifacts)


def test_stage1_artifact_validation_detects_missing_files(tmp_path: Path) -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "moe_experts.yaml"
    artifacts = build_gate_artifacts(manifest_path, stage1_root=tmp_path)

    missing = validate_stage1_artifacts(artifacts)
    assert len(missing) > 0


def test_joint_timestep_allocation_prefers_high_usage() -> None:
    usage = {
        "E1": 0.40,
        "E2": 0.10,
        "E3": 0.05,
    }
    allocation = allocate_expert_timesteps(usage, base_timesteps=10000)

    assert allocation["E1"] > allocation["E2"] > allocation["E3"]
    assert all(v >= 5000 for v in allocation.values())
