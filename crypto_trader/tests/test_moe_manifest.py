from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from moe.manifest import load_expert_manifest, resolve_feature_mask


def test_manifest_loads_eight_experts() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "moe_experts.yaml"
    manifest = load_expert_manifest(manifest_path)

    assert len(manifest.experts) == 8
    ids = [e.expert_id for e in manifest.experts]
    assert len(ids) == len(set(ids))


def test_feature_mask_resolution() -> None:
    keep = resolve_feature_mask("trend")
    assert 4 in keep
    assert 10 in keep
    assert 1 not in keep
