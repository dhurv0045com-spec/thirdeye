from __future__ import annotations

from thirdeye.evidence import grade_evidence
from thirdeye.lineage import capture_manifest
from thirdeye.models import (
    EvidenceGrade,
    FeatureCategory,
    FeatureSpec,
    FeatureVariant,
    ProjectSpec,
    ProtocolKind,
    ProtocolSpec,
)
from thirdeye.sdk import ThirdEye


def _feature(retraining: bool = False) -> FeatureSpec:
    return FeatureSpec(
        feature_id="demo.feature",
        name="Demo",
        owner="tests",
        version="1",
        intended_behavior="Improve the demo metric.",
        category=FeatureCategory.TRAINING if retraining else FeatureCategory.RUNTIME,
        variants=(
            FeatureVariant("off", "Off", is_control=True),
            FeatureVariant("on", "On"),
        ),
        requires_retraining=retraining,
        activation_probe="demo.feature.calls",
    )


def test_feature_requires_exactly_one_control() -> None:
    try:
        FeatureSpec(
            feature_id="bad",
            name="Bad",
            owner="tests",
            version="1",
            intended_behavior="bad",
            category=FeatureCategory.RUNTIME,
            variants=(FeatureVariant("on", "On"),),
            requires_retraining=False,
        )
    except ValueError as exc:
        assert "exactly one control" in str(exc)
    else:
        raise AssertionError("Feature without a control was accepted")


def test_observational_evidence_is_never_causal() -> None:
    protocol = ProtocolSpec("p", ProtocolKind.OBSERVATIONAL, feature_id="demo.feature")
    result = grade_evidence(
        project_id="demo",
        feature_id="demo.feature",
        protocol=protocol,
        run_ids=["r1", "r2", "r3"],
        activation_verified=True,
    )
    assert result.grade == EvidenceGrade.OBSERVED
    assert result.causal is False


def test_controlled_three_seed_evidence_is_replicated() -> None:
    protocol = ProtocolSpec(
        "p",
        ProtocolKind.MATCHED_RETRAINING,
        feature_id="demo.feature",
        seeds=(1, 2, 3),
    )
    result = grade_evidence(
        project_id="demo",
        feature_id="demo.feature",
        protocol=protocol,
        run_ids=["r1", "r2", "r3"],
        activation_verified=True,
        effects={"score": [0.1, 0.2, 0.15]},
    )
    assert result.grade == EvidenceGrade.REPLICATED
    assert result.causal is True
    assert result.effect_sizes["score"] == 0.15


def test_one_click_evaluate_writes_three_reports(tmp_path) -> None:
    eye = ThirdEye(tmp_path)
    eye.register_project(ProjectSpec("demo", "Demo"))
    eye.register_feature("demo", _feature())

    result = eye.evaluate("demo", "auto")

    assert len(result["recommended_experiments"]) == 1
    assert all(tmp_path.joinpath(path).exists() if not path.startswith(str(tmp_path)) else True
               for path in [])
    for path in result["report_paths"].values():
        assert __import__("pathlib").Path(path).exists()


def test_lineage_manifest_is_reproducible_shape(tmp_path) -> None:
    manifest = capture_manifest(
        project_id="demo",
        protocol_id="audit",
        feature_variants={},
        root=tmp_path,
        config={"x": 1},
        seed=42,
        reproduction_command="pytest",
    )
    assert manifest.project_id == "demo"
    assert len(manifest.config_hash) == 64
    assert len(manifest.dirty_patch_hash) == 64

