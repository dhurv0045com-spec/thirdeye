from __future__ import annotations

from thirdeye.intelligence import TrainingInsightEngine, TrainingSignalCollector
from thirdeye.models import ProjectSpec
from thirdeye.sdk import ThirdEye


def _collector() -> TrainingSignalCollector:
    collector = TrainingSignalCollector()
    for step in range(8):
        collector.record_training_step(
            step=step,
            loss=1.0 + step * 0.08,
            validation_loss=1.0 + step * 0.12,
            learning_rate=1e-3,
            gradient_norm=0.4 + step * 0.01,
            tokens_per_second=1000.0 - step * 60.0,
            token_utilization=0.25,
        )
    return collector


def test_insight_engine_explains_observed_training_patterns_without_causality() -> None:
    report = TrainingInsightEngine().analyze(_collector().signals)
    finding_ids = {item.finding_id for item in report.findings}

    assert report.status == "attention_required"
    assert report.causal is False
    assert {"loss_rising", "throughput_regression", "low_token_utilization"} <= finding_ids
    assert all(item.causal is False for item in report.findings)
    loss = next(item for item in report.findings if item.finding_id == "loss_rising")
    assert "training.loss points=" in loss.basis[0]
    assert "fixed validation" in loss.recommended_action


def test_insight_engine_does_not_overinterpret_short_history() -> None:
    collector = TrainingSignalCollector()
    collector.record_training_step(
        step=1,
        loss=1.0,
        learning_rate=1e-3,
        gradient_norm=1.0,
    )

    report = TrainingInsightEngine().analyze(collector.signals)

    assert report.status == "monitoring"
    assert any(item.finding_id == "loss_insufficient_history" for item in report.findings)


def test_store_snapshot_and_report_expose_insight(tmp_path) -> None:
    eye = ThirdEye(tmp_path)
    eye.register_project(ProjectSpec("demo", "Demo"))
    collector = _collector()
    eye.record_intelligence_signals(
        "demo",
        checkpoint_id="checkpoint-1",
        signals=collector.signals,
    )

    snapshot = eye.intelligence_snapshot("demo")
    evaluation = eye.evaluate("demo")

    assert snapshot["insight"]["status"] == "attention_required"
    scorecard = __import__("pathlib").Path(evaluation["report_paths"]["scorecard"]).read_text(
        encoding="utf-8"
    )
    dashboard = __import__("pathlib").Path(evaluation["report_paths"]["dashboard"]).read_text(
        encoding="utf-8"
    )
    assert "What ThirdEye Observed" in scorecard
    assert "Training loss moved" in scorecard
    assert "Confidence" in scorecard
    assert "What ThirdEye Observed" in dashboard


def test_capability_calibration_finding_requires_real_targets() -> None:
    collector = TrainingSignalCollector()
    for step in range(6):
        collector.record_training_step(
            step=step,
            loss=5.0 - step,
            learning_rate=1e-3,
            gradient_norm=1.0,
        )
    report = TrainingInsightEngine().analyze(collector.signals)

    assert any(item.finding_id == "capability_uncalibrated" for item in report.findings)
