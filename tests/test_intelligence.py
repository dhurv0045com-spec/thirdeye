from __future__ import annotations

import pytest

from thirdeye.intelligence import (
    IntelligenceCalibrator,
    IntelligenceMonitor,
    PyTorchSubsystemCollector,
    TrainingSignalCollector,
)
from thirdeye.models import (
    CapabilityTarget,
    MetricDirection,
    ProjectSpec,
    SignalKind,
    SubsystemSpec,
)
from thirdeye.sdk import ThirdEye


def test_lower_is_better_signals_are_oriented() -> None:
    collector = TrainingSignalCollector()
    collector.record(
        "loss",
        2.0,
        step=1,
        direction=MetricDirection.LOWER,
    )
    collector.record(
        "loss",
        1.0,
        step=2,
        direction=MetricDirection.LOWER,
    )

    vector = collector.vector()

    assert vector["model:loss:latest"] == -1.0
    assert vector["model:loss:trend"] > 0.0


def test_uncalibrated_monitor_refuses_to_invent_overall_intelligence() -> None:
    monitor = IntelligenceMonitor()
    monitor.collector.record("loss", 1.0, step=1, direction=MetricDirection.LOWER)

    estimate = monitor.checkpoint("checkpoint-1")

    assert estimate.overall is None
    assert estimate.calibrated is False
    assert estimate.confidence == 0.0


def test_synthetic_calibration_recovers_capability_ordering() -> None:
    calibrator = IntelligenceCalibrator(minimum_samples=5)
    rows = []
    for index in range(8):
        vector = {
            "model:loss:latest": -float(8 - index),
            "reasoning:activation.rms:mean": float(index) / 10.0,
        }
        target = 10.0 + 3.0 * index
        rows.append((vector, target))
    calibrator.fit(rows)

    low = calibrator.predict(rows[1][0], checkpoint_id="low")
    high = calibrator.predict(rows[7][0], checkpoint_id="high")

    assert low.calibrated and high.calibrated
    assert high.overall is not None and low.overall is not None
    assert high.overall > low.overall
    assert high.confidence > 0.0
    assert any("not universal intelligence" in item for item in high.limitations)


def test_monitor_persists_subsystems_signals_targets_and_estimates(tmp_path) -> None:
    subsystem = SubsystemSpec(
        subsystem_id="reasoning",
        name="Reasoning",
        owner="tests",
        kind="architecture",
    )
    monitor = IntelligenceMonitor([subsystem], minimum_calibration_checkpoints=2)
    for checkpoint, loss, target in (("a", 2.0, 0.2), ("b", 1.0, 0.8)):
        monitor.collector.record(
            "loss",
            loss,
            step=len(monitor.checkpoints),
            direction=MetricDirection.LOWER,
        )
        monitor.checkpoint(checkpoint, last_n=1)
        monitor.record_capability(
            CapabilityTarget(
                target_id="held_out",
                value=target,
                checkpoint_id=checkpoint,
                evaluator="tests",
                sample_count=10,
            )
        )

    eye = ThirdEye(tmp_path)
    eye.register_project(ProjectSpec("demo", "Demo"))
    monitor.persist(eye, "demo")
    snapshot = eye.intelligence_snapshot("demo")

    assert len(snapshot["subsystems"]) == 1
    assert len(snapshot["signals"]) == 2
    assert len(snapshot["capability_targets"]) == 2
    assert any(row["calibrated"] for row in snapshot["estimates"])


def test_store_backed_calibration_uses_multiple_training_sessions(tmp_path) -> None:
    eye = ThirdEye(tmp_path)
    eye.register_project(ProjectSpec("demo", "Demo"))
    for index in range(5):
        monitor = IntelligenceMonitor(minimum_calibration_checkpoints=5)
        monitor.collector.record(
            "loss",
            5.0 - index,
            step=index,
            direction=MetricDirection.LOWER,
        )
        checkpoint = f"checkpoint-{index}"
        monitor.checkpoint(checkpoint)
        monitor.targets.append(
            CapabilityTarget(
                target_id="quality",
                value=float(index),
                checkpoint_id=checkpoint,
                evaluator="tests",
                sample_count=20,
            )
        )
        monitor.persist(eye, "demo")

    estimate = eye.calibrate_intelligence("demo", target_id="quality")

    assert estimate.calibrated is True
    assert estimate.overall is not None
    assert estimate.checkpoint_id == "checkpoint-4"


def test_pytorch_hooks_collect_signals_without_changing_outputs_or_gradients() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    baseline = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    instrumented = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    instrumented.load_state_dict(baseline.state_dict())
    x = torch.randn(3, 4)

    baseline_output = baseline(x)
    baseline_output.square().mean().backward()
    baseline_gradients = [parameter.grad.clone() for parameter in baseline.parameters()]

    collector = TrainingSignalCollector()
    subsystem = SubsystemSpec(
        subsystem_id="mlp",
        name="MLP",
        owner="tests",
        kind="architecture",
        module_patterns=("0", "2"),
    )
    hooks = PyTorchSubsystemCollector(
        instrumented,
        [subsystem],
        collector,
        sample_every=1,
    )
    hooks.begin_step(0)
    instrumented_output = instrumented(x)
    instrumented_output.square().mean().backward()
    hooks.capture_gradients(learning_rate=1e-3)
    hooks.close()

    assert torch.equal(baseline_output, instrumented_output)
    for expected, parameter in zip(baseline_gradients, instrumented.parameters()):
        assert torch.equal(expected, parameter.grad)
    assert {signal.kind for signal in collector.signals} >= {
        SignalKind.REPRESENTATION,
        SignalKind.RELIABILITY,
        SignalKind.OPTIMIZATION,
    }
