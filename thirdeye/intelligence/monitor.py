from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from thirdeye.intelligence.calibration import IntelligenceCalibrator
from thirdeye.intelligence.signals import TrainingSignalCollector
from thirdeye.models import (
    CapabilityTarget,
    IntelligenceEstimate,
    IntelligenceSignal,
    SubsystemSpec,
)


class IntelligenceMonitor:
    """Coordinates telemetry, held-out targets, calibration, and estimates."""

    def __init__(
        self,
        subsystems: Iterable[SubsystemSpec] = (),
        *,
        minimum_calibration_checkpoints: int = 5,
        ridge: float = 1e-3,
    ) -> None:
        self.collector = TrainingSignalCollector()
        self.calibrator = IntelligenceCalibrator(
            ridge=ridge,
            minimum_samples=minimum_calibration_checkpoints,
        )
        self.subsystems = {item.subsystem_id: item for item in subsystems}
        self.targets: list[CapabilityTarget] = []
        self.checkpoints: dict[str, dict[str, float]] = {}
        self.checkpoint_signals: dict[str, tuple[IntelligenceSignal, ...]] = {}
        self.estimates: list[IntelligenceEstimate] = []

    def register_subsystem(self, subsystem: SubsystemSpec) -> None:
        self.subsystems[subsystem.subsystem_id] = subsystem

    def record_signal(self, signal: IntelligenceSignal) -> None:
        self.collector.extend((signal,))

    def checkpoint(
        self,
        checkpoint_id: str,
        *,
        last_n: int | None = None,
    ) -> IntelligenceEstimate:
        signals = (
            self.collector.signals[-last_n:]
            if last_n is not None
            else self.collector.signals
        )
        checkpoint_collector = TrainingSignalCollector()
        checkpoint_collector.extend(signals)
        vector = checkpoint_collector.vector()
        self.checkpoints[checkpoint_id] = vector
        self.checkpoint_signals[checkpoint_id] = signals
        estimate = self.calibrator.predict(
            vector,
            checkpoint_id=checkpoint_id,
            subsystem_vectors=checkpoint_collector.subsystem_vectors(),
        )
        self.estimates.append(estimate)
        return estimate

    def record_capability(
        self,
        target: CapabilityTarget,
        *,
        refit: bool = True,
    ) -> IntelligenceEstimate | None:
        if target.checkpoint_id not in self.checkpoints:
            raise KeyError(
                f"No telemetry snapshot exists for checkpoint {target.checkpoint_id!r}."
            )
        self.targets.append(target)
        if refit and len(self.targets) >= self.calibrator.minimum_samples:
            self.fit()
            estimate = self.calibrator.predict(
                self.checkpoints[target.checkpoint_id],
                checkpoint_id=target.checkpoint_id,
            )
            self.estimates.append(estimate)
            return estimate
        return None

    def fit(self) -> None:
        observations = [
            (self.checkpoints[target.checkpoint_id], target.value)
            for target in self.targets
            if target.checkpoint_id in self.checkpoints
        ]
        self.calibrator.fit(observations)

    def estimate(self, checkpoint_id: str) -> IntelligenceEstimate:
        try:
            vector = self.checkpoints[checkpoint_id]
        except KeyError as exc:
            raise KeyError(f"Unknown checkpoint telemetry: {checkpoint_id}") from exc
        estimate = self.calibrator.predict(
            vector,
            checkpoint_id=checkpoint_id,
            subsystem_vectors=_split_subsystems(vector),
        )
        self.estimates.append(estimate)
        return estimate

    def persist(self, eye: object, project_id: str) -> None:
        for subsystem in self.subsystems.values():
            eye.register_subsystem(project_id, subsystem)
        if self.checkpoint_signals:
            for checkpoint_id, signals in self.checkpoint_signals.items():
                eye.record_intelligence_signals(
                    project_id,
                    checkpoint_id=checkpoint_id,
                    signals=signals,
                )
        else:
            eye.record_intelligence_signals(
                project_id,
                checkpoint_id="live",
                signals=self.collector.signals,
            )
        for target in self.targets:
            eye.record_capability_target(project_id, target)
        for estimate in self.estimates:
            eye.record_intelligence_estimate(project_id, estimate)

    def write_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "scientific_status": {
                "telemetry": "observational",
                "overall_estimate": (
                    "calibrated_prediction"
                    if self.calibrator.model is not None
                    else "unavailable_until_calibrated"
                ),
                "causal_claim": False,
            },
            "subsystems": [item.to_dict() for item in self.subsystems.values()],
            "signals": [item.to_dict() for item in self.collector.signals],
            "targets": [item.to_dict() for item in self.targets],
            "estimates": [item.to_dict() for item in self.estimates],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def _split_subsystems(vector: dict[str, float]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key, value in vector.items():
        subsystem, signal = key.split(":", 1)
        result.setdefault(subsystem, {})[signal] = value
    return result
