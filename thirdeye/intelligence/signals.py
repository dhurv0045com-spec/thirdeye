from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, pstdev
from typing import Iterable

from thirdeye.models import IntelligenceSignal, MetricDirection, SignalKind


class TrainingSignalCollector:
    """Collect model-agnostic optimization, behavior, and efficiency signals."""

    def __init__(self) -> None:
        self._signals: list[IntelligenceSignal] = []

    @property
    def signals(self) -> tuple[IntelligenceSignal, ...]:
        return tuple(self._signals)

    def extend(self, signals: Iterable[IntelligenceSignal]) -> None:
        self._signals.extend(signals)

    def record(
        self,
        signal_id: str,
        value: float,
        *,
        step: int,
        subsystem_id: str = "model",
        kind: SignalKind = SignalKind.OPTIMIZATION,
        direction: MetricDirection = MetricDirection.HIGHER,
        unit: str = "",
        sample_count: int = 1,
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> IntelligenceSignal:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Signal must be finite: {signal_id}={number}")
        signal = IntelligenceSignal(
            signal_id=signal_id,
            subsystem_id=subsystem_id,
            kind=kind,
            value=number,
            step=int(step),
            direction=direction,
            unit=unit,
            sample_count=max(1, int(sample_count)),
            confidence=max(0.0, min(1.0, float(confidence))),
            metadata=metadata or {},
        )
        self._signals.append(signal)
        return signal

    def record_training_step(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float,
        gradient_norm: float,
        update_ratio: float | None = None,
        tokens_per_second: float | None = None,
        token_utilization: float | None = None,
        validation_loss: float | None = None,
    ) -> None:
        self.record(
            "training.loss",
            loss,
            step=step,
            direction=MetricDirection.LOWER,
        )
        self.record(
            "training.learning_rate",
            learning_rate,
            step=step,
            direction=MetricDirection.TARGET,
        )
        self.record(
            "training.gradient_norm",
            gradient_norm,
            step=step,
            direction=MetricDirection.TARGET,
        )
        if update_ratio is not None:
            self.record(
                "training.update_ratio",
                update_ratio,
                step=step,
                direction=MetricDirection.TARGET,
            )
        if tokens_per_second is not None:
            self.record(
                "efficiency.tokens_per_second",
                tokens_per_second,
                step=step,
                kind=SignalKind.EFFICIENCY,
                unit="tokens/second",
            )
        if token_utilization is not None:
            utilization = float(token_utilization)
            if not 0.0 <= utilization <= 1.0:
                raise ValueError("token_utilization must be between 0 and 1")
            self.record(
                "efficiency.token_utilization",
                utilization,
                step=step,
                kind=SignalKind.EFFICIENCY,
                unit="fraction",
            )
        if validation_loss is not None:
            self.record(
                "validation.loss",
                validation_loss,
                step=step,
                kind=SignalKind.BEHAVIOR,
                direction=MetricDirection.LOWER,
            )

    def vector(self, *, last_n: int | None = None) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        rows: Iterable[IntelligenceSignal] = (
            self._signals[-last_n:] if last_n else self._signals
        )
        for signal in rows:
            key = f"{signal.subsystem_id}:{signal.signal_id}"
            grouped[key].append(_oriented(signal))
        vector: dict[str, float] = {}
        for key, values in grouped.items():
            vector[f"{key}:latest"] = values[-1]
            vector[f"{key}:mean"] = mean(values)
            vector[f"{key}:stability"] = 1.0 / (1.0 + pstdev(values))
            if len(values) > 1:
                vector[f"{key}:trend"] = _slope(values)
        return vector

    def subsystem_vectors(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = defaultdict(dict)
        for key, value in self.vector().items():
            subsystem, signal = key.split(":", 1)
            result[subsystem][signal] = value
        return dict(result)


def _oriented(signal: IntelligenceSignal) -> float:
    if signal.direction == MetricDirection.LOWER:
        return -signal.value
    return signal.value


def _slope(values: list[float]) -> float:
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    numerator = sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / max(denominator, 1e-12)
