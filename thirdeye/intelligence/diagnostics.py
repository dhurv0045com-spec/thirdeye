from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, Mapping

from thirdeye.models import (
    InsightFinding,
    InsightSeverity,
    IntelligenceEstimate,
    IntelligenceSignal,
    MetricDirection,
    SignalKind,
    TrainingInsightReport,
)


class TrainingInsightEngine:
    """Turn training telemetry into bounded, observational explanations.

    The engine only describes measured patterns. It deliberately recommends a
    next measurement or controlled experiment instead of assigning a cause to
    a loss change, internal activation, or capability estimate.
    """

    def __init__(self, *, minimum_points: int = 4) -> None:
        self.minimum_points = max(2, int(minimum_points))

    def analyze(
        self,
        signals: Iterable[IntelligenceSignal],
        *,
        estimates: Iterable[IntelligenceEstimate] = (),
    ) -> TrainingInsightReport:
        rows = tuple(sorted(signals, key=lambda item: (item.step, item.signal_id)))
        if not rows:
            return TrainingInsightReport(
                status="insufficient_telemetry",
                signal_count=0,
                step_range=None,
                findings=(
                    self._finding(
                        "telemetry_missing",
                        InsightSeverity.WATCH,
                        "No training telemetry is available",
                        "ThirdEye cannot explain optimization behavior until the trainer emits signals.",
                        ("signal_count=0",),
                        "Record loss, learning rate, gradient norm, throughput, and fixed held-out metrics.",
                        1.0,
                    ),
                ),
                limitations=(
                    "No diagnostic can be formed without telemetry.",
                    "ThirdEye telemetry is observational and cannot establish causality.",
                ),
            )

        series = _series_by_key(rows)
        findings: list[InsightFinding] = []
        loss = series.get(("model", "training.loss"), ())
        validation_loss = series.get(("model", "validation.loss"), ())
        if len(loss) < self.minimum_points:
            findings.append(
                self._finding(
                    "loss_insufficient_history",
                    InsightSeverity.WATCH,
                    "Too little loss history to judge a trend",
                    f"Only {len(loss)} loss observations are available; short-term loss movement is noisy.",
                    (f"training.loss points={len(loss)}",),
                    "Collect more steps and compare against a fixed held-out evaluation before changing the model.",
                    _coverage_confidence(len(loss), self.minimum_points),
                )
            )
        elif loss:
            findings.append(self._loss_finding(loss))
            if len(validation_loss) >= self.minimum_points:
                validation = self._validation_finding(loss, validation_loss)
                if validation is not None:
                    findings.append(validation)

        gradient = series.get(("model", "training.gradient_norm"), ())
        if len(gradient) >= self.minimum_points:
            findings.append(self._gradient_finding(gradient))

        learning_rate = series.get(("model", "training.learning_rate"), ())
        if len(learning_rate) >= self.minimum_points:
            finding = self._learning_rate_finding(learning_rate)
            if finding is not None:
                findings.append(finding)

        throughput = series.get(("model", "efficiency.tokens_per_second"), ())
        if len(throughput) >= self.minimum_points:
            findings.append(self._throughput_finding(throughput))

        utilization = (
            series.get(("model", "efficiency.token_utilization"), ())
            or series.get(("model", "training.token_utilization"), ())
        )
        if utilization:
            findings.append(self._utilization_finding(utilization))

        findings.extend(self._subsystem_findings(series))
        findings.append(self._calibration_finding(tuple(estimates)))
        findings.sort(key=lambda item: (-_severity_rank(item.severity), item.finding_id))
        status = _status_for(findings)
        steps = (min(row.step for row in rows), max(row.step for row in rows))
        return TrainingInsightReport(
            status=status,
            signal_count=len(rows),
            step_range=steps,
            findings=tuple(findings[:12]),
            limitations=(
                "Findings describe observed telemetry patterns, not root causes.",
                "A loss trend is not a capability measurement; use versioned held-out evaluation.",
                "Subsystem findings are diagnostic prompts, not causal attribution.",
            ),
        )

    def analyze_payloads(
        self,
        signals: Iterable[Mapping[str, object]],
        *,
        estimates: Iterable[Mapping[str, object]] = (),
    ) -> TrainingInsightReport:
        return self.analyze(
            (_signal_from_payload(row) for row in signals),
            estimates=(_estimate_from_payload(row) for row in estimates),
        )

    def _loss_finding(self, values: tuple[tuple[int, float, float], ...]) -> InsightFinding:
        first, last, change = _relative_change(values)
        confidence = _series_confidence(values)
        basis = _basis("training.loss", values, first, last, change)
        if change > 0.05:
            return self._finding(
                "loss_rising",
                InsightSeverity.WARNING,
                "Training loss is rising over the observed window",
                _trend_summary("Training loss", first, last, change),
                basis,
                "Run a fixed validation slice and compare data mix, learning rate, and resume lineage before changing architecture.",
                confidence,
            )
        if change < -0.03:
            return self._finding(
                "loss_improving",
                InsightSeverity.INFO,
                "Training loss is improving",
                _trend_summary("Training loss", first, last, change),
                basis,
                "Continue the planned budget and save a checkpoint with a fixed held-out evaluation.",
                confidence,
            )
        severity = InsightSeverity.WATCH if abs(change) <= 0.01 and len(values) >= 8 else InsightSeverity.INFO
        title = "Training loss is plateauing" if severity == InsightSeverity.WATCH else "Training loss is broadly stable"
        action = (
            "Check fixed validation, gradient health, and data distribution before increasing model size or changing architecture."
            if severity == InsightSeverity.WATCH
            else "Collect a longer window or a held-out evaluation before concluding that progress has stopped."
        )
        return self._finding(
            "loss_plateau" if severity == InsightSeverity.WATCH else "loss_stable",
            severity,
            title,
            _trend_summary("Training loss", first, last, change),
            basis,
            action,
            confidence,
        )

    def _validation_finding(
        self,
        loss: tuple[tuple[int, float, float], ...],
        validation: tuple[tuple[int, float, float], ...],
    ) -> InsightFinding | None:
        _, _, train_change = _relative_change(loss)
        first, last, validation_change = _relative_change(validation)
        if train_change < -0.02 and validation_change > 0.02:
            return self._finding(
                "validation_regression",
                InsightSeverity.WARNING,
                "Held-out loss worsens while training loss improves",
                _trend_summary("Validation loss", first, last, validation_change),
                _basis("validation.loss", validation, first, last, validation_change),
                "Preserve this checkpoint, verify the held-out split, and run a regularization or data-mix experiment with matched seeds.",
                min(_series_confidence(loss), _series_confidence(validation)),
            )
        if validation_change < -0.02:
            return self._finding(
                "validation_improving",
                InsightSeverity.INFO,
                "Held-out loss is improving",
                _trend_summary("Validation loss", first, last, validation_change),
                _basis("validation.loss", validation, first, last, validation_change),
                "Keep the evaluation protocol fixed so future checkpoints remain comparable.",
                _series_confidence(validation),
            )
        return None

    def _gradient_finding(self, values: tuple[tuple[int, float, float], ...]) -> InsightFinding:
        first, last, change = _relative_change(values)
        minimum = min(value for _, value, _ in values)
        maximum = max(value for _, value, _ in values)
        confidence = _series_confidence(values)
        basis = _basis("training.gradient_norm", values, first, last, change)
        if maximum <= 1e-10:
            return self._finding(
                "gradients_near_zero",
                InsightSeverity.WARNING,
                "Gradient norms are near zero",
                f"Observed gradient norms remain at or below {maximum:.3g}.",
                basis,
                "Verify backward execution, mixed-precision unscaling, trainable parameters, and loss masking before changing the learning rate.",
                confidence,
            )
        if maximum / max(minimum, 1e-12) > 20.0:
            return self._finding(
                "gradient_instability",
                InsightSeverity.WATCH,
                "Gradient norms vary sharply",
                f"Observed gradient norm range is {minimum:.3g} to {maximum:.3g}.",
                basis,
                "Correlate spikes with data batches and learning-rate transitions; use a controlled clipping or LR experiment if the pattern persists.",
                confidence,
            )
        return self._finding(
            "gradient_stable",
            InsightSeverity.INFO,
            "Gradient norms are within a stable observed range",
            f"Observed gradient norm range is {minimum:.3g} to {maximum:.3g}.",
            basis,
            "Continue monitoring; this does not prove the optimizer is optimal.",
            confidence,
        )

    def _learning_rate_finding(
        self, values: tuple[tuple[int, float, float], ...]
    ) -> InsightFinding | None:
        first, last, change = _relative_change(values)
        if first <= 0.0 or last <= 0.0:
            return self._finding(
                "learning_rate_nonpositive",
                InsightSeverity.WARNING,
                "Learning rate reached a non-positive value",
                f"Observed learning rate moved from {first:.3g} to {last:.3g}.",
                _basis("training.learning_rate", values, first, last, change),
                "Inspect the scheduler state and resume metadata before continuing training.",
                _series_confidence(values),
            )
        if abs(change) >= 0.9:
            return self._finding(
                "learning_rate_transition",
                InsightSeverity.WATCH,
                "Learning rate changed by more than 90%",
                f"Observed learning rate moved from {first:.3g} to {last:.3g}.",
                _basis("training.learning_rate", values, first, last, change),
                "Treat loss movement across this scheduler transition separately from steady-state training.",
                _series_confidence(values),
            )
        return None

    def _throughput_finding(self, values: tuple[tuple[int, float, float], ...]) -> InsightFinding:
        first, last, change = _relative_change(values)
        severity = InsightSeverity.WARNING if change < -0.2 else InsightSeverity.INFO
        title = "Training throughput regressed" if severity == InsightSeverity.WARNING else "Training throughput is stable or improving"
        action = (
            "Profile data loading, checkpoint I/O, sequence padding, and accelerator utilization before changing model capacity."
            if severity == InsightSeverity.WARNING
            else "Keep recording throughput alongside capability so efficiency trade-offs remain visible."
        )
        return self._finding(
            "throughput_regression" if severity == InsightSeverity.WARNING else "throughput_stable",
            severity,
            title,
            _trend_summary("Throughput", first, last, change, unit=" tokens/s"),
            _basis("efficiency.tokens_per_second", values, first, last, change),
            action,
            _series_confidence(values),
        )

    def _utilization_finding(self, values: tuple[tuple[int, float, float], ...]) -> InsightFinding:
        _, last, _ = _relative_change(values)
        severity = InsightSeverity.WARNING if last < 0.5 else InsightSeverity.INFO
        return self._finding(
            "low_token_utilization" if severity == InsightSeverity.WARNING else "token_utilization_healthy",
            severity,
            "Token utilization leaves compute idle" if severity == InsightSeverity.WARNING else "Token utilization is healthy",
            f"Latest non-padding token utilization is {last:.1%}.",
            (f"efficiency.token_utilization latest={last:.6g}",),
            (
                "Use same-bucket sequence packing or packed token streams, then compare capability per compute under a controlled protocol."
                if severity == InsightSeverity.WARNING
                else "Continue tracking utilization whenever data formatting or context length changes."
            ),
            _series_confidence(values),
        )

    def _subsystem_findings(
        self,
        series: dict[tuple[str, str], tuple[tuple[int, float, float], ...]],
    ) -> list[InsightFinding]:
        findings: list[InsightFinding] = []
        for (subsystem, signal_id), values in sorted(series.items()):
            if subsystem == "model" or len(values) < self.minimum_points:
                continue
            first, last, change = _relative_change(values)
            if signal_id == "activation.saturation" and last > 0.01:
                findings.append(
                    self._finding(
                        f"subsystem_saturation:{subsystem}",
                        InsightSeverity.WATCH,
                        f"{subsystem} shows activation saturation",
                        f"Latest sampled activation saturation is {last:.2%}.",
                        _basis(f"{subsystem}:{signal_id}", values, first, last, change),
                        "Inspect this subsystem around the affected steps and compare a targeted stabilization change against a matched control.",
                        _series_confidence(values),
                    )
                )
            elif signal_id in {"activation.rms", "gradients.norm", "update.ratio"} and abs(change) > 2.0:
                findings.append(
                    self._finding(
                        f"subsystem_drift:{subsystem}:{signal_id}",
                        InsightSeverity.WATCH,
                        f"{subsystem} {signal_id} drifted sharply",
                        _trend_summary(f"{subsystem} {signal_id}", first, last, change),
                        _basis(f"{subsystem}:{signal_id}", values, first, last, change),
                        "Inspect the affected subsystem and correlate the drift with validation, data, and optimizer events before attributing a cause.",
                        _series_confidence(values),
                    )
                )
        return findings

    def _calibration_finding(
        self, estimates: tuple[IntelligenceEstimate, ...]
    ) -> InsightFinding:
        calibrated = [item for item in estimates if item.calibrated and item.overall is not None]
        if not calibrated:
            return self._finding(
                "capability_uncalibrated",
                InsightSeverity.WATCH,
                "No calibrated capability estimate is available",
                "Internal telemetry can diagnose training dynamics but cannot measure intelligence by itself.",
                (f"calibrated_estimates={len(calibrated)}",),
                "Record the same versioned held-out capability target for at least five checkpoints, then fit calibration.",
                1.0,
            )
        latest = calibrated[-1]
        return self._finding(
            "capability_calibrated",
            InsightSeverity.INFO,
            "Calibrated capability estimate is available",
            f"Latest predicted held-out target is {latest.overall:.6g} with confidence {latest.confidence:.3f}.",
            (
                f"checkpoint={latest.checkpoint_id}",
                f"calibration_error={latest.calibration_error if latest.calibration_error is not None else 'unknown'}",
            ),
            "Confirm the estimate on an unseen held-out evaluation; predictive calibration is not a promotion decision.",
            latest.confidence,
        )

    @staticmethod
    def _finding(
        finding_id: str,
        severity: InsightSeverity,
        title: str,
        summary: str,
        basis: tuple[str, ...],
        recommended_action: str,
        confidence: float,
    ) -> InsightFinding:
        return InsightFinding(
            finding_id=finding_id,
            severity=severity,
            title=title,
            summary=summary,
            basis=basis,
            recommended_action=recommended_action,
            confidence=max(0.0, min(1.0, confidence)),
        )


def _series_by_key(
    signals: Iterable[IntelligenceSignal],
) -> dict[tuple[str, str], tuple[tuple[int, float, float], ...]]:
    grouped: dict[tuple[str, str], dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for signal in signals:
        grouped[(signal.subsystem_id, signal.signal_id)][signal.step].append(
            (signal.value, signal.confidence)
        )
    return {
        key: tuple(
            (step, mean(value for value, _ in entries), mean(confidence for _, confidence in entries))
            for step, entries in sorted(by_step.items())
        )
        for key, by_step in grouped.items()
    }


def _relative_change(values: tuple[tuple[int, float, float], ...]) -> tuple[float, float, float]:
    midpoint = max(1, len(values) // 2)
    first = mean(value for _, value, _ in values[:midpoint])
    last = mean(value for _, value, _ in values[midpoint:])
    return first, last, (last - first) / max(abs(first), 1e-12)


def _basis(
    signal_id: str,
    values: tuple[tuple[int, float, float], ...],
    first: float,
    last: float,
    change: float,
) -> tuple[str, ...]:
    return (
        f"{signal_id} points={len(values)}",
        f"first_window_mean={first:.6g}",
        f"last_window_mean={last:.6g}",
        f"relative_change={change:+.2%}",
    )


def _trend_summary(label: str, first: float, last: float, change: float, *, unit: str = "") -> str:
    return f"{label} moved from {first:.6g}{unit} to {last:.6g}{unit} ({change:+.1%})."


def _series_confidence(values: tuple[tuple[int, float, float], ...]) -> float:
    return min(0.95, _coverage_confidence(len(values), 12) * mean(item[2] for item in values))


def _coverage_confidence(points: int, target: int) -> float:
    return min(1.0, max(0.0, points / max(1, target)))


def _severity_rank(severity: InsightSeverity) -> int:
    return {
        InsightSeverity.INFO: 0,
        InsightSeverity.WATCH: 1,
        InsightSeverity.WARNING: 2,
        InsightSeverity.CRITICAL: 3,
    }[severity]


def _status_for(findings: Iterable[InsightFinding]) -> str:
    level = max((_severity_rank(item.severity) for item in findings), default=0)
    return ("healthy", "monitoring", "attention_required", "critical_attention")[level]


def _signal_from_payload(payload: Mapping[str, object]) -> IntelligenceSignal:
    return IntelligenceSignal(
        signal_id=str(payload["signal_id"]),
        subsystem_id=str(payload["subsystem_id"]),
        kind=SignalKind(str(payload["kind"])),
        value=float(payload["value"]),
        step=int(payload["step"]),
        direction=MetricDirection(str(payload["direction"])),
        unit=str(payload.get("unit", "")),
        sample_count=int(payload.get("sample_count", 1)),
        confidence=float(payload.get("confidence", 1.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _estimate_from_payload(payload: Mapping[str, object]) -> IntelligenceEstimate:
    return IntelligenceEstimate(
        checkpoint_id=str(payload["checkpoint_id"]),
        overall=(None if payload.get("overall") is None else float(payload["overall"])),
        confidence=float(payload.get("confidence", 0.0)),
        calibrated=bool(payload.get("calibrated", False)),
        subsystem_scores=dict(payload.get("subsystem_scores", {})),
        signal_vector=dict(payload.get("signal_vector", {})),
        calibration_error=(
            None if payload.get("calibration_error") is None else float(payload["calibration_error"])
        ),
        limitations=tuple(payload.get("limitations", ())),
    )
