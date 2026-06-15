from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Iterable

from thirdeye.models import IntelligenceEstimate


@dataclass
class CalibrationModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    mae: float
    samples: int


class IntelligenceCalibrator:
    """Small ridge regressor linking telemetry to held-out capability."""

    def __init__(self, ridge: float = 1e-3, minimum_samples: int = 5) -> None:
        self.ridge = float(ridge)
        self.minimum_samples = int(minimum_samples)
        self.model: CalibrationModel | None = None

    def fit(
        self,
        observations: Iterable[tuple[dict[str, float], float]],
    ) -> CalibrationModel:
        rows = list(observations)
        if len(rows) < self.minimum_samples:
            raise ValueError(
                f"Calibration requires at least {self.minimum_samples} checkpoints."
            )
        names = tuple(sorted(set.intersection(*(set(vector) for vector, _ in rows))))
        if not names:
            raise ValueError("Calibration checkpoints share no telemetry signals.")
        raw = [[float(vector[name]) for name in names] for vector, _ in rows]
        targets = [float(target) for _, target in rows]
        means = [mean(column) for column in zip(*raw)]
        scales = [
            math.sqrt(mean((value - center) ** 2 for value in column)) or 1.0
            for column, center in zip(zip(*raw), means)
        ]
        x = [
            [(value - center) / scale for value, center, scale in zip(row, means, scales)]
            for row in raw
        ]
        x_aug = [[1.0, *row] for row in x]
        xtx = _matmul(_transpose(x_aug), x_aug)
        for index in range(1, len(xtx)):
            xtx[index][index] += self.ridge
        xty = _matvec(_transpose(x_aug), targets)
        beta = _solve(xtx, xty)
        predictions = [_dot(beta, row) for row in x_aug]
        mae = mean(abs(predicted - target) for predicted, target in zip(predictions, targets))
        self.model = CalibrationModel(
            feature_names=names,
            means=tuple(means),
            scales=tuple(scales),
            coefficients=tuple(beta[1:]),
            intercept=beta[0],
            mae=mae,
            samples=len(rows),
        )
        return self.model

    def predict(
        self,
        vector: dict[str, float],
        *,
        checkpoint_id: str,
        subsystem_vectors: dict[str, dict[str, float]] | None = None,
    ) -> IntelligenceEstimate:
        if self.model is None:
            return IntelligenceEstimate(
                checkpoint_id=checkpoint_id,
                overall=None,
                confidence=0.0,
                calibrated=False,
                subsystem_scores={
                    name: None for name in (subsystem_vectors or {})
                },
                signal_vector=dict(vector),
                limitations=("No post-training capability calibration is fitted.",),
            )
        missing = [name for name in self.model.feature_names if name not in vector]
        if missing:
            return IntelligenceEstimate(
                checkpoint_id=checkpoint_id,
                overall=None,
                confidence=0.0,
                calibrated=False,
                subsystem_scores={
                    name: None for name in (subsystem_vectors or {})
                },
                signal_vector=dict(vector),
                calibration_error=self.model.mae,
                limitations=(f"Missing {len(missing)} calibrated signals.",),
            )
        standardized = [
            (vector[name] - center) / scale
            for name, center, scale in zip(
                self.model.feature_names, self.model.means, self.model.scales
            )
        ]
        overall = self.model.intercept + _dot(self.model.coefficients, standardized)
        confidence = min(0.99, (1.0 - math.exp(-self.model.samples / 10.0)) / (1.0 + self.model.mae))
        subsystem_scores = self._subsystem_scores(vector, subsystem_vectors or {})
        return IntelligenceEstimate(
            checkpoint_id=checkpoint_id,
            overall=overall,
            confidence=confidence,
            calibrated=True,
            subsystem_scores=subsystem_scores,
            signal_vector=dict(vector),
            calibration_error=self.model.mae,
            limitations=(
                "Estimate predicts the configured held-out capability target, not universal intelligence.",
                "Subsystem scores are contribution estimates, not standalone causal effects.",
            ),
        )

    def _subsystem_scores(
        self,
        vector: dict[str, float],
        subsystem_vectors: dict[str, dict[str, float]],
    ) -> dict[str, float | None]:
        if self.model is None:
            return {name: None for name in subsystem_vectors}
        contributions: dict[str, list[float]] = {}
        for name, center, scale, coefficient in zip(
            self.model.feature_names,
            self.model.means,
            self.model.scales,
            self.model.coefficients,
        ):
            subsystem = name.split(":", 1)[0]
            value = (vector[name] - center) / scale
            contributions.setdefault(subsystem, []).append(coefficient * value)
        return {
            subsystem: sum(values) / max(1, len(values))
            for subsystem, values in contributions.items()
        }


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_t = _transpose(right)
    return [[_dot(row, column) for column in right_t] for row in left]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(n)]

