from __future__ import annotations

import time
from typing import Any, Callable

from thirdeye.models import MetricDirection, MetricObservation


class PyTorchAdapter:
    """Optional, dependency-light helpers for PyTorch models and training loops."""

    adapter_id = "pytorch"

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def model_metrics(model: Any) -> list[MetricObservation]:
        parameters = list(model.parameters())
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        )
        return [
            _metric("model.parameters", total, "count", deterministic=True),
            _metric("model.trainable_parameters", trainable, "count", deterministic=True),
            _metric("model.parameter_bytes", parameter_bytes, "bytes", deterministic=True),
        ]

    @staticmethod
    def profile_callable(
        fn: Callable[[], Any],
        *,
        tokens: int = 0,
        scorer: str = "thirdeye.pytorch",
    ) -> tuple[Any, list[MetricObservation]]:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = fn()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = [
            _metric(
                "runtime.duration_seconds",
                elapsed,
                "seconds",
                deterministic=False,
                scorer=scorer,
                direction=MetricDirection.LOWER,
            )
        ]
        if tokens > 0:
            metrics.append(
                _metric(
                    "runtime.tokens_per_second",
                    tokens / max(elapsed, 1e-9),
                    "tokens/second",
                    deterministic=False,
                    scorer=scorer,
                )
            )
        if cuda:
            metrics.append(
                _metric(
                    "runtime.peak_memory_bytes",
                    int(torch.cuda.max_memory_allocated()),
                    "bytes",
                    deterministic=False,
                    scorer=scorer,
                    direction=MetricDirection.LOWER,
                )
            )
        return result, metrics


def _metric(
    metric_id: str,
    value: float,
    unit: str,
    *,
    deterministic: bool,
    scorer: str = "thirdeye.pytorch",
    direction: MetricDirection = MetricDirection.HIGHER,
) -> MetricObservation:
    return MetricObservation(
        metric_id=metric_id,
        value=float(value),
        direction=direction,
        sample_count=1,
        scorer=scorer,
        scorer_version="1",
        deterministic=deterministic,
        unit=unit,
    )

