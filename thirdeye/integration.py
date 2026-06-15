from __future__ import annotations

import json
from typing import Any

from thirdeye.adapters.command import METRIC_PREFIX
from thirdeye.models import MetricDirection


def emit_metric(
    metric_id: str,
    value: float,
    *,
    direction: MetricDirection | str = MetricDirection.HIGHER,
    sample_count: int = 1,
    scorer: str = "project",
    scorer_version: str = "1",
    deterministic: bool = False,
    unit: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a metric from any CLI program through stdout."""
    payload = {
        "metric_id": metric_id,
        "value": float(value),
        "direction": MetricDirection(direction).value,
        "sample_count": int(sample_count),
        "scorer": scorer,
        "scorer_version": scorer_version,
        "deterministic": bool(deterministic),
        "unit": unit,
        "metadata": metadata or {},
    }
    print(METRIC_PREFIX + json.dumps(payload, sort_keys=True), flush=True)

