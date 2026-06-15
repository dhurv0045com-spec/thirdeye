from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
import uuid
from typing import Any, Callable, Iterator, TypeVar

from thirdeye.models import (
    LifecycleEvent,
    LifecyclePhase,
    MetricDirection,
    MetricObservation,
)


T = TypeVar("T")


@dataclass
class RuntimeSession:
    eye: Any
    project_id: str
    run_id: str
    phase: LifecyclePhase

    def metric(
        self,
        metric_id: str,
        value: float,
        *,
        direction: MetricDirection = MetricDirection.HIGHER,
        sample_count: int = 1,
        unit: str = "",
        deterministic: bool = False,
        scorer: str = "project",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.eye.emit_metrics(
            self.run_id,
            [
                MetricObservation(
                    metric_id=metric_id,
                    value=float(value),
                    direction=direction,
                    sample_count=sample_count,
                    scorer=scorer,
                    scorer_version="1",
                    deterministic=deterministic,
                    unit=unit,
                    metadata=metadata or {},
                )
            ],
        )

    def event(
        self,
        name: str,
        *,
        severity: str = "info",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            project_id=self.project_id,
            run_id=self.run_id,
            phase=self.phase,
            name=name,
            timestamp=time.time(),
            severity=severity,
            attributes=attributes or {},
            trace_id=self.run_id.replace("-", "")[:32],
            span_id=uuid.uuid4().hex[:16],
        )
        self.eye.record_event(event)


@contextmanager
def lifecycle(
    eye: Any,
    *,
    project_id: str,
    run_id: str,
    phase: LifecyclePhase,
    name: str,
) -> Iterator[RuntimeSession]:
    session = RuntimeSession(eye, project_id, run_id, phase)
    started = time.perf_counter()
    session.event(f"{name}.started")
    try:
        yield session
    except Exception as exc:
        session.event(
            f"{name}.failed",
            severity="error",
            attributes={"error.type": type(exc).__name__, "error.message": str(exc)},
        )
        session.metric(
            "runtime.success",
            0.0,
            deterministic=True,
            scorer="thirdeye.runtime",
        )
        raise
    else:
        session.event(f"{name}.completed")
        session.metric(
            "runtime.success",
            1.0,
            deterministic=True,
            scorer="thirdeye.runtime",
        )
    finally:
        session.metric(
            "runtime.duration_seconds",
            time.perf_counter() - started,
            direction=MetricDirection.LOWER,
            unit="seconds",
            scorer="thirdeye.runtime",
        )


def instrument(
    eye: Any,
    *,
    project_id: str,
    run_id: str,
    phase: LifecyclePhase,
    name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with lifecycle(
                eye,
                project_id=project_id,
                run_id=run_id,
                phase=phase,
                name=name or fn.__qualname__,
            ):
                return fn(*args, **kwargs)

        return wrapper

    return decorator

