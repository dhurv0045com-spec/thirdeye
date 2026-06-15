from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from thirdeye.adapters.base import AdapterContext
from thirdeye.models import (
    CommandSpec,
    MetricDirection,
    MetricObservation,
    RunResult,
)


METRIC_PREFIX = "THIRDEYE_METRIC "


class CommandAdapter:
    adapter_id = "command"

    def supports(self, command: CommandSpec) -> bool:
        return bool(command.argv)

    def execute(self, command: CommandSpec, context: AdapterContext) -> RunResult:
        started = time.time()
        run_dir = context.work_dir / command.command_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        cwd = (context.project_root / command.cwd).resolve()
        if context.project_root not in (cwd, *cwd.parents):
            raise ValueError(f"Command cwd escapes project root: {cwd}")
        environment = {
            **os.environ,
            **command.env,
            "THIRDEYE_RUN_ID": context.manifest.run_id,
            "THIRDEYE_PROJECT_ID": context.manifest.project_id,
            "THIRDEYE_PHASE": command.phase.value,
        }
        status = "completed"
        return_code: int | None = None
        error: str | None = None
        stdout = ""
        stderr = ""
        try:
            process = subprocess.run(
                list(command.argv),
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
                check=False,
            )
            return_code = int(process.returncode)
            stdout = process.stdout
            stderr = process.stderr
            if return_code != 0:
                status = "failed"
                error = f"Command exited with code {return_code}"
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = f"Command exceeded {command.timeout_seconds:.1f}s"
            stdout = _decode(exc.stdout)
            stderr = _decode(exc.stderr)
        except OSError as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            stderr = error
        if command.retain_output:
            stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
            stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        metrics = _parse_metrics(stdout)
        if command.metrics_file:
            metrics.extend(_load_metrics_file(cwd / command.metrics_file))
        ended = time.time()
        metrics.append(
            MetricObservation(
                metric_id="runtime.duration_seconds",
                value=ended - started,
                direction=MetricDirection.LOWER,
                sample_count=1,
                scorer="thirdeye.command",
                scorer_version="1",
                deterministic=False,
                unit="seconds",
            )
        )
        metrics.append(
            MetricObservation(
                metric_id="runtime.success",
                value=1.0 if status == "completed" else 0.0,
                direction=MetricDirection.HIGHER,
                sample_count=1,
                scorer="thirdeye.command",
                scorer_version="1",
                deterministic=True,
            )
        )
        return RunResult(
            run_id=context.manifest.run_id,
            command_id=command.command_id,
            phase=command.phase,
            status=status,
            return_code=return_code,
            started_at=started,
            ended_at=ended,
            stdout_path=str(stdout_path) if command.retain_output else "",
            stderr_path=str(stderr_path) if command.retain_output else "",
            metrics=tuple(metrics),
            error=error,
        )


def _parse_metrics(stdout: str) -> list[MetricObservation]:
    metrics: list[MetricObservation] = []
    for line in stdout.splitlines():
        if not line.startswith(METRIC_PREFIX):
            continue
        try:
            metrics.append(_metric_from_dict(json.loads(line[len(METRIC_PREFIX) :])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return metrics


def _load_metrics_file(path: Path) -> list[MetricObservation]:
    if not path.exists():
        return []
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("metrics", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    metrics: list[MetricObservation] = []
    for row in rows:
        try:
            metrics.append(_metric_from_dict(row))
        except (KeyError, TypeError, ValueError):
            continue
    return metrics


def _metric_from_dict(payload: dict[str, Any]) -> MetricObservation:
    return MetricObservation(
        metric_id=str(payload["metric_id"]),
        value=float(payload["value"]),
        direction=MetricDirection(str(payload.get("direction", "higher"))),
        sample_count=int(payload.get("sample_count", 1)),
        scorer=str(payload.get("scorer", "project")),
        scorer_version=str(payload.get("scorer_version", "1")),
        deterministic=bool(payload.get("deterministic", False)),
        uncertainty_low=_optional_float(payload.get("uncertainty_low")),
        uncertainty_high=_optional_float(payload.get("uncertainty_high")),
        unit=str(payload.get("unit", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
