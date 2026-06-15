from __future__ import annotations

import fnmatch
import math
from typing import Any, Iterable

from thirdeye.intelligence.signals import TrainingSignalCollector
from thirdeye.models import MetricDirection, SignalKind, SubsystemSpec


class PyTorchSubsystemCollector:
    """Sample compact per-subsystem signals through non-mutating PyTorch hooks."""

    def __init__(
        self,
        model: Any,
        subsystems: Iterable[SubsystemSpec],
        collector: TrainingSignalCollector,
        *,
        sample_every: int = 10,
        saturation_threshold: float = 6.0,
    ) -> None:
        self.model = model
        self.subsystems = tuple(subsystems)
        self.collector = collector
        self.sample_every = max(1, int(sample_every))
        self.saturation_threshold = float(saturation_threshold)
        self.step = -1
        self.active = False
        self._handles: list[Any] = []
        self._modules: dict[str, list[tuple[str, Any]]] = {}
        self._attach()

    def _attach(self) -> None:
        for subsystem in self.subsystems:
            matches: list[tuple[str, Any]] = []
            for name, module in self.model.named_modules():
                if name and any(
                    fnmatch.fnmatchcase(name, pattern)
                    for pattern in subsystem.module_patterns
                ):
                    matches.append((name, module))
                    self._handles.append(
                        module.register_forward_hook(
                            self._make_hook(subsystem.subsystem_id, name)
                        )
                    )
            self._modules[subsystem.subsystem_id] = matches

    def begin_step(self, step: int) -> bool:
        self.step = int(step)
        self.active = self.step % self.sample_every == 0
        return self.active

    def _make_hook(self, subsystem_id: str, module_name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.active:
                return
            tensor = _first_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return
            values = tensor.detach().float()
            metadata = {"module": module_name, "shape": list(tensor.shape)}
            self.collector.record(
                "activation.mean",
                values.mean().item(),
                step=self.step,
                subsystem_id=subsystem_id,
                kind=SignalKind.REPRESENTATION,
                direction=MetricDirection.TARGET,
                metadata=metadata,
            )
            self.collector.record(
                "activation.rms",
                values.square().mean().sqrt().item(),
                step=self.step,
                subsystem_id=subsystem_id,
                kind=SignalKind.REPRESENTATION,
                direction=MetricDirection.TARGET,
                metadata=metadata,
            )
            self.collector.record(
                "activation.std",
                values.std(unbiased=False).item(),
                step=self.step,
                subsystem_id=subsystem_id,
                kind=SignalKind.REPRESENTATION,
                direction=MetricDirection.TARGET,
                metadata=metadata,
            )
            self.collector.record(
                "activation.sparsity",
                (values.abs() < 1e-6).float().mean().item(),
                step=self.step,
                subsystem_id=subsystem_id,
                kind=SignalKind.REPRESENTATION,
                direction=MetricDirection.TARGET,
                metadata=metadata,
            )
            self.collector.record(
                "activation.saturation",
                (values.abs() > self.saturation_threshold).float().mean().item(),
                step=self.step,
                subsystem_id=subsystem_id,
                kind=SignalKind.RELIABILITY,
                direction=MetricDirection.LOWER,
                metadata=metadata,
            )

        return hook

    def capture_gradients(self, *, learning_rate: float) -> dict[str, float]:
        if not self.active:
            return {}
        result: dict[str, float] = {}
        for subsystem_id, modules in self._modules.items():
            parameters = _unique_parameters(module for _, module in modules)
            parameter_norm, gradient_norm = _norms(parameters)
            update_ratio = (
                abs(float(learning_rate)) * gradient_norm / max(parameter_norm, 1e-12)
            )
            result[f"{subsystem_id}.parameter_norm"] = parameter_norm
            result[f"{subsystem_id}.gradient_norm"] = gradient_norm
            result[f"{subsystem_id}.update_ratio"] = update_ratio
            self.collector.record(
                "parameters.norm",
                parameter_norm,
                step=self.step,
                subsystem_id=subsystem_id,
                direction=MetricDirection.TARGET,
            )
            self.collector.record(
                "gradients.norm",
                gradient_norm,
                step=self.step,
                subsystem_id=subsystem_id,
                direction=MetricDirection.TARGET,
            )
            self.collector.record(
                "update.ratio",
                update_ratio,
                step=self.step,
                subsystem_id=subsystem_id,
                direction=MetricDirection.TARGET,
            )
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> PyTorchSubsystemCollector:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _first_tensor(value: Any) -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _unique_parameters(modules: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters(recurse=True):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def _norms(parameters: Iterable[Any]) -> tuple[float, float]:
    parameter_sq = 0.0
    gradient_sq = 0.0
    for parameter in parameters:
        parameter_sq += float(parameter.detach().float().square().sum().item())
        if parameter.grad is not None:
            gradient_sq += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(parameter_sq), math.sqrt(gradient_sq)
