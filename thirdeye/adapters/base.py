from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from thirdeye.models import CommandSpec, RunManifest, RunResult
from thirdeye.store import EvidenceStore


@dataclass
class AdapterContext:
    store: EvidenceStore
    manifest: RunManifest
    project_root: Path
    work_dir: Path
    attributes: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    adapter_id: str

    def supports(self, command: CommandSpec) -> bool: ...

    def execute(self, command: CommandSpec, context: AdapterContext) -> RunResult: ...


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: list[Adapter] = []

    def register(self, adapter: Adapter) -> None:
        if any(item.adapter_id == adapter.adapter_id for item in self._adapters):
            raise ValueError(f"Adapter already registered: {adapter.adapter_id}")
        self._adapters.append(adapter)

    def resolve(self, command: CommandSpec) -> Adapter:
        for adapter in self._adapters:
            if adapter.adapter_id == command.adapter and adapter.supports(command):
                return adapter
        raise LookupError(
            f"No '{command.adapter}' adapter supports command: {command.command_id}"
        )

    @property
    def adapters(self) -> tuple[str, ...]:
        return tuple(adapter.adapter_id for adapter in self._adapters)
