from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any

from thirdeye.adapters import AdapterContext, AdapterRegistry, CommandAdapter
from thirdeye.lineage import capture_manifest
from thirdeye.manifest import load_manifest
from thirdeye.models import CommandSpec, LifecyclePhase, ProjectManifest
from thirdeye.plugins import load_adapter_plugins


class ProjectRunner:
    def __init__(self, eye: Any, manifest: ProjectManifest) -> None:
        self.eye = eye
        self.project = manifest
        self.registry = AdapterRegistry()
        self.registry.register(CommandAdapter())
        load_adapter_plugins(self.registry)

    @classmethod
    def from_file(cls, eye: Any, path: str | Path) -> "ProjectRunner":
        return cls(eye, load_manifest(path))

    def run(
        self,
        *,
        command_id: str | None = None,
        phase: LifecyclePhase | None = None,
    ) -> list[dict[str, Any]]:
        commands = self._select(command_id=command_id, phase=phase)
        if not commands:
            raise ValueError("No commands matched the requested selection.")
        root = Path(self.project.project.root).resolve()
        results: list[dict[str, Any]] = []
        for command in commands:
            reproduction = shlex.join(command.argv)
            manifest = capture_manifest(
                project_id=self.project.project.project_id,
                protocol_id=f"lifecycle:{command.phase.value}",
                feature_variants={},
                root=root,
                config=command.to_dict(),
                datasets={
                    name: root / path for name, path in self.project.datasets.items()
                },
                tokenizer=root / self.project.tokenizer if self.project.tokenizer else None,
                checkpoint=root / self.project.checkpoint if self.project.checkpoint else None,
                reproduction_command=reproduction,
            )
            self.eye.start_run(manifest)
            context = AdapterContext(
                store=self.eye.store,
                manifest=manifest,
                project_root=root,
                work_dir=self.eye.store.home / "runs" / manifest.run_id,
            )
            result = self.registry.resolve(command).execute(command, context)
            self.eye.emit_metrics(manifest.run_id, result.metrics)
            self.eye.record_run_result(self.project.project.project_id, result)
            for path, kind in (
                (result.stdout_path, "stdout"),
                (result.stderr_path, "stderr"),
            ):
                if not path:
                    continue
                self.eye.store.add_artifact(
                    project_id=self.project.project.project_id,
                    run_id=manifest.run_id,
                    source=path,
                    kind=kind,
                )
            results.append(result.to_dict())
            if command.required and result.status != "completed":
                break
        return results

    def _select(
        self,
        *,
        command_id: str | None,
        phase: LifecyclePhase | None,
    ) -> list[CommandSpec]:
        commands = list(self.project.commands)
        if command_id:
            commands = [item for item in commands if item.command_id == command_id]
        if phase:
            commands = [item for item in commands if item.phase == phase]
        return commands
