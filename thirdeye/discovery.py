from __future__ import annotations

import re
from pathlib import Path

from thirdeye.models import (
    CommandSpec,
    LifecyclePhase,
    ProjectKind,
    ProjectManifest,
    ProjectSpec,
)


def discover_project(root: str | Path, project_id: str | None = None) -> ProjectManifest:
    target = Path(root).resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    files = {path.name for path in target.iterdir() if path.is_file()}
    text = _dependency_text(target)
    commands: list[CommandSpec] = []
    kind = ProjectKind.SOFTWARE

    if "pyproject.toml" in files or "requirements.txt" in files:
        commands.append(
            CommandSpec(
                "python-tests",
                LifecyclePhase.TEST,
                ("python", "-m", "pytest", "-q"),
                description="Run the Python test suite.",
                required=False,
            )
        )
    if "package.json" in files:
        commands.extend(
            [
                CommandSpec(
                    "web-test",
                    LifecyclePhase.TEST,
                    ("npm", "test", "--", "--run"),
                    required=False,
                    description="Run JavaScript tests.",
                ),
                CommandSpec(
                    "web-build",
                    LifecyclePhase.BUILD,
                    ("npm", "run", "build"),
                    required=False,
                    description="Build the web application.",
                ),
            ]
        )
    if re.search(r"\b(torch|tensorflow|jax|transformers|sklearn)\b", text, re.I):
        kind = ProjectKind.MACHINE_LEARNING
    if re.search(r"\b(agent|llm|language model|inference)\b", text, re.I):
        kind = ProjectKind.AI if kind == ProjectKind.SOFTWARE else ProjectKind.HYBRID

    name = target.name
    return ProjectManifest(
        project=ProjectSpec(
            project_id=project_id or _slug(name),
            name=name,
            description="Discovered by ThirdEye. Review commands and feature contracts.",
            kind=kind,
            root=str(target),
        ),
        commands=tuple(commands),
        tags=("auto-discovered",),
    )


def _dependency_text(root: Path) -> str:
    chunks: list[str] = []
    for name in ("pyproject.toml", "requirements.txt", "package.json", "README.md"):
        path = root / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:100_000])
    return "\n".join(chunks)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "project"

