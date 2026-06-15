from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from thirdeye.hashing import hash_bytes, hash_file, hash_json
from thirdeye.models import RunManifest


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return b""


def capture_manifest(
    *,
    project_id: str,
    protocol_id: str,
    feature_variants: dict[str, str],
    root: str | Path = ".",
    config: dict[str, Any] | None = None,
    datasets: dict[str, str | Path] | None = None,
    tokenizer: str | Path | None = None,
    checkpoint: str | Path | None = None,
    seed: int = 0,
    precision: str = "unknown",
    reproduction_command: str = "",
) -> RunManifest:
    repo = Path(root).resolve()
    commit = _git(repo, "rev-parse", "HEAD").decode("utf-8", errors="replace").strip()
    patch = _git(repo, "diff", "--binary", "HEAD")
    dependency_data = _git(repo, "ls-files", "requirements*.txt", "pyproject.toml", "*.lock")
    dataset_hashes = {
        name: hash_file(path)
        for name, path in (datasets or {}).items()
        if Path(path).exists()
    }
    return RunManifest(
        run_id=str(uuid.uuid4()),
        project_id=project_id,
        protocol_id=protocol_id,
        feature_variants=dict(feature_variants),
        git_commit=commit or "unversioned",
        dirty_patch_hash=hash_bytes(patch),
        config_hash=hash_json(config or {}),
        dependency_hash=hash_bytes(dependency_data),
        dataset_hashes=dataset_hashes,
        tokenizer_hash=hash_file(tokenizer) if tokenizer and Path(tokenizer).exists() else None,
        checkpoint_hash=hash_file(checkpoint) if checkpoint and Path(checkpoint).exists() else None,
        seed=int(seed),
        hardware={
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        runtime={
            "python": sys.version,
            "implementation": platform.python_implementation(),
        },
        precision=precision,
        reproduction_command=reproduction_command,
        created_at=time.time(),
    )

