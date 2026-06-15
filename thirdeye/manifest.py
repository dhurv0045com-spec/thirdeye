from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thirdeye.models import (
    CommandSpec,
    FeatureCategory,
    FeatureSpec,
    FeatureVariant,
    LifecyclePhase,
    ProjectKind,
    ProjectManifest,
    ProjectSpec,
)


MANIFEST_NAME = "thirdeye.json"


def load_manifest(path: str | Path = MANIFEST_NAME) -> ProjectManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return manifest_from_dict(payload)


def manifest_from_dict(payload: dict[str, Any]) -> ProjectManifest:
    project_data = dict(payload["project"])
    project_data["kind"] = ProjectKind(project_data.get("kind", "hybrid"))
    project = ProjectSpec(**project_data)
    commands = tuple(
        CommandSpec(
            **{
                **item,
                "phase": LifecyclePhase(item["phase"]),
                "argv": tuple(item["argv"]),
            }
        )
        for item in payload.get("commands", [])
    )
    features = tuple(_load_feature(item) for item in payload.get("features", []))
    return ProjectManifest(
        project=project,
        commands=commands,
        features=features,
        datasets=dict(payload.get("datasets", {})),
        tokenizer=payload.get("tokenizer"),
        checkpoint=payload.get("checkpoint"),
        tags=tuple(payload.get("tags", ())),
        schema_version=int(payload.get("schema_version", 1)),
    )


def save_manifest(manifest: ProjectManifest, path: str | Path = MANIFEST_NAME) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def _load_feature(item: dict[str, Any]) -> FeatureSpec:
    payload = dict(item)
    payload["category"] = FeatureCategory(payload["category"])
    payload["variants"] = tuple(
        FeatureVariant(**variant) for variant in payload["variants"]
    )
    for field in (
        "expected_benefits",
        "possible_regressions",
        "protected_metrics",
        "required_scenarios",
        "resource_metrics",
        "incompatible_features",
    ):
        payload[field] = tuple(payload.get(field, ()))
    return FeatureSpec(**payload)
