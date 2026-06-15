from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from thirdeye.controller import plan_missing_evidence
from thirdeye.models import (
    FeatureSpec,
    LifecycleEvent,
    MetricObservation,
    ProjectManifest,
    ProjectSpec,
    RunManifest,
    RunResult,
)
from thirdeye.reports import build_bundle, write_reports
from thirdeye.store import EvidenceStore


class ThirdEye:
    def __init__(self, home: str | Path | None = None) -> None:
        self.store = EvidenceStore(home)

    def register_project(self, project: ProjectSpec) -> None:
        self.store.put("project", project.project_id, project.project_id, project.to_dict())

    def register_manifest(self, manifest: ProjectManifest) -> None:
        self.register_project(manifest.project)
        self.store.put(
            "project_manifest",
            manifest.project.project_id,
            manifest.project.project_id,
            manifest.to_dict(),
        )
        for feature in manifest.features:
            self.register_feature(manifest.project.project_id, feature)

    def register_feature(self, project_id: str, feature: FeatureSpec) -> None:
        self.store.put("feature", feature.feature_id, project_id, feature.to_dict())

    def start_run(self, manifest: RunManifest) -> None:
        self.store.put("run", manifest.run_id, manifest.project_id, manifest.to_dict())

    def emit_metrics(
        self, run_id: str, metrics: Iterable[MetricObservation]
    ) -> None:
        self.store.put_metrics(run_id, (metric.to_dict() for metric in metrics))

    def record_evidence(self, evidence: dict) -> None:
        self.store.put(
            "evidence",
            str(evidence["evidence_id"]),
            str(evidence["project_id"]),
            evidence,
        )

    def record_event(self, event: LifecycleEvent) -> None:
        self.store.put(
            "event",
            event.event_id,
            event.project_id,
            event.to_dict(),
        )

    def record_run_result(self, project_id: str, result: RunResult) -> None:
        self.store.put("run_result", result.run_id, project_id, result.to_dict())

    def project_snapshot(self, project_id: str) -> dict:
        project = self.store.get("project", project_id)
        if project is None:
            raise KeyError(f"Unknown project: {project_id}")
        runs = self.store.list("run", project_id)
        run_results = self.store.list("run_result", project_id)
        return {
            "project": project,
            "features": self.store.list("feature", project_id),
            "runs": runs,
            "run_results": run_results,
            "events": self.store.list("event", project_id),
            "evidence": self.store.list("evidence", project_id),
            "artifacts": self.store.list("artifact", project_id),
            "metrics": {
                run["run_id"]: self.store.metrics(run["run_id"]) for run in runs
            },
        }

    def evaluate(self, project_id: str, profile: str = "auto") -> dict:
        project = self.store.get("project", project_id)
        if project is None:
            raise KeyError(f"Unknown project: {project_id}")
        feature_rows = self.store.list("feature", project_id)
        evidence_rows = self.store.list("evidence", project_id)
        run_rows = self.store.list("run", project_id)
        run_results = self.store.list("run_result", project_id)
        latest = {row["feature_id"]: row for row in evidence_rows}
        features = [FeatureSpec(**_inflate_feature(row)) for row in feature_rows]
        plans = plan_missing_evidence(features=features, latest_evidence=latest, profile=profile)
        planned = [
            {"priority": plan.priority, "reason": plan.reason, "protocol": asdict(plan.protocol)}
            for plan in plans
        ]
        bundle = build_bundle(
            project=project,
            features=feature_rows,
            evidence=evidence_rows,
            planned=planned,
            runs=run_rows,
            run_results=run_results,
            metrics={
                run["run_id"]: self.store.metrics(run["run_id"]) for run in run_rows
            },
        )
        report_paths = write_reports(bundle, self.store.home / "reports" / project_id)
        return {**bundle, "report_paths": report_paths}

    def assess(self, manifest: ProjectManifest, profile: str = "auto") -> dict:
        """Run declared project lifecycle checks, then synthesize evidence reports."""
        from thirdeye.runner import ProjectRunner

        self.register_manifest(manifest)
        run_results = ProjectRunner(self, manifest).run() if manifest.commands else []
        evaluation = self.evaluate(manifest.project.project_id, profile)
        return {"run_results": run_results, "evaluation": evaluation}


def _inflate_feature(row: dict) -> dict:
    from thirdeye.models import FeatureCategory, FeatureVariant

    payload = dict(row)
    payload["category"] = FeatureCategory(payload["category"])
    payload["variants"] = tuple(FeatureVariant(**item) for item in payload["variants"])
    for name in (
        "expected_benefits",
        "possible_regressions",
        "protected_metrics",
        "required_scenarios",
        "resource_metrics",
        "incompatible_features",
    ):
        payload[name] = tuple(payload.get(name, ()))
    return payload
