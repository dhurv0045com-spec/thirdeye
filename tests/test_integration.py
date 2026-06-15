from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

from thirdeye.api import app
from thirdeye.discovery import discover_project
from thirdeye.lineage import capture_manifest
from thirdeye.manifest import load_manifest, save_manifest
from thirdeye.models import (
    CommandSpec,
    LifecyclePhase,
    ProjectManifest,
    ProjectSpec,
)
from thirdeye.runner import ProjectRunner
from thirdeye.runtime import lifecycle
from thirdeye.sdk import ThirdEye


def test_discovery_and_manifest_roundtrip(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="demo"\ndependencies=["torch"]\n',
        encoding="utf-8",
    )
    manifest = discover_project(tmp_path, "demo")
    path = save_manifest(manifest, tmp_path / "thirdeye.json")
    loaded = load_manifest(path)

    assert loaded.project.project_id == "demo"
    assert loaded.project.kind.value == "machine_learning"
    assert loaded.commands[0].phase == LifecyclePhase.TEST


def test_command_runner_captures_metrics_without_retaining_raw_output(tmp_path) -> None:
    script = tmp_path / "metric_script.py"
    script.write_text(
        "from thirdeye.integration import emit_metric\n"
        "emit_metric('quality.score', 0.9, deterministic=True)\n",
        encoding="utf-8",
    )
    project = ProjectManifest(
        project=ProjectSpec("demo", "Demo", root=str(tmp_path)),
        commands=(
            CommandSpec(
                "evaluate",
                LifecyclePhase.EVALUATE,
                (sys.executable, str(script)),
            ),
        ),
    )
    eye = ThirdEye(tmp_path / "state")
    eye.register_manifest(project)

    results = ProjectRunner(eye, project).run()

    assert results[0]["status"] == "completed"
    assert results[0]["stdout_path"] == ""
    metrics = eye.store.metrics(results[0]["run_id"])
    assert {row["metric_id"] for row in metrics} >= {
        "quality.score",
        "runtime.duration_seconds",
        "runtime.success",
    }
    assert eye.store.list("artifact", "demo") == []


def test_command_runner_can_retain_hashed_logs(tmp_path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('hello')\n", encoding="utf-8")
    project = ProjectManifest(
        project=ProjectSpec("demo", "Demo", root=str(tmp_path)),
        commands=(
            CommandSpec(
                "test",
                LifecyclePhase.TEST,
                (sys.executable, str(script)),
                retain_output=True,
            ),
        ),
    )
    eye = ThirdEye(tmp_path / "state")
    eye.register_manifest(project)

    result = ProjectRunner(eye, project).run()[0]

    assert Path(result["stdout_path"]).read_text(encoding="utf-8").strip() == "hello"
    artifacts = eye.store.list("artifact", "demo")
    assert {row["kind"] for row in artifacts} == {"stdout", "stderr"}
    assert all(len(row["sha256"]) == 64 for row in artifacts)


def test_one_click_assessment_runs_and_reports(tmp_path) -> None:
    script = tmp_path / "evaluate.py"
    script.write_text(
        "from thirdeye import emit_metric\n"
        "emit_metric('quality.score', 0.95, deterministic=True)\n",
        encoding="utf-8",
    )
    manifest = ProjectManifest(
        project=ProjectSpec("demo", "Demo", root=str(tmp_path)),
        commands=(
            CommandSpec(
                "evaluate",
                LifecyclePhase.EVALUATE,
                (sys.executable, str(script)),
            ),
        ),
    )
    eye = ThirdEye(tmp_path / "state")

    result = eye.assess(manifest, "quick")

    assert result["run_results"][0]["status"] == "completed"
    assert Path(result["evaluation"]["report_paths"]["dashboard"]).exists()
    assert result["evaluation"]["summary"]["completed_runs"] == 1
    assert result["evaluation"]["summary"]["metric_count"] == 3
    scorecard = Path(result["evaluation"]["report_paths"]["scorecard"]).read_text(
        encoding="utf-8"
    )
    assert "quality.score" in scorecard


def test_python_lifecycle_records_events_and_metrics(tmp_path) -> None:
    eye = ThirdEye(tmp_path)
    eye.register_project(ProjectSpec("demo", "Demo"))
    manifest = capture_manifest(
        project_id="demo",
        protocol_id="runtime",
        feature_variants={},
        root=tmp_path,
    )
    eye.start_run(manifest)

    with lifecycle(
        eye,
        project_id="demo",
        run_id=manifest.run_id,
        phase=LifecyclePhase.RUNTIME,
        name="demo.operation",
    ) as session:
        session.metric("quality.score", 1.0, deterministic=True)

    assert len(eye.store.list("event", "demo")) == 2
    assert {row["metric_id"] for row in eye.store.metrics(manifest.run_id)} == {
        "quality.score",
        "runtime.duration_seconds",
        "runtime.success",
    }


def test_api_dashboard_and_project_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    eye = ThirdEye()
    eye.register_project(ProjectSpec("demo", "Demo"))
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert "Universal Evidence Control Plane" in client.get("/").text
    response = client.get("/projects/demo")
    assert response.status_code == 200
    assert response.json()["project"]["name"] == "Demo"
