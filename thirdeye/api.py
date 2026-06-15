from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install thirdeye-evidence[server] to use the API.") from exc

from thirdeye.models import ProjectSpec
from thirdeye.sdk import ThirdEye

app = FastAPI(title="ThirdEye", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "thirdeye"}


@app.post("/projects")
def create_project(project: ProjectSpec) -> dict:
    ThirdEye().register_project(project)
    return project.to_dict()


@app.post("/projects/{project_id}/evaluate")
def evaluate(project_id: str, profile: str = "auto") -> dict:
    try:
        return ThirdEye().evaluate(project_id, profile)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/projects/{project_id}/features")
def features(project_id: str) -> dict:
    return {"features": ThirdEye().store.list("feature", project_id)}

