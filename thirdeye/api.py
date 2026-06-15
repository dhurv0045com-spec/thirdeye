from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install thirdeye-evidence[server] to use the API.") from exc

from thirdeye.manifest import manifest_from_dict
from thirdeye.models import ProjectSpec
from thirdeye.sdk import ThirdEye

app = FastAPI(title="ThirdEye", version="0.3.0")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _dashboard()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "thirdeye"}


@app.post("/projects")
def create_project(project: ProjectSpec) -> dict:
    ThirdEye().register_project(project)
    return project.to_dict()


@app.get("/projects")
def projects() -> dict:
    return {"projects": ThirdEye().store.list_all("project")}


@app.post("/projects/{project_id}/evaluate")
def evaluate(project_id: str, profile: str = "auto") -> dict:
    try:
        return ThirdEye().evaluate(project_id, profile)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/projects/{project_id}/assess")
def assess(project_id: str, profile: str = "auto") -> dict:
    if os.environ.get("THIRDEYE_ALLOW_EXECUTION") != "1":
        raise HTTPException(
            status_code=403,
            detail=(
                "Server-side command execution is disabled. Set "
                "THIRDEYE_ALLOW_EXECUTION=1 only on a trusted local control plane."
            ),
        )
    eye = ThirdEye()
    payload = eye.store.get("project_manifest", project_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Project manifest not found.")
    return eye.assess(manifest_from_dict(payload), profile)


@app.get("/projects/{project_id}/features")
def features(project_id: str) -> dict:
    return {"features": ThirdEye().store.list("feature", project_id)}


@app.get("/projects/{project_id}/intelligence")
def intelligence(project_id: str) -> dict:
    return ThirdEye().intelligence_snapshot(project_id)


@app.get("/projects/{project_id}")
def project(project_id: str) -> dict:
    try:
        return ThirdEye().project_snapshot(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _dashboard() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThirdEye</title><style>
:root{color-scheme:dark;--bg:#071018;--panel:#0e1b26;--line:#203747;--text:#edf7ff;--muted:#8ba5b7;--cyan:#4de3ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#123047,transparent 36%),var(--bg);
color:var(--text);font:14px Inter,Segoe UI,sans-serif}main{max-width:1240px;margin:auto;padding:32px}h1{font-size:42px;margin:4px 0}
.eyebrow{color:var(--cyan);letter-spacing:.2em;font-weight:800}.muted{color:var(--muted)}.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:24px}.card{background:rgba(14,27,38,.9);
border:1px solid var(--line);border-radius:16px;padding:20px;cursor:pointer}.card:hover{border-color:var(--cyan)}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 9px;color:var(--cyan)}
button{background:var(--cyan);color:#041019;border:0;border-radius:9px;padding:9px 13px;font-weight:800;cursor:pointer}
pre{white-space:pre-wrap;background:#071018;padding:16px;border-radius:12px;max-height:420px;overflow:auto}</style></head>
<body><main><div class="eyebrow">Universal Evidence Control Plane</div><h1>ThirdEye</h1>
<div class="muted">See what works, why it works, what it costs, and what to test next.</div>
<div id="projects" class="grid"></div><section id="detail"></section></main><script>
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){const d=await fetch('/projects').then(r=>r.json());document.querySelector('#projects').innerHTML=
d.projects.map(p=>`<article class="card" onclick="openProject('${esc(p.project_id)}')"><span class="pill">${esc(p.kind||'hybrid')}</span>
<h2>${esc(p.name)}</h2><div class="muted">${esc(p.description)}</div></article>`).join('')||'<div class="card">No projects yet. Run <code>thirdeye onboard</code>.</div>'}
async function openProject(id){const d=await fetch('/projects/'+encodeURIComponent(id)).then(r=>r.json());
document.querySelector('#detail').innerHTML=`<div class="card" style="margin-top:18px"><button onclick="assessProject('${esc(id)}')">
Run Full Assessment</button> <button onclick="evaluateProject('${esc(id)}')">Refresh Reports</button><h2>${esc(d.project.name)}</h2><p>${d.features.length} features, ${d.runs.length} runs, ${d.evidence.length} evidence records</p>
<pre>${esc(JSON.stringify(d.run_results.slice(-10),null,2))}</pre></div>`}
async function assessProject(id){const r=await fetch(`/projects/${encodeURIComponent(id)}/assess?profile=auto`,{method:'POST'});
const d=await r.json();document.querySelector('#detail').innerHTML=`<div class="card" style="margin-top:18px"><h2>${r.ok?'Assessment complete':'Assessment blocked'}</h2>
<pre>${esc(JSON.stringify(d,null,2))}</pre></div>`}
async function evaluateProject(id){const d=await fetch(`/projects/${encodeURIComponent(id)}/evaluate?profile=auto`,{method:'POST'}).then(r=>r.json());
document.querySelector('#detail').innerHTML=`<div class="card" style="margin-top:18px"><h2>Evaluation complete</h2>
<pre>${esc(JSON.stringify(d.report_paths,null,2))}</pre></div>`} load();</script></body></html>"""
