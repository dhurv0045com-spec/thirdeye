from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_bundle(
    *,
    project: dict[str, Any],
    features: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    runs: list[dict[str, Any]] | None = None,
    run_results: list[dict[str, Any]] | None = None,
    metrics: dict[str, list[dict[str, Any]]] | None = None,
    intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_by_feature = {row["feature_id"]: row for row in evidence}
    metric_summary = _metric_summary(run_results or [], metrics or {})
    return {
        "schema_version": 1,
        "project": project,
        "features": [
            {**feature, "latest_evidence": evidence_by_feature.get(feature["feature_id"])}
            for feature in features
        ],
        "evidence": evidence,
        "recommended_experiments": planned,
        "runs": runs or [],
        "run_results": run_results or [],
        "metrics": metrics or {},
        "metric_summary": metric_summary,
        "intelligence": intelligence
        or {
            "subsystems": [],
            "signals": [],
            "capability_targets": [],
            "estimates": [],
        },
        "summary": _summary(
            features=features,
            evidence=evidence,
            run_results=run_results or [],
            metric_summary=metric_summary,
        ),
    }


def write_reports(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle_path = target / "evidence-bundle.json"
    scorecard_path = target / "decision-scorecard.md"
    scientific_path = target / "scientific-report.md"
    html_path = target / "decision-dashboard.html"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    feature_lines = []
    for feature in bundle["features"]:
        evidence = feature.get("latest_evidence")
        grade = evidence.get("grade") if evidence else "untested"
        missing = ", ".join((evidence or {}).get("missing_requirements", [])) or "none"
        feature_lines.append(f"| {feature['name']} | {grade} | {missing} |")
    intelligence = bundle["intelligence"]
    estimates = intelligence.get("estimates", [])
    latest_estimate = estimates[-1] if estimates else None
    subsystem_lines = [
        (
            f"| {subsystem['name']} | {subsystem['kind']} | "
            f"{_subsystem_score(latest_estimate, subsystem['subsystem_id'])} |"
        )
        for subsystem in intelligence.get("subsystems", [])
    ]
    scorecard_path.write_text(
        "\n".join(
            [
                f"# {bundle['project']['name']} Decision Scorecard",
                "",
                "## Operational Status",
                "",
                "| Command | Phase | Status | Duration |",
                "| --- | --- | --- | ---: |",
                *[
                    (
                        f"| {row['command_id']} | {row['phase']} | {row['status']} | "
                        f"{float(row.get('duration_seconds', 0.0)):.2f}s |"
                    )
                    for row in bundle["run_results"][-20:]
                ],
                "",
                "## Latest Metrics",
                "",
                "| Metric | Value | Direction | Delta vs previous | Scorer |",
                "| --- | ---: | --- | ---: | --- |",
                *[
                    (
                        f"| {row['metric_id']} | {row['value']:.6g} {row['unit']} | "
                        f"{row['direction']} | "
                        f"{_format_delta(row['delta'])} | "
                        f"{row['scorer']}@{row['scorer_version']} |"
                    )
                    for row in bundle["metric_summary"]
                ],
                "",
                "## Feature Evidence",
                "",
                "| Feature | Evidence grade | Missing evidence |",
                "| --- | --- | --- |",
                *feature_lines,
                "",
                "## Model Intelligence Telemetry",
                "",
                (
                    "Overall calibrated capability estimate: "
                    + (
                        f"{latest_estimate['overall']:.6g} "
                        f"(confidence {latest_estimate['confidence']:.3f})"
                        if latest_estimate and latest_estimate.get("overall") is not None
                        else "unavailable until telemetry is calibrated against held-out evaluations"
                    )
                ),
                "",
                "| Subsystem | Kind | Calibrated contribution |",
                "| --- | --- | ---: |",
                *subsystem_lines,
                "",
                "Telemetry is observational. Subsystem contributions are predictive, not causal.",
                "",
                f"Recommended experiments: {len(bundle['recommended_experiments'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scientific_path.write_text(
        "\n".join(
            [
                f"# {bundle['project']['name']} Scientific Report",
                "",
                "## Method",
                "ThirdEye distinguishes observational evidence from controlled protocols.",
                "Only controlled protocols may produce causal claims.",
                "",
                "## Evidence",
                "```json",
                json.dumps(bundle["evidence"], indent=2, sort_keys=True),
                "```",
                "",
                "## Runs and Metrics",
                "```json",
                json.dumps(
                    {
                        "runs": bundle["runs"],
                        "run_results": bundle["run_results"],
                        "metric_summary": bundle["metric_summary"],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
                "## Intelligence Telemetry",
                "Training dynamics and internal signals are observational. The overall estimate "
                "is emitted only after fitting against held-out capability targets.",
                "```json",
                json.dumps(bundle["intelligence"], indent=2, sort_keys=True),
                "```",
                "",
                "## Recommended Experiments",
                "```json",
                json.dumps(bundle["recommended_experiments"], indent=2, sort_keys=True),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    html_path.write_text(_dashboard_html(bundle), encoding="utf-8")
    return {
        "scorecard": str(scorecard_path),
        "scientific_report": str(scientific_path),
        "evidence_bundle": str(bundle_path),
        "dashboard": str(html_path),
    }


def _summary(
    *,
    features: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
    metric_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    grades: dict[str, int] = {}
    for row in evidence:
        grade = str(row.get("grade", "unknown"))
        grades[grade] = grades.get(grade, 0) + 1
    completed = sum(row.get("status") == "completed" for row in run_results)
    failed = sum(row.get("status") not in {"completed"} for row in run_results)
    return {
        "feature_count": len(features),
        "evidence_count": len(evidence),
        "evidence_grades": grades,
        "run_count": len(run_results),
        "completed_runs": completed,
        "failed_runs": failed,
        "metric_count": len(metric_summary),
    }


def _metric_summary(
    run_results: list[dict[str, Any]],
    metrics: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered_run_ids = [row["run_id"] for row in run_results]
    history: dict[str, list[dict[str, Any]]] = {}
    for run_id in ordered_run_ids:
        for metric in metrics.get(run_id, []):
            history.setdefault(metric["metric_id"], []).append(
                {**metric, "run_id": run_id}
            )
    summary: list[dict[str, Any]] = []
    for metric_id, rows in sorted(history.items()):
        latest = rows[-1]
        previous = rows[-2] if len(rows) > 1 else None
        summary.append(
            {
                "metric_id": metric_id,
                "value": float(latest["value"]),
                "unit": str(latest.get("unit", "")),
                "direction": str(latest.get("direction", "")),
                "scorer": str(latest.get("scorer", "")),
                "scorer_version": str(latest.get("scorer_version", "")),
                "sample_count": int(latest.get("sample_count", 0)),
                "run_id": latest["run_id"],
                "delta": (
                    float(latest["value"]) - float(previous["value"])
                    if previous is not None
                    else None
                ),
            }
        )
    return summary


def _format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.6g}"


def _subsystem_score(estimate: dict[str, Any] | None, subsystem_id: str) -> str:
    if not estimate:
        return "uncalibrated"
    value = estimate.get("subsystem_scores", {}).get(subsystem_id)
    return "uncalibrated" if value is None else f"{float(value):+.6g}"


def _dashboard_html(bundle: dict[str, Any]) -> str:
    data = json.dumps(bundle, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThirdEye - {bundle['project']['name']}</title>
<style>
:root {{ color-scheme: dark; --bg:#071018; --panel:#0e1b26; --line:#203747;
--text:#edf7ff; --muted:#8ba5b7; --cyan:#4de3ff; --green:#62e6a7; --amber:#ffc66d; }}
* {{ box-sizing:border-box }} body {{ margin:0; font:14px Inter,Segoe UI,sans-serif;
background:radial-gradient(circle at 15% 0,#123047 0,transparent 35%),var(--bg); color:var(--text) }}
main {{ max-width:1400px; margin:auto; padding:32px }} header {{ display:flex; justify-content:space-between;
gap:24px; align-items:end; margin-bottom:24px }} h1 {{ margin:0; font-size:36px }} h2 {{ margin:0 0 14px }}
.eyebrow {{ color:var(--cyan); letter-spacing:.18em; text-transform:uppercase; font-weight:700 }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px }} .card {{ background:rgba(14,27,38,.88);
border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 16px 50px #0005 }}
.value {{ font:700 28px ui-monospace,monospace; margin-top:8px }} .muted {{ color:var(--muted) }}
section {{ margin-top:18px }} table {{ width:100%; border-collapse:collapse }} th,td {{ text-align:left;
padding:11px; border-bottom:1px solid var(--line) }} th {{ color:var(--muted); font-size:11px;
text-transform:uppercase; letter-spacing:.1em }} .pill {{ display:inline-block; padding:4px 9px;
border-radius:999px; border:1px solid var(--line); color:var(--cyan) }} .ok {{ color:var(--green) }}
.warn {{ color:var(--amber) }} @media(max-width:850px) {{ .grid {{ grid-template-columns:1fr 1fr }}
header {{ align-items:start; flex-direction:column }} }} </style>
</head>
<body><main>
<header><div><div class="eyebrow">ThirdEye Evidence Control Plane</div>
<h1>{bundle['project']['name']}</h1><div class="muted">{bundle['project'].get('description','')}</div></div>
<div class="pill">Schema v{bundle.get('schema_version',1)}</div></header>
<div id="app"></div>
<script>const d={data};
const s=d.summary; const evidence=Object.fromEntries(d.evidence.map(x=>[x.feature_id,x]));
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
document.querySelector('#app').innerHTML=`
<div class="grid">
<div class="card"><div class="muted">Registered Features</div><div class="value">${{s.feature_count}}</div></div>
<div class="card"><div class="muted">Evidence Records</div><div class="value">${{s.evidence_count}}</div></div>
<div class="card"><div class="muted">Completed Runs</div><div class="value ok">${{s.completed_runs}}</div></div>
<div class="card"><div class="muted">Tracked Metrics</div><div class="value">${{s.metric_count}}</div></div>
</div>
<section class="card"><h2>Model Intelligence Telemetry</h2>${{(()=>{{const x=d.intelligence;
const e=x.estimates.at(-1);return `<div class="grid"><div><div class="muted">Subsystems</div>
<div class="value">${{x.subsystems.length}}</div></div><div><div class="muted">Signals</div>
<div class="value">${{x.signals.length}}</div></div><div><div class="muted">Capability Estimate</div>
<div class="value">${{e?.overall==null?'Uncalibrated':Number(e.overall).toPrecision(5)}}</div></div>
<div><div class="muted">Confidence</div><div class="value">${{e?Number(e.confidence).toFixed(3):'0.000'}}</div></div></div>
<p class="muted">Internal telemetry is observational. Calibrated contributions are predictive, not causal.</p>`}})()}}</section>
<section class="card"><h2>Latest Metrics</h2><table><thead><tr><th>Metric</th><th>Value</th>
<th>Direction</th><th>Delta</th><th>Scorer</th></tr></thead><tbody>${{d.metric_summary.map(m=>`<tr>
<td><strong>${{esc(m.metric_id)}}</strong></td><td>${{Number(m.value).toPrecision(6)}} ${{esc(m.unit)}}</td>
<td>${{esc(m.direction)}}</td><td>${{m.delta===null?'n/a':Number(m.delta).toPrecision(4)}}</td>
<td>${{esc(m.scorer)}}@${{esc(m.scorer_version)}}</td></tr>`).join('')}}</tbody></table></section>
<section class="card"><h2>Feature Evidence</h2><table><thead><tr><th>Feature</th><th>Category</th>
<th>Grade</th><th>Causal</th><th>Missing</th></tr></thead><tbody>${{d.features.map(f=>{{const e=evidence[f.feature_id];
return `<tr><td><strong>${{esc(f.name)}}</strong><div class="muted">${{esc(f.feature_id)}}</div></td>
<td>${{esc(f.category)}}</td><td><span class="pill">${{esc(e?.grade||'untested')}}</span></td>
<td>${{e?.causal?'yes':'no'}}</td><td>${{esc(e?.missing_requirements?.join(', ')||'controlled evidence')}}</td></tr>`}}).join('')}}
</tbody></table></section>
<section class="card"><h2>Recent Runs</h2><table><thead><tr><th>Command</th><th>Phase</th><th>Status</th>
<th>Duration</th></tr></thead><tbody>${{d.run_results.slice(-20).reverse().map(r=>`<tr><td>${{esc(r.command_id)}}</td>
<td>${{esc(r.phase)}}</td><td class="${{r.status==='completed'?'ok':'warn'}}">${{esc(r.status)}}</td>
<td>${{Number(r.duration_seconds||0).toFixed(2)}}s</td></tr>`).join('')}}</tbody></table></section>`;
</script></main></body></html>"""
