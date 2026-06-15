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
) -> dict[str, Any]:
    evidence_by_feature = {row["feature_id"]: row for row in evidence}
    return {
        "schema_version": 1,
        "project": project,
        "features": [
            {**feature, "latest_evidence": evidence_by_feature.get(feature["feature_id"])}
            for feature in features
        ],
        "evidence": evidence,
        "recommended_experiments": planned,
    }


def write_reports(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle_path = target / "evidence-bundle.json"
    scorecard_path = target / "decision-scorecard.md"
    scientific_path = target / "scientific-report.md"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    feature_lines = []
    for feature in bundle["features"]:
        evidence = feature.get("latest_evidence")
        grade = evidence.get("grade") if evidence else "untested"
        missing = ", ".join((evidence or {}).get("missing_requirements", [])) or "none"
        feature_lines.append(f"| {feature['name']} | {grade} | {missing} |")
    scorecard_path.write_text(
        "\n".join(
            [
                f"# {bundle['project']['name']} Decision Scorecard",
                "",
                "| Feature | Evidence grade | Missing evidence |",
                "| --- | --- | --- |",
                *feature_lines,
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
                "## Recommended Experiments",
                "```json",
                json.dumps(bundle["recommended_experiments"], indent=2, sort_keys=True),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "scorecard": str(scorecard_path),
        "scientific_report": str(scientific_path),
        "evidence_bundle": str(bundle_path),
    }

