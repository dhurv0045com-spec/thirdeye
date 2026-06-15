from __future__ import annotations

import argparse
import json
from pathlib import Path

from thirdeye.models import FeatureCategory, FeatureSpec, FeatureVariant, ProjectSpec
from thirdeye.sdk import ThirdEye


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="thirdeye")
    parser.add_argument("--home", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("--project", required=True)
    init.add_argument("--name", default=None)
    init.add_argument("--description", default="")

    register = commands.add_parser("register-feature")
    register.add_argument("--project", required=True)
    register.add_argument("--spec", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--project", required=True)
    evaluate.add_argument(
        "--profile", choices=["quick", "standard", "exhaustive", "auto"], default="auto"
    )

    commands.add_parser("serve").add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    eye = ThirdEye(args.home)
    if args.command == "init":
        project = ProjectSpec(
            project_id=args.project,
            name=args.name or args.project,
            description=args.description,
        )
        eye.register_project(project)
        print(json.dumps(project.to_dict(), indent=2))
    elif args.command == "register-feature":
        payload = _read_json(args.spec)
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
        feature = FeatureSpec(**payload)
        eye.register_feature(args.project, feature)
        print(json.dumps(feature.to_dict(), indent=2))
    elif args.command == "evaluate":
        result = eye.evaluate(args.project, args.profile)
        print(json.dumps(result["report_paths"], indent=2))
    elif args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("Install thirdeye-evidence[server] to run the API.") from exc
        uvicorn.run("thirdeye.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

