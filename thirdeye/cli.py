from __future__ import annotations

import argparse
import json
from pathlib import Path

from thirdeye.discovery import discover_project
from thirdeye.manifest import load_manifest, save_manifest
from thirdeye.models import FeatureCategory, FeatureSpec, FeatureVariant, ProjectSpec
from thirdeye.models import LifecyclePhase
from thirdeye.runner import ProjectRunner
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

    onboard = commands.add_parser("onboard")
    onboard.add_argument("root", nargs="?", default=".")
    onboard.add_argument("--project", default=None)
    onboard.add_argument("--output", default=None)

    register_project = commands.add_parser("register-project")
    register_project.add_argument("--manifest", default="thirdeye.json")

    register = commands.add_parser("register-feature")
    register.add_argument("--project", required=True)
    register.add_argument("--spec", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--project", required=True)
    evaluate.add_argument(
        "--profile", choices=["quick", "standard", "exhaustive", "auto"], default="auto"
    )

    assess = commands.add_parser("assess")
    assess.add_argument("--manifest", default="thirdeye.json")
    assess.add_argument(
        "--profile", choices=["quick", "standard", "exhaustive", "auto"], default="auto"
    )

    run = commands.add_parser("run")
    run.add_argument("--manifest", default="thirdeye.json")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--command", default=None)
    selection.add_argument(
        "--phase",
        choices=[phase.value for phase in LifecyclePhase],
        default=None,
    )

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--project", required=True)

    intelligence = commands.add_parser("intelligence")
    intelligence.add_argument("--project", required=True)

    calibrate = commands.add_parser("calibrate-intelligence")
    calibrate.add_argument("--project", required=True)
    calibrate.add_argument("--target", default=None)
    calibrate.add_argument("--minimum-checkpoints", type=int, default=5)

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
    elif args.command == "onboard":
        manifest = discover_project(args.root, args.project)
        output = args.output or str(Path(args.root) / "thirdeye.json")
        path = save_manifest(manifest, output)
        print(
            json.dumps(
                {
                    "manifest": str(path.resolve()),
                    "project": manifest.project.to_dict(),
                    "discovered_commands": [
                        command.to_dict() for command in manifest.commands
                    ],
                    "next": (
                        f"thirdeye register-project --manifest {path} && "
                        f"thirdeye run --manifest {path}"
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "register-project":
        manifest = load_manifest(args.manifest)
        eye.register_manifest(manifest)
        print(json.dumps(manifest.to_dict(), indent=2))
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
    elif args.command == "assess":
        manifest = load_manifest(args.manifest)
        result = eye.assess(manifest, args.profile)
        print(
            json.dumps(
                {
                    "run_results": result["run_results"],
                    "reports": result["evaluation"]["report_paths"],
                    "recommended_experiments": len(
                        result["evaluation"]["recommended_experiments"]
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "run":
        manifest = load_manifest(args.manifest)
        eye.register_manifest(manifest)
        results = ProjectRunner(eye, manifest).run(
            command_id=args.command,
            phase=LifecyclePhase(args.phase) if args.phase else None,
        )
        print(json.dumps(results, indent=2))
    elif args.command == "inspect":
        print(json.dumps(eye.project_snapshot(args.project), indent=2))
    elif args.command == "intelligence":
        print(json.dumps(eye.intelligence_snapshot(args.project), indent=2))
    elif args.command == "calibrate-intelligence":
        estimate = eye.calibrate_intelligence(
            args.project,
            target_id=args.target,
            minimum_checkpoints=args.minimum_checkpoints,
        )
        print(json.dumps(estimate.to_dict(), indent=2))
    elif args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("Install thirdeye-evidence[server] to run the API.") from exc
        uvicorn.run("thirdeye.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
