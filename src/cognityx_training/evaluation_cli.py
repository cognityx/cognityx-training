"""Command-line entry point for saved-output candidate evaluation."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from cognityx_training.evaluation_configuration import EvaluationConfig
from cognityx_training.evaluation_pipeline import EvaluationPipeline, show_evaluation
from cognityx_training.human import render_human
from cognityx_training.storage_runtime import resolve_storage_runtime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Cognityx Training candidate outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        selected = subparsers.add_parser(command)
        selected.add_argument(
            "--config",
            type=Path,
            required=True,
            help="TOML evaluation configuration.",
        )
        selected.add_argument("--human", action="store_true")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--evaluation-request", required=True)
    show = subparsers.add_parser("show")
    show.add_argument("--evaluation-manifest", required=True)
    for selected in (resume, show):
        selected.add_argument("--storage-config", type=Path)
        selected.add_argument("--storage-root")
        selected.add_argument("--human", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> EvaluationConfig:
    with path.open("rb") as source:
        return EvaluationConfig.from_mapping(tomllib.load(source))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command in {"plan", "run"}:
        pipeline = EvaluationPipeline(load_config(args.config))
        result = pipeline.plan() if args.command == "plan" else pipeline.run()
    else:
        runtime = resolve_storage_runtime(
            storage_config=args.storage_config,
            storage_root=args.storage_root,
        )
        if args.command == "resume":
            result = EvaluationPipeline.from_request(
                args.evaluation_request,
                storage_runtime=runtime,
            ).run(resume=True)
        else:
            result = show_evaluation(
                args.evaluation_manifest,
                storage_runtime=runtime,
            )
    if args.human:
        print(render_human(result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
