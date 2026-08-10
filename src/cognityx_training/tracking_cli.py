"""Backfill a completed Cognityx Training publication into a configured tracker."""

from __future__ import annotations

import argparse
import json

from cognityx_training.storage_runtime import resolve_storage_runtime
from cognityx_training.tracking import create_tracker, payload_from_publication, track_with_policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cognityx-track-publication")
    parser.add_argument("publication_manifest_uri")
    parser.add_argument("--backend", choices=("none", "mlflow"), default="mlflow")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--failure-policy", choices=("warn", "error"), default="warn")
    parser.add_argument("--storage-config")
    parser.add_argument("--storage-root")
    args = parser.parse_args(argv)
    runtime = resolve_storage_runtime(storage_config=args.storage_config, storage_root=args.storage_root)
    payload = payload_from_publication(runtime, args.publication_manifest_uri)
    tracker = create_tracker(
        backend=args.backend,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        parent_run_id=args.parent_run_id,
    )
    result = track_with_policy(tracker, payload, failure_policy=args.failure_policy)
    print(json.dumps({
        "status": result.status,
        "backend": result.backend,
        "external_run_id": result.external_run_id,
    }, indent=2, sort_keys=True))
