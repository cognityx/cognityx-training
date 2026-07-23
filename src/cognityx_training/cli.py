"""Command-line entry point for configuration-driven training."""

import argparse
from dataclasses import replace
import json
import os
import tomllib
from pathlib import Path

from cognityx_core import Dataset, TrainingRequest

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.factory import create_training_backend
from cognityx_training.reporting import jsonable_configuration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training configuration path."""
    parser = argparse.ArgumentParser(description="Run a Cognityx training backend.")
    parser.add_argument(
        "--config", type=Path, required=True, help="TOML training configuration."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the configured output root."
    )
    parser.add_argument(
        "--run-id", help="Stable run directory name; defaults to a UTC timestamp."
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and dataset without loading a model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load TOML configuration and run the selected backend."""
    args = parse_args(argv)
    with args.config.open("rb") as source:
        values = tomllib.load(source)
    training = values.get("training", {})
    dataset_values = values.get("dataset", {})
    config = CustomPyTorchTrainingConfig.from_mapping(training)
    if args.output_dir is not None or args.run_id is not None:
        config = replace(
            config,
            output_dir=args.output_dir or config.output_dir,
            run_id=args.run_id or config.run_id,
        )
        config.validate()
    dataset = Dataset(
        name=dataset_values["name"],
        version=str(dataset_values.get("version", "1")),
        uri=dataset_values["uri"],
    )
    if args.print_config:
        print(json.dumps(jsonable_configuration(config), indent=2, sort_keys=True))
    if args.dry_run:
        from cognityx_training.dataset_pipeline import (
            load_jsonl_records,
            partition_records,
        )

        records = load_jsonl_records(dataset.uri)
        training_records, evaluation_records = partition_records(
            records, config.max_examples
        )
        effective_batch_size = (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        )
        print(
            "Configuration valid. "
            f"Dataset records: {len(records)} total, "
            f"{len(training_records)} training, {len(evaluation_records)} evaluation. "
            f"Micro batch: {config.per_device_train_batch_size}; "
            f"effective batch: {effective_batch_size}; "
            f"optimizer steps: {config.max_steps}."
        )
        return
    result = create_training_backend(config).train(TrainingRequest(dataset=dataset))
    report_uri = getattr(result, "report_uri", None) or result.metrics.get("report_uri")
    if os.environ.get("COGNITYX_AUTOTUNE_WORKER") != "1":
        print(f"Artifact: {result.artifact.uri}")
        print(f"Metrics: {dict(result.metrics)}")
        print(f"Report: {report_uri}")


if __name__ == "__main__":
    main()
