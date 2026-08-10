"""Command-line entry point for configuration-driven training."""

import argparse
from dataclasses import asdict, replace
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
        "--experiment-id",
        help="Experiment identity; defaults to a generated exp-* identifier.",
    )
    parser.add_argument(
        "--storage-config",
        type=Path,
        help="Optional StorageRuntime TOML config for DataForge dataset manifests.",
    )
    parser.add_argument(
        "--storage-root",
        help="Optional local storage root for temporary or built-in StorageRuntime config.",
    )
    parser.add_argument(
        "--dataset-input-mode",
        choices=["auto", "dataforge_manifest", "legacy_jsonl"],
        help="Override dataset interpretation mode.",
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
    experiment_values = values.get("experiment", {})
    publication_values = values.get("publication", {})
    dataset_values = values.get("dataset", {})
    tracking_values = values.get("tracking", {})
    config = CustomPyTorchTrainingConfig.from_mapping(
        {
            **training,
            "experiment_id": experiment_values.get("id"),
            "experiment_name": experiment_values.get("name"),
            "experiment_description": experiment_values.get("description"),
            "experiment_created_by": experiment_values.get("created_by"),
            "experiment_tags": experiment_values.get("tags", []),
            "publication_mode": publication_values.get("mode", "local"),
            "retain_local_staging": publication_values.get(
                "retain_local_staging",
                False,
            ),
            "tracking_backend": tracking_values.get("backend", "none"),
            "tracking_uri": tracking_values.get("uri"),
            "tracking_experiment_name": tracking_values.get("experiment_name"),
            "tracking_run_name": tracking_values.get("run_name"),
            "tracking_parent_run_id": tracking_values.get("parent_run_id"),
            "tracking_failure_policy": tracking_values.get("failure_policy", "warn"),
        }
    )
    if (
        args.output_dir is not None
        or args.run_id is not None
        or args.experiment_id is not None
    ):
        config = replace(
            config,
            output_dir=args.output_dir or config.output_dir,
            run_id=None if args.run_id is not None else config.run_id,
            training_run_id=args.run_id or config.training_run_id,
            experiment_id=args.experiment_id or config.experiment_id,
        )
        config.validate()
    if args.storage_config is not None or args.storage_root is not None or args.dataset_input_mode is not None:
        config = replace(
            config,
            storage_config=args.storage_config or config.storage_config,
            storage_root=args.storage_root or config.storage_root,
            dataset_input_mode=args.dataset_input_mode or config.dataset_input_mode,
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
        from cognityx_training.dataset_pipeline import DataForgeDatasetReader, preflight_dataset

        reader = DataForgeDatasetReader(
            dataset.uri,
            storage_config=config.storage_config,
            storage_root=config.storage_root,
            input_mode=config.dataset_input_mode,
        )
        tokenizer = None
        if config.model_name:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir=str(config.model_cache_dir),
                local_files_only=config.local_files_only,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        preflight = preflight_dataset(
            reader,
            tokenizer,
            max_examples=config.max_examples,
            max_sequence_length=config.max_sequence_length,
            overlength_policy=config.overlength_policy,
        )
        effective_batch_size = (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        )
        print(
            f"Dataset lineage: {json.dumps(asdict(preflight.lineage), sort_keys=True, default=str)}. "
            f"Dataset records: {preflight.statistics.total_records} total, "
            f"{preflight.statistics.accepted_training_examples} training, {preflight.statistics.evaluation_records} evaluation. "
            f"Micro batch: {config.per_device_train_batch_size}; "
            f"effective batch: {effective_batch_size}; "
            f"optimizer steps: {config.max_steps}."
        )
        return
    result = create_training_backend(config).train(TrainingRequest(dataset=dataset))
    report_uri = getattr(result, "report_uri", None) or result.metrics.get("report_uri")
    if os.environ.get("COGNITYX_AUTOTUNE_WORKER") != "1":
        structured = {
            "experiment_id": result.metrics.get("experiment_id"),
            "training_variant_id": result.metrics.get("training_variant_id"),
            "training_run_id": result.metrics.get("training_run_id"),
            "adapter_id": result.metrics.get("adapter_id"),
            "adapter_manifest_uri": result.metrics.get("adapter_manifest_uri"),
            "training_report_uri": report_uri,
            "publication_manifest_uri": result.metrics.get(
                "publication_manifest_uri"
            ),
            "artifact_uri": result.artifact.uri,
        }
        print(json.dumps(structured, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
