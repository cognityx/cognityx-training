"""Command-line entry point for configuration-driven training."""

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cognityx_core import Dataset, TrainingRequest

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.factory import create_training_backend
from cognityx_training.human import render_human
from cognityx_training.reporting import jsonable_configuration
from cognityx_training.runtime_check import check_training_runtime

CLI_RESULT_SCHEMA = "cognityx.training.cli-result/v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training configuration path."""
    arguments = list(argv) if argv is not None else list(sys.argv[1:])
    if arguments and arguments[0] == "config":
        parser = argparse.ArgumentParser(
            description="Inspect a Training run specification."
        )
        commands = parser.add_subparsers(dest="command", required=True)
        config = commands.add_parser("config")
        actions = config.add_subparsers(dest="config_action", required=True)
        for name in ("show", "validate"):
            selected = actions.add_parser(name)
            _add_inspection_arguments(selected)
        args = parser.parse_args(arguments)
        args.print_config = False
        args.dry_run = False
        args.check_runtime = False
        args.output_format = "json"
        return args
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
        "--dataset-uri",
        help="Override the configured authoritative DataForge package URI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the configured training seed for an orchestrated run.",
    )
    parser.add_argument(
        "--parent-run-id",
        help="Attach component tracking to a parent observation run.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration as JSON.",
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and dataset without loading a model.",
    )
    execution_mode.add_argument(
        "--check-runtime",
        action="store_true",
        help="Verify Training execution dependencies and CUDA without loading a model.",
    )
    presentation = parser.add_mutually_exclusive_group()
    presentation.add_argument(
        "--output-format",
        choices=("human", "json"),
        default="human",
        help="Choose interactive human output or one machine-readable JSON result.",
    )
    presentation.add_argument(
        "--human",
        dest="output_format",
        action="store_const",
        const="human",
        help="Explicit alias for --output-format human.",
    )
    args = parser.parse_args(arguments)
    args.command = "run"
    if args.output_format == "json" and args.print_config:
        parser.error("--print-config cannot be combined with --output-format json")
    return args


def _add_inspection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--storage-config", type=Path)
    parser.add_argument("--storage-root")
    parser.add_argument(
        "--dataset-input-mode",
        choices=["auto", "dataforge_manifest", "legacy_jsonl"],
    )
    parser.add_argument("--dataset-uri")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--human", action="store_true")


def resolve_training_config(
    values: Mapping[str, Any], args: argparse.Namespace
) -> CustomPyTorchTrainingConfig:
    """Compose TOML values and CLI overrides before validating once."""
    training = dict(values.get("training") or {})
    experiment_values = dict(values.get("experiment") or {})
    publication_values = dict(values.get("publication") or {})
    tracking_values = dict(values.get("tracking") or {})
    effective: dict[str, object] = {
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
    if args.output_dir is not None:
        effective["output_dir"] = args.output_dir
    if args.run_id is not None:
        effective["run_id"] = None
        effective["training_run_id"] = args.run_id
    if args.experiment_id is not None:
        effective["experiment_id"] = args.experiment_id
    if args.seed is not None:
        effective["seed"] = args.seed
    if args.parent_run_id is not None:
        effective["tracking_parent_run_id"] = args.parent_run_id
    if args.storage_config is not None:
        effective["storage_config"] = args.storage_config
    if args.storage_root is not None:
        effective["storage_root"] = args.storage_root
    if args.dataset_input_mode is not None:
        effective["dataset_input_mode"] = args.dataset_input_mode
    return CustomPyTorchTrainingConfig.from_mapping(effective)


@dataclass(frozen=True, slots=True)
class TrainingConfigResolution:
    configuration: CustomPyTorchTrainingConfig
    values: Mapping[str, Any]
    path: Path
    file_sha256: str
    field_sources: Mapping[str, str]
    overrides: tuple[Mapping[str, Any], ...]
    dataset_uri: str | None

    def to_dict(self) -> dict[str, Any]:
        effective = _secret_safe(jsonable_configuration(self.configuration))
        effective["dataset_uri"] = self.dataset_uri
        return {
            "component": "training",
            "configuration_kind": "scientific-workload",
            "valid": True,
            "master_config": {
                "kind": "file",
                "path": str(self.path),
                "selected_by": "explicit",
                "sha256": self.file_sha256,
            },
            "config_layers": [
                {
                    "path": str(self.path),
                    "selected_by": "explicit",
                    "sha256": self.file_sha256,
                    "changed_keys": sorted(
                        key
                        for key, source in self.field_sources.items()
                        if source == str(self.path)
                    ),
                }
            ],
            "field_sources": dict(sorted(self.field_sources.items())),
            "overrides": [dict(item) for item in self.overrides],
            "effective": effective,
            "warnings": [],
            "errors": [],
        }


def resolve_training_configuration(
    path: str | Path, args: argparse.Namespace
) -> TrainingConfigResolution:
    selected = Path(path).expanduser().resolve()
    raw = selected.read_bytes()
    values = tomllib.loads(raw.decode("utf-8"))
    baseline_args = argparse.Namespace(**vars(args))
    override_flags = {
        "output_dir": "--output-dir",
        "run_id": "--run-id",
        "experiment_id": "--experiment-id",
        "storage_config": "--storage-config",
        "storage_root": "--storage-root",
        "dataset_input_mode": "--dataset-input-mode",
        "seed": "--seed",
        "parent_run_id": "--parent-run-id",
    }
    for name in override_flags:
        setattr(baseline_args, name, None)
    baseline = resolve_training_config(values, baseline_args)
    configuration = resolve_training_config(values, args)
    baseline_values = jsonable_configuration(baseline)
    effective_values = jsonable_configuration(configuration)
    overrides: list[Mapping[str, Any]] = []
    overridden_fields: dict[str, str] = {}
    for name, flag in override_flags.items():
        supplied = getattr(args, name, None)
        if supplied is None:
            continue
        target = (
            "tracking_parent_run_id"
            if name == "parent_run_id"
            else ("training_run_id" if name == "run_id" else name)
        )
        previous = baseline_values.get(target)
        effective = effective_values.get(target)
        if previous != effective:
            overrides.append(
                {
                    "key": target,
                    "source": flag,
                    "previous": _secret_safe(previous, target),
                    "effective": _secret_safe(effective, target),
                    "changed": True,
                }
            )
            overridden_fields[target] = flag
    dataset_values = dict(values.get("dataset") or {})
    baseline_dataset_uri = dataset_values.get("uri")
    dataset_uri = getattr(args, "dataset_uri", None) or baseline_dataset_uri
    if (
        getattr(args, "dataset_uri", None) is not None
        and dataset_uri != baseline_dataset_uri
    ):
        overrides.append(
            {
                "key": "dataset_uri",
                "source": "--dataset-uri",
                "previous": _secret_safe(baseline_dataset_uri, "dataset_uri"),
                "effective": _secret_safe(dataset_uri, "dataset_uri"),
                "changed": True,
            }
        )
        overridden_fields["dataset_uri"] = "--dataset-uri"
    file_fields = set(dict(values.get("training") or {}))
    file_fields.update(
        {
            {
                "id": "experiment_id",
                "name": "experiment_name",
                "description": "experiment_description",
                "created_by": "experiment_created_by",
                "tags": "experiment_tags",
            }[name]
            for name in dict(values.get("experiment") or {})
            if name in {"id", "name", "description", "created_by", "tags"}
        }
    )
    file_fields.update(
        {
            {
                "mode": "publication_mode",
                "retain_local_staging": "retain_local_staging",
            }[name]
            for name in dict(values.get("publication") or {})
            if name in {"mode", "retain_local_staging"}
        }
    )
    file_fields.update(
        {
            {
                "backend": "tracking_backend",
                "uri": "tracking_uri",
                "experiment_name": "tracking_experiment_name",
                "run_name": "tracking_run_name",
                "parent_run_id": "tracking_parent_run_id",
                "failure_policy": "tracking_failure_policy",
            }[name]
            for name in dict(values.get("tracking") or {})
            if name
            in {
                "backend",
                "uri",
                "experiment_name",
                "run_name",
                "parent_run_id",
                "failure_policy",
            }
        }
    )
    field_sources = {
        name: overridden_fields.get(
            name, str(selected) if name in file_fields else "built-in"
        )
        for name in effective_values
    }
    field_sources["dataset_uri"] = overridden_fields.get(
        "dataset_uri", str(selected) if "uri" in dataset_values else "built-in"
    )
    return TrainingConfigResolution(
        configuration=configuration,
        values=values,
        path=selected,
        file_sha256=sha256(raw).hexdigest(),
        field_sources=field_sources,
        overrides=tuple(overrides),
        dataset_uri=str(dataset_uri) if dataset_uri is not None else None,
    )


def _secret_safe(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in ("secret", "password", "token", "api_key")):
        return "<redacted>" if value is not None else None
    if isinstance(value, dict):
        return {
            str(name): _secret_safe(item, str(name)) for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_secret_safe(item, key) for item in value]
    if isinstance(value, str) and "://" in value:
        return _redacted_uri(value)
    return value


def _redacted_uri(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = (
        host
        if parsed.username is not None or parsed.password is not None
        else parsed.netloc
    )
    query = urlencode(
        [
            (
                name,
                "<redacted>"
                if any(
                    marker in name.lower()
                    for marker in ("secret", "password", "token", "api_key")
                )
                else item,
            )
            for name, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def main(argv: list[str] | None = None) -> None:
    """Load TOML configuration and run the selected backend."""
    args = parse_args(argv)
    try:
        resolution = resolve_training_configuration(args.config, args)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        if args.command == "config":
            report = {
                "component": "training",
                "configuration_kind": "scientific-workload",
                "valid": False,
                "master_config": {
                    "kind": "file",
                    "path": str(args.config.expanduser().resolve()),
                    "selected_by": "explicit",
                    "sha256": None,
                },
                "config_layers": [],
                "field_sources": {},
                "overrides": [],
                "effective": {},
                "warnings": [],
                "errors": [{"code": "configuration_invalid", "message": str(exc)}],
            }
            _write(report, human=args.human)
            raise SystemExit(2) from None
        raise
    if args.command == "config":
        _write(resolution.to_dict(), human=args.human)
        return
    values = resolution.values
    dataset_values = values.get("dataset", {})
    config = resolution.configuration
    if args.check_runtime:
        runtime_result = check_training_runtime(require_cuda=config.load_in_4bit)
        if args.output_format == "json":
            print(json.dumps(runtime_result, indent=2, sort_keys=True))
        else:
            outcome = "ready" if runtime_result["passed"] else "not ready"
            print(
                f"Training runtime is {outcome}: "
                f"{json.dumps(runtime_result, sort_keys=True)}"
            )
        if not runtime_result["passed"]:
            raise SystemExit(1)
        return
    dataset = Dataset(
        name=dataset_values["name"],
        version=str(dataset_values.get("version", "1")),
        uri=args.dataset_uri or dataset_values["uri"],
    )
    if args.print_config:
        print(json.dumps(jsonable_configuration(config), indent=2, sort_keys=True))
    if args.dry_run:
        from cognityx_training.dataset_pipeline import (
            DataForgeDatasetReader,
            preflight_dataset,
        )

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
        if args.output_format == "json":
            print(
                json.dumps(
                    {
                        "schema": CLI_RESULT_SCHEMA,
                        "mode": "dry_run",
                        "experiment_id": config.experiment_id,
                        "training_run_id": config.training_run_id or config.run_id,
                        "total_records": preflight.statistics.total_records,
                        "accepted_training_examples": (
                            preflight.statistics.accepted_training_examples
                        ),
                        "evaluation_records": preflight.statistics.evaluation_records,
                        "micro_batch_size": config.per_device_train_batch_size,
                        "effective_batch_size": effective_batch_size,
                        "optimizer_steps": config.max_steps,
                        "dataset": {
                            "dataset_id": preflight.lineage.dataset_id,
                            "dataset_version": preflight.lineage.dataset_version,
                            "dataset_manifest_checksum": (
                                preflight.lineage.dataset_manifest_checksum
                            ),
                            "records_checksum": preflight.lineage.records_checksum,
                            "research_package_id": (
                                preflight.lineage.research_package_id
                            ),
                            "research_package_version": (
                                preflight.lineage.research_package_version
                            ),
                            "research_package_manifest_checksum": (
                                preflight.lineage.research_package_manifest_checksum
                            ),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
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
            "schema": CLI_RESULT_SCHEMA,
            "mode": "completed",
            "experiment_id": result.metrics.get("experiment_id"),
            "training_variant_id": result.metrics.get("training_variant_id"),
            "training_run_id": result.metrics.get("training_run_id"),
            "adapter_id": result.metrics.get("adapter_id"),
            "adapter_manifest_uri": result.metrics.get("adapter_manifest_uri"),
            "training_report_uri": report_uri,
            "publication_manifest_uri": result.metrics.get("publication_manifest_uri"),
            "artifact_uri": result.artifact.uri,
        }
        print(json.dumps(structured, indent=2, sort_keys=True))


def _write(value: Any, *, human: bool) -> None:
    if human:
        print(render_human(value))
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
