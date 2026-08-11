"""Immutable experiment metadata and atomic candidate-adapter publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

from cognityx_training.lineage import (
    TrainingLineageIds,
    stable_json,
    variant_identity_checksum,
)
from cognityx_training.reporting import utc_now
from cognityx_training.storage_uri import resolve_storage_uri

ADAPTER_SCHEMA = "cognityx.training.adapter/v1"
EXPERIMENT_SCHEMA = "cognityx.training.experiment/v1"
PUBLICATION_SCHEMA = "cognityx.training.publication/v1"
VARIANT_SCHEMA = "cognityx.training.variant/v1"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


@dataclass(frozen=True, slots=True)
class AdapterVerificationResult:
    """Provider-neutral result of verifying one published adapter bundle."""

    valid: bool
    adapter_id: str
    adapter_manifest_uri: str
    bundle_checksum: str
    verified_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """URIs produced after the terminal publication manifest is durable."""

    adapter_uri: str
    adapter_manifest_uri: str
    training_report_uri: str
    baseline_predictions_uri: str
    trained_predictions_uri: str
    publication_manifest_uri: str
    artifact_checksums: dict[str, str]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_stream(source: Any) -> tuple[str, int]:
    """Hash an open binary stream without requiring a native filesystem path."""
    hasher = hashlib.sha256()
    size_bytes = 0
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        hasher.update(chunk)
        size_bytes += len(chunk)
    return hasher.hexdigest(), size_bytes


def bundle_checksum(files: Iterable[Mapping[str, Any]]) -> str:
    """Hash sorted file identities, independent of staging location."""
    normalized = [
        {
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in files
    ]
    normalized.sort(key=lambda item: item["path"])
    return hashlib.sha256(stable_json(normalized).encode("utf-8")).hexdigest()


def inspect_adapter_files(staging_directory: Path) -> list[dict[str, Any]]:
    """Validate and hash a fixture-sized or real PEFT output directory."""
    root = staging_directory.resolve()
    if not root.is_dir():
        raise ValueError(f"Adapter staging directory does not exist: {root}")

    files: list[dict[str, Any]] = []
    for path in sorted(staging_directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Adapter staging contains a symlink: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Adapter file escapes staging directory: {path}")
        relative = path.relative_to(staging_directory).as_posix()
        if relative in {"checksums.json", "adapter-manifest.json"}:
            continue
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    names = {item["path"] for item in files}
    missing = sorted(set(REQUIRED_ADAPTER_FILES) - names)
    if missing:
        raise ValueError(
            "Adapter output is incomplete; missing required files: "
            + ", ".join(missing)
        )
    return files


def canonical_variant_identity(
    config: Any,
    dataset_lineage: Any,
    *,
    base_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only result-changing inputs for deterministic variant identity."""
    return {
        "schema_version": "cognityx.training.variant-identity/v1",
        "dataset": {
            "dataset_id": dataset_lineage.dataset_id,
            "dataset_version": dataset_lineage.dataset_version,
            "dataset_variant_id": dataset_lineage.dataset_variant_id,
            "manifest_checksum": dataset_lineage.dataset_manifest_checksum,
            "records_checksum": dataset_lineage.records_checksum,
            "recipe": dataset_lineage.recipe,
        },
        "base_model": dict(base_model_identity),
        "training": {
            "backend": config.backend,
            "adapter_type": "qlora" if config.load_in_4bit else "lora",
            "load_in_4bit": config.load_in_4bit,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": sorted(config.target_modules),
            "max_sequence_length": config.max_sequence_length,
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_examples": config.max_examples,
            "max_steps": config.max_steps,
            "seed": config.seed,
            "overlength_policy": config.overlength_policy,
            "data_order": config.data_order,
        },
    }


def research_package_lineage(dataset_lineage: Any) -> dict[str, Any] | None:
    """Return additive research lineage without making it training identity."""
    if dataset_lineage.research_package_manifest_uri is None:
        return None
    return {
        "research_package_id": dataset_lineage.research_package_id,
        "research_package_version": dataset_lineage.research_package_version,
        "manifest_uri": dataset_lineage.research_package_manifest_uri,
        "manifest_checksum": dataset_lineage.research_package_manifest_checksum,
        "evaluation_sets": [dict(item) for item in dataset_lineage.evaluation_sets],
    }


def runtime_environment(
    storage_runtime: Any,
    *,
    torch_module: Any | None = None,
    base_model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture reproducibility metadata without requiring CUDA."""
    packages: dict[str, str | None] = {}
    for name in (
        "cognityx-training",
        "cognityx-core",
        "cognityx-storage",
        "transformers",
        "peft",
        "torch",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_revision": discover_git_revision(),
        "storage": storage_runtime.describe(),
        "base_model": dict(base_model_identity or {}),
    }
    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        environment["torch"] = {
            "version": torch_module.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch_module.version.cuda,
            "gpu": (
                torch_module.cuda.get_device_name(torch_module.cuda.current_device())
                if cuda_available
                else None
            ),
        }
    return environment


def discover_git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _read_json(store: Any, key: str) -> dict[str, Any]:
    with store.open(key) as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Stored JSON object is not a mapping: {key}")
    return value


def verify_published_adapter(
    adapter_manifest_uri: str,
    *,
    storage_runtime: Any,
) -> AdapterVerificationResult:
    """Verify a stored PEFT adapter without loading a model or GPU."""
    resolution = resolve_storage_uri(storage_runtime, adapter_manifest_uri)
    if resolution.selected_role not in {"model", "shared"}:
        raise ValueError(
            "Adapter manifest URI must resolve through the model role or "
            f"legacy shared scope, got {resolution.selected_role}"
        )
    store = resolution.store
    manifest_key = resolution.key
    manifest = _read_json(store, manifest_key)
    if manifest.get("schema_version") != ADAPTER_SCHEMA:
        raise ValueError("Unsupported adapter manifest schema")
    bundle_root = manifest_key.rsplit("/", 1)[0]
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Adapter manifest files must be a list")
    names = {item.get("path") for item in files if isinstance(item, dict)}
    missing = sorted(set(REQUIRED_ADAPTER_FILES) - names)
    if missing:
        raise ValueError(
            "Stored adapter manifest is missing required files: " + ", ".join(missing)
        )
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Adapter manifest contains a malformed file entry")
        relative = str(item["path"])
        with store.open(f"{bundle_root}/{relative}") as source:
            calculated_checksum, calculated_size = sha256_stream(source)
        if calculated_size != int(item["size_bytes"]):
            raise ValueError(f"Stored adapter size mismatch: {relative}")
        if calculated_checksum != item["sha256"]:
            raise ValueError(f"Stored adapter checksum mismatch: {relative}")
    calculated_bundle = bundle_checksum(files)
    if calculated_bundle != manifest.get("bundle_checksum"):
        raise ValueError("Stored adapter bundle checksum mismatch")
    checksums = _read_json(store, f"{bundle_root}/checksums.json")
    if (
        checksums.get("files") != files
        or checksums.get("bundle_checksum") != calculated_bundle
    ):
        raise ValueError("Stored checksums manifest does not match adapter manifest")
    return AdapterVerificationResult(
        valid=True,
        adapter_id=str(manifest["adapter_id"]),
        adapter_manifest_uri=adapter_manifest_uri,
        bundle_checksum=calculated_bundle,
        verified_files=tuple(sorted(str(item["path"]) for item in files)),
    )


class TrainingPublisher:
    """Own immutable metadata and the terminal publication commit point."""

    def __init__(
        self,
        storage_runtime: Any,
        ids: TrainingLineageIds,
        *,
        experiment_name: str | None = None,
        experiment_description: str | None = None,
        experiment_created_by: str | None = None,
        experiment_tags: Iterable[str] = (),
    ) -> None:
        self.storage_runtime = storage_runtime
        self.ids = ids
        self.artifact_store = storage_runtime.for_role("artifact")
        self.model_store = storage_runtime.for_role("model")
        self.temporary_store = storage_runtime.for_role("temporary")
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.experiment_created_by = experiment_created_by
        self.experiment_tags = tuple(experiment_tags)

    @property
    def experiment_root(self) -> str:
        return f"experiments/{self.ids.experiment_id}"

    @property
    def run_root(self) -> str:
        return f"{self.experiment_root}/runs/{self.ids.training_run_id}"

    @property
    def adapter_root(self) -> str:
        return f"adapters/{self.ids.adapter_id}/1"

    def publish_experiment(self) -> str:
        key = f"{self.experiment_root}/experiment.json"
        existing = _read_json(self.artifact_store, key) if self.artifact_store.exists(key) else None
        created_at = existing.get("created_at") if existing else utc_now()
        payload = {
            "schema_version": EXPERIMENT_SCHEMA,
            "experiment_id": self.ids.experiment_id,
            "name": self.experiment_name,
            "description": self.experiment_description,
            "created_by": self.experiment_created_by,
            "tags": list(self.experiment_tags),
            "created_at": created_at,
        }
        return self.artifact_store.put_json_idempotent(key, payload).uri

    def publish_variant(
        self,
        canonical_identity: Mapping[str, Any],
        *,
        dataset_lineage: Any,
        base_model_identity: Mapping[str, Any],
    ) -> str:
        checksum = variant_identity_checksum(canonical_identity)
        payload = {
            "schema_version": VARIANT_SCHEMA,
            "experiment_id": self.ids.experiment_id,
            "training_variant_id": self.ids.training_variant_id,
            "canonical_identity": dict(canonical_identity),
            "dataset_lineage": asdict(dataset_lineage),
            "base_model": dict(base_model_identity),
            "semantic_training_configuration": canonical_identity["training"],
            "variant_identity_checksum": checksum,
        }
        key = f"{self.experiment_root}/variants/{self.ids.training_variant_id}.json"
        return self.artifact_store.put_json_idempotent(key, payload).uri

    def publish_training_request(
        self,
        *,
        dataset_lineage: Any,
        normalized_request: Mapping[str, Any],
        base_model_identity: Mapping[str, Any],
        publication_mode: str,
    ) -> str:
        payload = {
            "schema_version": "cognityx.training.request/v1",
            **self.ids.to_dict(),
            "dataset_manifest_uri": dataset_lineage.dataset_manifest_uri,
            "dataset_manifest_checksum": dataset_lineage.dataset_manifest_checksum,
            "training_variant_id": self.ids.training_variant_id,
            "base_model": dict(base_model_identity),
            "normalized_training_request": dict(normalized_request),
            "execution_context": {
                "pid": os.getpid(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "created_at": utc_now(),
            "package_version": _package_version(),
            "git_revision": discover_git_revision(),
            "publication_mode": publication_mode,
        }
        key = f"{self.run_root}/training-request.json"
        return self.artifact_store.put_json_idempotent(key, payload).uri

    def publish_failure(self, exc: BaseException, *, phase: str) -> str | None:
        payload = {
            "schema_version": "cognityx.training.failure/v1",
            **self.ids.to_dict(),
            "status": "failed",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": utc_now(),
        }
        try:
            return self.artifact_store.put_json_idempotent(
                f"{self.run_root}/failure.json",
                payload,
            ).uri
        except Exception:
            return None

    def publish_completed_run(
        self,
        *,
        staging_directory: Path,
        dataset_lineage: Any,
        base_model_identity: Mapping[str, Any],
        adapter_details: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        training_report: Mapping[str, Any],
        metrics: Mapping[str, Any],
        baseline_predictions: Iterable[Mapping[str, Any]],
        trained_predictions: Iterable[Mapping[str, Any]],
        retain_local_staging: bool,
    ) -> PublicationResult:
        files = inspect_adapter_files(staging_directory)
        calculated_bundle = bundle_checksum(files)
        checksums = {
            "schema_version": "cognityx.training.adapter-checksums/v1",
            "files": files,
            "bundle_checksum": calculated_bundle,
        }
        package_lineage = research_package_lineage(dataset_lineage)
        adapter_manifest = {
            "schema_version": ADAPTER_SCHEMA,
            **self.ids.to_dict(),
            "adapter_version": "1",
            "status": "candidate",
            "dataset": {
                "dataset_id": dataset_lineage.dataset_id,
                "dataset_version": dataset_lineage.dataset_version,
                "dataset_variant_id": dataset_lineage.dataset_variant_id,
                "manifest_uri": dataset_lineage.dataset_manifest_uri,
                "manifest_checksum": dataset_lineage.dataset_manifest_checksum,
                "records_checksum": dataset_lineage.records_checksum,
                "recipe": dataset_lineage.recipe,
            },
            "base_model": dict(base_model_identity),
            "adapter": dict(adapter_details),
            "files": files,
            "bundle_checksum": calculated_bundle,
            "created_at": utc_now(),
        }
        if package_lineage is not None:
            adapter_manifest["research_package"] = package_lineage
        _write_json(staging_directory / "checksums.json", checksums)
        _write_json(staging_directory / "adapter-manifest.json", adapter_manifest)

        adapter_object = self.model_store.put_directory(
            self.adapter_root,
            staging_directory,
        )
        adapter_manifest_uri = self.model_store.uri(
            f"{self.adapter_root}/adapter-manifest.json"
        )
        verification = verify_published_adapter(
            adapter_manifest_uri,
            storage_runtime=self.storage_runtime,
        )

        artifact_payloads = (
            ("resolved-config.json", dict(resolved_config)),
            ("dataset-lineage.json", asdict(dataset_lineage)),
            ("environment.json", dict(environment)),
            ("training-report.json", dict(training_report)),
            ("metrics.json", dict(metrics)),
        )
        artifact_uris: dict[str, str] = {}
        artifact_checksums: dict[str, str] = {}
        for filename, payload in artifact_payloads:
            stored = self.artifact_store.put_json_idempotent(
                f"{self.run_root}/{filename}",
                payload,
            )
            artifact_uris[filename] = stored.uri
            artifact_checksums[filename] = _json_checksum(payload)

        baseline_uri, baseline_checksum = self._publish_predictions(
            "baseline-predictions.jsonl",
            baseline_predictions,
        )
        trained_uri, trained_checksum = self._publish_predictions(
            "trained-predictions.jsonl",
            trained_predictions,
        )
        artifact_checksums.update(
            {
                "baseline-predictions.jsonl": baseline_checksum,
                "trained-predictions.jsonl": trained_checksum,
                "adapter_bundle": verification.bundle_checksum,
            }
        )

        terminal = {
            "schema_version": PUBLICATION_SCHEMA,
            "status": "completed",
            **self.ids.to_dict(),
            "adapter_uri": adapter_object.uri,
            "adapter_manifest_uri": adapter_manifest_uri,
            "training_report_uri": artifact_uris["training-report.json"],
            "dataset_lineage_uri": artifact_uris["dataset-lineage.json"],
            "dataset_lineage_checksum": artifact_checksums["dataset-lineage.json"],
            "baseline_predictions_uri": baseline_uri,
            "trained_predictions_uri": trained_uri,
            "artifact_checksums": artifact_checksums,
            "completed_at": utc_now(),
        }
        if package_lineage is not None:
            terminal["research_package"] = package_lineage
        terminal_object = self.artifact_store.put_json_idempotent(
            f"{self.run_root}/publication-manifest.json",
            terminal,
        )
        if not retain_local_staging:
            shutil.rmtree(staging_directory)
        return PublicationResult(
            adapter_uri=adapter_object.uri,
            adapter_manifest_uri=adapter_manifest_uri,
            training_report_uri=artifact_uris["training-report.json"],
            baseline_predictions_uri=baseline_uri,
            trained_predictions_uri=trained_uri,
            publication_manifest_uri=terminal_object.uri,
            artifact_checksums=dict(artifact_checksums),
        )

    def _publish_predictions(
        self,
        filename: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> tuple[str, str]:
        hasher = hashlib.sha256()
        temporary_root = getattr(self.temporary_store, "native_path", None)
        directory = None
        if callable(temporary_root):
            try:
                directory = temporary_root("training-publication")
                directory.mkdir(parents=True, exist_ok=True)
            except Exception:
                directory = None
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="predictions-",
            suffix=".jsonl",
            dir=directory,
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            try:
                for row in rows:
                    content = (
                        json.dumps(
                            dict(row),
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    target.write(content)
                    hasher.update(content)
                target.flush()
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            stored = self.artifact_store.put_file(
                f"{self.run_root}/{filename}",
                temporary_path,
                media_type="application/x-ndjson",
            )
        finally:
            temporary_path.unlink(missing_ok=True)
        return stored.uri, hasher.hexdigest()


def prediction_rows(
    evaluation: Mapping[str, Any],
    *,
    prediction_type: str,
    ids: TrainingLineageIds,
    base_model_identity: Mapping[str, Any],
    decoding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in evaluation.get("outputs", []):
        original_provenance = dict(output.get("provenance") or {})
        original_metadata = dict(output.get("metadata") or {})
        combined = {**original_provenance, **original_metadata, **dict(output)}
        singular_knowledge_unit = combined.get("knowledge_unit_id")
        knowledge_unit_ids = _normalized_string_list(
            combined.get("knowledge_unit_ids")
        )
        if singular_knowledge_unit is not None:
            normalized_singular = str(singular_knowledge_unit)
            if normalized_singular not in knowledge_unit_ids:
                knowledge_unit_ids.insert(0, normalized_singular)
        rows.append(
            {
                **ids.to_dict(),
                "adapter_id": ids.adapter_id if prediction_type == "trained" else None,
                "dataset_record_id": output.get("record_id"),
                "prompt": output.get("prompt"),
                "expected_answer": output.get("expected"),
                "generated_answer": output.get("generated"),
                "exact_match": output.get("exact_match"),
                "contains_expected": output.get("contains_expected"),
                "knowledge_unit_id": singular_knowledge_unit,
                "knowledge_unit_ids": knowledge_unit_ids,
                "evidence_ids": _normalized_string_list(
                    combined.get("evidence_ids")
                ),
                "source_asset_ids": _normalized_string_list(
                    combined.get("source_asset_ids")
                ),
                "document_ids": _normalized_string_list(
                    combined.get("document_ids")
                ),
                "probe_id": combined.get("probe_id"),
                "probe_class": combined.get("probe_class"),
                "recipe": combined.get("recipe"),
                "research_role": combined.get("research_role"),
                "evaluation_set_id": combined.get("evaluation_set_id"),
                "evaluation_set_version": combined.get("evaluation_set_version"),
                "source_record_id": combined.get("source_record_id"),
                "source_reference_id": combined.get("source_reference_id"),
                "fact_group_id": combined.get("fact_group_id"),
                "provenance": original_provenance,
                "metadata": original_metadata,
                "decoding": dict(decoding),
                "base_model": dict(base_model_identity),
                "prediction_type": prediction_type,
            }
        )
    return rows


def _normalized_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item) for item in items if item is not None))


def _package_version() -> str | None:
    try:
        return metadata.version("cognityx-training")
    except metadata.PackageNotFoundError:
        return None


def _json_checksum(payload: Mapping[str, Any]) -> str:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
