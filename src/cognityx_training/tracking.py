"""Tracker-neutral publication logging with Cognityx Storage as authority."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol
import warnings

from cognityx_training.storage_uri import resolve_storage_uri


@dataclass(frozen=True, slots=True)
class TrackingResult:
    status: str
    backend: str
    external_run_id: str | None = None


class RunTracker(Protocol):
    def log_completed_run(self, payload: Mapping[str, Any]) -> TrackingResult: ...


class NoOpTracker:
    """Default tracker that performs no network or filesystem writes."""

    def log_completed_run(self, payload: Mapping[str, Any]) -> TrackingResult:
        return TrackingResult(status="disabled", backend="none")


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(name, item))
        return flattened
    if isinstance(value, (list, tuple, set)):
        return {prefix: json.dumps(list(value), sort_keys=True, default=str)}
    return {prefix: value}


def _parameter_values(values: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value)[:500]
        for key, value in _flatten("", values).items()
        if value is not None
    }


def _metric_values(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in _flatten("", values).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class MLflowTracker:
    """Log compact run metadata without copying Cognityx artifacts into MLflow."""

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str,
        run_name: str | None = None,
        parent_run_id: str | None = None,
        mlflow_module: Any | None = None,
    ) -> None:
        if mlflow_module is None:
            try:
                import mlflow as mlflow_module
            except ImportError as exc:
                raise RuntimeError("MLflow tracking requires `cognityx-training[tracking]`.") from exc
        self.mlflow = mlflow_module
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.parent_run_id = parent_run_id

    def _existing_run_id(self, idempotency_key: str) -> str | None:
        tracking = getattr(self.mlflow, "tracking", None)
        client_type = getattr(tracking, "MlflowClient", None)
        if client_type is None:
            return None
        client = client_type()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        escaped = idempotency_key.replace("'", "\\'")
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.`cognityx.idempotency_key` = '{escaped}'",
            max_results=1,
        )
        if not runs:
            return None
        return str(runs[0].info.run_id)

    def log_completed_run(self, payload: Mapping[str, Any]) -> TrackingResult:
        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment_name)
        references = dict(payload.get("artifact_references") or {})
        identity = dict(payload.get("identity") or {})
        idempotency_key = str(
            references.get("publication_manifest_uri")
            or identity.get("training_run_id")
            or ""
        )
        if not idempotency_key:
            raise ValueError("Tracking payload requires a publication URI or training_run_id")
        existing = self._existing_run_id(idempotency_key)
        if existing:
            return TrackingResult(status="already_tracked", backend="mlflow", external_run_id=existing)
        tags = {
            "cognityx.idempotency_key": idempotency_key,
            "cognityx.artifacts_authority": "cognityx-storage",
        }
        if self.parent_run_id:
            tags["mlflow.parentRunId"] = self.parent_run_id
        with self.mlflow.start_run(run_name=self.run_name, tags=tags) as active:
            self.mlflow.log_params(_parameter_values({
                **identity,
                **dict(payload.get("parameters") or {}),
            }))
            self.mlflow.log_metrics(_metric_values({
                **dict(payload.get("metrics") or {}),
                **dict(payload.get("resources") or {}),
            }))
            self.mlflow.set_tags(_parameter_values({
                "artifact_references": references,
                "artifact_checksums": dict(payload.get("artifact_checksums") or {}),
            }))
            run_id = str(active.info.run_id)
        return TrackingResult(status="logged", backend="mlflow", external_run_id=run_id)


def create_tracker(
    *,
    backend: str,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    parent_run_id: str | None = None,
    mlflow_module: Any | None = None,
) -> RunTracker:
    if backend == "none":
        return NoOpTracker()
    if backend == "mlflow":
        if not experiment_name:
            raise ValueError("MLflow tracking requires experiment_name")
        return MLflowTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
            mlflow_module=mlflow_module,
        )
    raise ValueError(f"Unsupported tracking backend: {backend}")


def track_with_policy(
    tracker: RunTracker,
    payload: Mapping[str, Any],
    *,
    failure_policy: str,
) -> TrackingResult:
    try:
        return tracker.log_completed_run(payload)
    except Exception as exc:
        if failure_policy == "error":
            raise
        if failure_policy != "warn":
            raise ValueError("Tracking failure_policy must be warn or error") from exc
        warnings.warn(f"Training tracking failed: {exc}", RuntimeWarning, stacklevel=2)
        return TrackingResult(status="failed_warning", backend=type(tracker).__name__)


def track_configured_run(
    payload: Mapping[str, Any],
    *,
    backend: str,
    tracking_uri: str | None,
    experiment_name: str | None,
    run_name: str | None,
    parent_run_id: str | None,
    failure_policy: str,
) -> TrackingResult:
    try:
        tracker = create_tracker(
            backend=backend,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
        )
        return tracker.log_completed_run(payload)
    except Exception as exc:
        if failure_policy == "error":
            raise
        if failure_policy != "warn":
            raise ValueError("Tracking failure_policy must be warn or error") from exc
        warnings.warn(f"Training tracking failed: {exc}", RuntimeWarning, stacklevel=2)
        return TrackingResult(status="failed_warning", backend=backend)


def completed_run_payload(
    *,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
    resources: Mapping[str, Any],
    artifact_references: Mapping[str, Any],
    artifact_checksums: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "cognityx.training.tracking-payload/v1",
        "identity": dict(identity),
        "parameters": dict(parameters),
        "metrics": dict(metrics),
        "resources": dict(resources),
        "artifact_references": dict(artifact_references),
        "artifact_checksums": dict(artifact_checksums),
    }


def _read_json_uri(storage_runtime: Any, uri: str) -> tuple[dict[str, Any], Any, str]:
    resolution = resolve_storage_uri(storage_runtime, uri)
    with resolution.store.open(resolution.key) as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Stored JSON is not an object: {uri}")
    return value, resolution.store, resolution.key


def payload_from_publication(storage_runtime: Any, publication_manifest_uri: str) -> dict[str, Any]:
    publication, store, key = _read_json_uri(storage_runtime, publication_manifest_uri)
    if publication.get("schema_version") != "cognityx.training.publication/v1" or publication.get("status") != "completed":
        raise ValueError("Tracking backfill requires a completed training publication manifest")
    root = key.rsplit("/", 1)[0]
    with store.open(f"{root}/metrics.json") as source:
        metrics = json.load(source)
    with store.open(f"{root}/training-report.json") as source:
        report = json.load(source)
    return completed_run_payload(
        identity={name: publication.get(name) for name in (
            "experiment_id", "training_variant_id", "training_run_id", "adapter_id"
        )},
        parameters=dict(report.get("configuration") or {}),
        metrics=dict(metrics or {}),
        resources={
            "system_usage": dict(report.get("system_usage") or {}),
            "gpu_usage": list(report.get("gpu_usage") or []),
        },
        artifact_references={
            "publication_manifest_uri": publication_manifest_uri,
            "adapter_uri": publication.get("adapter_uri"),
            "adapter_manifest_uri": publication.get("adapter_manifest_uri"),
            "training_report_uri": publication.get("training_report_uri"),
            "baseline_predictions_uri": publication.get("baseline_predictions_uri"),
            "trained_predictions_uri": publication.get("trained_predictions_uri"),
        },
        artifact_checksums=dict(publication.get("artifact_checksums") or {}),
    )
