"""Tracker-neutral live run events with Cognityx Storage as authority."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from cognityx_observability import (
    ArtifactReference,
    MLflowExporter,
    ObservationContext,
    ObservationSession,
)

from cognityx_training.storage_uri import resolve_storage_uri


@dataclass(frozen=True, slots=True)
class TrackingResult:
    status: str
    backend: str
    external_run_id: str | None = None


class RunTracker(Protocol):
    """Small lifecycle contract implemented by optional tracking backends."""

    def start_run(self, context: Mapping[str, Any]) -> TrackingResult: ...

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None: ...

    def log_evaluation(
        self,
        suite_identity: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None: ...

    def log_artifact_references(
        self,
        references: Mapping[str, Any],
        checksums: Mapping[str, Any],
    ) -> None: ...

    def finish(
        self,
        status: str,
        final_metadata: Mapping[str, Any],
    ) -> TrackingResult: ...

    def fail(
        self,
        error: BaseException | str,
        failure_metadata: Mapping[str, Any],
    ) -> TrackingResult: ...

    def log_completed_run(self, payload: Mapping[str, Any]) -> TrackingResult: ...


class NoOpTracker:
    """Default tracker that performs no network or filesystem writes."""

    _result = TrackingResult(status="disabled", backend="none")

    def start_run(self, context: Mapping[str, Any]) -> TrackingResult:
        return self._result

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        return None

    def log_evaluation(
        self,
        suite_identity: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        return None

    def log_artifact_references(
        self,
        references: Mapping[str, Any],
        checksums: Mapping[str, Any],
    ) -> None:
        return None

    def finish(
        self,
        status: str,
        final_metadata: Mapping[str, Any],
    ) -> TrackingResult:
        return self._result

    def fail(
        self,
        error: BaseException | str,
        failure_metadata: Mapping[str, Any],
    ) -> TrackingResult:
        return self._result

    def log_completed_run(self, payload: Mapping[str, Any]) -> TrackingResult:
        """Retain the original completed-run entry point for compatibility."""
        return self._result


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


def _metric_segment(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return normalized[:120] or "unnamed"


class MLflowTracker:
    """Compatibility tracker delegating backend mechanics to Observability."""

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str,
        run_name: str | None = None,
        parent_run_id: str | None = None,
        mlflow_module: Any | None = None,
    ) -> None:
        self.exporter = MLflowExporter(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            mlflow_module=mlflow_module,
        )
        self.parent_run_id = parent_run_id
        self._session: ObservationSession | None = None

    @staticmethod
    def _result(value: Any) -> TrackingResult:
        return TrackingResult(
            status="logged" if value.status == "completed" else value.status,
            backend=value.backend,
            external_run_id=value.external_run_id,
        )

    def start_run(self, context: Mapping[str, Any]) -> TrackingResult:
        if self._session is not None:
            raise RuntimeError("Tracking run is already active")
        identity = dict(context.get("identity") or {})
        supplied_keys = [
            str(value) for value in context.get("idempotency_keys", []) if value
        ]
        idempotency_keys = list(
            dict.fromkeys(
                [
                    *supplied_keys,
                    str(identity.get("training_run_id") or ""),
                ]
            )
        )
        idempotency_keys = [value for value in idempotency_keys if value]
        if not idempotency_keys:
            raise ValueError(
                "Tracking context requires a training_run_id or idempotency key"
            )
        observation = ObservationContext(
            component="training",
            operation="optimize",
            run_id=str(identity.get("training_run_id") or "") or None,
            parent_run_id=self.parent_run_id,
            idempotency_key=idempotency_keys[0],
            attributes={
                **identity,
                "parameters": dict(context.get("parameters") or {}),
                "run_metadata": dict(context.get("metadata") or {}),
            },
        )
        self._session = ObservationSession(
            observation,
            self.exporter,
            failure_policy="error",
        )
        return self._result(self._session.start())

    def _active_session(self) -> ObservationSession | None:
        if self._session is None or self._session.result.status == "already_tracked":
            return None
        return self._session

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        if session := self._active_session():
            session.metrics(_metric_values(metrics), step=step)

    def log_evaluation(
        self,
        suite_identity: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        session = self._active_session()
        if session is None:
            return
        phase = _metric_segment(suite_identity.get("phase") or "evaluation")
        role = _metric_segment(
            suite_identity.get("research_role")
            or suite_identity.get("suite_id")
            or "suite"
        )
        prefix = f"evaluation.{phase}.{role}"
        session.metrics(
            {
                f"{prefix}.{key}": value
                for key, value in _metric_values(metrics).items()
            },
            step=step,
        )
        session.event(
            "training.evaluation",
            attributes={"metric_prefix": prefix, **dict(suite_identity)},
        )

    def log_artifact_references(
        self,
        references: Mapping[str, Any],
        checksums: Mapping[str, Any],
    ) -> None:
        session = self._active_session()
        if session is None:
            return
        session.artifacts(
            ArtifactReference(
                name=str(name),
                uri=str(uri),
                checksum=(
                    str(
                        checksums.get(name)
                        or checksums.get(str(name).removesuffix("_uri"))
                    )
                    if checksums.get(name)
                    or checksums.get(str(name).removesuffix("_uri"))
                    else None
                ),
            )
            for name, uri in references.items()
            if uri
        )
        if checksums:
            session.event(
                "training.artifact_checksums",
                attributes=dict(checksums),
            )

    def finish(
        self,
        status: str,
        final_metadata: Mapping[str, Any],
    ) -> TrackingResult:
        if self._session is None:
            raise RuntimeError("No active tracking run to finish")
        return self._result(self._session.finish(status, attributes=final_metadata))

    def fail(
        self,
        error: BaseException | str,
        failure_metadata: Mapping[str, Any],
    ) -> TrackingResult:
        if self._session is None:
            return TrackingResult(status="not_started", backend="mlflow")
        return self._result(self._session.fail(error, attributes=failure_metadata))

    def log_completed_run(self, payload: Mapping[str, Any]) -> TrackingResult:
        """Register a historical payload through the lifecycle API."""
        return track_with_policy(self, payload, failure_policy="error")


class TrackingSession:
    """Apply warn/error policy consistently across a tracker lifecycle."""

    def __init__(self, tracker: RunTracker, *, failure_policy: str) -> None:
        if failure_policy not in {"warn", "error"}:
            raise ValueError("Tracking failure_policy must be warn or error")
        self.tracker = tracker
        self.failure_policy = failure_policy
        self._disabled = False
        self._result = TrackingResult(
            status="not_started",
            backend=type(tracker).__name__,
        )

    @property
    def result(self) -> TrackingResult:
        return self._result

    def _failed(self, exc: Exception) -> TrackingResult:
        if self.failure_policy == "error":
            raise exc
        warnings.warn(f"Training tracking failed: {exc}", RuntimeWarning, stacklevel=3)
        self._disabled = True
        self._result = TrackingResult(
            status="failed_warning",
            backend=self._result.backend,
            external_run_id=self._result.external_run_id,
        )
        return self._result

    def start_run(self, context: Mapping[str, Any]) -> TrackingResult:
        try:
            self._result = self.tracker.start_run(context)
        except Exception as exc:  # noqa: BLE001 - configured safety boundary
            return self._failed(exc)
        if self._result.status in {"disabled", "already_tracked"}:
            self._disabled = True
        return self._result

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        if self._disabled:
            return
        try:
            self.tracker.log_metrics(metrics, step=step)
        except Exception as exc:  # noqa: BLE001 - configured safety boundary
            self._failed(exc)

    def log_evaluation(
        self,
        suite_identity: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        if self._disabled:
            return
        try:
            self.tracker.log_evaluation(suite_identity, metrics, step=step)
        except Exception as exc:  # noqa: BLE001 - configured safety boundary
            self._failed(exc)

    def log_artifact_references(
        self,
        references: Mapping[str, Any],
        checksums: Mapping[str, Any],
    ) -> None:
        if self._disabled:
            return
        try:
            self.tracker.log_artifact_references(references, checksums)
        except Exception as exc:  # noqa: BLE001 - configured safety boundary
            self._failed(exc)

    def finish(
        self,
        status: str,
        final_metadata: Mapping[str, Any],
    ) -> TrackingResult:
        if self._disabled:
            return self._result
        try:
            self._result = self.tracker.finish(status, final_metadata)
        except Exception as exc:  # noqa: BLE001 - configured safety boundary
            return self._failed(exc)
        return self._result

    def fail(
        self,
        error: BaseException | str,
        failure_metadata: Mapping[str, Any],
    ) -> TrackingResult:
        if self._disabled:
            return self._result
        try:
            self._result = self.tracker.fail(error, failure_metadata)
        except Exception as exc:  # noqa: BLE001 - configured safety boundary
            return self._failed(exc)
        return self._result


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


def create_tracking_session(
    *,
    backend: str,
    tracking_uri: str | None,
    experiment_name: str | None,
    run_name: str | None,
    parent_run_id: str | None,
    failure_policy: str,
    context: Mapping[str, Any],
) -> TrackingSession:
    """Create and start a configured live session under the failure policy."""
    try:
        tracker = create_tracker(
            backend=backend,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
        )
    except Exception as exc:  # noqa: BLE001 - configured safety boundary
        session = TrackingSession(NoOpTracker(), failure_policy=failure_policy)
        session._result = TrackingResult(status="not_started", backend=backend)
        session._failed(exc)
        return session
    session = TrackingSession(tracker, failure_policy=failure_policy)
    session.start_run(context)
    return session


def track_with_policy(
    tracker: RunTracker,
    payload: Mapping[str, Any],
    *,
    failure_policy: str,
) -> TrackingResult:
    """Replay a completed publication through the lifecycle for backfill."""
    session = TrackingSession(tracker, failure_policy=failure_policy)
    references = dict(payload.get("artifact_references") or {})
    identity = dict(payload.get("identity") or {})
    run_metadata = {
        "registration_mode": "backfill",
        **dict(payload.get("run_metadata") or {}),
    }
    idempotency_keys = [
        value
        for value in (
            references.get("publication_manifest_uri"),
            identity.get("training_run_id"),
        )
        if value
    ]
    session.start_run(
        {
            "identity": identity,
            "parameters": dict(payload.get("parameters") or {}),
            "metadata": run_metadata,
            "idempotency_keys": idempotency_keys,
        }
    )
    session.log_metrics(
        {
            **dict(payload.get("metrics") or {}),
            **dict(payload.get("resources") or {}),
        }
    )
    for evaluation in payload.get("evaluations") or []:
        session.log_evaluation(
            dict(evaluation.get("suite_identity") or {}),
            dict(evaluation.get("metrics") or {}),
            step=evaluation.get("step"),
        )
    session.log_artifact_references(
        references,
        dict(payload.get("artifact_checksums") or {}),
    )
    return session.finish("completed", run_metadata)


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
    """Backward-compatible completed-run registration used by callers/backfill."""
    try:
        tracker = create_tracker(
            backend=backend,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
        )
    except Exception as exc:
        if failure_policy == "error":
            raise
        if failure_policy != "warn":
            raise ValueError("Tracking failure_policy must be warn or error") from exc
        warnings.warn(f"Training tracking failed: {exc}", RuntimeWarning, stacklevel=2)
        return TrackingResult(status="failed_warning", backend=backend)
    return track_with_policy(tracker, payload, failure_policy=failure_policy)


def completed_run_payload(
    *,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
    resources: Mapping[str, Any],
    artifact_references: Mapping[str, Any],
    artifact_checksums: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "cognityx.training.tracking-payload/v1",
        "identity": dict(identity),
        "parameters": dict(parameters),
        "metrics": dict(metrics),
        "resources": dict(resources),
        "evaluations": [dict(value) for value in evaluations or []],
        "run_metadata": dict(run_metadata or {}),
        "artifact_references": dict(artifact_references),
        "artifact_checksums": dict(artifact_checksums),
    }


def _read_json_uri(storage_runtime: Any, uri: str) -> tuple[dict[str, Any], Any, str]:
    resolution = resolve_storage_uri(storage_runtime, uri)
    with resolution.store.open(resolution.key) as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"Stored JSON is not an object: {uri}")
    return value, resolution.store, resolution.key


def payload_from_publication(
    storage_runtime: Any, publication_manifest_uri: str
) -> dict[str, Any]:
    publication, store, key = _read_json_uri(storage_runtime, publication_manifest_uri)
    if (
        publication.get("schema_version") != "cognityx.training.publication/v1"
        or publication.get("status") != "completed"
    ):
        raise ValueError(
            "Tracking backfill requires a completed training publication manifest"
        )
    root = key.rsplit("/", 1)[0]
    with store.open(f"{root}/metrics.json") as source:
        metrics = json.load(source)
    with store.open(f"{root}/training-report.json") as source:
        report = json.load(source)
    evaluations: list[dict[str, Any]] = []
    for phase in ("baseline", "trained"):
        evaluation = dict((report.get("evaluation") or {}).get(phase) or {})
        for role, suite_metrics in (evaluation.get("suite_metrics") or {}).items():
            evaluations.append(
                {
                    "suite_identity": {
                        "phase": phase,
                        "research_role": role,
                        "evaluation_sets": list(
                            suite_metrics.get("evaluation_sets") or []
                        ),
                    },
                    "metrics": {
                        name: value
                        for name, value in suite_metrics.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    },
                    "step": 0 if phase == "baseline" else metrics.get("train_steps"),
                }
            )
    return completed_run_payload(
        identity={
            name: publication.get(name)
            for name in (
                "experiment_id",
                "training_variant_id",
                "training_run_id",
                "adapter_id",
            )
        },
        parameters=dict(report.get("configuration") or {}),
        metrics=dict(metrics or {}),
        resources={
            "system_usage": dict(report.get("system_usage") or {}),
            "gpu_usage": list(report.get("gpu_usage") or []),
        },
        evaluations=evaluations,
        run_metadata={
            "registration_mode": "backfill",
            "original_started_at": report.get("started_at"),
            "original_finished_at": report.get("finished_at"),
            "original_duration_seconds": report.get("duration_seconds"),
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
