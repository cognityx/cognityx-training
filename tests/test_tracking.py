from __future__ import annotations

from types import SimpleNamespace

import pytest
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_training.custom_pytorch import (
    _evaluation_tracking_events,
    _live_tracking_metrics,
)
from cognityx_training.tracking import (
    MLflowTracker,
    NoOpTracker,
    TrackingResult,
    TrackingSession,
    completed_run_payload,
    payload_from_publication,
    track_with_policy,
)


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.params = {}
        self.metric_events = []
        self.tags = {}
        self.start_tags = {}
        self.end_status = None
        self.existing_run_id = None
        owner = self

        class Client:
            def get_experiment_by_name(self, name):
                return SimpleNamespace(experiment_id="experiment-1")

            def search_runs(self, *args, **kwargs):
                if owner.existing_run_id:
                    return [SimpleNamespace(info=SimpleNamespace(run_id=owner.existing_run_id))]
                return []

        self.tracking = SimpleNamespace(MlflowClient=Client)

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, *, run_name, tags):
        self.start_tags = tags
        return SimpleNamespace(info=SimpleNamespace(run_id="external-run-1"))

    def log_params(self, values):
        self.params.update(values)

    def log_metrics(self, values, *, step=None):
        self.metric_events.append((step, dict(values)))

    def set_tags(self, values):
        self.tags.update(values)

    def end_run(self, *, status):
        self.end_status = status


def _payload():
    return completed_run_payload(
        identity={"experiment_id": "exp-1", "training_run_id": "run-1"},
        parameters={"training": {"learning_rate": 0.001}},
        metrics={"train_loss": 0.5, "evaluation_suite_counts": {"exact_recall": 2}},
        resources={"ram_peak_bytes": 10},
        evaluations=[{
            "suite_identity": {
                "phase": "trained",
                "research_role": "exact_recall",
                "evaluation_sets": [{"evaluation_set_id": "eval-1", "evaluation_set_version": "1"}],
            },
            "metrics": {"exact_match_accuracy": 1.0},
            "step": 2,
        }],
        run_metadata={
            "original_started_at": "2026-08-01T00:00:00+00:00",
            "original_finished_at": "2026-08-01T00:01:00+00:00",
        },
        artifact_references={
            "publication_manifest_uri": "storage://local-main/artifacts/runs/run-1/publication-manifest.json",
            "adapter_uri": "storage://local-main/models/adapters/adapter-1/1",
        },
        artifact_checksums={"adapter_bundle": "abc"},
    )


def test_noop_and_mlflow_lifecycle_are_compact_and_idempotent():
    noop = NoOpTracker()
    assert noop.start_run({"identity": {"training_run_id": "run-1"}}).status == "disabled"

    fake = FakeMLflow()
    tracker = MLflowTracker(
        tracking_uri="sqlite:///tracking.db",
        experiment_name="qualification",
        run_name="qualified-run",
        parent_run_id="parent-1",
        mlflow_module=fake,
    )
    started = tracker.start_run({
        "identity": {"experiment_id": "exp-1", "training_run_id": "run-1"},
        "parameters": {"training": {"learning_rate": 0.001}},
        "metadata": {"registration_mode": "live"},
    })
    tracker.log_metrics({"train_loss": 0.5}, step=1)
    tracker.log_evaluation(
        {
            "phase": "baseline",
            "research_role": "exact_recall",
            "evaluation_sets": [{"evaluation_set_id": "eval-1", "evaluation_set_version": "1"}],
        },
        {"exact_match_accuracy": 0.25},
        step=0,
    )
    tracker.log_artifact_references(
        {
            "publication_manifest_uri": "storage://local-main/runs/run-1/publication-manifest.json",
            "adapter_uri": "storage://local-main/adapters/adapter-1/1",
        },
        {"adapter_bundle": "abc"},
    )
    result = tracker.finish("completed", {"training_duration_seconds": 60.0})

    assert started.status == "started"
    assert result.status == "logged"
    assert result.external_run_id == "external-run-1"
    assert fake.tracking_uri == "sqlite:///tracking.db"
    assert fake.experiment_name == "qualification"
    assert fake.params["training.learning_rate"] == "0.001"
    assert fake.metric_events[0] == (1, {"train_loss": 0.5})
    assert fake.metric_events[1][0] == 0
    assert fake.metric_events[1][1]["evaluation.baseline.exact_recall.exact_match_accuracy"] == 0.25
    assert fake.start_tags["mlflow.parentRunId"] == "parent-1"
    assert fake.tags["artifact_references.adapter_uri"].startswith("storage://")
    assert fake.tags["cognityx.publication_manifest_uri"].startswith("storage://")
    assert fake.end_status == "FINISHED"
    assert not hasattr(fake, "log_artifact")

    repeated_fake = FakeMLflow()
    repeated_fake.existing_run_id = "existing-run"
    repeated = MLflowTracker(
        tracking_uri=None,
        experiment_name="qualification",
        mlflow_module=repeated_fake,
    ).start_run({
        "identity": {"training_run_id": "run-1"},
        "idempotency_keys": ["storage://local-main/runs/run-1/publication-manifest.json"],
    })
    assert repeated.status == "already_tracked"
    assert repeated.external_run_id == "existing-run"


class RecordingTracker:
    def __init__(self) -> None:
        self.events = []

    def start_run(self, context):
        self.events.append(("start", dict(context)))
        return TrackingResult(status="started", backend="fake", external_run_id="fake-1")

    def log_metrics(self, metrics, *, step=None):
        self.events.append(("metrics", step, dict(metrics)))

    def log_evaluation(self, suite_identity, metrics, *, step=None):
        self.events.append(("evaluation", step, dict(suite_identity), dict(metrics)))

    def log_artifact_references(self, references, checksums):
        self.events.append(("artifacts", dict(references), dict(checksums)))

    def finish(self, status, final_metadata):
        self.events.append(("finish", status, dict(final_metadata)))
        return TrackingResult(status="logged", backend="fake", external_run_id="fake-1")

    def fail(self, error, failure_metadata):
        self.events.append(("fail", str(error), dict(failure_metadata)))
        return TrackingResult(status="failed", backend="fake", external_run_id="fake-1")


def test_backfill_replays_required_event_order_and_original_times():
    tracker = RecordingTracker()
    result = track_with_policy(tracker, _payload(), failure_policy="warn")
    assert result.status == "logged"
    assert [event[0] for event in tracker.events] == [
        "start", "metrics", "evaluation", "artifacts", "finish"
    ]
    start_metadata = tracker.events[0][1]["metadata"]
    assert start_metadata["registration_mode"] == "backfill"
    assert start_metadata["original_started_at"] == "2026-08-01T00:00:00+00:00"


def test_session_emits_failure_event():
    tracker = RecordingTracker()
    session = TrackingSession(tracker, failure_policy="warn")
    session.start_run({"identity": {"training_run_id": "run-1"}})
    result = session.fail(RuntimeError("training stopped"), {"phase": "training"})
    assert result.status == "failed"
    assert [event[0] for event in tracker.events] == ["start", "fail"]


class BrokenMetricTracker(RecordingTracker):
    def log_metrics(self, metrics, *, step=None):
        self.events.append(("metrics", step, dict(metrics)))
        raise RuntimeError("tracker unavailable")


def test_tracking_failure_policy_warns_and_disables_or_raises():
    warn_tracker = BrokenMetricTracker()
    warn_session = TrackingSession(warn_tracker, failure_policy="warn")
    warn_session.start_run({"identity": {"training_run_id": "run-1"}})
    with pytest.warns(RuntimeWarning, match="tracker unavailable"):
        warn_session.log_metrics({"loss": 1.0}, step=1)
    assert warn_session.finish("completed", {}).status == "failed_warning"
    assert [event[0] for event in warn_tracker.events] == ["start", "metrics"]

    strict_tracker = BrokenMetricTracker()
    strict_session = TrackingSession(strict_tracker, failure_policy="error")
    strict_session.start_run({"identity": {"training_run_id": "run-1"}})
    with pytest.raises(RuntimeError, match="tracker unavailable"):
        strict_session.log_metrics({"loss": 1.0}, step=1)


def test_live_metrics_use_measured_values_and_explicit_resource_scope():
    metrics = _live_tracking_metrics(
        step=2,
        loss=0.25,
        examples_processed=4,
        input_tokens_processed=100,
        target_tokens_processed=40,
        elapsed_seconds=5.0,
        resource_snapshot={
            "cpu_percent": 10.0,
            "ram_bytes": 200,
            "disk_read_bytes": 5,
            "disk_write_bytes": 6,
            "host": {
                "scope": "wsl_vm",
                "cpu_percent": 30.0,
                "used_bytes": 300,
                "used_percent": 50.0,
                "total_bytes": 600,
            },
            "gpu": {
                "scope": "whole_device",
                "device_index": 0,
                "utilization_percent": 70.0,
                "memory_used_bytes": 400,
                "memory_total_bytes": 800,
                "power_draw_watts": 100.0,
            },
            "gpu_aggregate": {"energy_joules": 250.0},
        },
    )
    assert metrics["training.optimizer_step"] == 2
    assert metrics["training.examples_per_second"] == 0.8
    assert metrics["training.input_tokens_per_second"] == 20.0
    assert metrics["resource.host.wsl_vm.ram_used_bytes"] == 300
    assert metrics["resource.gpu.whole_device.device_0.accumulated_energy_joules"] == 250.0


def test_evaluation_events_retain_research_suite_identity():
    events = _evaluation_tracking_events(
        {
            "suite_metrics": {
                "paraphrase_evaluation": {
                    "example_count": 2,
                    "exact_match_accuracy": 0.5,
                    "record_ids": ["one", "two"],
                    "evaluation_sets": [{
                        "evaluation_set_id": "eval-paraphrase",
                        "evaluation_set_version": "version-1",
                    }],
                }
            }
        },
        phase="baseline",
    )
    identity, metrics = events[0]
    assert identity == {
        "phase": "baseline",
        "research_role": "paraphrase_evaluation",
        "evaluation_sets": [{
            "evaluation_set_id": "eval-paraphrase",
            "evaluation_set_version": "version-1",
        }],
    }
    assert metrics == {"example_count": 2.0, "exact_match_accuracy": 0.5}


def test_completed_publication_backfill_preserves_times_and_storage_references(tmp_path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    store = runtime.for_role("artifact")
    root = "experiments/exp-1/runs/run-1"
    store.put_json(f"{root}/metrics.json", {"train_loss": 0.25, "train_steps": 2})
    store.put_json(f"{root}/training-report.json", {
        "started_at": "2026-08-01T00:00:00+00:00",
        "finished_at": "2026-08-01T00:01:00+00:00",
        "duration_seconds": 60.0,
        "configuration": {"learning_rate": 0.001},
        "system_usage": {"ram_peak_bytes": 100},
        "gpu_usage": [],
        "evaluation": {
            "baseline": {
                "suite_metrics": {
                    "exact_recall": {
                        "example_count": 1,
                        "exact_match_accuracy": 1.0,
                        "evaluation_sets": [{
                            "evaluation_set_id": "eval-1",
                            "evaluation_set_version": "1",
                        }],
                    }
                }
            }
        },
    })
    publication_uri = store.put_json(f"{root}/publication-manifest.json", {
        "schema_version": "cognityx.training.publication/v1",
        "status": "completed",
        "experiment_id": "exp-1",
        "training_variant_id": "variant-1",
        "training_run_id": "run-1",
        "adapter_id": "adapter-1",
        "adapter_uri": "storage://local-main/models/adapters/adapter-1/1",
        "adapter_manifest_uri": "storage://local-main/models/adapters/adapter-1/1/adapter-manifest.json",
        "training_report_uri": store.uri(f"{root}/training-report.json"),
        "baseline_predictions_uri": store.uri(f"{root}/baseline-predictions.jsonl"),
        "trained_predictions_uri": store.uri(f"{root}/trained-predictions.jsonl"),
        "artifact_checksums": {"adapter_bundle": "abc"},
    }).uri
    payload = payload_from_publication(runtime, publication_uri)
    assert payload["identity"]["training_run_id"] == "run-1"
    assert payload["metrics"]["train_loss"] == 0.25
    assert payload["run_metadata"] == {
        "registration_mode": "backfill",
        "original_started_at": "2026-08-01T00:00:00+00:00",
        "original_finished_at": "2026-08-01T00:01:00+00:00",
        "original_duration_seconds": 60.0,
    }
    assert payload["evaluations"][0]["suite_identity"]["evaluation_sets"][0]["evaluation_set_id"] == "eval-1"
    assert payload["artifact_references"]["publication_manifest_uri"] == publication_uri
    assert payload["artifact_checksums"]["adapter_bundle"] == "abc"


def test_local_sqlite_mlflow_when_optional_dependency_is_installed(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="cognityx-test",
        mlflow_module=mlflow,
    )
    started = tracker.start_run({
        "identity": {"training_run_id": "local-file-backed-run"},
        "parameters": {"seed": 7},
    })
    tracker.log_metrics({"loss": 0.5}, step=1)
    result = tracker.finish("completed", {"registration_mode": "test"})
    assert started.status == "started"
    assert result.status == "logged"
