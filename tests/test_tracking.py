from __future__ import annotations

from types import SimpleNamespace

import pytest
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_training.tracking import (
    MLflowTracker,
    NoOpTracker,
    completed_run_payload,
    payload_from_publication,
    track_with_policy,
)


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.start_tags = {}
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
        active = SimpleNamespace(info=SimpleNamespace(run_id="external-run-1"))

        class Context:
            def __enter__(self):
                return active

            def __exit__(self, *args):
                return False

        return Context()

    def log_params(self, values):
        self.params.update(values)

    def log_metrics(self, values):
        self.metrics.update(values)

    def set_tags(self, values):
        self.tags.update(values)


def _payload():
    return completed_run_payload(
        identity={"experiment_id": "exp-1", "training_run_id": "run-1"},
        parameters={"training": {"learning_rate": 0.001}},
        metrics={"train_loss": 0.5, "evaluation_suite_counts": {"exact_recall": 2}},
        resources={"ram_peak_bytes": 10},
        artifact_references={
            "publication_manifest_uri": "storage://local-main/artifacts/runs/run-1/publication-manifest.json",
            "adapter_uri": "storage://local-main/models/adapters/adapter-1/1",
        },
        artifact_checksums={"adapter_bundle": "abc"},
    )


def test_noop_and_mlflow_tracking_are_compact_and_idempotent():
    assert NoOpTracker().log_completed_run(_payload()).status == "disabled"
    fake = FakeMLflow()
    tracker = MLflowTracker(
        tracking_uri="sqlite:///tracking.db",
        experiment_name="qualification",
        run_name="qualified-run",
        parent_run_id="parent-1",
        mlflow_module=fake,
    )
    result = tracker.log_completed_run(_payload())
    assert result.status == "logged"
    assert result.external_run_id == "external-run-1"
    assert fake.tracking_uri == "sqlite:///tracking.db"
    assert fake.experiment_name == "qualification"
    assert fake.params["training.learning_rate"] == "0.001"
    assert fake.metrics["train_loss"] == 0.5
    assert fake.metrics["evaluation_suite_counts.exact_recall"] == 2.0
    assert fake.start_tags["mlflow.parentRunId"] == "parent-1"
    assert fake.tags["artifact_references.adapter_uri"].startswith("storage://")
    assert not hasattr(fake, "log_artifact")

    fake.existing_run_id = "existing-run"
    repeated = tracker.log_completed_run(_payload())
    assert repeated.status == "already_tracked"
    assert repeated.external_run_id == "existing-run"


class BrokenTracker:
    def log_completed_run(self, payload):
        raise RuntimeError("tracker unavailable")


def test_tracking_failure_policy_warns_or_raises():
    with pytest.warns(RuntimeWarning, match="tracker unavailable"):
        result = track_with_policy(BrokenTracker(), _payload(), failure_policy="warn")
    assert result.status == "failed_warning"
    with pytest.raises(RuntimeError, match="tracker unavailable"):
        track_with_policy(BrokenTracker(), _payload(), failure_policy="error")


def test_completed_publication_backfill_payload_uses_storage_references(tmp_path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    store = runtime.for_role("artifact")
    root = "experiments/exp-1/runs/run-1"
    store.put_json(f"{root}/metrics.json", {"train_loss": 0.25})
    store.put_json(f"{root}/training-report.json", {
        "configuration": {"learning_rate": 0.001},
        "system_usage": {"ram_peak_bytes": 100},
        "gpu_usage": [],
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
    assert payload["artifact_references"]["publication_manifest_uri"] == publication_uri
    assert payload["artifact_checksums"]["adapter_bundle"] == "abc"
