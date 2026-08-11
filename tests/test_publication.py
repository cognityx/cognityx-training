from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognityx_core import Dataset, TrainingRequest
from cognityx_storage import (
    ObjectAlreadyExistsError,
    ObjectConsistencyError,
    StorageConfig,
    StorageRuntime,
)

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.custom_pytorch import CustomPyTorchTrainerBackend
from cognityx_training.dataset_pipeline import DatasetLineage
from cognityx_training.lineage import (
    TrainingLineageIds,
    adapter_id,
    build_lineage_ids,
    training_run_id,
)
from cognityx_training.publication import (
    PublicationResult,
    TrainingPublisher,
    bundle_checksum,
    canonical_variant_identity,
    inspect_adapter_files,
    prediction_rows,
    verify_published_adapter,
)


def _runtime(tmp_path: Path) -> StorageRuntime:
    return StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path))


def _lineage() -> DatasetLineage:
    return DatasetLineage(
        dataset_id="demo",
        dataset_name="Demo",
        dataset_version="3",
        dataset_variant_id="dvar-clean",
        dataset_manifest_uri="storage://local-main/datasets/demo/3/manifest.json",
        dataset_manifest_checksum="manifest-sha",
        records_uri="storage://local-main/datasets/demo/3/records.jsonl",
        records_checksum="records-sha",
        recipe="knowledge-unit-qa",
        source_manifest_uri="storage://local-main/ingest/runs/r1/manifest.json",
        source_manifest_checksum="source-sha",
        configuration_checksum="config-sha",
    )


def _research_lineage() -> DatasetLineage:
    return replace(
        _lineage(),
        research_package_manifest_uri=(
            "storage://local-main/datasets/dataforge/research-packages/pkg/4/manifest.json"
        ),
        research_package_manifest_checksum="package-manifest-sha",
        research_package_id="pkg",
        research_package_version="4",
        evaluation_sets=(
            {
                "research_role": "exact_recall",
                "evaluation_set_id": "eval-exact",
                "evaluation_set_version": "1",
                "manifest_uri": "storage://local-main/datasets/evaluations/exact/1/manifest.json",
                "manifest_checksum": "exact-manifest-sha",
                "records_uri": "storage://local-main/datasets/evaluations/exact/1/records.jsonl",
                "records_checksum": "exact-records-sha",
                "record_count": 2,
                "freeze_checksum": "exact-freeze-sha",
            },
        ),
    )


def _base_model() -> dict[str, str | None]:
    return {
        "name": "Qwen/Qwen3-8B",
        "requested_revision": "main",
        "resolved_revision": "commit-1",
        "tokenizer_name": "Qwen/Qwen3-8B",
        "tokenizer_revision": "commit-1",
        "chat_template_checksum": "template-sha",
    }


def _config(tmp_path: Path, **changes) -> CustomPyTorchTrainingConfig:
    config = CustomPyTorchTrainingConfig(
        model_cache_dir=tmp_path,
        output_dir=tmp_path / "staging",
        publication_mode="storage",
        max_steps=2,
    )
    return replace(config, **changes)


def _ids(
    tmp_path: Path,
    *,
    lineage: DatasetLineage | None = None,
) -> tuple[TrainingLineageIds, dict]:
    identity = canonical_variant_identity(
        _config(tmp_path),
        lineage or _lineage(),
        base_model_identity=_base_model(),
    )
    return (
        build_lineage_ids(
            identity,
            requested_experiment_id="exp-demo",
            requested_run_id="fixture-run",
        ),
        identity,
    )


def _adapter_staging(tmp_path: Path, name: str = "staging") -> Path:
    staging = tmp_path / name
    staging.mkdir()
    (staging / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}\n',
        encoding="utf-8",
    )
    (staging / "adapter_model.safetensors").write_bytes(b"fixture-weights")
    (staging / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return staging


def _publisher(
    tmp_path: Path,
    *,
    name: str = "Demo",
    lineage: DatasetLineage | None = None,
) -> tuple[TrainingPublisher, dict]:
    ids, identity = _ids(tmp_path, lineage=lineage)
    return (
        TrainingPublisher(
            _runtime(tmp_path / "storage"),
            ids,
            experiment_name=name,
            experiment_description="Fixture experiment",
        ),
        identity,
    )


def _prepare(
    publisher: TrainingPublisher,
    identity: dict,
    *,
    lineage: DatasetLineage | None = None,
) -> None:
    selected_lineage = lineage or _lineage()
    publisher.publish_experiment()
    publisher.publish_variant(
        identity,
        dataset_lineage=selected_lineage,
        base_model_identity=_base_model(),
    )
    publisher.publish_training_request(
        dataset_lineage=selected_lineage,
        normalized_request=identity["training"],
        base_model_identity=_base_model(),
        publication_mode="storage",
    )


def _complete(
    publisher: TrainingPublisher,
    staging: Path,
    *,
    retain: bool = False,
    lineage: DatasetLineage | None = None,
):
    ids = publisher.ids
    baseline = [
        {
            **ids.to_dict(),
            "adapter_id": None,
            "dataset_record_id": "eval-1",
            "prediction_type": "baseline",
        }
    ]
    trained = [
        {
            **ids.to_dict(),
            "dataset_record_id": "eval-1",
            "prediction_type": "trained",
        }
    ]
    return publisher.publish_completed_run(
        staging_directory=staging,
        dataset_lineage=lineage or _lineage(),
        base_model_identity=_base_model(),
        adapter_details={"type": "qlora", "format": "peft", "rank": 8},
        resolved_config={"data_order": "source"},
        environment={"python_version": "fixture"},
        training_report={"status": "completed", **ids.to_dict()},
        metrics={"train_loss": 0.5},
        baseline_predictions=baseline,
        trained_predictions=trained,
        retain_local_staging=retain,
    )


def test_variant_identity_is_stable_and_semantic(tmp_path: Path) -> None:
    first = canonical_variant_identity(
        _config(tmp_path),
        _lineage(),
        base_model_identity=_base_model(),
    )
    same_semantics = canonical_variant_identity(
        _config(
            tmp_path,
            output_dir=tmp_path / "other",
            model_cache_dir=tmp_path / "cache",
            storage_root=tmp_path / "storage-other",
            nvidia_smi_path="/different/nvidia-smi",
            progress_interval_steps=99,
        ),
        _lineage(),
        base_model_identity=_base_model(),
    )
    changed = canonical_variant_identity(
        _config(tmp_path, learning_rate=0.001),
        _lineage(),
        base_model_identity=_base_model(),
    )

    assert build_lineage_ids(
        first,
        requested_experiment_id="exp-demo",
        requested_run_id="one",
    ).training_variant_id == build_lineage_ids(
        same_semantics,
        requested_experiment_id="exp-demo",
        requested_run_id="two",
    ).training_variant_id
    assert build_lineage_ids(
        changed,
        requested_experiment_id="exp-demo",
        requested_run_id="three",
    ).training_variant_id != build_lineage_ids(
        first,
        requested_experiment_id="exp-demo",
        requested_run_id="four",
    ).training_variant_id


def test_evaluation_package_does_not_change_optimizer_identity(tmp_path: Path) -> None:
    dataset_only = canonical_variant_identity(
        _config(tmp_path),
        _lineage(),
        base_model_identity=_base_model(),
    )
    research_package = canonical_variant_identity(
        _config(tmp_path),
        _research_lineage(),
        base_model_identity=_base_model(),
    )

    assert research_package == dataset_only


def test_retries_have_unique_run_and_adapter_ids() -> None:
    first_run = training_run_id()
    second_run = training_run_id()
    assert first_run != second_run
    assert adapter_id("exp-one", first_run) != adapter_id("exp-one", second_run)
    assert adapter_id("exp-one", first_run) != adapter_id("exp-two", first_run)


def test_experiment_and_variant_manifests_are_idempotent(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    first_experiment = publisher.publish_experiment()
    second_experiment = publisher.publish_experiment()
    first_variant = publisher.publish_variant(
        identity,
        dataset_lineage=_lineage(),
        base_model_identity=_base_model(),
    )
    second_variant = publisher.publish_variant(
        identity,
        dataset_lineage=_lineage(),
        base_model_identity=_base_model(),
    )
    assert first_experiment == second_experiment
    assert first_variant == second_variant


def test_conflicting_experiment_and_variant_manifests_fail(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    publisher.publish_experiment()
    conflicting = TrainingPublisher(
        publisher.storage_runtime,
        publisher.ids,
        experiment_name="Different",
        experiment_description="Fixture experiment",
    )
    with pytest.raises(ObjectConsistencyError):
        conflicting.publish_experiment()

    publisher.publish_variant(
        identity,
        dataset_lineage=_lineage(),
        base_model_identity=_base_model(),
    )
    changed = dict(identity)
    changed["training"] = {**identity["training"], "max_steps": 999}
    with pytest.raises(ObjectConsistencyError):
        publisher.publish_variant(
            changed,
            dataset_lineage=_lineage(),
            base_model_identity=_base_model(),
        )


def test_adapter_checksums_and_bundle_are_stable(tmp_path: Path) -> None:
    staging = _adapter_staging(tmp_path)
    files = inspect_adapter_files(staging)
    assert {item["path"] for item in files} >= {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert bundle_checksum(files) == bundle_checksum(reversed(files))


def test_atomic_publication_verifies_and_removes_staging(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    staging = _adapter_staging(tmp_path)
    result = _complete(publisher, staging)

    verification = verify_published_adapter(
        result.adapter_manifest_uri,
        storage_runtime=publisher.storage_runtime,
    )
    assert verification.valid is True
    assert verification.adapter_id == publisher.ids.adapter_id
    assert result.adapter_uri == (
        f"storage://local-main/models/adapters/{publisher.ids.adapter_id}/1"
    )
    assert result.publication_manifest_uri == (
        "storage://local-main/artifacts/experiments/exp-demo/"
        "runs/trun-fixture-run/publication-manifest.json"
    )
    assert not staging.exists()

    artifact_store = publisher.artifact_store
    with artifact_store.open(
        f"{publisher.run_root}/publication-manifest.json"
    ) as source:
        terminal = json.load(source)
    assert terminal["status"] == "completed"
    assert terminal["training_run_id"] == publisher.ids.training_run_id
    with artifact_store.open(f"{publisher.run_root}/resolved-config.json") as source:
        assert json.load(source)["data_order"] == "source"
    with publisher.model_store.open(
        f"{publisher.adapter_root}/adapter-manifest.json"
    ) as source:
        adapter_manifest = json.load(source)
    assert adapter_manifest["dataset"]["dataset_id"] == "demo"
    assert adapter_manifest["dataset"]["dataset_variant_id"] == "dvar-clean"
    assert "research_package" not in adapter_manifest
    assert "research_package" not in terminal


def test_caller_stable_retry_reuses_verified_terminal_publication(
    tmp_path: Path,
) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    completed = _complete(publisher, _adapter_staging(tmp_path))

    found = TrainingPublisher.find_completed_run(
        publisher.storage_runtime,
        requested_experiment_id="exp-demo",
        requested_training_run_id="fixture-run",
    )

    assert found is not None
    retried_publisher, reused = found
    assert reused == completed
    assert retried_publisher.ids == publisher.ids
    assert reused.adapter_manifest_uri.endswith("/adapter-manifest.json")
    assert not (tmp_path / "second-adapter-staging").exists()


def test_backend_retry_returns_terminal_publication_before_model_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = PublicationResult(
        adapter_uri="storage://local-main/models/adapters/adp-existing/1",
        adapter_manifest_uri=(
            "storage://local-main/models/adapters/adp-existing/1/adapter-manifest.json"
        ),
        training_report_uri=(
            "storage://local-main/artifacts/runs/trun-fixture-run/training-report.json"
        ),
        baseline_predictions_uri=(
            "storage://local-main/artifacts/runs/trun-fixture-run/baseline.jsonl"
        ),
        trained_predictions_uri=(
            "storage://local-main/artifacts/runs/trun-fixture-run/trained.jsonl"
        ),
        publication_manifest_uri=(
            "storage://local-main/artifacts/runs/trun-fixture-run/publication-manifest.json"
        ),
        artifact_checksums={"adapter_bundle": "bundle-safe"},
    )
    monkeypatch.setattr(
        TrainingPublisher,
        "find_completed_run",
        lambda *args, **kwargs: (
            SimpleNamespace(
                ids=TrainingLineageIds(
                    experiment_id="exp-demo",
                    training_variant_id="tvar-existing",
                    training_run_id="trun-fixture-run",
                    adapter_id="adp-existing",
                )
            ),
            publication,
        ),
    )
    original_import = __import__
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "training libraries must not import during stable-run retry"
                )
            )
            if name in {"torch", "peft", "transformers"}
            else original_import(name, *args, **kwargs)
        ),
    )
    backend = CustomPyTorchTrainerBackend(
        _config(
            tmp_path,
            experiment_id="exp-demo",
            training_run_id="fixture-run",
        ),
        storage_runtime=_runtime(tmp_path / "retry-storage"),
    )

    result = backend.train(
        TrainingRequest(
            dataset=Dataset(
                name="fixture",
                version="1",
                uri="storage://local-main/datasets/fixture/manifest.json",
            )
        )
    )

    assert result.metrics["reused_completed_publication"] is True
    assert result.metrics["publication_manifest_uri"] == (
        publication.publication_manifest_uri
    )
    assert result.artifact.uri == publication.adapter_uri


def test_research_package_lineage_reaches_adapter_and_terminal(tmp_path: Path) -> None:
    lineage = _research_lineage()
    publisher, identity = _publisher(tmp_path, lineage=lineage)
    _prepare(publisher, identity, lineage=lineage)
    result = _complete(
        publisher,
        _adapter_staging(tmp_path),
        lineage=lineage,
    )

    with publisher.model_store.open(
        f"{publisher.adapter_root}/adapter-manifest.json"
    ) as source:
        adapter_manifest = json.load(source)
    with publisher.artifact_store.open(
        f"{publisher.run_root}/publication-manifest.json"
    ) as source:
        terminal = json.load(source)

    expected = {
        "research_package_id": "pkg",
        "research_package_version": "4",
        "manifest_uri": lineage.research_package_manifest_uri,
        "manifest_checksum": "package-manifest-sha",
        "evaluation_sets": [dict(lineage.evaluation_sets[0])],
    }
    assert adapter_manifest["research_package"] == expected
    assert terminal["research_package"] == expected
    assert terminal["dataset_lineage_uri"].endswith("/dataset-lineage.json")
    assert terminal["dataset_lineage_checksum"] == terminal["artifact_checksums"][
        "dataset-lineage.json"
    ]
    assert verify_published_adapter(
        result.adapter_manifest_uri,
        storage_runtime=publisher.storage_runtime,
    ).valid


def test_prediction_jsonl_is_published_with_lineage(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    result = _complete(publisher, _adapter_staging(tmp_path))

    store = publisher.artifact_store
    baseline_key = store_key(store, result.baseline_predictions_uri)
    trained_key = store_key(store, result.trained_predictions_uri)
    with store.open(baseline_key) as source:
        baseline = json.loads(source.readline())
    with store.open(trained_key) as source:
        trained = json.loads(source.readline())
    assert baseline["prediction_type"] == "baseline"
    assert baseline["adapter_id"] is None
    assert trained["prediction_type"] == "trained"
    assert trained["adapter_id"] == publisher.ids.adapter_id


def test_prediction_jsonl_uses_streamed_file_publication(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)

    class StreamOnlyStore:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped
            self.jsonl_files = 0

        def put_bytes(self, key, content, **kwargs):
            if key.endswith(".jsonl"):
                raise AssertionError("prediction JSONL must not use put_bytes")
            return self.wrapped.put_bytes(key, content, **kwargs)

        def put_file(self, key, source, **kwargs):
            if key.endswith(".jsonl"):
                self.jsonl_files += 1
            return self.wrapped.put_file(key, source, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    wrapped = StreamOnlyStore(publisher.artifact_store)
    publisher.artifact_store = wrapped
    _complete(publisher, _adapter_staging(tmp_path))
    assert wrapped.jsonl_files == 2


def test_terminal_manifest_is_written_last(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    writes: list[str] = []
    publisher.artifact_store = RecordingStore(publisher.artifact_store, writes)
    _complete(publisher, _adapter_staging(tmp_path))
    assert writes[-1].endswith("/publication-manifest.json")


def test_report_failure_has_no_terminal_and_retains_staging(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    staging = _adapter_staging(tmp_path)
    publisher.artifact_store = FailingStore(
        publisher.artifact_store,
        suffix="training-report.json",
    )
    with pytest.raises(RuntimeError, match="simulated"):
        _complete(publisher, staging)
    assert staging.exists()
    assert not publisher.artifact_store.exists(
        f"{publisher.run_root}/publication-manifest.json"
    )


def test_adapter_failure_has_no_terminal_and_retains_staging(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    staging = _adapter_staging(tmp_path)
    publisher.model_store = FailingStore(
        publisher.model_store,
        suffix=publisher.adapter_root,
    )
    with pytest.raises(RuntimeError, match="simulated"):
        _complete(publisher, staging)
    assert staging.exists()
    assert not publisher.artifact_store.exists(
        f"{publisher.run_root}/publication-manifest.json"
    )


def test_existing_adapter_collision_never_overwrites(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    _complete(publisher, _adapter_staging(tmp_path, "first"), retain=True)
    stored = publisher.model_store.materialize(
        f"{publisher.adapter_root}/adapter_model.safetensors"
    )
    original = stored.read_bytes()
    second = _adapter_staging(tmp_path, "second")
    (second / "adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(ObjectAlreadyExistsError):
        _complete(publisher, second)
    assert stored.read_bytes() == original
    assert second.exists()


def test_retain_local_staging_after_success(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    staging = _adapter_staging(tmp_path)
    _complete(publisher, staging, retain=True)
    assert staging.exists()


def test_failure_record_is_not_a_completed_manifest(tmp_path: Path) -> None:
    publisher, _ = _publisher(tmp_path)
    failure_uri = publisher.publish_failure(
        RuntimeError("training failed"),
        phase="training",
    )
    assert failure_uri and failure_uri.endswith("/failure.json")
    assert not publisher.artifact_store.exists(
        f"{publisher.run_root}/publication-manifest.json"
    )


def test_prediction_rows_preserve_evidence_provenance(tmp_path: Path) -> None:
    ids, _ = _ids(tmp_path)
    rows = prediction_rows(
        {
            "outputs": [
                {
                    "record_id": "eval-1",
                    "prompt": "Question",
                    "expected": "Answer",
                    "generated": "Answer",
                    "exact_match": True,
                    "contains_expected": True,
                    "knowledge_unit_ids": ["ku-1"],
                    "evidence_ids": ["ev-1"],
                    "metadata": {
                        "knowledge_unit_id": "ku-singular",
                        "source_asset_ids": ["asset-1"],
                        "document_ids": ["doc-1"],
                        "probe_id": "probe-1",
                        "probe_class": "knowledge",
                        "recipe": "knowledge-unit-qa",
                    },
                    "provenance": {"source": "asset-1"},
                }
            ]
        },
        prediction_type="trained",
        ids=ids,
        base_model_identity=_base_model(),
        decoding={"do_sample": False},
    )
    assert rows[0]["knowledge_unit_id"] == "ku-singular"
    assert rows[0]["knowledge_unit_ids"] == ["ku-singular", "ku-1"]
    assert rows[0]["evidence_ids"] == ["ev-1"]
    assert rows[0]["source_asset_ids"] == ["asset-1"]
    assert rows[0]["document_ids"] == ["doc-1"]
    assert rows[0]["probe_id"] == "probe-1"
    assert rows[0]["adapter_id"] == ids.adapter_id


def test_adapter_verification_does_not_materialize(tmp_path: Path) -> None:
    publisher, identity = _publisher(tmp_path)
    _prepare(publisher, identity)
    result = _complete(publisher, _adapter_staging(tmp_path))
    original_for_profile = publisher.storage_runtime.for_profile

    class OpenOnlyStore:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped

        def materialize(self, key):
            raise AssertionError(f"materialize must not be called: {key}")

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    publisher.storage_runtime.for_profile = lambda *args, **kwargs: OpenOnlyStore(
        original_for_profile(*args, **kwargs)
    )
    assert verify_published_adapter(
        result.adapter_manifest_uri,
        storage_runtime=publisher.storage_runtime,
    ).valid


class RecordingStore:
    def __init__(self, wrapped, writes: list[str]) -> None:
        self.wrapped = wrapped
        self.writes = writes

    def put_json_idempotent(self, key, value):
        self.writes.append(key)
        return self.wrapped.put_json_idempotent(key, value)

    def put_bytes(self, key, content, **kwargs):
        self.writes.append(key)
        return self.wrapped.put_bytes(key, content, **kwargs)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class FailingStore:
    def __init__(self, wrapped, *, suffix: str) -> None:
        self.wrapped = wrapped
        self.suffix = suffix

    def put_json_idempotent(self, key, value):
        if key.endswith(self.suffix):
            raise RuntimeError("simulated publication failure")
        return self.wrapped.put_json_idempotent(key, value)

    def put_directory(self, key, source):
        if key.endswith(self.suffix):
            raise RuntimeError("simulated publication failure")
        return self.wrapped.put_directory(key, source)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def store_key(store, uri: str) -> str:
    key = uri.split("/", 3)[-1]
    namespace = store.namespace.strip("/")
    if namespace and key.startswith(namespace + "/"):
        key = key[len(namespace) + 1 :]
    return key
