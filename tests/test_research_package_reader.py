from __future__ import annotations

import json
from pathlib import Path

import pytest
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_training.dataset_pipeline import (
    DataForgeDatasetReader,
    dataforge_checksum,
    dataforge_manifest_checksum,
)


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _package(tmp_path: Path, *, trainable_evaluation: bool = False) -> tuple[StorageRuntime, str]:
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    store = runtime.for_role("dataset")
    dataset_rows = [
        {"record_id": "train-1", "messages": [{"role": "user", "content": "Train Q"}, {"role": "assistant", "content": "Train A"}], "split": "train", "metadata": {"research_role": "training", "training_eligible": True}},
        {"record_id": "validation-1", "messages": [{"role": "user", "content": "Validation Q"}, {"role": "assistant", "content": "Validation A"}], "split": "validation", "metadata": {}},
        {"record_id": "test-1", "messages": [{"role": "user", "content": "Test Q"}, {"role": "assistant", "content": "Test A"}], "split": "test", "metadata": {}},
    ]
    dataset_raw = _jsonl(dataset_rows)
    dataset_records_uri = store.put_bytes("research/dataset/records.jsonl", dataset_raw, media_type="application/x-ndjson").uri
    dataset_manifest = {
        "schema_version": "cognityx.dataforge.dataset/v1",
        "dataset_id": "dataset-1",
        "dataset_version": "version-1",
        "dataset_name": "research-fixture",
        "recipe": "paragraph-qa-qualified",
        "accepted_count": 3,
        "candidate_count": 0,
        "train_count": 1,
        "validation_count": 1,
        "test_count": 1,
        "eval_count": 2,
        "records_uri": dataset_records_uri,
        "records_checksum": dataforge_checksum(dataset_raw.decode()),
    }
    dataset_manifest_uri = store.put_json("research/dataset/manifest.json", dataset_manifest).uri

    exact_rows = [{
        "record_id": "exact-1",
        "source_record_id": "train-1",
        "messages": dataset_rows[0]["messages"],
        "split": "evaluation",
        "research_role": "exact_recall",
        "training_eligible": False,
    }]
    paraphrase_rows = [{
        "record_id": "paraphrase-1",
        "question": "Paraphrased question",
        "gold_reference": "Paraphrased answer",
        "split": "evaluation",
        "research_role": "paraphrase_evaluation",
        "training_eligible": trainable_evaluation,
        "fact_group_id": "fact-1",
        "record_provenance": {"evidence_ids": ["evidence-1"]},
    }]
    evaluation_refs = []
    for role, rows in (("exact_recall", exact_rows), ("paraphrase_evaluation", paraphrase_rows)):
        raw = _jsonl(rows)
        records_uri = store.put_bytes(f"research/{role}/records.jsonl", raw, media_type="application/x-ndjson").uri
        manifest = {
            "schema": "cognityx.dataforge.evaluation-set/v1",
            "evaluation_set_id": f"{role}-id",
            "evaluation_set_version": "1",
            "research_role": role,
            "training_eligible": False,
            "record_count": len(rows),
            "records_uri": records_uri,
            "records_checksum": dataforge_checksum(raw.decode()),
            "freeze_checksum": f"freeze-{role}",
        }
        manifest["manifest_checksum"] = dataforge_manifest_checksum(manifest)
        uri = store.put_json(f"research/{role}/manifest.json", manifest).uri
        evaluation_refs.append({
            "manifest_uri": uri,
            "evaluation_set_id": manifest["evaluation_set_id"],
            "evaluation_set_version": manifest["evaluation_set_version"],
            "research_role": role,
            "records_checksum": manifest["records_checksum"],
            "record_count": manifest["record_count"],
            "freeze_checksum": manifest["freeze_checksum"],
            "manifest_checksum": manifest["manifest_checksum"],
        })
    package = {
        "schema": "cognityx.dataforge.research-package/v1",
        "research_package_id": "package-1",
        "research_package_version": "version-1",
        "dataset": {
            "manifest_uri": dataset_manifest_uri,
            "records_checksum": dataset_manifest["records_checksum"],
        },
        "evaluation_sets": evaluation_refs,
    }
    package["manifest_checksum"] = dataforge_manifest_checksum(package)
    return runtime, store.put_json("research/package/manifest.json", package).uri


def test_research_package_optimizer_and_evaluation_roles_are_disjoint(tmp_path: Path):
    runtime, package_uri = _package(tmp_path)
    reader = DataForgeDatasetReader(package_uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    training = list(reader.iter_training_records())
    evaluation = list(reader.iter_evaluation_records())
    assert [record.record_id for record in training] == ["train-1"]
    assert {record.record_id for record in evaluation} == {
        "validation-1", "test-1", "exact-1", "paraphrase-1",
    }
    roles = {record.record_id: record.metadata["research_role"] for record in evaluation}
    assert roles["validation-1"] == "legacy_validation"
    assert roles["test-1"] == "legacy_test"
    assert roles["exact-1"] == "exact_recall"
    assert roles["paraphrase-1"] == "paraphrase_evaluation"
    assert next(record for record in evaluation if record.record_id == "validation-1").metadata["original_split"] == "validation"

    lineage = reader.lineage()
    assert lineage.research_package_manifest_uri == package_uri
    assert lineage.research_package_id == "package-1"
    assert lineage.research_package_version == "version-1"
    assert lineage.research_package_manifest_checksum
    assert {item["research_role"] for item in lineage.evaluation_sets} == {
        "exact_recall", "paraphrase_evaluation",
    }
    assert all(item["evaluation_set_id"] for item in lineage.evaluation_sets)
    assert all(item["evaluation_set_version"] for item in lineage.evaluation_sets)
    assert all(item["manifest_uri"] for item in lineage.evaluation_sets)
    assert all(item["manifest_checksum"] for item in lineage.evaluation_sets)
    assert all(item["records_checksum"] for item in lineage.evaluation_sets)
    assert all(item["freeze_checksum"] for item in lineage.evaluation_sets)
    assert lineage.record_counts["candidates"] == 0
    assert "rejected" not in lineage.record_counts

    statistics = reader.statistics()
    assert statistics.training_records == 1
    assert statistics.evaluation_records == 4
    assert statistics.evaluation_suite_counts == {
        "exact_recall": 1,
        "legacy_test": 1,
        "legacy_validation": 1,
        "paraphrase_evaluation": 1,
    }


def test_research_package_rejects_trainable_evaluation_record(tmp_path: Path):
    runtime, package_uri = _package(tmp_path, trainable_evaluation=True)
    reader = DataForgeDatasetReader(package_uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    with pytest.raises(ValueError, match="is trainable"):
        list(reader.iter_records())


def test_research_package_rejects_changed_manifest(tmp_path: Path):
    runtime, package_uri = _package(tmp_path)
    store = runtime.for_role("dataset")
    key = package_uri.removeprefix("storage://local-main/datasets/")
    with store.open(key) as source:
        package = json.load(source)
    package["research_package_version"] = "tampered"
    store.native_path(key).write_text(json.dumps(package), encoding="utf-8")

    reader = DataForgeDatasetReader(
        package_uri,
        storage_runtime=runtime,
        input_mode="dataforge_manifest",
    )
    with pytest.raises(ValueError, match="manifest checksum verification failed"):
        reader.manifest()
