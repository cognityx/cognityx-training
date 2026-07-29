from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognityx_storage import StorageConfig, StorageRuntime
from cognityx_training.dataset_pipeline import (
    DataForgeDatasetReader,
    collate_supervised_batch,
    encode_supervised_example,
    legacy_jsonl_records,
    messages_for_record,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(ch) % 10 + 1 for ch in text]}

    def apply_chat_template(self, messages, **kwargs):
        ids = []
        mask = []
        for message in messages:
            token_ids = [ord(ch) % 10 + 1 for ch in message["content"]]
            ids.extend(token_ids)
            mask.extend([1 if message["role"] == "assistant" else 0] * len(token_ids))
        if kwargs.get("return_dict"):
            return {"input_ids": ids, "assistant_tokens_mask": mask}
        return ids


def test_legacy_jsonl_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"instruction": "Say hello", "output": "Hello!"}) + "\n",
        encoding="utf-8",
    )

    records = legacy_jsonl_records(path)
    assert records[0]["instruction"] == "Say hello"
    assert messages_for_record(records[0])[1]["content"] == "Hello!"


def test_assistant_only_labels_and_padding() -> None:
    encoded = encode_supervised_example(
        FakeTokenizer(),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    batch = collate_supervised_batch([encoded], pad_token_id=0)
    labels = batch["labels"][0].tolist()
    assert any(label != -100 for label in labels)
    assert labels[:3] == [-100, -100, -100]


def test_dataforge_manifest_reader_streams_and_checksums(tmp_path: Path) -> None:
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path))
    dataset_store = runtime.for_role("dataset")
    records = [
        {"record_id": "r1", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], "split": "train", "metadata": {"evidence_ids": ["e1"]}},
        {"record_id": "r2", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}], "split": "evaluation", "metadata": {"evidence_ids": ["e2"]}},
    ]
    records_bytes = b"".join(json.dumps(row).encode() + b"\n" for row in records)
    records_obj = dataset_store.put_bytes("demo/1/records.jsonl", records_bytes)
    manifest = {
        "dataset_id": "demo",
        "dataset_version": "1",
        "records_uri": records_obj.uri,
        "records_checksum": "sha256:" + __import__("hashlib").sha256(json.dumps(records_bytes.decode("utf-8")).encode("utf-8")).hexdigest(),
        "recipe": "paragraph-qa",
        "source_manifest_uri": "storage://local-main/ingest/runs/r1/manifest.json",
        "source_manifest_checksum": "sha256:abc",
        "split_summary": {"train": 1, "evaluation": 1},
        "record_counts": {"train": 1, "evaluation": 1},
        "overlength_policy": "error",
    }
    manifest_obj = dataset_store.put_json("demo/1/manifest.json", manifest)

    reader = DataForgeDatasetReader(manifest_obj.uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    lineage = reader.lineage()
    assert lineage.dataset_id == "demo"
    assert list(reader.iter_training_records())[0].record_id == "r1"
    assert list(reader.iter_evaluation_records())[0].record_id == "r2"


def test_records_checksum_mismatch_fails(tmp_path: Path) -> None:
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path))
    dataset_store = runtime.for_role("dataset")
    records_obj = dataset_store.put_bytes("demo/1/records.jsonl", b'{"instruction":"a","output":"b"}\n')
    manifest_obj = dataset_store.put_json(
        "demo/1/manifest.json",
        {
            "dataset_id": "demo",
            "dataset_version": "1",
            "records_uri": records_obj.uri,
            "records_checksum": "sha256:bad",
        },
    )
    reader = DataForgeDatasetReader(manifest_obj.uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(reader.iter_records())
