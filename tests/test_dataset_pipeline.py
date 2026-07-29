from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cognityx_storage import StorageConfig, StorageRuntime
from cognityx_training.dataset_pipeline import (
    DataForgeDatasetReader,
    collate_supervised_batch,
    dataforge_checksum,
    encode_supervised_example,
    legacy_jsonl_records,
    normalize_checksum,
    preflight_dataset,
)


class MaskTokenizer:
    pad_token_id = 0

    def __init__(self, mask: list[int] | None = None, template_mode: str = "mask") -> None:
        self.mask = mask
        self.template_mode = template_mode

    def apply_chat_template(self, messages, **kwargs):
        ids = []
        mask = []
        for message in messages:
            token_ids = [ord(ch) % 13 + 1 for ch in message["content"]]
            ids.extend(token_ids)
            if self.template_mode == "prefix":
                mask.extend([1 if message["role"] == "assistant" else 0] * len(token_ids))
            else:
                mask.extend([1 if message["role"] == "assistant" else 0] * len(token_ids))
        if self.mask is not None:
            mask = list(self.mask)
        if kwargs.get("return_dict"):
            result = {"input_ids": ids}
            if kwargs.get("return_assistant_tokens_mask"):
                result["assistant_tokens_mask"] = mask
            return result
        return ids


class PrefixTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        text = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        ids = [ord(ch) % 17 + 1 for ch in text]
        if kwargs.get("return_dict"):
            return {"input_ids": ids}
        return ids


def _storage_runtime(tmp_path: Path) -> StorageRuntime:
    return StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path))


def test_dataforge_checksum_matches_real_algorithm() -> None:
    records_text = (
        '{"record_id":"r1","messages":[{"role":"user","content":"Hello Ω"},'
        '{"role":"assistant","content":"Hi \\"there\\"\\\\"}],"split":"train","metadata":{}}\n'
        '{"record_id":"r2","messages":[{"role":"user","content":"Line1\\tLine2"},'
        '{"role":"assistant","content":"CR\\rLF\\n"}],"split":"evaluation","metadata":{"note":"x"}}\n'
    )
    assert dataforge_checksum(records_text) == "262068295605a39e1f6dcf5eda645306f691daa83b0dbdd274e156f6023cb346"
    assert normalize_checksum("sha256:abc") == "abc"


def test_legacy_jsonl_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"instruction": "Say hello", "output": "Hello!"}) + "\n", encoding="utf-8")
    records = list(legacy_jsonl_records(path))
    assert records[0].messages[1]["content"] == "Hello!"


def test_assistant_masking_and_collation() -> None:
    encoded = encode_supervised_example(
        MaskTokenizer(),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    batch = collate_supervised_batch([encoded], pad_token_id=0)
    assert any(label != -100 for label in batch["labels"][0].tolist())
    assert batch["labels"][0].tolist()[:5].count(-100) >= 3


def test_prefix_fallback_masks_multi_turn_assistant() -> None:
    encoded = encode_supervised_example(
        PrefixTokenizer(),
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ],
    )
    labels = encoded["labels"]
    assert any(label != -100 for label in labels)
    assert labels.count(-100) < len(labels)


def test_all_zero_or_mismatched_masks_fail() -> None:
    with pytest.raises(ValueError, match="did not mark any target tokens"):
        encode_supervised_example(MaskTokenizer(mask=[0, 0, 0]), [{"role": "assistant", "content": "abc"}])

    class BadTokenizer(MaskTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if kwargs.get("return_dict"):
                result["assistant_tokens_mask"] = [1]
            return result

    with pytest.raises(ValueError, match="does not match token length"):
        encode_supervised_example(BadTokenizer(), [{"role": "assistant", "content": "abc"}])


def test_streaming_reader_and_preflight(tmp_path: Path) -> None:
    runtime = _storage_runtime(tmp_path)
    dataset_store = runtime.for_role("dataset")
    records = [
        {"record_id": "r1", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], "split": "train", "metadata": {"evidence_ids": ["e1"]}},
        {"record_id": "r2", "messages": [{"role": "user", "content": "v"}, {"role": "assistant", "content": "b"}], "split": "evaluation", "metadata": {"evidence_ids": ["e2"]}},
        {"record_id": "r3", "messages": [{"role": "user", "content": "w"}, {"role": "assistant", "content": "c"}], "split": "train", "metadata": {"evidence_ids": ["e3"]}},
    ]
    records_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in records
    )
    records_bytes = records_text.encode("utf-8")
    records_obj = dataset_store.put_bytes("demo/1/records.jsonl", records_bytes)
    manifest = {
        "dataset_id": "demo",
        "dataset_name": "demo-name",
        "dataset_version": "1",
        "schema_version": "cognityx.dataforge.dataset/v1",
        "recipe": "paragraph-qa",
        "source_manifest_uri": "storage://local-main/ingest/runs/r1/manifest.json",
        "source_manifest_checksum": "abc",
        "configuration_checksum": "cfg",
        "records_uri": records_obj.uri,
        "records_checksum": hashlib.sha256(
            json.dumps(records_text, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "accepted_count": 3,
        "train_count": 2,
        "eval_count": 1,
    }
    manifest_obj = dataset_store.put_json("demo/1/manifest.json", manifest)
    reader = DataForgeDatasetReader(manifest_obj.uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    assert [record.record_id for record in reader.iter_training_records()] == ["r1", "r3"]
    assert [record.record_id for record in reader.iter_evaluation_records()] == ["r2"]
    preflight = preflight_dataset(reader, MaskTokenizer(), max_examples=1, max_sequence_length=32, overlength_policy="error")
    assert preflight.statistics.total_records == 3
    assert preflight.statistics.accepted_training_examples == 1
    assert preflight.lineage.dataset_name == "demo-name"


def test_checksum_mismatch_fails_before_training(tmp_path: Path) -> None:
    runtime = _storage_runtime(tmp_path)
    dataset_store = runtime.for_role("dataset")
    records_text = '{"instruction":"a","output":"b"}\n'
    records_obj = dataset_store.put_bytes("demo/1/records.jsonl", records_text.encode("utf-8"))
    manifest_obj = dataset_store.put_json(
        "demo/1/manifest.json",
        {
            "dataset_id": "demo",
            "dataset_version": "1",
            "records_uri": records_obj.uri,
            "records_checksum": "deadbeef",
        },
    )
    reader = DataForgeDatasetReader(manifest_obj.uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(reader.iter_records())


def test_unknown_split_fails(tmp_path: Path) -> None:
    runtime = _storage_runtime(tmp_path)
    dataset_store = runtime.for_role("dataset")
    records_text = '{"record_id":"r1","messages":[{"role":"user","content":"u"},{"role":"assistant","content":"a"}],"split":"mystery"}\n'
    records_obj = dataset_store.put_bytes("demo/1/records.jsonl", records_text.encode("utf-8"))
    manifest_obj = dataset_store.put_json(
        "demo/1/manifest.json",
        {"dataset_id": "demo", "dataset_version": "1", "records_uri": records_obj.uri, "records_checksum": dataforge_checksum(records_text)},
    )
    reader = DataForgeDatasetReader(manifest_obj.uri, storage_runtime=runtime, input_mode="dataforge_manifest")
    with pytest.raises(ValueError, match="unsupported split"):
        list(reader.iter_records())
