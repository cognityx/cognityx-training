"""Dataset resolution, validation, and token masking for training."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse


SUPPORTED_INPUT_MODES = {"auto", "dataforge_manifest", "legacy_jsonl"}


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    dataset_id: str | None
    dataset_version: str | None
    dataset_variant_id: str | None
    dataset_manifest_uri: str | None
    dataset_manifest_checksum: str | None
    records_uri: str | None
    records_checksum: str | None
    recipe: str | None
    source_manifest_uri: str | None
    source_manifest_checksum: str | None
    split_summary: dict[str, int] = field(default_factory=dict)
    record_counts: dict[str, int] = field(default_factory=dict)
    overlength_policy: str = "error"
    skipped_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    training_records: int
    evaluation_records: int
    selected_training_records: int
    skipped_records: int
    maximum_observed_token_length: int | None
    configured_max_sequence_length: int | None
    split_summary: dict[str, int]
    validation_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetPreflightResult:
    lineage: DatasetLineage
    statistics: DatasetStatistics
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    record_id: str | None
    messages: tuple[dict[str, str], ...]
    split: str
    metadata: dict[str, Any]
    source_uri: str | None = None
    line_number: int | None = None


def stable_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _json_object_at_line(text: str, *, source_uri: str | None, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {line_number} in {source_uri or 'dataset'}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source_uri or 'dataset'}:{line_number} must contain a JSON object")
    return value


def _open_text_stream(store: Any, key: str):
    handle = store.open(key)
    return handle


def _unscope_key(store: Any, key: str) -> str:
    namespace = getattr(store, "namespace", "").strip("/")
    if namespace and key.startswith(namespace + "/"):
        return key[len(namespace) + 1 :]
    return key


def resolve_dataset_source(
    dataset_uri: str,
    *,
    storage_runtime: Any | None = None,
    storage_config: str | Path | None = None,
    storage_root: str | Path | None = None,
    input_mode: str = "auto",
) -> tuple[str, Any | None, dict[str, Any] | None]:
    mode = input_mode or "auto"
    if mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(f"Unsupported dataset input mode: {mode}")
    parsed = urlparse(dataset_uri)
    if mode == "auto":
        mode = "dataforge_manifest" if parsed.scheme == "storage" else "legacy_jsonl"
    if mode == "legacy_jsonl":
        if parsed.scheme not in ("", "file"):
            raise ValueError(f"Legacy JSONL mode only supports local files, got: {dataset_uri}")
        return "legacy_jsonl", Path(parsed.path if parsed.scheme else dataset_uri).expanduser(), None
    runtime = storage_runtime
    if runtime is None:
        from cognityx_storage import StorageConfig, StorageRuntime

        runtime = (
            StorageRuntime.load(config_file=storage_config)
            if storage_config
            else StorageRuntime.from_config(
                StorageConfig.built_in(root=storage_root or "/tmp/cognityx-training-storage")
            )
        )
    if parsed.scheme != "storage":
        raise ValueError(f"DataForge manifest mode requires a storage:// URI, got: {dataset_uri}")
    profile = parsed.netloc
    key = parsed.path.lstrip("/")
    store = runtime.for_profile(profile, role_name="dataset")
    return "dataforge_manifest", store, {"key": key, "runtime": runtime}


def _normalize_messages(record: dict[str, Any], *, source_uri: str | None, line_number: int) -> tuple[dict[str, str], ...]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                raise ValueError(f"{source_uri or 'dataset'}:{line_number} message {index} must be an object")
            role = str(message.get("role", "")).strip()
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"{source_uri or 'dataset'}:{line_number} has unsupported role '{role}'"
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"{source_uri or 'dataset'}:{line_number} message {index} must have non-empty string content"
                )
            normalized.append({"role": role, "content": content})
        if not any(item["role"] == "assistant" for item in normalized):
            raise ValueError(f"{source_uri or 'dataset'}:{line_number} requires at least one assistant message")
        return tuple(normalized)
    instruction = record.get("instruction") or record.get("prompt")
    output = record.get("output") or record.get("response")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{source_uri or 'dataset'}:{line_number} requires instruction/prompt text")
    if not isinstance(output, str) or not output.strip():
        raise ValueError(f"{source_uri or 'dataset'}:{line_number} requires output/response text")
    return (
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    )


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    for key in ("record_id", "knowledge_unit_id", "evidence_ids", "source_asset_ids", "document_ids", "recipe", "split"):
        if key in record and key not in metadata:
            metadata[key] = record[key]
    return metadata


def normalize_record(record: dict[str, Any], *, source_uri: str | None = None, line_number: int | None = None) -> NormalizedRecord:
    split = str(record.get("split", "train")).lower()
    if split in {"eval", "evaluation"}:
        split = "evaluation"
    elif split != "train":
        split = "train"
    return NormalizedRecord(
        record_id=str(record.get("record_id")) if record.get("record_id") is not None else None,
        messages=_normalize_messages(record, source_uri=source_uri, line_number=line_number or 0),
        split=split,
        metadata=_record_metadata(record),
        source_uri=source_uri,
        line_number=line_number,
    )


def messages_for_record(record: dict[str, Any]) -> list[dict[str, str]]:
    return list(_normalize_messages(record, source_uri=None, line_number=0))


def legacy_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            records.append(_json_object_at_line(line, source_uri=str(path), line_number=line_number))
    if not records:
        raise ValueError(f"Dataset contains no examples: {path}")
    return records


def _stream_jsonl_from_store(store: Any, key: str):
    with _open_text_stream(store, key) as handle:
        source_uri = store.uri(key) if hasattr(store, "uri") else None
        for line_number, raw in enumerate(handle, start=1):
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if text.strip():
                yield line_number, _json_object_at_line(text, source_uri=source_uri, line_number=line_number)


def load_manifest(store: Any, key: str) -> tuple[dict[str, Any], str]:
    with _open_text_stream(store, key) as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    manifest_checksum = stable_checksum(manifest)
    return manifest, manifest_checksum


def _lineage_from_manifest(manifest_uri: str, manifest: dict[str, Any], manifest_checksum: str) -> DatasetLineage:
    return DatasetLineage(
        dataset_id=manifest.get("dataset_id"),
        dataset_version=manifest.get("dataset_version"),
        dataset_variant_id=manifest.get("dataset_variant_id"),
        dataset_manifest_uri=manifest_uri,
        dataset_manifest_checksum=manifest_checksum,
        records_uri=manifest.get("records_uri"),
        records_checksum=manifest.get("records_checksum"),
        recipe=manifest.get("recipe"),
        source_manifest_uri=manifest.get("source_manifest_uri"),
        source_manifest_checksum=manifest.get("source_manifest_checksum"),
        split_summary=dict(manifest.get("split_summary", {})),
        record_counts=dict(manifest.get("record_counts", {})),
        overlength_policy=str(manifest.get("overlength_policy", "error")),
        skipped_records=list(manifest.get("skipped_records", [])),
    )


class DataForgeDatasetReader:
    def __init__(
        self,
        dataset_uri: str,
        *,
        storage_runtime: Any | None = None,
        storage_config: str | Path | None = None,
        storage_root: str | Path | None = None,
        input_mode: str = "auto",
    ) -> None:
        self.dataset_uri = dataset_uri
        self.mode, self._source, self._context = resolve_dataset_source(
            dataset_uri,
            storage_runtime=storage_runtime,
            storage_config=storage_config,
            storage_root=storage_root,
            input_mode=input_mode,
        )
        self._manifest: dict[str, Any] | None = None
        self._manifest_checksum: str | None = None

    def _load_manifest(self) -> tuple[dict[str, Any], str]:
        if self.mode != "dataforge_manifest":
            raise ValueError("Legacy JSONL datasets do not have a DataForge manifest")
        assert self._source is not None and self._context is not None
        key = _unscope_key(self._source, self._context["key"])
        manifest, manifest_checksum = load_manifest(self._source, key)
        if manifest.get("records_uri") is None or manifest.get("records_checksum") is None:
            raise ValueError(f"Malformed DataForge manifest at {self.dataset_uri}")
        if manifest.get("dataset_manifest_uri") and manifest["dataset_manifest_uri"] != self.dataset_uri:
            manifest = dict(manifest)
            manifest["dataset_manifest_uri"] = self.dataset_uri
        self._manifest = manifest
        self._manifest_checksum = manifest_checksum
        return manifest, manifest_checksum

    def lineage(self) -> DatasetLineage:
        if self.mode == "legacy_jsonl":
            return DatasetLineage(None, None, None, self.dataset_uri, None, self.dataset_uri, None, None, None, None)
        manifest, manifest_checksum = self._manifest or self._load_manifest()
        return _lineage_from_manifest(self.dataset_uri, manifest, manifest_checksum)

    def iter_records(self):
        if self.mode == "legacy_jsonl":
            assert isinstance(self._source, Path)
            for line_number, record in enumerate(legacy_jsonl_records(self._source), start=1):
                yield normalize_record(record, source_uri=str(self._source), line_number=line_number)
            return
        assert self._source is not None
        manifest = self._manifest or self._load_manifest()[0]
        records_uri = manifest["records_uri"]
        parsed = urlparse(records_uri)
        if parsed.scheme != "storage":
            raise ValueError(f"Records URI must be storage://, got: {records_uri}")
        store = self._context["runtime"].for_profile(parsed.netloc, role_name="dataset")
        key = _unscope_key(store, parsed.path.lstrip("/"))
        with store.open(key) as handle:
            raw = handle.read()
        if stable_checksum(raw.decode("utf-8")) != manifest["records_checksum"]:
            raise ValueError(f"Records checksum mismatch for {records_uri}")
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            yield normalize_record(
                _json_object_at_line(line, source_uri=records_uri, line_number=line_number),
                source_uri=records_uri,
                line_number=line_number,
            )

    def iter_training_records(self, *, max_examples: int | None = None):
        count = 0
        for record in self.iter_records():
            if record.split == "evaluation":
                continue
            if max_examples is not None and count >= max_examples:
                break
            count += 1
            yield record

    def iter_evaluation_records(self):
        for record in self.iter_records():
            if record.split == "evaluation":
                yield record

    def statistics(self, *, max_examples: int | None = None) -> DatasetStatistics:
        training = evaluation = selected = skipped = 0
        split_summary: dict[str, int] = {}
        max_tokens: int | None = None
        for record in self.iter_records():
            split_summary[record.split] = split_summary.get(record.split, 0) + 1
            if record.split == "evaluation":
                evaluation += 1
            else:
                training += 1
        selected = min(training, max_examples) if max_examples is not None else training
        lineage = self.lineage()
        return DatasetStatistics(training, evaluation, selected, skipped, max_tokens, None, split_summary, ())


def _assistant_token_mask(tokenizer: Any, messages: Sequence[dict[str, str]]) -> tuple[list[int], list[int]]:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = list(rendered["input_ids"])
        mask = list(rendered["assistant_tokens_mask"])
        return input_ids, mask
    except Exception:
        input_ids: list[int] = []
        labels: list[int] = []
        for message in messages:
            encoded = tokenizer(
                message["content"],
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            if message["role"] == "assistant":
                input_ids.extend(encoded)
                labels.extend(encoded)
            else:
                input_ids.extend(encoded)
                labels.extend([-100] * len(encoded))
        return input_ids, labels


def encode_supervised_example(tokenizer: Any, messages: Sequence[dict[str, str]], *, max_sequence_length: int | None = None) -> dict[str, list[int]]:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = list(rendered["input_ids"])
        mask = rendered.get("assistant_tokens_mask") or rendered.get("assistant_mask")
        if mask is None:
            raise KeyError
        labels = [token if bool(flag) else -100 for token, flag in zip(input_ids, mask)]
    except Exception:
        input_ids, labels = _assistant_token_mask(tokenizer, messages)
    if max_sequence_length is not None and len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Example exceeds maximum sequence length: {len(input_ids)} > {max_sequence_length}"
        )
    return {"input_ids": input_ids, "labels": labels}


def collate_supervised_batch(
    batch: Sequence[dict[str, list[int]]],
    *,
    pad_token_id: int,
    max_sequence_length: int | None = None,
):
    import torch

    longest = max(len(item["input_ids"]) for item in batch)
    if max_sequence_length is not None:
        longest = min(longest, max_sequence_length)
    input_ids = []
    attention_masks = []
    labels = []
    for item in batch:
        ids = item["input_ids"][:longest]
        lab = item["labels"][:longest]
        pad = longest - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_masks.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
