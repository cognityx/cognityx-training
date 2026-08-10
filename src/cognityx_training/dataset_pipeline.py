"""Dataset resolution, validation, and streaming tokenization for training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urlparse

DATASET_INPUT_MODES = {"auto", "dataforge_manifest", "legacy_jsonl"}
SUPPORTED_SPLITS = {"train", "validation", "test", "eval", "evaluation"}
RESEARCH_PACKAGE_SCHEMA = "cognityx.dataforge.research-package/v1"
EVALUATION_SET_SCHEMA = "cognityx.dataforge.evaluation-set/v1"
DEFAULT_SKIPPED_SAMPLE_LIMIT = 10


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def dataforge_checksum(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def normalize_checksum(value: str | None) -> str | None:
    if value is None:
        return None
    return value.removeprefix("sha256:")


def incremental_dataforge_checksum(handle: Iterable[bytes | str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b'"')
    for chunk in handle:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        hasher.update(json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8"))
    hasher.update(b'"')
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    dataset_id: str | None
    dataset_name: str | None
    dataset_version: str | None
    dataset_variant_id: str | None
    dataset_manifest_uri: str | None
    dataset_manifest_checksum: str | None
    records_uri: str | None
    records_checksum: str | None
    recipe: str | None
    source_manifest_uri: str | None
    source_manifest_checksum: str | None
    configuration_checksum: str | None
    split_summary: dict[str, int] = field(default_factory=dict)
    record_counts: dict[str, int] = field(default_factory=dict)
    overlength_policy: str = "error"
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    research_package_manifest_uri: str | None = None
    research_package_manifest_checksum: str | None = None
    research_package_id: str | None = None
    research_package_version: str | None = None
    evaluation_sets: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    total_records: int
    training_records: int
    evaluation_records: int
    selected_training_candidates: int
    accepted_training_examples: int
    skipped_overlength_count: int
    maximum_observed_token_length: int | None
    maximum_accepted_token_length: int | None
    configured_max_sequence_length: int | None
    split_summary: dict[str, int]
    evaluation_suite_counts: dict[str, int] = field(default_factory=dict)
    skipped_record_samples: tuple[dict[str, Any], ...] = ()
    skipped_record_sample_limit: int = DEFAULT_SKIPPED_SAMPLE_LIMIT
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetPreflightResult:
    lineage: DatasetLineage
    statistics: DatasetStatistics
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    record_id: str | None
    messages: tuple[dict[str, str], ...]
    split: str
    metadata: dict[str, Any]
    line_number: int
    source_uri: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedTrainingExample:
    record_id: str | None
    line_number: int
    input_ids: list[int]
    labels: list[int]
    source_uri: str | None = None


def _json_object_at_line(text: str, *, source_uri: str | None, line_number: int) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{source_uri or 'dataset'}:{line_number} must contain a JSON object")
    return value


def _iter_lines(handle: Iterable[bytes | str]) -> Iterator[tuple[int, str]]:
    for line_number, raw in enumerate(handle, start=1):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if text.strip():
            yield line_number, text


def _normalize_split(split: Any, *, record_id: str | None, line_number: int, source_uri: str | None) -> str:
    value = "train" if split is None else str(split).strip().lower()
    if not value:
        value = "train"
    if value not in SUPPORTED_SPLITS:
        raise ValueError(
            f"{source_uri or 'dataset'}:{line_number} record {record_id or '?'} has unsupported split '{split}'"
        )
    return "evaluation" if value in {"validation", "test", "eval", "evaluation"} else "train"


def _normalize_messages(record: dict[str, Any], *, source_uri: str | None, line_number: int) -> tuple[dict[str, str], ...]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        assistant_count = 0
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
            assistant_count += int(role == "assistant")
            normalized.append({"role": role, "content": content})
        if assistant_count == 0:
            raise ValueError(f"{source_uri or 'dataset'}:{line_number} requires at least one assistant message")
        return tuple(normalized)
    instruction = record.get("instruction") or record.get("prompt") or record.get("question")
    output = record.get("output") or record.get("response") or record.get("gold_reference")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{source_uri or 'dataset'}:{line_number} requires instruction/prompt text")
    if not isinstance(output, str) or not output.strip():
        raise ValueError(f"{source_uri or 'dataset'}:{line_number} requires output/response text")
    return (
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    )


def normalize_record(record: dict[str, Any], *, source_uri: str | None, line_number: int) -> NormalizedRecord:
    record_id = record.get("record_id")
    raw_split = "train" if record.get("split") is None else str(record.get("split")).strip().lower()
    metadata = dict(record.get("metadata") or {})
    for name in (
        "research_role",
        "training_eligible",
        "evaluation_set_id",
        "evaluation_set_version",
        "source_record_id",
        "source_reference_id",
        "fact_group_id",
        "knowledge_unit_id",
        "variant_id",
        "record_provenance",
        "source_evidence",
    ):
        if name in record and name not in metadata:
            metadata[name] = record[name]
    if raw_split in {"validation", "test"}:
        metadata.setdefault("original_split", raw_split)
        metadata.setdefault("research_role", f"legacy_{raw_split}")
        metadata.setdefault("training_eligible", False)
    return NormalizedRecord(
        record_id=str(record_id) if record_id is not None else None,
        messages=_normalize_messages(record, source_uri=source_uri, line_number=line_number),
        split=_normalize_split(
            record.get("split"),
            record_id=str(record_id) if record_id is not None else None,
            line_number=line_number,
            source_uri=source_uri,
        ),
        metadata=metadata,
        line_number=line_number,
        source_uri=source_uri,
    )


def messages_for_record(record: dict[str, Any]) -> list[dict[str, str]]:
    return list(_normalize_messages(record, source_uri=None, line_number=0))


def iter_normalized_records(handle: Iterable[bytes | str], *, source_uri: str | None) -> Iterator[NormalizedRecord]:
    for line_number, text in _iter_lines(handle):
        yield normalize_record(
            _json_object_at_line(text, source_uri=source_uri, line_number=line_number),
            source_uri=source_uri,
            line_number=line_number,
        )


def legacy_jsonl_records(path: Path) -> Iterator[NormalizedRecord]:
    with path.open(encoding="utf-8") as source:
        yield from iter_normalized_records(source, source_uri=str(path))


def resolve_dataset_source(
    dataset_uri: str,
    *,
    storage_runtime: Any | None = None,
    storage_config: str | Path | None = None,
    storage_root: str | Path | None = None,
    input_mode: str = "auto",
) -> tuple[str, Any, dict[str, Any] | None]:
    if input_mode not in DATASET_INPUT_MODES:
        raise ValueError(f"Unsupported dataset input mode: {input_mode}")
    parsed = urlparse(dataset_uri)
    mode = input_mode if input_mode != "auto" else ("dataforge_manifest" if parsed.scheme == "storage" else "legacy_jsonl")
    if mode == "legacy_jsonl":
        if parsed.scheme not in {"", "file"}:
            raise ValueError(f"Legacy JSONL mode only supports local files, got: {dataset_uri}")
        return mode, Path(parsed.path if parsed.scheme else dataset_uri).expanduser(), None
    if parsed.scheme != "storage":
        raise ValueError(f"DataForge manifest mode requires a storage:// URI, got: {dataset_uri}")
    from cognityx_training.storage_runtime import resolve_storage_runtime

    runtime = resolve_storage_runtime(
        storage_runtime=storage_runtime,
        storage_config=storage_config,
        storage_root=storage_root,
    )
    store = runtime.for_profile(parsed.netloc, role_name="dataset")
    return mode, store, {"runtime": runtime, "key": parsed.path.lstrip("/")}


def _split_manifest_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    split_summary = dict(manifest.get("split_summary") or {})
    if not split_summary:
        for name, field_name in (
            ("train", "train_count"),
            ("validation", "validation_count"),
            ("test", "test_count"),
        ):
            if manifest.get(field_name) is not None:
                split_summary[name] = int(manifest[field_name])
        if manifest.get("eval_count") is not None:
            split_summary["evaluation_total"] = int(manifest["eval_count"])
    record_counts = dict(manifest.get("record_counts") or {})
    if not record_counts:
        for name, field_name in (
            ("accepted", "accepted_count"),
            ("candidates", "candidate_count"),
            ("rejected", "rejected_count"),
            ("needs_review", "needs_review_count"),
            ("duplicates", "duplicate_count"),
        ):
            if manifest.get(field_name) is not None:
                record_counts[name] = int(manifest[field_name])
    return {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("dataset_version"),
        "dataset_variant_id": manifest.get("dataset_variant_id"),
        "dataset_manifest_uri": manifest.get("dataset_manifest_uri"),
        "dataset_manifest_checksum": normalize_checksum(manifest.get("dataset_manifest_checksum")),
        "records_uri": manifest.get("records_uri"),
        "records_checksum": normalize_checksum(manifest.get("records_checksum")),
        "recipe": manifest.get("recipe"),
        "source_manifest_uri": manifest.get("source_manifest_uri"),
        "source_manifest_checksum": normalize_checksum(manifest.get("source_manifest_checksum")),
        "configuration_checksum": normalize_checksum(manifest.get("configuration_checksum")),
        "split_summary": split_summary,
        "record_counts": record_counts,
        "overlength_policy": str(manifest.get("overlength_policy", "error")),
        "skipped_records": list(manifest.get("skipped_records") or []),
    }


def _load_json_object(store: Any, key: str) -> dict[str, Any]:
    namespace = getattr(store, "namespace", "").strip("/")
    if namespace and key.startswith(namespace + "/"):
        key = key[len(namespace) + 1 :]
    with store.open(key) as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    return manifest


def _records_key_from_uri(records_uri: str, runtime: Any) -> tuple[Any, str]:
    parsed = urlparse(records_uri)
    if parsed.scheme != "storage":
        raise ValueError(f"Records URI must be storage://, got: {records_uri}")
    store = runtime.for_profile(parsed.netloc, role_name="dataset")
    key = parsed.path.lstrip("/")
    namespace = getattr(store, "namespace", "").strip("/")
    if namespace and key.startswith(namespace + "/"):
        key = key[len(namespace) + 1 :]
    return store, key


def _json_from_uri(uri: str, runtime: Any) -> tuple[dict[str, Any], str]:
    store, key = _records_key_from_uri(uri, runtime)
    value = _load_json_object(store, key)
    return value, dataforge_checksum(value)


def _stream_records(store: Any, key: str, *, source_uri: str | None) -> Iterator[NormalizedRecord]:
    with store.open(key) as handle:
        for line_number, text in _iter_lines(handle):
            yield normalize_record(
                _json_object_at_line(text, source_uri=source_uri, line_number=line_number),
                source_uri=source_uri,
                line_number=line_number,
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
        self._research_package: dict[str, Any] | None = None
        self._research_package_checksum: str | None = None
        self._evaluation_manifests: list[tuple[str, dict[str, Any]]] = []

    def _load_manifest(self) -> tuple[dict[str, Any], str]:
        if self.mode != "dataforge_manifest":
            raise ValueError("Legacy JSONL datasets do not have a DataForge manifest")
        assert self._source is not None and self._context is not None
        selected = _load_json_object(self._source, self._context["key"])
        selected_checksum = dataforge_checksum(selected)
        if selected.get("schema") == RESEARCH_PACKAGE_SCHEMA:
            runtime = self._context["runtime"]
            dataset_ref = selected.get("dataset")
            if not isinstance(dataset_ref, dict) or not dataset_ref.get("manifest_uri"):
                raise ValueError("Research package requires one dataset manifest reference")
            manifest, checksum = _json_from_uri(str(dataset_ref["manifest_uri"]), runtime)
            if manifest.get("schema_version") != "cognityx.dataforge.dataset/v1":
                raise ValueError("Research package dataset must use cognityx.dataforge.dataset/v1")
            if manifest.get("records_checksum") != dataset_ref.get("records_checksum"):
                raise ValueError("Research package dataset records checksum does not match its dataset manifest")
            evaluation_manifests: list[tuple[str, dict[str, Any]]] = []
            roles: set[str] = set()
            for item in selected.get("evaluation_sets", []):
                if not isinstance(item, dict) or not item.get("manifest_uri"):
                    raise ValueError("Research package contains a malformed evaluation-set reference")
                uri = str(item["manifest_uri"])
                evaluation_manifest, _ = _json_from_uri(uri, runtime)
                if evaluation_manifest.get("schema") != EVALUATION_SET_SCHEMA:
                    raise ValueError(f"Unsupported evaluation-set schema at {uri}")
                role = str(evaluation_manifest.get("research_role"))
                if role in roles:
                    raise ValueError(f"Research package repeats evaluation role: {role}")
                roles.add(role)
                if evaluation_manifest.get("training_eligible") is not False:
                    raise ValueError(f"Evaluation set {role} is marked trainable")
                if evaluation_manifest.get("records_checksum") != item.get("records_checksum"):
                    raise ValueError(f"Research package checksum mismatch for evaluation role {role}")
                evaluation_manifests.append((uri, evaluation_manifest))
            if "exact_recall" not in roles:
                raise ValueError("Research package requires an exact_recall evaluation set")
            self._research_package = selected
            self._research_package_checksum = selected_checksum
            self._evaluation_manifests = evaluation_manifests
        else:
            manifest = selected
            checksum = selected_checksum
        self._manifest = manifest
        self._manifest_checksum = checksum
        return manifest, checksum

    def manifest(self) -> dict[str, Any]:
        return self._manifest or self._load_manifest()[0]

    def lineage(self) -> DatasetLineage:
        if self.mode == "legacy_jsonl":
            return DatasetLineage(
                dataset_id=None,
                dataset_name=None,
                dataset_version=None,
                dataset_variant_id=None,
                dataset_manifest_uri=self.dataset_uri,
                dataset_manifest_checksum=None,
                records_uri=self.dataset_uri,
                records_checksum=None,
                recipe=None,
                source_manifest_uri=None,
                source_manifest_checksum=None,
                configuration_checksum=None,
            )
        manifest = self.manifest()
        fields = _split_manifest_fields(manifest)
        package_dataset = self._research_package.get("dataset", {}) if self._research_package else {}
        manifest_uri = fields["dataset_manifest_uri"] or package_dataset.get("manifest_uri") or self.dataset_uri
        return DatasetLineage(
            dataset_id=fields["dataset_id"],
            dataset_name=fields["dataset_name"],
            dataset_version=fields["dataset_version"],
            dataset_variant_id=fields["dataset_variant_id"],
            dataset_manifest_uri=manifest_uri,
            dataset_manifest_checksum=self._manifest_checksum or dataforge_checksum(manifest),
            records_uri=fields["records_uri"],
            records_checksum=fields["records_checksum"],
            recipe=fields["recipe"],
            source_manifest_uri=fields["source_manifest_uri"],
            source_manifest_checksum=fields["source_manifest_checksum"],
            configuration_checksum=fields["configuration_checksum"],
            split_summary=fields["split_summary"],
            record_counts=fields["record_counts"],
            overlength_policy=fields["overlength_policy"],
            skipped_records=fields["skipped_records"],
            research_package_manifest_uri=self.dataset_uri if self._research_package else None,
            research_package_manifest_checksum=self._research_package_checksum,
            research_package_id=(self._research_package or {}).get("research_package_id"),
            research_package_version=(self._research_package or {}).get("research_package_version"),
            evaluation_sets=tuple(
                {
                    "manifest_uri": uri,
                    "evaluation_set_id": item.get("evaluation_set_id"),
                    "evaluation_set_version": item.get("evaluation_set_version"),
                    "research_role": item.get("research_role"),
                    "records_uri": item.get("records_uri"),
                    "records_checksum": item.get("records_checksum"),
                    "record_count": item.get("record_count"),
                }
                for uri, item in self._evaluation_manifests
            ),
        )

    def iter_records(self) -> Iterator[NormalizedRecord]:
        if self.mode == "legacy_jsonl":
            assert isinstance(self._source, Path)
            yield from legacy_jsonl_records(self._source)
            return
        manifest = self.manifest()
        assert self._context is not None
        records_uri = manifest["records_uri"]
        records_checksum = normalize_checksum(manifest.get("records_checksum"))
        if records_checksum is None:
            raise ValueError(f"Malformed DataForge manifest at {self.dataset_uri}")
        runtime = self._context["runtime"]
        store, key = _records_key_from_uri(records_uri, runtime)
        with store.open(key) as handle:
            checksum = incremental_dataforge_checksum(handle)
        if checksum != records_checksum:
            raise ValueError(f"Records checksum mismatch for {records_uri}")
        seen_record_ids: set[str] = set()
        for record in _stream_records(store, key, source_uri=records_uri):
            if record.record_id is not None:
                if record.record_id in seen_record_ids:
                    raise ValueError(f"Duplicate record_id in research input: {record.record_id}")
                seen_record_ids.add(record.record_id)
            yield record
        for manifest_uri, evaluation_manifest in self._evaluation_manifests:
            evaluation_uri = str(evaluation_manifest["records_uri"])
            evaluation_checksum = normalize_checksum(evaluation_manifest.get("records_checksum"))
            evaluation_store, evaluation_key = _records_key_from_uri(evaluation_uri, runtime)
            with evaluation_store.open(evaluation_key) as handle:
                calculated = incremental_dataforge_checksum(handle)
            if calculated != evaluation_checksum:
                raise ValueError(f"Records checksum mismatch for {evaluation_uri}")
            count = 0
            for record in _stream_records(evaluation_store, evaluation_key, source_uri=evaluation_uri):
                count += 1
                if record.record_id is not None:
                    if record.record_id in seen_record_ids:
                        raise ValueError(f"Duplicate record_id in research input: {record.record_id}")
                    seen_record_ids.add(record.record_id)
                if record.split != "evaluation" or record.metadata.get("training_eligible") is not False:
                    raise ValueError(f"Evaluation record {record.record_id or count} from {manifest_uri} is trainable")
                if record.metadata.get("research_role") != evaluation_manifest.get("research_role"):
                    raise ValueError(f"Evaluation record {record.record_id or count} has inconsistent research_role")
                yield record
            if count != int(evaluation_manifest.get("record_count", -1)):
                raise ValueError(f"Evaluation set record_count mismatch for {manifest_uri}")

    @staticmethod
    def _is_trainable(record: NormalizedRecord) -> bool:
        if record.split != "train":
            return False
        role = record.metadata.get("research_role")
        eligible = record.metadata.get("training_eligible")
        if role is not None and role != "training":
            return False
        return eligible is not False

    def iter_training_records(self) -> Iterator[NormalizedRecord]:
        for record in self.iter_records():
            if self._is_trainable(record):
                yield record

    def iter_evaluation_records(self) -> Iterator[NormalizedRecord]:
        for record in self.iter_records():
            if not self._is_trainable(record):
                yield record

    def statistics(self) -> DatasetStatistics:
        total = training = evaluation = 0
        evaluation_suite_counts: dict[str, int] = {}
        for record in self.iter_records():
            total += 1
            if self._is_trainable(record):
                training += 1
            else:
                evaluation += 1
                role = str(record.metadata.get("research_role") or "legacy_evaluation")
                evaluation_suite_counts[role] = evaluation_suite_counts.get(role, 0) + 1
        return DatasetStatistics(
            total_records=total,
            training_records=training,
            evaluation_records=evaluation,
            selected_training_candidates=training,
            accepted_training_examples=training,
            skipped_overlength_count=0,
            maximum_observed_token_length=None,
            maximum_accepted_token_length=None,
            configured_max_sequence_length=None,
            split_summary={"train": training, "evaluation": evaluation},
            evaluation_suite_counts=evaluation_suite_counts,
        )


def _assistant_mask_from_template(tokenizer: Any, messages: Sequence[dict[str, str]]) -> tuple[list[int], list[int]]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and "{% generation" not in chat_template:
        raise LookupError(
            "Tokenizer chat template does not declare assistant generation boundaries"
        )
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = rendered.get("input_ids")
    mask = rendered.get("assistant_tokens_mask")
    if mask is None:
        mask = rendered.get("assistant_mask")
    if mask is None:
        mask = rendered.get("assistant_masks")
    if input_ids is None or mask is None:
        raise LookupError("Tokenizer did not provide assistant token mask")
    if len(input_ids) != len(mask):
        raise ValueError("Assistant mask length does not match token length")
    labels = [token if bool(flag) else -100 for token, flag in zip(input_ids, mask)]
    if all(label == -100 for label in labels):
        raise ValueError("Assistant mask did not mark any target tokens")
    return list(input_ids), labels


def _fallback_mask_from_prefixes(tokenizer: Any, messages: Sequence[dict[str, str]]) -> tuple[list[int], list[int]]:
    prefixes: list[dict[str, str]] = []
    input_ids: list[int] = []
    labels: list[int] = []
    assistant_tokens = 0
    for index, message in enumerate(messages):
        prefixes.append(message)
        rendered = tokenizer.apply_chat_template(
            list(prefixes),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if not isinstance(rendered, (list, tuple)):
            raise ValueError("Tokenizer chat template fallback must return token ids")
        rendered_ids = list(rendered)
        if index == 0:
            input_ids = rendered_ids
            labels = [-100] * len(rendered_ids)
            continue
        if rendered_ids[: len(input_ids)] != input_ids:
            raise ValueError(
                "Tokenizer fallback rewrote previously rendered chat template tokens"
            )
        new_tokens = rendered_ids[len(input_ids) :]
        input_ids = rendered_ids
        if message["role"] == "assistant":
            labels.extend(new_tokens)
            assistant_tokens += len(new_tokens)
        else:
            labels.extend([-100] * len(new_tokens))
    if assistant_tokens == 0:
        raise ValueError("Assistant boundaries could not be determined safely")
    if len(input_ids) != len(labels):
        raise ValueError("Assistant mask length does not match token length")
    return input_ids, labels


def encode_supervised_example(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    try:
        input_ids, labels = _assistant_mask_from_template(tokenizer, messages)
    except LookupError:
        input_ids, labels = _fallback_mask_from_prefixes(tokenizer, messages)
    except TypeError as exc:
        message = str(exc)
        unsupported_mask_option = (
            "return_assistant_tokens_mask" in message
            and (
                "unexpected keyword" in message
                or "unexpected keyword argument" in message
                or "unsupported" in message.lower()
            )
        )
        if not unsupported_mask_option:
            raise
        input_ids, labels = _fallback_mask_from_prefixes(tokenizer, messages)
    if max_sequence_length is not None and len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Example exceeds maximum sequence length: {len(input_ids)} > {max_sequence_length}"
        )
    return {"input_ids": input_ids, "labels": labels}


def iter_selected_training_examples(
    reader: DataForgeDatasetReader,
    tokenizer: Any,
    *,
    max_examples: int | None,
    max_sequence_length: int,
    overlength_policy: str,
) -> Iterator[SelectedTrainingExample]:
    accepted = 0
    for record in reader.iter_records():
        if not reader._is_trainable(record):
            continue
        encoded = encode_supervised_example(tokenizer, record.messages)
        if len(encoded["input_ids"]) > max_sequence_length:
            if overlength_policy == "skip":
                continue
            raise ValueError(
                f"Record {record.record_id or record.line_number} at line {record.line_number} in {record.source_uri} exceeds maximum sequence length {max_sequence_length}."
            )
        yield SelectedTrainingExample(
            record_id=record.record_id,
            line_number=record.line_number,
            input_ids=list(encoded["input_ids"]),
            labels=list(encoded["labels"]),
            source_uri=record.source_uri,
        )
        accepted += 1
        if max_examples is not None and accepted >= max_examples:
            break


def collate_supervised_batch(batch: Sequence[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    import torch

    longest = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_masks = []
    labels = []
    for item in batch:
        ids = list(item["input_ids"])
        lab = list(item["labels"])
        pad = longest - len(ids)
        assert pad >= 0
        input_ids.append(ids + [pad_token_id] * pad)
        attention_masks.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _skip_sample(
    record: NormalizedRecord,
    *,
    length: int,
    max_sequence_length: int,
    dataset_manifest_uri: str | None,
    records_uri: str | None,
) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "line_number": record.line_number,
        "actual_token_count": length,
        "maximum_sequence_length": max_sequence_length,
        "dataset_manifest_uri": dataset_manifest_uri,
        "records_uri": records_uri,
        "reason": "overlength",
    }


def preflight_dataset(
    reader: DataForgeDatasetReader,
    tokenizer: Any,
    *,
    max_examples: int | None,
    max_sequence_length: int,
    overlength_policy: str,
    skipped_sample_limit: int = DEFAULT_SKIPPED_SAMPLE_LIMIT,
) -> DatasetPreflightResult:
    manifest = reader.manifest() if reader.mode == "dataforge_manifest" else {}
    lineage = reader.lineage()
    manifest_train = manifest.get("train_count")
    manifest_eval = manifest.get("eval_count")
    manifest_accepted = manifest.get("accepted_count")
    total = training = evaluation = 0
    selected_candidates = 0
    accepted = 0
    skipped_count = 0
    max_observed: int | None = None
    max_accepted: int | None = None
    skipped_samples: list[dict[str, Any]] = []
    evaluation_suite_counts: dict[str, int] = {}
    for record in reader.iter_records():
        total += 1
        if not reader._is_trainable(record):
            evaluation += 1
            role = str(record.metadata.get("research_role") or "legacy_evaluation")
            evaluation_suite_counts[role] = evaluation_suite_counts.get(role, 0) + 1
            continue
        training += 1
        tokenized = encode_supervised_example(tokenizer, record.messages)
        length = len(tokenized["input_ids"])
        max_observed = length if max_observed is None else max(max_observed, length)
        selected_candidates += 1
        if length > max_sequence_length:
            skipped_count += 1
            if len(skipped_samples) < skipped_sample_limit:
                skipped_samples.append(
                    _skip_sample(
                        record,
                        length=length,
                        max_sequence_length=max_sequence_length,
                        dataset_manifest_uri=lineage.dataset_manifest_uri,
                        records_uri=lineage.records_uri,
                    )
                )
            if overlength_policy == "skip":
                continue
            raise ValueError(
                f"Record {record.record_id or record.line_number} at line {record.line_number} in {lineage.records_uri} exceeds maximum sequence length {max_sequence_length}; "
                f"manifest {lineage.dataset_manifest_uri}. Suggest increasing max_sequence_length or using overlength_policy=skip."
            )
        if max_examples is None or accepted < max_examples:
            accepted += 1
            max_accepted = length if max_accepted is None else max(max_accepted, length)
    if manifest_train is not None and int(manifest_train) != training:
        raise ValueError(
            f"Manifest train_count {manifest_train} disagrees with streamed training records {training}."
        )
    if manifest_eval is not None and reader._research_package is None and int(manifest_eval) != evaluation:
        raise ValueError(
            f"Manifest eval_count {manifest_eval} disagrees with streamed evaluation records {evaluation}."
        )
    if manifest_accepted is not None and int(manifest_accepted) < accepted:
        raise ValueError(
            f"Manifest accepted_count {manifest_accepted} is smaller than accepted training examples {accepted}."
        )
    if accepted == 0:
        raise ValueError(
            f"No trainable examples remain after applying max_examples={max_examples} and overlength_policy={overlength_policy}."
        )
    lineage_payload = asdict(lineage)
    lineage_payload["overlength_policy"] = overlength_policy
    lineage_payload["skipped_records"] = list(skipped_samples)
    return DatasetPreflightResult(
        lineage=DatasetLineage(**lineage_payload),
        statistics=DatasetStatistics(
            total_records=total,
            training_records=training,
            evaluation_records=evaluation,
            selected_training_candidates=selected_candidates if max_examples is None else min(selected_candidates, max_examples),
            accepted_training_examples=accepted,
            skipped_overlength_count=skipped_count,
            maximum_observed_token_length=max_observed,
            maximum_accepted_token_length=max_accepted,
            configured_max_sequence_length=max_sequence_length,
            split_summary={"train": training, "evaluation": evaluation},
            evaluation_suite_counts=evaluation_suite_counts,
            skipped_record_samples=tuple(skipped_samples),
            skipped_record_sample_limit=skipped_sample_limit,
            warnings=(),
        ),
        manifest=manifest,
    )
