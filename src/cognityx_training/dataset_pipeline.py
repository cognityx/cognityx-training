"""Convert simple JSONL instruction records into chat-template training text."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def dataset_path(uri: str) -> Path:
    """Resolve a local path or ``file://`` dataset URI.

    Raises:
        ValueError: If the URI uses a provider not implemented by this slice.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError(
            f"Unsupported dataset URI '{uri}'. The first slice supports local files only."
        )
    return Path(parsed.path if parsed.scheme else uri).expanduser()


def load_jsonl_records(uri: str) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a local JSONL dataset."""
    path = dataset_path(uri)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            records.append(value)
    if not records:
        raise ValueError(f"Dataset contains no examples: {path}")
    return records


def messages_for_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize either chat messages or instruction/output fields."""
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
    instruction = record.get("instruction") or record.get("prompt")
    output = record.get("output") or record.get("response")
    if instruction is None or output is None:
        raise ValueError(
            "Each record needs non-empty 'messages' or instruction/prompt and output/response."
        )
    return [
        {"role": "user", "content": str(instruction)},
        {"role": "assistant", "content": str(output)},
    ]


def render_ift_examples(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
) -> list[str]:
    """Apply the selected model's chat template to instruction examples."""
    return [
        tokenizer.apply_chat_template(
            messages_for_record(record),
            tokenize=False,
            add_generation_prompt=False,
        )
        for record in records
    ]
