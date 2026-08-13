"""Deterministic human presentation for already-safe CLI payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def render_human(value: Any) -> str:
    """Render one JSON-safe value without resolving or enriching it."""
    return "\n".join(_render(value))


def _render(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}No fields."]
        lines: list[str] = []
        for key, item in value.items():
            label = _label(str(key))
            if str(key) == "overrides" and isinstance(item, list):
                lines.extend(_render_overrides(label, item, indent=indent))
            elif isinstance(item, Mapping):
                lines.append(f"{prefix}{label}:")
                lines.extend(_render(item, indent=indent + 2))
            elif _is_sequence(item):
                lines.append(f"{prefix}{label}:")
                lines.extend(_render_sequence(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{label}: {_scalar(item)}")
        return lines
    if _is_sequence(value):
        return _render_sequence(value, indent=indent)
    return [f"{prefix}{_scalar(value)}"]


def _render_sequence(value: Sequence[Any], *, indent: int) -> list[str]:
    prefix = " " * indent
    if not value:
        return [f"{prefix}No records."]
    if all(isinstance(item, Mapping) for item in value):
        records = [item for item in value if isinstance(item, Mapping)]
        columns = _columns(records)
        if columns and all(
            not isinstance(item.get(column), (Mapping, list, tuple))
            for item in records
            for column in columns
        ):
            return _table(records, columns, indent=indent)
    lines: list[str] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, Mapping):
            lines.append(f"{prefix}Record {index}:")
            lines.extend(_render(item, indent=indent + 2))
        elif _is_sequence(item):
            lines.append(f"{prefix}Record {index}:")
            lines.extend(_render_sequence(item, indent=indent + 2))
        else:
            lines.append(f"{prefix}- {_scalar(item)}")
    return lines


def _render_overrides(label: str, value: list[Any], *, indent: int) -> list[str]:
    prefix = " " * indent
    lines = [f"{prefix}{label}:"]
    if not value:
        lines.append(f"{prefix}  No records.")
        return lines
    for item in value:
        if not isinstance(item, Mapping):
            lines.append(f"{prefix}  - {_scalar(item)}")
            continue
        key = _scalar(item.get("key"))
        previous = _scalar(item.get("previous"))
        effective = _scalar(item.get("effective"))
        source = _scalar(item.get("source"))
        lines.append(f"{prefix}  {key}: {previous} -> {effective} ({source})")
    return lines


def _columns(records: list[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    for record in records:
        for key in record:
            name = str(key)
            if name not in columns:
                columns.append(name)
    return columns


def _table(
    records: list[Mapping[str, Any]],
    columns: list[str],
    *,
    indent: int,
) -> list[str]:
    prefix = " " * indent
    headers = [_label(column) for column in columns]
    rows = [[_scalar(record.get(column)) for column in columns] for record in records]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    header = "  ".join(
        headers[index].ljust(widths[index]) for index in range(len(columns))
    )
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in rows
    ]
    return [
        f"{prefix}{header}",
        f"{prefix}{divider}",
        *(f"{prefix}{row}" for row in body),
    ]


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _label(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))
