"""Stable identities for training experiments, variants, runs, and adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import uuid
from typing import Any, Mapping

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def stable_json(value: Any) -> str:
    """Serialize a semantic identity payload deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_lineage_id(value: str, *, prefix: str) -> str:
    """Validate one prefixed, portable Storage path segment."""
    if not value.startswith(prefix) or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"Identifier must start with '{prefix}' and be one safe path segment: {value!r}"
        )
    return value


def _sortable_unique_value() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def experiment_id(value: str | None = None) -> str:
    return validate_lineage_id(value or f"exp-{_sortable_unique_value()}", prefix="exp-")


def training_run_id(value: str | None = None) -> str:
    if value and not value.startswith("trun-"):
        value = f"trun-{value}"
    return validate_lineage_id(
        value or f"trun-{_sortable_unique_value()}",
        prefix="trun-",
    )


def adapter_id(experiment: str, run_id: str) -> str:
    """Build a globally safe adapter identity for one experiment run."""
    validated_experiment = validate_lineage_id(experiment, prefix="exp-")
    validated_run = validate_lineage_id(run_id, prefix="trun-")
    digest = hashlib.sha256(
        f"{validated_experiment}\x1f{validated_run}".encode("utf-8")
    ).hexdigest()
    return validate_lineage_id(f"adp-{digest[:24]}", prefix="adp-")


def variant_identity_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def training_variant_id(payload: Mapping[str, Any]) -> str:
    return f"tvar-{variant_identity_checksum(payload)[:20]}"


@dataclass(frozen=True, slots=True)
class TrainingLineageIds:
    """Typed identity set for one physical training execution."""

    experiment_id: str
    training_variant_id: str
    training_run_id: str
    adapter_id: str

    def __post_init__(self) -> None:
        validate_lineage_id(self.experiment_id, prefix="exp-")
        validate_lineage_id(self.training_variant_id, prefix="tvar-")
        validate_lineage_id(self.training_run_id, prefix="trun-")
        validate_lineage_id(self.adapter_id, prefix="adp-")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_lineage_ids(
    canonical_variant_identity: Mapping[str, Any],
    *,
    requested_experiment_id: str | None = None,
    requested_run_id: str | None = None,
) -> TrainingLineageIds:
    exp_id = experiment_id(requested_experiment_id)
    run_id = training_run_id(requested_run_id)
    return TrainingLineageIds(
        experiment_id=exp_id,
        training_variant_id=training_variant_id(canonical_variant_identity),
        training_run_id=run_id,
        adapter_id=adapter_id(exp_id, run_id),
    )
