"""Typed configuration for saved-output candidate evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    provider: str = "local"
    model: str = "Qwen/Qwen3-14B"
    revision: str | None = None
    backend: str = "vllm"
    profile: str = "int4"
    server_profile: str | None = None
    base_url: str = "http://127.0.0.1:8013"
    manager_url: str | None = None
    auto_start: bool = False
    context_limit_tokens: int = 32768
    max_output_tokens: int = 768
    temperature: float = 0.0
    seed: int | None = 42
    timeout_seconds: float = 300.0
    discovery_policy: str = "require_existing"
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    required: bool = False
    missing_policy: str = "reference-only"
    store_evidence_text: bool = False


@dataclass(frozen=True, slots=True)
class GateConfig:
    minimum_record_coverage: float = 0.95
    minimum_candidate_win_rate: float = 0.55
    maximum_regression_rate: float = 0.05
    minimum_reference_correctness: float = 0.75
    minimum_evidence_faithfulness: float = 0.75
    maximum_unjudgeable_rate: float = 0.10
    require_deterministic_non_regression: bool = True
    require_human_approval: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    publication_manifests: tuple[str, ...]
    name: str
    storage_config: Path | None = None
    storage_root: str | Path | None = None
    unload_judge_when_done: bool = True
    maximum_judge_retries: int = 2
    checkpoint_interval: int = 10
    prompt_version: str = "cognityx.training.judge-rubric/v1"
    metric_version: str = "cognityx.training.deterministic-metrics/v1"
    minimum_comparable_overlap: float = 0.80
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationConfig":
        evaluation = dict(values.get("evaluation") or {})
        judge = JudgeConfig(**dict(values.get("judge") or {}))
        evidence = EvidenceConfig(**dict(values.get("evidence") or {}))
        gates = GateConfig(**dict(values.get("gates") or {}))
        manifests = tuple(str(item) for item in evaluation.pop("publication_manifests", ()))
        storage_config = evaluation.get("storage_config")
        if storage_config is not None:
            evaluation["storage_config"] = Path(storage_config)
        config = cls(
            publication_manifests=manifests,
            name=str(evaluation.pop("name", "")).strip(),
            judge=judge,
            evidence=evidence,
            gates=gates,
            **evaluation,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name:
            raise ValueError("evaluation.name must be non-empty")
        if not self.publication_manifests:
            raise ValueError("evaluation.publication_manifests must not be empty")
        if len(set(self.publication_manifests)) != len(self.publication_manifests):
            raise ValueError("evaluation.publication_manifests contains duplicates")
        for uri in self.publication_manifests:
            if not uri.startswith("storage://"):
                raise ValueError(
                    "Candidate publication manifests must use storage:// URIs"
                )
        if self.maximum_judge_retries < 0:
            raise ValueError("maximum_judge_retries cannot be negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if not 0 < self.minimum_comparable_overlap <= 1:
            raise ValueError("minimum_comparable_overlap must be in (0, 1]")
        if self.storage_config is not None and not self.storage_config.exists():
            raise ValueError(f"storage_config does not exist: {self.storage_config}")
        if self.evidence.missing_policy not in {
            "reference-only",
            "unjudgeable",
            "error",
        }:
            raise ValueError(
                "evidence.missing_policy must be reference-only, unjudgeable, or error"
            )
        if self.evidence.required and self.evidence.missing_policy == "reference-only":
            raise ValueError(
                "evidence.required=true is incompatible with reference-only"
            )
        if not self.judge.provider or not self.judge.model:
            raise ValueError("judge.provider and judge.model must be non-empty")
        if self.judge.context_limit_tokens <= 0 or self.judge.max_output_tokens <= 0:
            raise ValueError("judge token limits must be positive")
        if self.judge.max_output_tokens >= self.judge.context_limit_tokens:
            raise ValueError(
                "judge.max_output_tokens must be smaller than context_limit_tokens"
            )
        if self.judge.temperature < 0 or self.judge.timeout_seconds <= 0:
            raise ValueError("judge temperature cannot be negative and timeout must be positive")
        for name, value in asdict(self.gates).items():
            if name.startswith(("minimum_", "maximum_")) and not 0 <= value <= 1:
                raise ValueError(f"gates.{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["publication_manifests"] = list(self.publication_manifests)
        value["storage_config"] = (
            str(self.storage_config) if self.storage_config is not None else None
        )
        value["storage_root"] = (
            str(self.storage_root) if self.storage_root is not None else None
        )
        return value
