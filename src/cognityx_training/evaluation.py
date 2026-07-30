"""Pure contracts and calculations for saved-output candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import uuid
from typing import Any, Iterable, Mapping

from cognityx_training.lineage import stable_json

EVALUATION_REQUEST_SCHEMA = "cognityx.training.evaluation-request/v1"
EVALUATION_MANIFEST_SCHEMA = "cognityx.training.evaluation/v1"
JUDGE_RUBRIC_VERSION = "cognityx.training.judge-rubric/v1"
DETERMINISTIC_METRIC_VERSION = "cognityx.training.deterministic-metrics/v1"
JUDGE_DECISIONS = {
    "candidate_better",
    "baseline_better",
    "tie",
    "unjudgeable",
}
JUDGE_DIMENSIONS = (
    "reference_correctness",
    "evidence_faithfulness",
    "completeness",
    "relevance",
    "instruction_following",
    "format_validity",
)


@dataclass(frozen=True, slots=True)
class PredictionPair:
    record_id: str
    baseline: Mapping[str, Any] | None
    candidate: Mapping[str, Any] | None
    baseline_duplicate: bool = False
    candidate_duplicate: bool = False


def evaluation_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"eval-{timestamp}-{uuid.uuid4().hex[:12]}"


def evaluation_variant_id(identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()
    return f"evar-{digest[:20]}"


def request_id(
    evaluation_id: str,
    candidate_id: str,
    record_id: str,
    attempt: int,
) -> str:
    payload = f"{evaluation_id}\x1f{candidate_id}\x1f{record_id}\x1f{attempt}"
    return f"jreq-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _format_valid(answer: Any, required_format: Any) -> bool | None:
    if required_format in (None, "", {}):
        return None
    text = "" if answer is None else str(answer).strip()
    if required_format in {"json", "application/json"}:
        try:
            json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return False
        return True
    if isinstance(required_format, Mapping) and required_format.get("type") == "json_object":
        try:
            return isinstance(json.loads(text), dict)
        except (TypeError, json.JSONDecodeError):
            return False
    return bool(text)


def deterministic_result(pair: PredictionPair, *, candidate_id: str) -> dict[str, Any]:
    baseline_answer = (
        None if pair.baseline is None else pair.baseline.get("generated_answer")
    )
    candidate_answer = (
        None if pair.candidate is None else pair.candidate.get("generated_answer")
    )
    source = pair.candidate or pair.baseline or {}
    expected = source.get("expected_answer")
    normalized_expected = normalize_answer(expected)
    normalized_baseline = normalize_answer(baseline_answer)
    normalized_candidate = normalize_answer(candidate_answer)
    metadata = source.get("metadata")
    required_format = source.get("required_format") or (
        metadata.get("required_format") if isinstance(metadata, Mapping) else None
    )

    def score(answer: Any, normalized: str) -> dict[str, Any]:
        exact = bool(normalized_expected) and normalized == normalized_expected
        contains = bool(normalized_expected) and normalized_expected in normalized
        return {
            "normalized_exact_match": exact,
            "expected_answer_containment": contains,
            "empty_answer": not bool(normalized),
            "format_validity": _format_valid(answer, required_format),
            "response_characters": len("" if answer is None else str(answer)),
            "response_token_estimate": len(
                re.findall(r"\w+|[^\w\s]", "" if answer is None else str(answer))
            ),
        }

    baseline_score = score(baseline_answer, normalized_baseline)
    candidate_score = score(candidate_answer, normalized_candidate)
    warnings: list[str] = []
    for label, row, calculated in (
        ("baseline", pair.baseline, baseline_score),
        ("candidate", pair.candidate, candidate_score),
    ):
        if row is None:
            continue
        for stored, recomputed in (
            ("exact_match", "normalized_exact_match"),
            ("contains_expected", "expected_answer_containment"),
        ):
            if row.get(stored) is not None and bool(row[stored]) != calculated[recomputed]:
                warnings.append(f"{label}_{stored}_mismatch")
    return {
        "schema_version": DETERMINISTIC_METRIC_VERSION,
        "candidate_id": candidate_id,
        "dataset_record_id": pair.record_id,
        "baseline": baseline_score,
        "candidate": candidate_score,
        "answer_changed": normalized_baseline != normalized_candidate,
        "missing_prediction": (
            "baseline"
            if pair.baseline is None
            else "candidate"
            if pair.candidate is None
            else None
        ),
        "duplicate_prediction": {
            "baseline": pair.baseline_duplicate,
            "candidate": pair.candidate_duplicate,
        },
        "integrity_warnings": warnings,
    }


def pair_predictions(
    baseline_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[PredictionPair], list[dict[str, Any]]]:
    baseline, baseline_duplicates = _index_predictions(baseline_rows, "baseline")
    candidate, candidate_duplicates = _index_predictions(candidate_rows, "candidate")
    identities = sorted(set(baseline) | set(candidate))
    pairs = [
        PredictionPair(
            record_id=record_id,
            baseline=baseline.get(record_id),
            candidate=candidate.get(record_id),
            baseline_duplicate=record_id in baseline_duplicates,
            candidate_duplicate=record_id in candidate_duplicates,
        )
        for record_id in identities
    ]
    issues = [
        {
            "dataset_record_id": pair.record_id,
            "missing": (
                "baseline"
                if pair.baseline is None
                else "candidate"
                if pair.candidate is None
                else None
            ),
            "baseline_duplicate": pair.baseline_duplicate,
            "candidate_duplicate": pair.candidate_duplicate,
        }
        for pair in pairs
        if pair.baseline is None
        or pair.candidate is None
        or pair.baseline_duplicate
        or pair.candidate_duplicate
    ]
    return pairs, issues


def _index_predictions(
    rows: Iterable[Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        record_id = row.get("dataset_record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} prediction requires dataset_record_id")
        if record_id in indexed:
            duplicates.add(record_id)
            if stable_json(indexed[record_id]) != stable_json(row):
                raise ValueError(
                    f"Conflicting {label} prediction for dataset record {record_id}"
                )
            continue
        indexed[record_id] = row
    return indexed, duplicates


def validate_judge_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("judge response must be a JSON object")
    result = dict(value)
    for side in ("baseline", "candidate"):
        scores = result.get(side)
        if not isinstance(scores, Mapping):
            raise ValueError(f"judge response requires {side} scores")
        for dimension in JUDGE_DIMENSIONS:
            score = scores.get(dimension)
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(f"{side}.{dimension} must be numeric")
            if not 0 <= float(score) <= 1:
                raise ValueError(f"{side}.{dimension} must be in [0, 1]")
    decision = result.get("decision")
    if decision not in JUDGE_DECISIONS:
        raise ValueError("judge decision is invalid")
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("judge confidence must be in [0, 1]")
    if not isinstance(result.get("regression"), bool):
        raise ValueError("judge regression must be boolean")
    if not isinstance(result.get("rationale"), str) or not result["rationale"].strip():
        raise ValueError("judge rationale must be a non-empty string")
    if not isinstance(result.get("failure_categories", []), list):
        raise ValueError("failure_categories must be a list")
    if result.get("evidence_sufficiency") not in {
        "sufficient",
        "insufficient",
        "not_provided",
    }:
        raise ValueError("evidence_sufficiency is invalid")
    return result


def aggregate_candidate(
    candidate_id: str,
    deterministic: Iterable[Mapping[str, Any]],
    judgments: Iterable[Mapping[str, Any]],
    *,
    rejection_count: int = 0,
) -> dict[str, Any]:
    judge_rows = list(judgments)
    total = 0
    paired = 0
    missing_baseline = 0
    missing_candidate = 0
    duplicate_baseline = 0
    duplicate_candidate = 0
    candidate_exact = 0
    baseline_exact = 0
    for row in deterministic:
        total += 1
        missing = row.get("missing_prediction")
        missing_baseline += int(missing == "baseline")
        missing_candidate += int(missing == "candidate")
        duplicates = row.get("duplicate_prediction") or {}
        duplicate_baseline += int(bool(duplicates.get("baseline")))
        duplicate_candidate += int(bool(duplicates.get("candidate")))
        if missing is None:
            paired += 1
            candidate_exact += int(
                bool(row["candidate"]["normalized_exact_match"])
            )
            baseline_exact += int(
                bool(row["baseline"]["normalized_exact_match"])
            )
    valid_judgments = [row for row in judge_rows if row.get("result")]
    decisions = [row["result"]["decision"] for row in valid_judgments]

    def mean_score(side: str, dimension: str) -> float | None:
        values = [
            float(row["result"][side][dimension])
            for row in valid_judgments
            if row["result"].get(side, {}).get(dimension) is not None
        ]
        return sum(values) / len(values) if values else None

    denominator = len(valid_judgments)
    candidate_accuracy = candidate_exact / paired if paired else 0.0
    baseline_accuracy = baseline_exact / paired if paired else 0.0
    latency = [
        float(row.get("latency_seconds", 0))
        for row in valid_judgments
        if row.get("latency_seconds") is not None
    ]
    usage = [row.get("usage") or {} for row in valid_judgments]
    return {
        "candidate_id": candidate_id,
        "record_count": total,
        "paired_record_count": paired,
        "record_coverage": paired / total if total else 0.0,
        "missing_baseline_count": missing_baseline,
        "missing_candidate_count": missing_candidate,
        "duplicate_baseline_count": duplicate_baseline,
        "duplicate_candidate_count": duplicate_candidate,
        "candidate_exact_match_count": candidate_exact,
        "baseline_exact_match_count": baseline_exact,
        "paired_exact_match_delta": candidate_accuracy - baseline_accuracy,
        "deterministic_accuracy": candidate_accuracy,
        "baseline_deterministic_accuracy": baseline_accuracy,
        "deterministic_regression": candidate_exact < baseline_exact,
        "judge_reference_correctness_mean": mean_score(
            "candidate", "reference_correctness"
        ),
        "judge_evidence_faithfulness_mean": mean_score(
            "candidate", "evidence_faithfulness"
        ),
        "candidate_win_rate": (
            decisions.count("candidate_better") / denominator if denominator else 0.0
        ),
        "baseline_win_rate": (
            decisions.count("baseline_better") / denominator if denominator else 0.0
        ),
        "tie_rate": decisions.count("tie") / denominator if denominator else 0.0,
        "regression_rate": (
            sum(bool(row["result"]["regression"]) for row in valid_judgments)
            / denominator
            if denominator
            else 0.0
        ),
        "unjudgeable_rate": (
            decisions.count("unjudgeable") / denominator if denominator else 1.0
        ),
        "invalid_judge_response_rate": (
            rejection_count / (denominator + rejection_count)
            if denominator + rejection_count
            else 0.0
        ),
        "evidence_grounded_coverage": (
            sum(row.get("judgment_basis") == "evidence-grounded" for row in valid_judgments)
            / denominator
            if denominator
            else 0.0
        ),
        "average_latency_seconds": sum(latency) / len(latency) if latency else None,
        "input_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "output_tokens": sum(int(item.get("completion_tokens") or 0) for item in usage),
    }


def recommendation_for(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_record_coverage": metrics["record_coverage"]
        >= gates["minimum_record_coverage"],
        "minimum_candidate_win_rate": metrics["candidate_win_rate"]
        >= gates["minimum_candidate_win_rate"],
        "maximum_regression_rate": metrics["regression_rate"]
        <= gates["maximum_regression_rate"],
        "minimum_reference_correctness": (
            metrics["judge_reference_correctness_mean"] is not None
            and metrics["judge_reference_correctness_mean"]
            >= gates["minimum_reference_correctness"]
        ),
        "minimum_evidence_faithfulness": (
            metrics["judge_evidence_faithfulness_mean"] is not None
            and metrics["judge_evidence_faithfulness_mean"]
            >= gates["minimum_evidence_faithfulness"]
        ),
        "maximum_unjudgeable_rate": metrics["unjudgeable_rate"]
        <= gates["maximum_unjudgeable_rate"],
        "require_deterministic_non_regression": (
            not gates["require_deterministic_non_regression"]
            or not metrics["deterministic_regression"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if metrics["record_coverage"] == 0 or metrics["unjudgeable_rate"] == 1:
        status = "insufficient_evidence"
    elif failed:
        status = "not_recommended"
    elif gates["require_human_approval"]:
        status = "manual_review_required"
    else:
        status = "recommended_for_promotion"
    return {
        "candidate_id": metrics["candidate_id"],
        "recommendation": status,
        "passed_gates": sorted(name for name, passed in checks.items() if passed),
        "failed_gates": failed,
        "supporting_metrics": dict(metrics),
        "warnings": [],
        "recommended_next_action": {
            "recommended_for_promotion": "Submit for the configured promotion workflow.",
            "manual_review_required": "Obtain human approval before any promotion action.",
            "not_recommended": "Review failed gates and run a new training variant.",
            "insufficient_evidence": "Resolve coverage or evidence gaps and evaluate again.",
        }[status],
    }
