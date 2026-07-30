"""Resumable sequential evaluation of already-saved training predictions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import urlparse

from cognityx_training.evaluation import (
    EVALUATION_MANIFEST_SCHEMA,
    EVALUATION_REQUEST_SCHEMA,
    PredictionPair,
    aggregate_candidate,
    deterministic_result,
    evaluation_run_id,
    evaluation_variant_id,
    recommendation_for,
    request_id,
    validate_judge_response,
)
from cognityx_training.evaluation_pairing import PredictionPairingStore
from cognityx_training.evaluation_configuration import EvaluationConfig
from cognityx_training.evaluation_judge import (
    CognityxJudgeClient,
    JudgeClient,
    judge_messages,
    response_content,
)
from cognityx_training.lineage import stable_json
from cognityx_training.publication import (
    PUBLICATION_SCHEMA,
    verify_published_adapter,
)
from cognityx_training.reporting import utc_now
from cognityx_training.storage_runtime import resolve_storage_runtime
from cognityx_training.storage_uri import (
    StorageUriResolutionError,
    resolve_storage_uri,
)


class EvaluationPipeline:
    """Coordinate immutable preflight, scoring, judging, and recommendation."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        storage_runtime: Any | None = None,
        judge_client: JudgeClient | None = None,
        run_id: str | None = None,
        variant_id: str | None = None,
        root: str | None = None,
    ) -> None:
        self.config = config
        self.storage_runtime = resolve_storage_runtime(
            storage_runtime=storage_runtime,
            storage_config=config.storage_config,
            storage_root=config.storage_root,
        )
        self.artifact_store = self.storage_runtime.for_role("artifact")
        self.judge = judge_client
        self.run_id = run_id
        self.variant_id = variant_id
        self.root = root
        self.pairing_store: PredictionPairingStore | None = None

    def plan(self) -> dict[str, Any]:
        """Perform complete artifact preflight without loading or calling a judge."""
        try:
            candidates = self._preflight_candidates()
            identity = self._variant_identity(candidates)
            evidence_summary = self._evidence_resolution_summary(candidates)
            pairing_summary = {
                item["adapter_id"]: dict(item["pairing_summary"])
                for item in candidates
            }
            diagnostic: Mapping[str, Any]
            capabilities: Mapping[str, Any]
            judge = self._judge_client()
            try:
                diagnostic = judge.diagnose()
            except Exception as exc:
                diagnostic = {"reachable": False, "error": type(exc).__name__, "detail": str(exc)}
            try:
                capabilities = judge.capabilities()
            except Exception as exc:
                capabilities = {"available": False, "error": type(exc).__name__, "detail": str(exc)}
            advertised = capabilities.get("capabilities", capabilities)
            if (
                isinstance(advertised, Mapping)
                and "structured_output" in advertised
                and advertised["structured_output"] is False
            ):
                raise ValueError("Configured judge does not support structured output")
            context = capabilities.get("context")
            if isinstance(context, Mapping):
                advertised_context = context.get("max_context_tokens")
                advertised_output = context.get("max_output_tokens_limit")
                if (
                    isinstance(advertised_context, int)
                    and self.config.judge.context_limit_tokens > advertised_context
                ):
                    raise ValueError(
                        "Configured context_limit_tokens exceeds judge capability"
                    )
                if (
                    isinstance(advertised_output, int)
                    and self.config.judge.max_output_tokens > advertised_output
                ):
                    raise ValueError(
                        "Configured max_output_tokens exceeds judge capability"
                    )
            estimated_max_input = max(
                (
                    _estimate_pair_tokens(pair)
                    for candidate in candidates
                    for pair in self._pairs(candidate)
                ),
                default=0,
            )
            if estimated_max_input + self.config.judge.max_output_tokens > self.config.judge.context_limit_tokens:
                raise ValueError(
                    "Estimated judge input plus output budget exceeds context_limit_tokens"
                )
            return {
                "status": "planned",
                "name": self.config.name,
                "evaluation_variant_id": evaluation_variant_id(identity),
                "candidate_count": len(candidates),
                "record_count": sum(
                    item["pairing_summary"]["record_count"] for item in candidates
                ),
                "estimated_max_input_tokens": estimated_max_input,
                "exact_token_count_available": False,
                "token_budget_check_mode": "estimated",
                "evidence_resolution_summary": evidence_summary,
                "candidate_pairing_summary": pairing_summary,
                "judge_diagnostic": dict(diagnostic),
                "judge_capabilities": dict(capabilities),
                "execution_mode": "saved-output-sequential-judge",
                "candidate_model_loaded": False,
                "base_student_model_loaded": False,
                "warnings": [
                    issue
                    for candidate in candidates
                    for issue in candidate["pairing_issues"]
                ],
            }
        finally:
            self._close_pairing()

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        """Execute or resume one sequential evaluation."""
        candidates: list[dict[str, Any]] = []
        acquired = False
        try:
            candidates = self._preflight_candidates()
            identity = self._variant_identity(candidates)
            calculated_variant = evaluation_variant_id(identity)
            if self.variant_id is not None and self.variant_id != calculated_variant:
                raise ValueError("Evaluation variant identity changed since the request")
            self.variant_id = calculated_variant
            self.run_id = self.run_id or evaluation_run_id()
            self.root = self.root or self._evaluation_root(candidates)
            self._publish_request(identity)
            self._publish_candidate_set(candidates)
            self._checkpoint(
                "preflight",
                row_counts={"candidates": len(candidates)},
                artifact_checksums={},
            )

            deterministic_count = self._deterministic_stage(candidates)
            evidence_count = self._evidence_stage(candidates)

            judge = self._judge_client()
            lifecycle = dict(judge.acquire())
            acquired = bool(lifecycle.get("loaded_by_evaluation"))
            judgment_rows, rejection_rows = self._judge_stage(
                candidates,
                judge,
            )
            aggregate = self._aggregation_stage(
                candidates,
                judgment_rows,
                rejection_rows,
            )
            recommendations = self._recommendation_stage(aggregate)
            if self.config.unload_judge_when_done and acquired:
                try:
                    lifecycle["release"] = judge.release()
                except Exception as exc:
                    lifecycle["release"] = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                acquired = False
            elif self.config.unload_judge_when_done:
                lifecycle["release"] = "not_owned"
            else:
                lifecycle["release"] = "retained_by_configuration"
                acquired = False
            manifest = self._finalize(
                candidates=candidates,
                deterministic_count=deterministic_count,
                evidence_count=evidence_count,
                judgment_rows=judgment_rows,
                rejection_rows=rejection_rows,
                aggregate=aggregate,
                recommendations=recommendations,
                lifecycle=lifecycle,
                resumed=resume,
            )
            return manifest
        except BaseException as exc:
            self._publish_failure(exc)
            raise
        finally:
            if acquired and self.judge is not None:
                try:
                    self.judge.release()
                except Exception:
                    pass
            self._close_pairing()

    @classmethod
    def from_request(
        cls,
        evaluation_request_uri: str,
        *,
        storage_runtime: Any | None = None,
        judge_client: JudgeClient | None = None,
    ) -> "EvaluationPipeline":
        runtime = resolve_storage_runtime(storage_runtime=storage_runtime)
        request = _read_uri_json(runtime, evaluation_request_uri)
        if request.get("schema_version") != EVALUATION_REQUEST_SCHEMA:
            raise ValueError("Unsupported evaluation request schema")
        config_value = dict(request["configuration"])
        config = EvaluationConfig.from_mapping(
            {
                "evaluation": {
                    **{
                        key: value
                        for key, value in config_value.items()
                        if key not in {"judge", "evidence", "gates"}
                    },
                },
                "judge": config_value["judge"],
                "evidence": config_value["evidence"],
                "gates": config_value["gates"],
            }
        )
        return cls(
            config,
            storage_runtime=runtime,
            judge_client=judge_client,
            run_id=str(request["evaluation_run_id"]),
            variant_id=str(request["evaluation_variant_id"]),
            root=str(request["artifact_root"]),
        )

    def _judge_client(self) -> JudgeClient:
        if self.judge is None:
            self.judge = CognityxJudgeClient(self.config.judge)
        return self.judge

    def _preflight_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        run_ids: set[str] = set()
        adapter_ids: set[str] = set()
        pairing = self._ensure_pairing()
        for manifest_uri in self.config.publication_manifests:
            resolution = resolve_storage_uri(self.storage_runtime, manifest_uri)
            store, key = resolution.store, resolution.key
            manifest_bytes = _read_uri_bytes(
                self.storage_runtime,
                manifest_uri,
            )
            manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = _json_object(manifest_bytes, manifest_uri)
            if manifest.get("schema_version") != PUBLICATION_SCHEMA:
                raise ValueError(f"Unsupported publication manifest: {manifest_uri}")
            if manifest.get("status") != "completed":
                raise ValueError(f"Publication is not completed: {manifest_uri}")
            run_id = _required_string(manifest, "training_run_id")
            adapter_id = _required_string(manifest, "adapter_id")
            if run_id in run_ids:
                raise ValueError(f"Duplicate training_run_id: {run_id}")
            if adapter_id in adapter_ids:
                raise ValueError(f"Duplicate adapter_id: {adapter_id}")
            run_ids.add(run_id)
            adapter_ids.add(adapter_id)
            for field in (
                "adapter_manifest_uri",
                "training_report_uri",
                "baseline_predictions_uri",
                "trained_predictions_uri",
            ):
                _require_storage_uri(_required_string(manifest, field))

            root = key.rsplit("/", 1)[0]
            checksums = manifest.get("artifact_checksums")
            if not isinstance(checksums, Mapping):
                raise ValueError("Publication artifact_checksums must be an object")
            artifacts = {
                "training-report.json": manifest["training_report_uri"],
                "baseline-predictions.jsonl": manifest["baseline_predictions_uri"],
                "trained-predictions.jsonl": manifest["trained_predictions_uri"],
            }
            for filename, expected in checksums.items():
                if filename == "adapter_bundle":
                    continue
                uri = artifacts.get(str(filename)) or store.uri(f"{root}/{filename}")
                actual = _sha256_uri(
                    self.storage_runtime,
                    uri,
                )
                if actual != _normalize_checksum(str(expected)):
                    raise ValueError(f"Artifact checksum mismatch: {uri}")
            verification = verify_published_adapter(
                str(manifest["adapter_manifest_uri"]),
                storage_runtime=self.storage_runtime,
            )
            if verification.bundle_checksum != _normalize_checksum(
                str(checksums.get("adapter_bundle", ""))
            ):
                raise ValueError("Adapter bundle checksum does not match publication")

            pairing.ingest(
                adapter_id,
                "baseline",
                _iter_uri_jsonl(
                    self.storage_runtime,
                    str(manifest["baseline_predictions_uri"]),
                ),
            )
            pairing.ingest(
                adapter_id,
                "candidate",
                _iter_uri_jsonl(
                    self.storage_runtime,
                    str(manifest["trained_predictions_uri"]),
                ),
            )
            pairing_summary = pairing.summary(adapter_id)
            issues = list(pairing.issues(adapter_id))
            lineage_uri = store.uri(f"{root}/dataset-lineage.json")
            lineage = _read_uri_json(
                self.storage_runtime,
                lineage_uri,
            )
            candidates.append(
                {
                    "publication_manifest_uri": manifest_uri,
                    "publication_manifest_checksum": manifest_checksum,
                    "manifest": manifest,
                    "experiment_id": _required_string(manifest, "experiment_id"),
                    "training_run_id": run_id,
                    "adapter_id": adapter_id,
                    "dataset_lineage_uri": lineage_uri,
                    "dataset_lineage": lineage,
                    "pairing_summary": pairing_summary,
                    "pairing_issues": issues,
                    "adapter_verification": asdict(verification),
                }
            )
        return candidates

    def _ensure_pairing(self) -> PredictionPairingStore:
        if self.pairing_store is None:
            self.pairing_store = PredictionPairingStore()
        return self.pairing_store

    def _pairs(self, candidate: Mapping[str, Any]) -> Iterator[PredictionPair]:
        return self._ensure_pairing().iter_pairs(str(candidate["adapter_id"]))

    def _close_pairing(self) -> None:
        if self.pairing_store is not None:
            self.pairing_store.close()
            self.pairing_store = None

    def _variant_identity(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        dataset_identities = sorted(
            {
                stable_json(
                    {
                        "manifest_uri": item["dataset_lineage"].get(
                            "dataset_manifest_uri"
                        ),
                        "manifest_checksum": item["dataset_lineage"].get(
                            "dataset_manifest_checksum"
                        ),
                        "records_checksum": item["dataset_lineage"].get(
                            "records_checksum"
                        ),
                    }
                )
                for item in candidates
            }
        )
        return {
            "schema_version": "cognityx.training.evaluation-variant-identity/v1",
            "candidates": sorted(
                [
                    {
                        "publication_manifest_uri": item[
                            "publication_manifest_uri"
                        ],
                        "publication_manifest_checksum": item[
                            "publication_manifest_checksum"
                        ],
                        "training_run_id": item["training_run_id"],
                        "adapter_id": item["adapter_id"],
                    }
                    for item in candidates
                ],
                key=lambda item: item["publication_manifest_uri"],
            ),
            "datasets": dataset_identities,
            "judge": {
                "provider": self.config.judge.provider,
                "model": self.config.judge.model,
                "revision": self.config.judge.revision,
                "backend": self.config.judge.backend,
                "profile": self.config.judge.profile,
            },
            "prompt_version": self.config.prompt_version,
            "metric_version": self.config.metric_version,
            "evidence_policy": asdict(self.config.evidence),
            "token_budget": {
                "context_limit_tokens": self.config.judge.context_limit_tokens,
                "max_output_tokens": self.config.judge.max_output_tokens,
            },
            "gates": asdict(self.config.gates),
        }

    def _evaluation_root(self, candidates: list[dict[str, Any]]) -> str:
        experiments = {item["experiment_id"] for item in candidates}
        if len(experiments) == 1:
            return f"experiments/{next(iter(experiments))}/evaluations/{self.run_id}"
        return f"evaluations/{self.run_id}"

    def _publish_request(self, identity: Mapping[str, Any]) -> None:
        key = f"{self.root}/evaluation-request.json"
        if self.artifact_store.exists(key):
            existing = _read_store_json(self.artifact_store, key)
            if (
                existing.get("evaluation_run_id") != self.run_id
                or existing.get("evaluation_variant_id") != self.variant_id
            ):
                raise ValueError("Existing evaluation request has conflicting identity")
            return
        payload = {
            "schema_version": EVALUATION_REQUEST_SCHEMA,
            "status": "accepted",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "name": self.config.name,
            "artifact_root": self.root,
            "configuration": self.config.to_dict(),
            "variant_identity": dict(identity),
            "execution_mode": "saved-output-sequential-judge",
            "candidate_model_loaded": False,
            "base_student_model_loaded": False,
            "created_at": utc_now(),
        }
        self.artifact_store.put_json_idempotent(key, payload)

    def _publish_candidate_set(self, candidates: list[dict[str, Any]]) -> None:
        payload = {
            "schema_version": "cognityx.training.evaluation-candidates/v1",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "candidates": [
                {
                    key: item[key]
                    for key in (
                        "publication_manifest_uri",
                        "publication_manifest_checksum",
                        "experiment_id",
                        "training_run_id",
                        "adapter_id",
                        "dataset_lineage_uri",
                        "adapter_verification",
                        "pairing_issues",
                    )
                }
                | {"record_count": item["pairing_summary"]["record_count"]}
                for item in candidates
            ],
        }
        self.artifact_store.put_json_idempotent(
            f"{self.root}/candidate-set.json",
            payload,
        )

    def _deterministic_stage(
        self,
        candidates: list[dict[str, Any]],
    ) -> int:
        count = 0

        def rows() -> Iterator[dict[str, Any]]:
            nonlocal count
            for candidate in candidates:
                for pair in self._pairs(candidate):
                    count += 1
                    yield deterministic_result(
                        pair,
                        candidate_id=candidate["adapter_id"],
                    )

        uri, checksum = self._put_jsonl("deterministic-results.jsonl", rows())
        self._checkpoint(
            "deterministic-scoring",
            row_counts={"deterministic_results": count},
            artifact_checksums={uri: checksum},
        )
        return count

    def _evidence_stage(
        self,
        candidates: list[dict[str, Any]],
    ) -> int:
        count = 0

        def rows() -> Iterator[dict[str, Any]]:
            nonlocal count
            for candidate in candidates:
                lookup, resolution_failures, _ = self._load_evidence_lookup(
                    candidate["dataset_lineage"]
                )
                for pair in self._pairs(candidate):
                    source = pair.candidate or pair.baseline or {}
                    evidence_ids = _string_list(source.get("evidence_ids"))
                    found = [lookup[item] for item in evidence_ids if item in lookup]
                    missing = [item for item in evidence_ids if item not in lookup]
                    if missing and (
                        self.config.evidence.required
                        or self.config.evidence.missing_policy == "error"
                    ):
                        raise ValueError(
                            f"Required evidence unavailable for {pair.record_id}: {missing}"
                        )
                    if evidence_ids and found and not missing:
                        basis = "evidence-grounded"
                    elif self.config.evidence.missing_policy == "unjudgeable":
                        basis = "unjudgeable"
                    else:
                        basis = "reference-only"
                    stored_evidence = [
                        {
                            "evidence_id": item["evidence_id"],
                            "content_sha256": item["content_sha256"],
                            **(
                                {"text": item["text"]}
                                if self.config.evidence.store_evidence_text
                                else {}
                            ),
                        }
                        for item in found
                    ]
                    row = {
                        "candidate_id": candidate["adapter_id"],
                        "dataset_record_id": pair.record_id,
                        "judgment_basis": basis,
                        "evidence_ids": evidence_ids,
                        "resolved_evidence": stored_evidence,
                        "missing_evidence_ids": missing,
                        "resolution_failures": resolution_failures,
                    }
                    self._ensure_pairing().put_evidence(
                        candidate["adapter_id"],
                        pair.record_id,
                        {**row, "judge_evidence": found},
                    )
                    count += 1
                    yield row

        uri, checksum = self._put_jsonl("evidence-resolution.jsonl", rows())
        self._checkpoint(
            "evidence-resolution",
            row_counts={"evidence_resolutions": count},
            artifact_checksums={uri: checksum},
        )
        return count

    def _load_evidence_lookup(
        self,
        lineage: Mapping[str, Any],
    ) -> tuple[
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        failures: list[dict[str, Any]] = []
        resolutions: list[dict[str, Any]] = []
        dataset_uri = lineage.get("dataset_manifest_uri")
        if not isinstance(dataset_uri, str) or not dataset_uri.startswith("storage://"):
            failures.append(_missing_uri_failure(dataset_uri, "dataset_manifest_uri"))
            return {}, failures, resolutions
        dataset = self._resolve_evidence_json(
            dataset_uri,
            failures=failures,
            resolutions=resolutions,
        )
        if dataset is None:
            return {}, failures, resolutions
        source_uri = dataset.get("source_manifest_uri") or lineage.get(
            "source_manifest_uri"
        )
        if not isinstance(source_uri, str) or not source_uri.startswith("storage://"):
            failures.append(_missing_uri_failure(source_uri, "source_manifest_uri"))
            return {}, failures, resolutions
        source = self._resolve_evidence_json(
            source_uri,
            failures=failures,
            resolutions=resolutions,
        )
        if source is None:
            return {}, failures, resolutions
        refs = source.get("evidence_refs")
        if not isinstance(refs, list):
            failures.append(
                {
                    "uri": source_uri,
                    "detected_namespace": resolutions[-1]["detected_namespace"],
                    "selected_role": resolutions[-1]["selected_role"],
                    "selected_profile": resolutions[-1]["selected_profile"],
                    "failure_category": "evidence_refs_missing",
                    "detail": "Ingest manifest does not contain evidence_refs",
                }
            )
            return {}, failures, resolutions
        lookup: dict[str, dict[str, Any]] = {}
        for uri in refs:
            if not isinstance(uri, str) or not uri.startswith("storage://"):
                failures.append(_missing_uri_failure(uri, "evidence_ref"))
                continue
            resolution = None
            try:
                resolution = resolve_storage_uri(self.storage_runtime, uri)
                resolutions.append(resolution.describe())
                for item in _iter_resolved_jsonl(resolution):
                    evidence_id = item.get("evidence_id") or item.get("id")
                    if evidence_id is None:
                        continue
                    text = _evidence_text(item)
                    normalized = {
                        "evidence_id": str(evidence_id),
                        "text": text,
                        "content_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                    }
                    existing = lookup.get(str(evidence_id))
                    if existing is not None and existing != normalized:
                        raise ValueError(f"Conflicting evidence_id: {evidence_id}")
                    lookup[str(evidence_id)] = normalized
            except StorageUriResolutionError as exc:
                failures.append(exc.to_dict())
            except Exception as exc:
                failures.append(
                    _resolved_failure(
                        uri,
                        resolution,
                        "evidence_artifact_unavailable",
                        exc,
                    )
                )
        return lookup, failures, resolutions

    def _resolve_evidence_json(
        self,
        uri: str,
        *,
        failures: list[dict[str, Any]],
        resolutions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        resolution = None
        try:
            resolution = resolve_storage_uri(self.storage_runtime, uri)
            resolutions.append(resolution.describe())
            with resolution.store.open(resolution.key) as source:
                return _json_object(source.read(), uri)
        except StorageUriResolutionError as exc:
            failures.append(exc.to_dict())
        except Exception as exc:
            failures.append(
                _resolved_failure(uri, resolution, "artifact_unavailable", exc)
            )
        return None

    def _evidence_resolution_summary(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summaries = []
        for candidate in candidates:
            lookup, failures, resolutions = self._load_evidence_lookup(
                candidate["dataset_lineage"]
            )
            summaries.append(
                {
                    "candidate_id": candidate["adapter_id"],
                    "resolved_evidence_count": len(lookup),
                    "resolution_failures": failures,
                    "resolved_uris": resolutions,
                }
            )
        return {
            "candidate_count": len(summaries),
            "resolvable": all(not item["resolution_failures"] for item in summaries),
            "candidates": summaries,
        }

    def _judge_stage(
        self,
        candidates: list[dict[str, Any]],
        judge: JudgeClient,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        completed = 0
        for candidate in candidates:
            candidate_id = candidate["adapter_id"]
            for pair in self._pairs(candidate):
                result_key = self._row_key("judge-results", candidate_id, pair.record_id)
                if self.artifact_store.exists(result_key):
                    results.append(_read_store_json(self.artifact_store, result_key))
                    completed += 1
                    continue
                next_attempt, terminal, prior_rejections = self._attempt_state(
                    candidate_id,
                    pair.record_id,
                )
                if terminal:
                    rejections.extend(prior_rejections)
                    continue
                evidence = self._ensure_pairing().get_evidence(
                    candidate_id,
                    pair.record_id,
                )
                if evidence["judgment_basis"] == "unjudgeable":
                    row = self._unjudgeable_result(candidate_id, pair)
                    self.artifact_store.put_json_idempotent(result_key, row)
                    results.append(row)
                    completed += 1
                    continue
                messages = self._messages(pair, evidence)
                input_tokens = judge.count_tokens(messages)
                if input_tokens is None:
                    raise ValueError("Judge token count endpoint returned no token count")
                if (
                    input_tokens + self.config.judge.max_output_tokens
                    > self.config.judge.context_limit_tokens
                ):
                    rejection = self._rejection(
                        candidate_id,
                        pair.record_id,
                        attempt=next_attempt,
                        reason="token_budget_exceeded",
                        detail=(
                            f"{input_tokens} input + "
                            f"{self.config.judge.max_output_tokens} output > "
                            f"{self.config.judge.context_limit_tokens}"
                        ),
                    )
                    self._put_attempt("rejections", rejection["request_id"], rejection)
                    rejections.append(rejection)
                    continue
                accepted: dict[str, Any] | None = None
                for attempt in range(
                    next_attempt,
                    self.config.maximum_judge_retries + 1,
                ):
                    current_request_id = request_id(
                        str(self.run_id),
                        candidate_id,
                        pair.record_id,
                        attempt,
                    )
                    request_row = {
                        "request_id": current_request_id,
                        "candidate_id": candidate_id,
                        "dataset_record_id": pair.record_id,
                        "attempt": attempt,
                        "input_tokens": input_tokens,
                        "max_output_tokens": self.config.judge.max_output_tokens,
                        "prompt_sha256": hashlib.sha256(
                            stable_json(messages).encode("utf-8")
                        ).hexdigest(),
                        "evidence_ids": evidence["evidence_ids"],
                        "evidence_hashes": [
                            item["content_sha256"]
                            for item in evidence["resolved_evidence"]
                        ],
                        **(
                            {"messages": messages}
                            if self.config.evidence.store_evidence_text
                            else {}
                        ),
                    }
                    self._put_attempt("judge-requests", current_request_id, request_row)
                    started = time.monotonic()
                    try:
                        raw = judge.judge(messages, request_id=current_request_id)
                        content, telemetry = response_content(raw)
                        parsed = validate_judge_response(_parse_json_content(content))
                        accepted = {
                            "request_id": current_request_id,
                            "candidate_id": candidate_id,
                            "dataset_record_id": pair.record_id,
                            "attempt": attempt,
                            "judgment_basis": evidence["judgment_basis"],
                            "result": parsed,
                            "latency_seconds": time.monotonic() - started,
                            **telemetry,
                        }
                        break
                    except Exception as exc:
                        rejection = self._rejection(
                            candidate_id,
                            pair.record_id,
                            attempt=attempt,
                            reason=(
                                "malformed_judge_response"
                                if isinstance(exc, (ValueError, json.JSONDecodeError))
                                else "judge_request_failed"
                            ),
                            detail=str(exc),
                            request_identity=current_request_id,
                        )
                        self._put_attempt(
                            "rejections",
                            current_request_id,
                            rejection,
                        )
                        rejections.append(rejection)
                if accepted is not None:
                    self.artifact_store.put_json_idempotent(result_key, accepted)
                    results.append(accepted)
                    completed += 1
                if completed and completed % self.config.checkpoint_interval == 0:
                    self._checkpoint(
                        f"judge-evaluation-{completed:08d}",
                        row_counts={
                            "completed_judge_results": completed,
                            "rejections": len(rejections),
                        },
                        artifact_checksums={},
                        completed_request_ids=[
                            row["request_id"] for row in results
                        ],
                    )
        request_rows = self._attempt_rows("judge-requests")
        rejection_rows = self._attempt_rows("rejections")
        result_uri, result_checksum = self._put_jsonl("judge-results.jsonl", results)
        request_uri, request_checksum = self._put_jsonl(
            "judge-requests.jsonl",
            request_rows,
        )
        rejection_uri, rejection_checksum = self._put_jsonl(
            "rejections.jsonl",
            rejection_rows,
        )
        self._checkpoint(
            "judge-evaluation",
            row_counts={
                "judge_requests": len(request_rows),
                "judge_results": len(results),
                "rejections": len(rejection_rows),
            },
            artifact_checksums={
                result_uri: result_checksum,
                request_uri: request_checksum,
                rejection_uri: rejection_checksum,
            },
            completed_request_ids=[row["request_id"] for row in results],
        )
        return results, rejection_rows

    def _attempt_state(
        self,
        candidate_id: str,
        record_id: str,
    ) -> tuple[int, bool, list[dict[str, Any]]]:
        used_attempts: set[int] = set()
        rejections: list[dict[str, Any]] = []
        terminal = False
        maximum_attempt = self.config.maximum_judge_retries
        for attempt in range(maximum_attempt + 1):
            identity = request_id(
                str(self.run_id),
                candidate_id,
                record_id,
                attempt,
            )
            request_key = f"{self.root}/rows/judge-requests/{identity}.json"
            rejection_key = f"{self.root}/rows/rejections/{identity}.json"
            request_exists = self.artifact_store.exists(request_key)
            rejection_exists = self.artifact_store.exists(rejection_key)
            if rejection_exists:
                rejection = _read_store_json(self.artifact_store, rejection_key)
                rejections.append(rejection)
                used_attempts.add(attempt)
                terminal = terminal or (
                    rejection.get("reason") == "token_budget_exceeded"
                )
            elif request_exists:
                interrupted = self._rejection(
                    candidate_id,
                    record_id,
                    attempt=attempt,
                    reason="interrupted_judge_attempt",
                    detail=(
                        "A durable judge request had no result or rejection; "
                        "the attempt will not be reused."
                    ),
                    request_identity=identity,
                )
                self._put_attempt("rejections", identity, interrupted)
                rejections.append(interrupted)
                used_attempts.add(attempt)
        next_attempt = next(
            (
                attempt
                for attempt in range(maximum_attempt + 1)
                if attempt not in used_attempts
            ),
            maximum_attempt + 1,
        )
        exhausted = next_attempt > maximum_attempt
        return next_attempt, terminal or exhausted, rejections

    def _messages(
        self,
        pair: PredictionPair,
        evidence: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        source = pair.candidate or pair.baseline or {}
        return judge_messages(
            question=source.get("prompt"),
            reference_answer=source.get("expected_answer"),
            baseline_answer=(
                None
                if pair.baseline is None
                else pair.baseline.get("generated_answer")
            ),
            candidate_answer=(
                None
                if pair.candidate is None
                else pair.candidate.get("generated_answer")
            ),
            evidence=evidence.get("judge_evidence", []),
            judgment_basis=str(evidence["judgment_basis"]),
        )

    def _unjudgeable_result(
        self,
        candidate_id: str,
        pair: PredictionPair,
    ) -> dict[str, Any]:
        scores = {
            dimension: 0.0
            for dimension in (
                "reference_correctness",
                "evidence_faithfulness",
                "completeness",
                "relevance",
                "instruction_following",
                "format_validity",
            )
        }
        return {
            "request_id": request_id(
                str(self.run_id), candidate_id, pair.record_id, 0
            ),
            "candidate_id": candidate_id,
            "dataset_record_id": pair.record_id,
            "attempt": 0,
            "judgment_basis": "unjudgeable",
            "result": {
                "baseline": scores,
                "candidate": scores,
                "decision": "unjudgeable",
                "regression": False,
                "confidence": 0.0,
                "rationale": "Required evidence was unavailable.",
                "failure_categories": ["missing_evidence"],
                "evidence_sufficiency": "insufficient",
            },
            "latency_seconds": 0.0,
            "usage": {},
        }

    def _aggregation_stage(
        self,
        candidates: list[dict[str, Any]],
        judgment_rows: list[dict[str, Any]],
        rejection_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metrics = []
        record_sets: dict[str, set[str]] = {}
        for candidate in candidates:
            candidate_id = candidate["adapter_id"]
            deterministic_uri = self.artifact_store.uri(
                f"{self.root}/deterministic-results.jsonl"
            )
            candidate_deterministic = (
                row
                for row in _iter_uri_jsonl(
                    self.storage_runtime,
                    deterministic_uri,
                )
                if row["candidate_id"] == candidate_id
            )
            candidate_judgments = [
                row for row in judgment_rows if row["candidate_id"] == candidate_id
            ]
            record_sets[candidate_id] = {
                row["dataset_record_id"] for row in candidate_judgments
            }
            metrics.append(
                aggregate_candidate(
                    candidate_id,
                    candidate_deterministic,
                    candidate_judgments,
                    rejection_count=sum(
                        row["candidate_id"] == candidate_id for row in rejection_rows
                    ),
                )
            )
        comparable, overlap = _comparability(
            record_sets,
            minimum=self.config.minimum_comparable_overlap,
        )
        ranked = (
            [
                item["candidate_id"]
                for item in sorted(
                    metrics,
                    key=lambda value: (
                        value["candidate_win_rate"],
                        value["deterministic_accuracy"],
                    ),
                    reverse=True,
                )
            ]
            if comparable
            else []
        )
        payload = {
            "schema_version": "cognityx.training.evaluation-aggregate/v1",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "candidates": metrics,
            "comparable": comparable,
            "minimum_pairwise_overlap": overlap,
            "ranking": ranked,
            "warnings": [] if comparable else ["candidate_record_sets_incomparable"],
        }
        stored = self.artifact_store.put_json_idempotent(
            f"{self.root}/aggregate.json",
            payload,
        )
        self._checkpoint(
            "aggregation",
            row_counts={"candidate_aggregates": len(metrics)},
            artifact_checksums={stored.uri: _json_checksum(payload)},
        )
        return payload

    def _recommendation_stage(
        self,
        aggregate: Mapping[str, Any],
    ) -> dict[str, Any]:
        recommendations = [
            recommendation_for(item, asdict(self.config.gates))
            for item in aggregate["candidates"]
        ]
        if not aggregate["comparable"] and len(recommendations) > 1:
            for item in recommendations:
                item["warnings"].append("candidate_record_sets_incomparable")
                if item["recommendation"] == "recommended_for_promotion":
                    item["recommendation"] = "manual_review_required"
        payload = {
            "schema_version": "cognityx.training.promotion-recommendation/v1",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "recommendations": recommendations,
            "promotion_performed": False,
            "deployment_performed": False,
        }
        stored = self.artifact_store.put_json_idempotent(
            f"{self.root}/recommendation.json",
            payload,
        )
        self._checkpoint(
            "recommendation",
            row_counts={"recommendations": len(recommendations)},
            artifact_checksums={stored.uri: _json_checksum(payload)},
        )
        return payload

    def _finalize(
        self,
        *,
        candidates: list[dict[str, Any]],
        deterministic_count: int,
        evidence_count: int,
        judgment_rows: list[dict[str, Any]],
        rejection_rows: list[dict[str, Any]],
        aggregate: Mapping[str, Any],
        recommendations: Mapping[str, Any],
        lifecycle: Mapping[str, Any],
        resumed: bool,
    ) -> dict[str, Any]:
        artifact_uris = {
            filename: self.artifact_store.uri(f"{self.root}/{filename}")
            for filename in (
                "evaluation-request.json",
                "candidate-set.json",
                "deterministic-results.jsonl",
                "evidence-resolution.jsonl",
                "judge-requests.jsonl",
                "judge-results.jsonl",
                "rejections.jsonl",
                "aggregate.json",
                "recommendation.json",
            )
        }
        artifact_checksums = {
            filename: _sha256_uri(
                self.storage_runtime,
                uri,
            )
            for filename, uri in artifact_uris.items()
        }
        manifest = {
            "schema_version": EVALUATION_MANIFEST_SCHEMA,
            "status": "completed",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "name": self.config.name,
            "artifact_root": self.artifact_store.uri(str(self.root)),
            "participating_experiment_ids": sorted(
                {item["experiment_id"] for item in candidates}
            ),
            "candidate_ids": [item["adapter_id"] for item in candidates],
            "artifacts": artifact_uris,
            "artifact_checksums": artifact_checksums,
            "row_counts": {
                "deterministic_results": deterministic_count,
                "evidence_resolutions": evidence_count,
                "judge_results": len(judgment_rows),
                "rejections": len(rejection_rows),
            },
            "aggregate": dict(aggregate),
            "recommendation": dict(recommendations),
            "judge_lifecycle": dict(lifecycle),
            "resumed": resumed,
            "execution_mode": "saved-output-sequential-judge",
            "candidate_model_loaded": False,
            "base_student_model_loaded": False,
            "promotion_performed": False,
            "deployment_performed": False,
            "completed_at": utc_now(),
        }
        self._checkpoint(
            "finalization",
            row_counts=manifest["row_counts"],
            artifact_checksums=artifact_checksums,
        )
        self.artifact_store.put_json_idempotent(
            f"{self.root}/evaluation-manifest.json",
            manifest,
        )
        return manifest

    def _checkpoint(
        self,
        stage: str,
        *,
        row_counts: Mapping[str, int],
        artifact_checksums: Mapping[str, str],
        completed_request_ids: Iterable[str] = (),
    ) -> None:
        key = f"{self.root}/checkpoints/{stage}.json"
        if self.artifact_store.exists(key):
            return
        payload = {
            "schema_version": "cognityx.training.evaluation-checkpoint/v1",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "stage": stage,
            "completed_request_ids": sorted(set(completed_request_ids)),
            "artifact_checksums": dict(artifact_checksums),
            "row_counts": dict(row_counts),
            "status": "completed",
            "completed_at": utc_now(),
        }
        self.artifact_store.put_json_idempotent(key, payload)

    def _put_jsonl(
        self,
        filename: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> tuple[str, str]:
        key = f"{self.root}/{filename}"
        hasher = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="cognityx-evaluation-",
            suffix=".jsonl",
            delete=False,
        ) as target:
            path = Path(target.name)
            for row in rows:
                content = (
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                target.write(content)
                hasher.update(content)
        checksum = hasher.hexdigest()
        try:
            if self.artifact_store.exists(key):
                with self.artifact_store.open(key) as existing:
                    existing_checksum = _sha256_stream(existing)
                if existing_checksum != checksum:
                    raise ValueError(f"Immutable evaluation artifact conflicts: {key}")
                return self.artifact_store.uri(key), checksum
            stored = self.artifact_store.put_file(
                key,
                path,
                media_type="application/x-ndjson",
            )
            return stored.uri, checksum
        finally:
            path.unlink(missing_ok=True)

    def _row_key(self, group: str, candidate_id: str, record_id: str) -> str:
        digest = hashlib.sha256(
            f"{candidate_id}\x1f{record_id}".encode("utf-8")
        ).hexdigest()
        return f"{self.root}/rows/{group}/{digest}.json"

    def _put_attempt(
        self,
        group: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.artifact_store.put_json_idempotent(
            f"{self.root}/rows/{group}/{identity}.json",
            dict(payload),
        )

    def _attempt_rows(self, group: str) -> list[dict[str, Any]]:
        prefix = f"{self.root}/rows/{group}"
        if not self.artifact_store.exists(prefix):
            return []
        rows = []
        for item in self.artifact_store.list(prefix):
            if item.is_directory or not item.uri.endswith(".json"):
                continue
            rows.append(
                _read_uri_json(
                    self.storage_runtime,
                    item.uri,
                )
            )
        return sorted(rows, key=lambda row: str(row.get("request_id", "")))

    def _rejection(
        self,
        candidate_id: str,
        record_id: str,
        *,
        attempt: int,
        reason: str,
        detail: str,
        request_identity: str | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_identity
            or request_id(str(self.run_id), candidate_id, record_id, attempt),
            "candidate_id": candidate_id,
            "dataset_record_id": record_id,
            "attempt": attempt,
            "reason": reason,
            "detail": detail,
        }

    def _publish_failure(self, exc: BaseException) -> None:
        if self.root is None:
            return
        payload = {
            "schema_version": "cognityx.training.evaluation-failure/v1",
            "status": "failed",
            "evaluation_run_id": self.run_id,
            "evaluation_variant_id": self.variant_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": utc_now(),
        }
        try:
            if not self.artifact_store.exists(f"{self.root}/failure.json"):
                self.artifact_store.put_json_idempotent(
                    f"{self.root}/failure.json",
                    payload,
                )
        except Exception:
            pass


def show_evaluation(
    evaluation_manifest_uri: str,
    *,
    storage_runtime: Any | None = None,
) -> dict[str, Any]:
    runtime = resolve_storage_runtime(storage_runtime=storage_runtime)
    manifest = _read_uri_json(runtime, evaluation_manifest_uri)
    if manifest.get("schema_version") != EVALUATION_MANIFEST_SCHEMA:
        raise ValueError("Unsupported evaluation manifest schema")
    return {
        "evaluation_run_id": manifest["evaluation_run_id"],
        "evaluation_variant_id": manifest["evaluation_variant_id"],
        "status": manifest["status"],
        "aggregate": manifest["aggregate"],
        "recommendation": manifest["recommendation"],
    }


def _read_uri_bytes(runtime: Any, uri: str) -> bytes:
    resolution = resolve_storage_uri(runtime, uri)
    with resolution.store.open(resolution.key) as source:
        return source.read()


def _read_uri_json(runtime: Any, uri: str) -> dict[str, Any]:
    return _json_object(_read_uri_bytes(runtime, uri), uri)


def _read_store_json(store: Any, key: str) -> dict[str, Any]:
    with store.open(key) as source:
        return _json_object(source.read(), key)


def _json_object(content: bytes, source: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _iter_uri_jsonl(
    runtime: Any,
    uri: str,
) -> Iterator[dict[str, Any]]:
    yield from _iter_resolved_jsonl(resolve_storage_uri(runtime, uri))


def _iter_resolved_jsonl(resolution: Any) -> Iterator[dict[str, Any]]:
    with resolution.store.open(resolution.key) as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{resolution.uri}:{line_number} must be a JSON object"
                )
            yield value


def _sha256_uri(runtime: Any, uri: str) -> str:
    resolution = resolve_storage_uri(runtime, uri)
    with resolution.store.open(resolution.key) as source:
        return _sha256_stream(source)


def _sha256_stream(source: Any) -> str:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def _missing_uri_failure(value: Any, field: str) -> dict[str, Any]:
    return {
        "uri": value if isinstance(value, str) else None,
        "detected_namespace": None,
        "selected_role": None,
        "selected_profile": None,
        "failure_category": "missing_or_invalid_uri",
        "detail": f"{field} must be a storage:// URI",
    }


def _resolved_failure(
    uri: str,
    resolution: Any,
    category: str,
    exc: BaseException,
) -> dict[str, Any]:
    description = (
        resolution.describe()
        if resolution is not None
        else {
            "detected_namespace": None,
            "selected_role": None,
            "selected_profile": None,
            "shared_compatibility": False,
        }
    )
    return {
        **description,
        "uri": uri,
        "failure_category": category,
        "detail": str(exc),
    }


def _required_string(value: Mapping[str, Any], field: str) -> str:
    selected = value.get(field)
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"Required string field is missing: {field}")
    return selected


def _require_storage_uri(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "storage" or not parsed.netloc:
        raise ValueError(f"Production artifact identity must use storage://: {uri}")


def _normalize_checksum(value: str) -> str:
    return value.removeprefix("sha256:")


def _json_checksum(value: Mapping[str, Any]) -> str:
    content = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item) for item in items if item is not None))


def _evidence_text(value: Mapping[str, Any]) -> str:
    for field in ("text", "content", "chunk_text", "normalized_text"):
        selected = value.get(field)
        if isinstance(selected, str):
            return selected
    return ""


def _estimate_pair_tokens(pair: PredictionPair) -> int:
    source = pair.candidate or pair.baseline or {}
    values = (
        source.get("prompt"),
        source.get("expected_answer"),
        None if pair.baseline is None else pair.baseline.get("generated_answer"),
        None if pair.candidate is None else pair.candidate.get("generated_answer"),
    )
    characters = sum(len(str(value or "")) for value in values)
    return max(1, characters // 3) + 512


def _parse_json_content(content: str) -> Any:
    selected = content.strip()
    if selected.startswith("```") and selected.endswith("```"):
        selected = selected[3:-3].strip()
        if selected.startswith("json"):
            selected = selected[4:].strip()
    return json.loads(selected)


def _comparability(
    record_sets: Mapping[str, set[str]],
    *,
    minimum: float,
) -> tuple[bool, float]:
    values = list(record_sets.values())
    if len(values) < 2:
        return True, 1.0
    overlaps = []
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            denominator = max(len(left), len(right))
            overlaps.append(len(left & right) / denominator if denominator else 1.0)
    lowest = min(overlaps, default=1.0)
    return lowest >= minimum, lowest
