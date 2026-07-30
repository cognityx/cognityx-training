from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from cognityx_storage import StorageConfig, StorageRuntime
from cognityx_training.dataset_pipeline import DatasetLineage
from cognityx_training.evaluation import (
    PredictionPair,
    aggregate_candidate,
    deterministic_result,
    evaluation_variant_id,
    pair_predictions,
    recommendation_for,
    validate_judge_response,
    request_id,
)
from cognityx_training.evaluation_configuration import (
    EvidenceConfig,
    EvaluationConfig,
    GateConfig,
    JudgeConfig,
)
from cognityx_training.evaluation_pipeline import (
    EvaluationPipeline,
    show_evaluation,
)
from cognityx_training.evaluation_pairing import PredictionPairingStore
from cognityx_training.lineage import (
    TrainingLineageIds,
    adapter_id,
)
from cognityx_training.publication import TrainingPublisher
from cognityx_training.storage_uri import (
    StorageUriResolutionError,
    resolve_storage_uri,
)


def _runtime(root: Path) -> StorageRuntime:
    return StorageRuntime.from_config(StorageConfig.built_in(root=root))


def _valid_judgment(decision: str = "candidate_better") -> dict[str, Any]:
    dimensions = {
        "reference_correctness": 0.9,
        "evidence_faithfulness": 0.9,
        "completeness": 0.8,
        "relevance": 0.9,
        "instruction_following": 1.0,
        "format_validity": 1.0,
    }
    return {
        "baseline": {**dimensions, "reference_correctness": 0.5},
        "candidate": dimensions,
        "decision": decision,
        "regression": decision == "baseline_better",
        "confidence": 0.9,
        "rationale": "The candidate is more accurate.",
        "failure_categories": [],
        "evidence_sufficiency": "sufficient",
    }


class FakeJudge:
    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        owned: bool = True,
        token_count: int = 100,
        fail_count_on: int | None = None,
    ) -> None:
        self.responses = list(responses or [_valid_judgment()])
        self.owned = owned
        self.token_count = token_count
        self.fail_count_on = fail_count_on
        self.diagnose_calls = 0
        self.capability_calls = 0
        self.acquire_calls = 0
        self.count_calls = 0
        self.judge_calls = 0
        self.release_calls = 0
        self.request_ids: list[str] = []

    def diagnose(self) -> Mapping[str, Any]:
        self.diagnose_calls += 1
        return {"reachable": True}

    def capabilities(self) -> Mapping[str, Any]:
        self.capability_calls += 1
        return {"structured_output": True}

    def acquire(self) -> Mapping[str, Any]:
        self.acquire_calls += 1
        return {
            "loaded_by_evaluation": self.owned,
            "found_resident": not self.owned,
        }

    def count_tokens(self, messages) -> int:
        self.count_calls += 1
        if self.fail_count_on == self.count_calls:
            raise RuntimeError("simulated token endpoint failure")
        return self.token_count

    def judge(self, messages, *, request_id: str) -> Mapping[str, Any]:
        self.judge_calls += 1
        self.request_ids.append(request_id)
        selected = self.responses[min(self.judge_calls - 1, len(self.responses) - 1)]
        if isinstance(selected, BaseException):
            raise selected
        content = selected if isinstance(selected, str) else json.dumps(selected)
        return {
            "id": f"provider-{self.judge_calls}",
            "model": "fake-judge",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": self.token_count, "completion_tokens": 50},
        }

    def release(self) -> Mapping[str, Any]:
        self.release_calls += 1
        return {"unloaded": True}


def _lineage(dataset_manifest_uri: str) -> DatasetLineage:
    return DatasetLineage(
        dataset_id="dataset-eval",
        dataset_name="Evaluation",
        dataset_version="1",
        dataset_variant_id="dvar-eval",
        dataset_manifest_uri=dataset_manifest_uri,
        dataset_manifest_checksum="fixture",
        records_uri="storage://local-main/datasets/eval/records.jsonl",
        records_checksum="fixture-records",
        recipe="knowledge-unit-qa",
        source_manifest_uri="storage://local-main/artifacts/ingest/runs/run/manifest.json",
        source_manifest_checksum="fixture-source",
        configuration_checksum="fixture-config",
    )


def _adapter_staging(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
    (path / "adapter_model.safetensors").write_bytes(f"weights-{name}".encode())
    return path


def _prediction(
    ids: TrainingLineageIds,
    record_id: str,
    answer: str,
    *,
    prediction_type: str,
    exact_match: bool | None = None,
) -> dict[str, Any]:
    return {
        **ids.to_dict(),
        "adapter_id": ids.adapter_id if prediction_type == "trained" else None,
        "dataset_record_id": record_id,
        "prompt": f"Question {record_id}?",
        "expected_answer": "correct",
        "generated_answer": answer,
        "exact_match": exact_match,
        "contains_expected": "correct" in answer,
        "knowledge_unit_id": f"ku-{record_id}",
        "knowledge_unit_ids": [f"ku-{record_id}"],
        "evidence_ids": [f"evidence-{record_id}"],
        "source_asset_ids": ["asset-1"],
        "document_ids": ["doc-1"],
        "probe_id": "probe-1",
        "probe_class": "known",
        "recipe": "knowledge-unit-qa",
        "prediction_type": prediction_type,
    }


def _publication(
    runtime: StorageRuntime,
    tmp_path: Path,
    *,
    experiment: str = "exp-evaluation",
    run: str = "trun-candidate-one",
    record_ids: tuple[str, ...] = ("one",),
    publish_evidence: bool = True,
    shared_evidence: bool = False,
) -> str:
    artifact_store = runtime.for_role("artifact")
    evidence_store = (
        artifact_store._client.for_shared_data()
        if shared_evidence
        else artifact_store
    )
    evidence_key = f"ingest/documents/{run}/evidence.jsonl"
    stored_evidence = evidence_store.put_bytes(
        evidence_key,
        b"".join(
            (
                json.dumps(
                    {
                        "schema_version": "cognityx.ingest.evidence/v2",
                        "evidence_id": f"evidence-{record_id}",
                        "text": f"Evidence says correct for {record_id}.",
                    }
                )
                + "\n"
            ).encode()
            for record_id in record_ids
            if publish_evidence
        ),
        media_type="application/x-ndjson",
    )
    evidence_ref = (
        evidence_store.uri(evidence_key)
        if shared_evidence
        else stored_evidence.uri
    )
    source_key = f"ingest/runs/{run}/manifest.json"
    stored_source = evidence_store.put_json_idempotent(
        source_key,
        {
            "schema_version": "cognityx.ingest.run/v1",
            "evidence_refs": [evidence_ref],
        },
    )
    source_manifest_uri = (
        evidence_store.uri(source_key)
        if shared_evidence
        else stored_source.uri
    )
    dataset_manifest_uri = runtime.for_role("dataset").put_json_idempotent(
        f"eval/{run}/manifest.json",
        {
            "schema_version": "cognityx.dataforge.dataset/v1",
            "dataset_id": "dataset-eval",
            "source_manifest_uri": source_manifest_uri,
        },
    ).uri
    ids = TrainingLineageIds(
        experiment_id=experiment,
        training_variant_id=f"tvar-{run.removeprefix('trun-')}",
        training_run_id=run,
        adapter_id=adapter_id(experiment, run),
    )
    publisher = TrainingPublisher(runtime, ids)
    baseline = [
        _prediction(ids, record_id, "wrong", prediction_type="baseline")
        for record_id in record_ids
    ]
    trained = [
        _prediction(
            ids,
            record_id,
            "correct",
            prediction_type="trained",
            exact_match=False,
        )
        for record_id in record_ids
    ]
    result = publisher.publish_completed_run(
        staging_directory=_adapter_staging(tmp_path, run),
        dataset_lineage=_lineage(dataset_manifest_uri),
        base_model_identity={"name": "student", "revision": "fixture"},
        adapter_details={"type": "lora", "format": "peft"},
        resolved_config={"seed": 42},
        environment={"python": "fixture"},
        training_report={"status": "completed"},
        metrics={"loss": 0.1},
        baseline_predictions=baseline,
        trained_predictions=trained,
        retain_local_staging=False,
    )
    return result.publication_manifest_uri


def _config(
    publication_manifests: tuple[str, ...],
    tmp_path: Path,
    **changes: Any,
) -> EvaluationConfig:
    config = EvaluationConfig(
        publication_manifests=publication_manifests,
        name="Fixture evaluation",
        storage_root=tmp_path / "storage",
        judge=JudgeConfig(
            provider="local",
            model="fixture-judge",
            context_limit_tokens=4096,
            max_output_tokens=256,
        ),
        checkpoint_interval=1,
    )
    return replace(config, **changes)


def _deterministic_row(
    record_id: str,
    *,
    baseline: bool | None,
    candidate: bool | None,
    duplicate_baseline: bool = False,
    duplicate_candidate: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_id": "adp-one",
        "dataset_record_id": record_id,
        "baseline": (
            {"normalized_exact_match": baseline}
            if baseline is not None
            else None
        ),
        "candidate": (
            {"normalized_exact_match": candidate}
            if candidate is not None
            else None
        ),
        "missing_prediction": (
            "baseline"
            if baseline is None
            else "candidate"
            if candidate is None
            else None
        ),
        "duplicate_prediction": {
            "baseline": duplicate_baseline,
            "candidate": duplicate_candidate,
        },
    }


def test_evaluation_config_requires_storage_candidate_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="storage://"):
        EvaluationConfig(
            publication_manifests=("/tmp/publication.json",),
            name="invalid",
        ).validate()


def test_evaluation_variant_identity_is_deterministic() -> None:
    assert evaluation_variant_id({"b": 2, "a": 1}) == evaluation_variant_id(
        {"a": 1, "b": 2}
    )


def test_pairing_records_missing_and_identical_duplicates() -> None:
    baseline = [
        {"dataset_record_id": "one"},
        {"dataset_record_id": "one"},
        {"dataset_record_id": "baseline-only"},
    ]
    candidate = [{"dataset_record_id": "one"}, {"dataset_record_id": "candidate-only"}]
    pairs, issues = pair_predictions(baseline, candidate)
    assert len(pairs) == 3
    assert any(item["baseline_duplicate"] for item in issues)
    assert {item["missing"] for item in issues} >= {"baseline", "candidate"}


def test_pairing_rejects_conflicting_duplicates() -> None:
    with pytest.raises(ValueError, match="Conflicting baseline"):
        pair_predictions(
            [
                {"dataset_record_id": "one", "generated_answer": "a"},
                {"dataset_record_id": "one", "generated_answer": "b"},
            ],
            [],
        )


def test_deterministic_metrics_recompute_and_warn() -> None:
    result = deterministic_result(
        PredictionPair(
            "one",
            {
                "generated_answer": "wrong",
                "expected_answer": "Correct",
                "exact_match": False,
            },
            {
                "generated_answer": "  CORRECT ",
                "expected_answer": "Correct",
                "exact_match": False,
            },
        ),
        candidate_id="adp-one",
    )
    assert result["candidate"]["normalized_exact_match"] is True
    assert result["answer_changed"] is True
    assert "candidate_exact_match_mismatch" in result["integrity_warnings"]
    invalid_format = deterministic_result(
        PredictionPair(
            "json",
            None,
            {
                "generated_answer": "not json",
                "expected_answer": "{}",
                "metadata": {"required_format": "json"},
            },
        ),
        candidate_id="adp-one",
    )
    assert invalid_format["candidate"]["format_validity"] is False


def test_structured_judge_validation_rejects_malformed_response() -> None:
    assert validate_judge_response(_valid_judgment())["decision"] == "candidate_better"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_judge_response({"baseline": {}, "decision": "tie"})


def test_aggregate_and_gates() -> None:
    deterministic = [
        deterministic_result(
            PredictionPair(
                "one",
                {"generated_answer": "wrong", "expected_answer": "correct"},
                {"generated_answer": "correct", "expected_answer": "correct"},
            ),
            candidate_id="adp-one",
        )
    ]
    judgments = [
        {
            "candidate_id": "adp-one",
            "dataset_record_id": "one",
            "judgment_basis": "evidence-grounded",
            "result": _valid_judgment(),
            "latency_seconds": 0.1,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    ]
    aggregate = aggregate_candidate("adp-one", deterministic, judgments)
    recommendation = recommendation_for(aggregate, asdict(GateConfig()))
    assert aggregate["candidate_win_rate"] == 1.0
    assert recommendation["recommendation"] == "manual_review_required"


def test_aggregate_uses_only_paired_deterministic_denominator() -> None:
    rows = [
        _deterministic_row("paired", baseline=True, candidate=False),
        _deterministic_row("candidate-only", baseline=None, candidate=True),
        _deterministic_row("baseline-only", baseline=True, candidate=None),
    ]
    aggregate = aggregate_candidate("adp-one", rows, [])
    assert aggregate["record_count"] == 3
    assert aggregate["paired_record_count"] == 1
    assert aggregate["deterministic_accuracy"] == 0.0
    assert aggregate["baseline_deterministic_accuracy"] == 1.0
    assert aggregate["paired_exact_match_delta"] == -1.0
    assert aggregate["missing_baseline_count"] == 1
    assert aggregate["missing_candidate_count"] == 1
    assert aggregate["candidate_exact_match_count"] == 0
    assert aggregate["baseline_exact_match_count"] == 1
    assert aggregate["deterministic_regression"] is True
    assert all(
        0.0 <= aggregate[field] <= 1.0
        for field in (
            "record_coverage",
            "deterministic_accuracy",
            "baseline_deterministic_accuracy",
            "candidate_win_rate",
            "baseline_win_rate",
            "tie_rate",
            "regression_rate",
            "unjudgeable_rate",
        )
    )


def test_aggregate_handles_no_paired_rows_and_duplicate_counts() -> None:
    rows = [
        _deterministic_row(
            "candidate-only",
            baseline=None,
            candidate=True,
            duplicate_candidate=True,
        ),
        _deterministic_row(
            "baseline-only",
            baseline=True,
            candidate=None,
            duplicate_baseline=True,
        ),
    ]
    aggregate = aggregate_candidate("adp-one", rows, [])
    assert aggregate["paired_record_count"] == 0
    assert aggregate["deterministic_accuracy"] == 0.0
    assert aggregate["baseline_deterministic_accuracy"] == 0.0
    assert aggregate["paired_exact_match_delta"] == 0.0
    assert aggregate["deterministic_regression"] is False
    assert aggregate["duplicate_baseline_count"] == 1
    assert aggregate["duplicate_candidate_count"] == 1


def test_plan_validates_artifacts_without_acquiring_or_calling_judge(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge()
    result = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    ).plan()
    assert result["status"] == "planned"
    assert result["candidate_model_loaded"] is False
    assert result["token_budget_check_mode"] == "estimated"
    assert result["exact_token_count_available"] is False
    assert result["evidence_resolution_summary"]["resolvable"] is True
    resolved_roles = {
        item["selected_role"]
        for candidate in result["evidence_resolution_summary"]["candidates"]
        for item in candidate["resolved_uris"]
    }
    assert {"dataset", "artifact"} <= resolved_roles
    assert judge.diagnose_calls == judge.capability_calls == 1
    assert judge.acquire_calls == judge.count_calls == judge.judge_calls == 0


def test_plan_rejects_judge_without_structured_output(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge()
    judge.capabilities = lambda: {"capabilities": {"structured_output": False}}
    with pytest.raises(ValueError, match="structured output"):
        EvaluationPipeline(
            _config((publication,), tmp_path),
            storage_runtime=runtime,
            judge_client=judge,
        ).plan()


def test_preflight_rejects_incomplete_publication(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    store = runtime.for_role("artifact")
    uri = store.put_json_idempotent(
        "bad/publication-manifest.json",
        {"schema_version": "cognityx.training.publication/v1", "status": "failed"},
    ).uri
    with pytest.raises(ValueError, match="not completed"):
        EvaluationPipeline(
            _config((uri,), tmp_path),
            storage_runtime=runtime,
            judge_client=FakeJudge(),
        ).plan()


def test_preflight_rejects_artifact_checksum_mismatch(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    manifest = _read_uri(runtime, publication)
    store = runtime.for_role("artifact")
    trained_key = _key(store, manifest["trained_predictions_uri"])
    store.materialize(trained_key).write_text('{"corrupt":true}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        EvaluationPipeline(
            _config((publication,), tmp_path),
            storage_runtime=runtime,
            judge_client=FakeJudge(),
        ).plan()


def test_run_resolves_evidence_and_writes_terminal_last(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge()
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    )
    manifest = pipeline.run()
    assert manifest["status"] == "completed"
    assert manifest["candidate_model_loaded"] is False
    assert manifest["base_student_model_loaded"] is False
    assert manifest["promotion_performed"] is False
    assert judge.acquire_calls == judge.judge_calls == judge.release_calls == 1
    root = str(pipeline.root)
    store = runtime.for_role("artifact")
    assert store.exists(f"{root}/evaluation-manifest.json")
    evidence = _read_jsonl(store, f"{root}/evidence-resolution.jsonl")
    assert evidence[0]["judgment_basis"] == "evidence-grounded"
    assert "text" not in evidence[0]["resolved_evidence"][0]
    request = _read_store(store, f"{root}/evaluation-request.json")
    assert request["execution_mode"] == "saved-output-sequential-judge"
    shown = show_evaluation(
        store.uri(f"{root}/evaluation-manifest.json"),
        storage_runtime=runtime,
    )
    assert shown["status"] == "completed"


def test_shared_legacy_evidence_uri_resolves_through_default_profile(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(
        runtime,
        tmp_path,
        shared_evidence=True,
    )
    plan = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=FakeJudge(),
    ).plan()
    resolutions = [
        item
        for candidate in plan["evidence_resolution_summary"]["candidates"]
        for item in candidate["resolved_uris"]
    ]
    assert any(
        item["selected_role"] == "shared"
        and item["selected_profile"] == "local-main"
        for item in resolutions
    )


def test_storage_uri_resolution_is_namespace_authoritative(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    manifest = _read_uri(runtime, publication)
    dataset_uri = _read_uri(runtime, publication.replace(
        "publication-manifest.json", "dataset-lineage.json"
    ))["dataset_manifest_uri"]
    assert resolve_storage_uri(runtime, publication).selected_role == "artifact"
    assert resolve_storage_uri(runtime, dataset_uri).selected_role == "dataset"
    assert (
        resolve_storage_uri(runtime, manifest["adapter_manifest_uri"]).selected_role
        == "model"
    )
    with pytest.raises(StorageUriResolutionError, match="not configured"):
        resolve_storage_uri(runtime, "storage://local-main/unknown/value.json")
    with pytest.raises(StorageUriResolutionError, match="not override"):
        resolve_storage_uri(runtime, dataset_uri, role_override="artifact")


def test_large_pairing_store_is_disk_backed_ordered_and_incremental(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.sqlite3"
    store = PredictionPairingStore(path)
    observed = 0

    def rows(prediction_type: str):
        nonlocal observed
        for index in range(2000, 0, -1):
            observed += 1
            yield {
                "dataset_record_id": f"record-{index:05d}",
                "prediction_type": prediction_type,
            }

    store.ingest("adp-large", "baseline", rows("baseline"))
    assert observed == 2000
    store.ingest("adp-large", "candidate", rows("trained"))
    assert observed == 4000
    pairs = store.iter_pairs("adp-large")
    first = next(pairs)
    assert first.record_id == "record-00001"
    assert store.summary("adp-large")["paired_count"] == 2000
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="Conflicting baseline"):
        store.ingest(
            "adp-large",
            "baseline",
            [{"dataset_record_id": "record-00001", "changed": True}],
        )
    store.close(remove=False)


def test_malformed_judge_response_is_retried(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge(["not-json", _valid_judgment()])
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    )
    manifest = pipeline.run()
    assert judge.judge_calls == 2
    assert manifest["row_counts"]["judge_results"] == 1
    assert manifest["row_counts"]["rejections"] == 1


def test_permanent_malformed_response_is_recorded_without_aborting(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge(["not-json"])
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path, maximum_judge_retries=1),
        storage_runtime=runtime,
        judge_client=judge,
    )
    manifest = pipeline.run()
    assert manifest["status"] == "completed"
    assert manifest["row_counts"]["judge_results"] == 0
    assert manifest["row_counts"]["rejections"] == 2


def test_token_budget_rejects_before_inference(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge(token_count=4000)
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    )
    manifest = pipeline.run()
    assert judge.count_calls == 1
    assert judge.judge_calls == 0
    assert manifest["row_counts"]["rejections"] == 1


def test_missing_evidence_policy_can_mark_record_unjudgeable(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path, publish_evidence=False)
    judge = FakeJudge()
    config = _config((publication,), tmp_path)
    config = replace(
        config,
        evidence=EvidenceConfig(
            required=False,
            missing_policy="unjudgeable",
        ),
    )
    pipeline = EvaluationPipeline(
        config,
        storage_runtime=runtime,
        judge_client=judge,
    )
    manifest = pipeline.run()
    assert judge.judge_calls == 0
    assert manifest["aggregate"]["candidates"][0]["unjudgeable_rate"] == 1.0
    evidence = _read_jsonl(
        runtime.for_role("artifact"),
        f"{pipeline.root}/evidence-resolution.jsonl",
    )
    assert evidence[0]["judgment_basis"] == "unjudgeable"


def test_multiple_candidates_share_one_judge_lifecycle_and_rank(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    first = _publication(
        runtime,
        tmp_path,
        run="trun-candidate-one",
    )
    second = _publication(
        runtime,
        tmp_path,
        run="trun-candidate-two",
    )
    judge = FakeJudge([_valid_judgment(), _valid_judgment("tie")])
    manifest = EvaluationPipeline(
        _config((first, second), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    ).run()
    assert judge.acquire_calls == judge.release_calls == 1
    assert judge.judge_calls == 2
    assert len(manifest["aggregate"]["ranking"]) == 2


def test_incomparable_candidates_are_not_ranked(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    first = _publication(
        runtime,
        tmp_path,
        run="trun-candidate-one",
        record_ids=("one",),
    )
    second = _publication(
        runtime,
        tmp_path,
        run="trun-candidate-two",
        record_ids=("two",),
    )
    manifest = EvaluationPipeline(
        _config((first, second), tmp_path),
        storage_runtime=runtime,
        judge_client=FakeJudge([_valid_judgment(), _valid_judgment()]),
    ).run()
    assert manifest["aggregate"]["comparable"] is False
    assert manifest["aggregate"]["ranking"] == []


def test_resume_skips_committed_judge_results(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(
        runtime,
        tmp_path,
        record_ids=("one", "two"),
    )
    first_judge = FakeJudge(fail_count_on=2)
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=first_judge,
    )
    with pytest.raises(RuntimeError, match="token endpoint"):
        pipeline.run()
    store = runtime.for_role("artifact")
    assert not store.exists(f"{pipeline.root}/evaluation-manifest.json")
    assert store.exists(f"{pipeline.root}/failure.json")
    request_uri = store.uri(f"{pipeline.root}/evaluation-request.json")

    resumed_judge = FakeJudge()
    resumed = EvaluationPipeline.from_request(
        request_uri,
        storage_runtime=runtime,
        judge_client=resumed_judge,
    ).run(resume=True)
    assert resumed["status"] == "completed"
    assert resumed_judge.judge_calls == 1
    assert resumed["resumed"] is True


@pytest.mark.parametrize(
    ("failed_attempts", "expected_attempt"),
    [((0,), 1), ((0, 1), 2)],
)
def test_resume_starts_at_next_unused_attempt(
    tmp_path: Path,
    failed_attempts: tuple[int, ...],
    expected_attempt: int,
) -> None:
    pipeline, candidates, judge = _prepared_attempt_pipeline(tmp_path)
    candidate_id = candidates[0]["adapter_id"]
    for attempt in failed_attempts:
        identity = request_id(
            str(pipeline.run_id), candidate_id, "one", attempt
        )
        pipeline._put_attempt(
            "judge-requests",
            identity,
            {"request_id": identity, "attempt": attempt},
        )
        pipeline._put_attempt(
            "rejections",
            identity,
            pipeline._rejection(
                candidate_id,
                "one",
                attempt=attempt,
                reason="judge_request_failed",
                detail="fixture failure",
                request_identity=identity,
            ),
        )
    pipeline._judge_stage(candidates, judge)
    assert judge.judge_calls == 1
    assert judge.request_ids == [
        request_id(
            str(pipeline.run_id),
            candidate_id,
            "one",
            expected_attempt,
        )
    ]
    assert len(set(judge.request_ids)) == len(judge.request_ids)
    pipeline._close_pairing()


def test_orphan_request_is_rejected_and_not_reused(tmp_path: Path) -> None:
    pipeline, candidates, judge = _prepared_attempt_pipeline(tmp_path)
    candidate_id = candidates[0]["adapter_id"]
    orphan = request_id(str(pipeline.run_id), candidate_id, "one", 0)
    pipeline._put_attempt(
        "judge-requests",
        orphan,
        {"request_id": orphan, "attempt": 0},
    )
    _, rejections = pipeline._judge_stage(candidates, judge)
    assert judge.request_ids == [
        request_id(str(pipeline.run_id), candidate_id, "one", 1)
    ]
    assert any(
        item["request_id"] == orphan
        and item["reason"] == "interrupted_judge_attempt"
        for item in rejections
    )
    pipeline._close_pairing()


@pytest.mark.parametrize("terminal_reason", ["token_budget_exceeded", "exhausted"])
def test_terminal_attempt_state_causes_zero_new_calls(
    tmp_path: Path,
    terminal_reason: str,
) -> None:
    pipeline, candidates, judge = _prepared_attempt_pipeline(tmp_path)
    candidate_id = candidates[0]["adapter_id"]
    attempts = (0,) if terminal_reason == "token_budget_exceeded" else (0, 1, 2)
    for attempt in attempts:
        identity = request_id(
            str(pipeline.run_id), candidate_id, "one", attempt
        )
        reason = (
            "token_budget_exceeded"
            if terminal_reason == "token_budget_exceeded"
            else "judge_request_failed"
        )
        pipeline._put_attempt(
            "rejections",
            identity,
            pipeline._rejection(
                candidate_id,
                "one",
                attempt=attempt,
                reason=reason,
                detail="terminal fixture",
                request_identity=identity,
            ),
        )
    pipeline._judge_stage(candidates, judge)
    assert judge.judge_calls == 0
    pipeline._close_pairing()


def test_existing_success_causes_zero_new_judge_calls(tmp_path: Path) -> None:
    pipeline, candidates, judge = _prepared_attempt_pipeline(tmp_path)
    candidate_id = candidates[0]["adapter_id"]
    identity = request_id(str(pipeline.run_id), candidate_id, "one", 0)
    pipeline.artifact_store.put_json_idempotent(
        pipeline._row_key("judge-results", candidate_id, "one"),
        {
            "request_id": identity,
            "candidate_id": candidate_id,
            "dataset_record_id": "one",
            "result": _valid_judgment(),
        },
    )
    pipeline._judge_stage(candidates, judge)
    assert judge.judge_calls == 0
    pipeline._close_pairing()


def test_resident_judge_is_not_released(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge(owned=False)
    manifest = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    ).run()
    assert judge.release_calls == 0
    assert manifest["judge_lifecycle"]["release"] == "not_owned"


def test_failure_releases_owned_judge_and_has_no_terminal(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge(fail_count_on=1)
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path),
        storage_runtime=runtime,
        judge_client=judge,
    )
    with pytest.raises(RuntimeError):
        pipeline.run()
    store = runtime.for_role("artifact")
    assert judge.release_calls == 1
    assert store.exists(f"{pipeline.root}/failure.json")
    assert not store.exists(f"{pipeline.root}/evaluation-manifest.json")


def _read_uri(runtime: StorageRuntime, uri: str) -> dict[str, Any]:
    store = runtime.for_role("artifact")
    return _read_store(store, _key(store, uri))


def _key(store: Any, uri: str) -> str:
    key = uri.split("/", 3)[-1]
    namespace = store.namespace.strip("/")
    return key[len(namespace) + 1 :] if key.startswith(namespace + "/") else key


def _read_store(store: Any, key: str) -> dict[str, Any]:
    with store.open(key) as source:
        return json.load(source)


def _read_jsonl(store: Any, key: str) -> list[dict[str, Any]]:
    with store.open(key) as source:
        return [json.loads(line) for line in source if line.strip()]


def _prepared_attempt_pipeline(
    tmp_path: Path,
) -> tuple[EvaluationPipeline, list[dict[str, Any]], FakeJudge]:
    runtime = _runtime(tmp_path / "storage")
    publication = _publication(runtime, tmp_path)
    judge = FakeJudge()
    pipeline = EvaluationPipeline(
        _config((publication,), tmp_path, maximum_judge_retries=2),
        storage_runtime=runtime,
        judge_client=judge,
        run_id="eval-resume-fixture",
        variant_id="evar-resume-fixture",
        root="experiments/exp-evaluation/evaluations/eval-resume-fixture",
    )
    candidates = pipeline._preflight_candidates()
    pipeline._evidence_stage(candidates)
    return pipeline, candidates, judge
