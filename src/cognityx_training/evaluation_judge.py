"""Judge prompting and the Cognityx Inference integration boundary."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, runtime_checkable

from cognityx_training.evaluation import JUDGE_DIMENSIONS, JUDGE_RUBRIC_VERSION
from cognityx_training.evaluation_configuration import JudgeConfig


@runtime_checkable
class JudgeClient(Protocol):
    """Minimal lifecycle and inference seam used by the evaluation pipeline."""

    def diagnose(self) -> Mapping[str, Any]: ...

    def capabilities(self) -> Mapping[str, Any]: ...

    def acquire(self) -> Mapping[str, Any]: ...

    def count_tokens(self, messages: list[Mapping[str, Any]]) -> int | None: ...

    def judge(
        self,
        messages: list[Mapping[str, Any]],
        *,
        request_id: str,
    ) -> Mapping[str, Any]: ...

    def release(self) -> Mapping[str, Any] | None: ...


def judge_messages(
    *,
    question: Any,
    reference_answer: Any,
    baseline_answer: Any,
    candidate_answer: Any,
    evidence: list[Mapping[str, Any]],
    judgment_basis: str,
) -> list[dict[str, Any]]:
    """Create one stateless rubric request without asking for chain-of-thought."""
    schema = {
        "baseline": {dimension: "number from 0 to 1" for dimension in JUDGE_DIMENSIONS},
        "candidate": {dimension: "number from 0 to 1" for dimension in JUDGE_DIMENSIONS},
        "decision": "candidate_better | baseline_better | tie | unjudgeable",
        "regression": "boolean",
        "confidence": "number from 0 to 1",
        "rationale": "short decision rationale only",
        "failure_categories": ["short category strings"],
        "evidence_sufficiency": "sufficient | insufficient | not_provided",
    }
    material = {
        "question": question,
        "reference_answer": reference_answer,
        "baseline_answer": baseline_answer,
        "candidate_answer": candidate_answer,
        "judgment_basis": judgment_basis,
        "evidence": list(evidence),
    }
    return [
        {
            "role": "system",
            "content": (
                f"You are a strict saved-output evaluator using {JUDGE_RUBRIC_VERSION}. "
                "Score each answer independently. Use only the supplied material. "
                "Return JSON matching the requested shape. Do not provide hidden "
                "reasoning or chain-of-thought; provide only a concise rationale."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"required_response": schema, "evaluation_material": material},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


class CognityxJudgeClient:
    """Adapt the optional Cognityx Inference client to the judge protocol."""

    def __init__(self, config: JudgeConfig) -> None:
        try:
            from cognityx_inference import CognityxInferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "Evaluation requires cognityx-inference; install the evaluation extra."
            ) from exc
        self.config = config
        self._client = CognityxInferenceClient(
            config.base_url,
            timeout_seconds=config.timeout_seconds,
            discovery_policy=config.discovery_policy,
            backend=config.provider,
            profile=config.server_profile,
            auto_start=config.auto_start,
            manager_url=config.manager_url,
        )
        self._worker = self._client
        self._owned_load = False
        self._acquired = False

    @property
    def owned_load(self) -> bool:
        return self._owned_load

    def diagnose(self) -> Mapping[str, Any]:
        return self._client.diagnose_server(
            model=self.config.model,
            backend=self.config.backend,
            profile=self.config.profile,
        )

    def capabilities(self) -> Mapping[str, Any]:
        return self._client.provider_capabilities(
            self.config.provider,
            self.config.model,
        )

    def acquire(self) -> Mapping[str, Any]:
        if self._acquired:
            return {"loaded_by_evaluation": self._owned_load, "already_acquired": True}
        if self.config.auto_start and self.config.provider == "local":
            worker_url = self._client.ensure_server_ready()
            self._client.set_base_url(worker_url)
            self._worker = self._client
        statuses = self._worker.model_status() if self.config.provider == "local" else []
        resident = any(self._matches_status(item) for item in statuses)
        if self.config.provider == "local" and not resident:
            self._worker.load_model(
                self.config.model,
                self.config.backend,
                self.config.profile,
                self.config.runtime,
                discovery_policy=self.config.discovery_policy,
                required_context_length=self.config.context_limit_tokens,
            )
            self._owned_load = True
        self._acquired = True
        return {
            "loaded_by_evaluation": self._owned_load,
            "found_resident": resident,
            "provider": self.config.provider,
            "model": self.config.model,
            "backend": self.config.backend,
            "profile": self.config.profile,
        }

    def count_tokens(self, messages: list[Mapping[str, Any]]) -> int | None:
        return self._worker.count_input_tokens(
            model=self.config.model,
            messages=messages,
            backend=self.config.backend,
            profile=self.config.profile,
        )

    def judge(
        self,
        messages: list[Mapping[str, Any]],
        *,
        request_id: str,
    ) -> Mapping[str, Any]:
        from cognityx_inference import InferenceRequest

        return self._worker.infer(
            InferenceRequest(
                model=self.config.model,
                messages=tuple(messages),
                response_format={"type": "json_object"},
                client_type="cognityx-training-evaluation",
                provider=self.config.provider,
                backend=self.config.backend,
                profile=self.config.profile,
                load_policy="require_loaded",
                discovery_policy="require_existing",
                required_context_length=self.config.context_limit_tokens,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                seed=self.config.seed,
                timeout_seconds=self.config.timeout_seconds,
                execution_context={
                    "execution_mode": "saved-output-sequential-judge",
                    "candidate_model_loaded": False,
                    "base_student_model_loaded": False,
                },
                request_metadata={
                    "evaluation_request_id": request_id,
                    "rubric_version": JUDGE_RUBRIC_VERSION,
                },
            )
        )

    def release(self) -> Mapping[str, Any] | None:
        if not self._owned_load:
            return None
        result = self._worker.unload_model(
            self.config.model,
            self.config.backend,
            self.config.runtime,
        )
        self._owned_load = False
        return result

    def _matches_status(self, status: Mapping[str, Any]) -> bool:
        identity = status.get("identity")
        source = identity if isinstance(identity, Mapping) else status
        return (
            str(source.get("model") or source.get("model_name") or "") == self.config.model
            and str(source.get("backend") or self.config.backend) == self.config.backend
            and str(source.get("profile") or self.config.profile) == self.config.profile
            and str(status.get("state", "")).upper() not in {"FAILED", "UNLOADED"}
        )


def response_content(response: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract OpenAI-compatible content and normalized request telemetry."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("judge response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("judge response choice is malformed")
    message = first.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("judge response has no message content")
    usage = dict(response.get("usage") or {})
    telemetry = {
        "provider_request_id": response.get("id"),
        "model": response.get("model"),
        "usage": usage,
        "finish_reason": first.get("finish_reason"),
    }
    return message["content"], telemetry
