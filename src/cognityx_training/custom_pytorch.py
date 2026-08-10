"""Custom Transformers/PyTorch LoRA and QLoRA training backend."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import gc
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from cognityx_core import Artifact, ModelArtifactRegistry, TrainingBackend
from cognityx_core.models import TrainingRequest, TrainingResult

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.dataset_pipeline import (
    DataForgeDatasetReader,
    collate_supervised_batch,
    encode_supervised_example,
    iter_selected_training_examples,
    preflight_dataset,
)
from cognityx_training.lineage import build_lineage_ids
from cognityx_training.publication import (
    TrainingPublisher,
    canonical_variant_identity,
    prediction_rows,
    runtime_environment,
)
from cognityx_training.reporting import (
    ResourceMonitor,
    directory_size,
    format_progress,
    jsonable_configuration,
    latency_summary,
    parameter_counts,
    utc_now,
    write_training_report,
)
from cognityx_training.telemetry import query_host, query_nvidia_gpus
from cognityx_training.storage_runtime import resolve_storage_runtime
from cognityx_training.tracking import TrackingSession, create_tracking_session


_PERSISTENT_MODEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}
try:
    import torch
except ImportError:  # pragma: no cover - imported lazily in normal training paths
    torch = None

_IterableDatasetBase = torch.utils.data.IterableDataset if torch is not None else object


def evaluation_changes(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    evaluation_count: int,
) -> dict[str, float | None]:
    if evaluation_count <= 0:
        return {"exact_match_change": None, "contains_expected_change": None}
    return {
        "exact_match_change": (
            trained["exact_match_accuracy"] - baseline["exact_match_accuracy"]
        ),
        "contains_expected_change": (
            trained["contains_expected_accuracy"] - baseline["contains_expected_accuracy"]
        ),
    }


def _evaluation_tracking_events(
    evaluation: dict[str, Any],
    *,
    phase: str,
) -> list[tuple[dict[str, Any], dict[str, float]]]:
    """Return scalar evaluation events with their research-suite identity."""
    events: list[tuple[dict[str, Any], dict[str, float]]] = []
    for role, values in sorted(evaluation.get("suite_metrics", {}).items()):
        metrics = {
            name: float(value)
            for name, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        events.append((
            {
                "phase": phase,
                "research_role": role,
                "evaluation_sets": list(values.get("evaluation_sets") or []),
            },
            metrics,
        ))
    return events


def _live_tracking_metrics(
    *,
    step: int,
    loss: float,
    examples_processed: int,
    input_tokens_processed: int,
    target_tokens_processed: int,
    elapsed_seconds: float,
    resource_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build live metrics only from values already measured by Training."""
    metrics: dict[str, Any] = {
        "training.global_step": step,
        "training.optimizer_step": step,
        "training.loss": loss,
        "training.examples_processed": examples_processed,
        "training.input_tokens_processed": input_tokens_processed,
        "training.target_tokens_processed": target_tokens_processed,
        "training.elapsed_seconds": elapsed_seconds,
    }
    if elapsed_seconds > 0:
        metrics.update({
            "training.examples_per_second": examples_processed / elapsed_seconds,
            "training.input_tokens_per_second": input_tokens_processed / elapsed_seconds,
            "training.target_tokens_per_second": target_tokens_processed / elapsed_seconds,
        })
    metrics.update({
        "resource.process.cpu_percent": resource_snapshot.get("cpu_percent"),
        "resource.process.ram_bytes": resource_snapshot.get("ram_bytes"),
        "resource.process.disk_read_bytes": resource_snapshot.get("disk_read_bytes"),
        "resource.process.disk_write_bytes": resource_snapshot.get("disk_write_bytes"),
    })
    host = dict(resource_snapshot.get("host") or {})
    if host:
        scope = str(host.get("scope") or "unspecified").replace(".", "_")
        prefix = f"resource.host.{scope}"
        metrics.update({
            f"{prefix}.cpu_percent": host.get("cpu_percent"),
            f"{prefix}.ram_used_bytes": host.get("used_bytes"),
            f"{prefix}.ram_used_percent": host.get("used_percent"),
            f"{prefix}.ram_total_bytes": host.get("total_bytes"),
        })
    gpu = dict(resource_snapshot.get("gpu") or {})
    if gpu:
        scope = str(gpu.get("scope") or "unspecified").replace(".", "_")
        index = gpu.get("device_index", 0)
        prefix = f"resource.gpu.{scope}.device_{index}"
        metrics.update({
            f"{prefix}.utilization_percent": gpu.get("utilization_percent"),
            f"{prefix}.memory_used_bytes": gpu.get("memory_used_bytes"),
            f"{prefix}.memory_total_bytes": gpu.get("memory_total_bytes"),
            f"{prefix}.framework_allocated_bytes": gpu.get("framework_allocated_bytes"),
            f"{prefix}.framework_reserved_bytes": gpu.get("framework_reserved_bytes"),
            f"{prefix}.power_draw_watts": gpu.get("power_draw_watts"),
        })
        gpu_aggregate = dict(resource_snapshot.get("gpu_aggregate") or {})
        if gpu_aggregate.get("energy_joules") is not None:
            metrics[f"{prefix}.accumulated_energy_joules"] = gpu_aggregate["energy_joules"]
    return {key: value for key, value in metrics.items() if value is not None}


class _StreamingTrainingDataset(_IterableDatasetBase):
    def __init__(
        self,
        reader: DataForgeDatasetReader,
        tokenizer: Any,
        *,
        max_examples: int | None,
        max_sequence_length: int,
        overlength_policy: str,
    ) -> None:
        super().__init__()
        self.reader = reader
        self.tokenizer = tokenizer
        self.max_examples = max_examples
        self.max_sequence_length = max_sequence_length
        self.overlength_policy = overlength_policy

    def __iter__(self):
        emitted = 0
        for selected in iter_selected_training_examples(
            self.reader,
            self.tokenizer,
            max_examples=self.max_examples,
            max_sequence_length=self.max_sequence_length,
            overlength_policy=self.overlength_policy,
        ):
            emitted += 1
            yield {"input_ids": selected.input_ids, "labels": selected.labels}
        if emitted == 0:
            raise ValueError("No trainable examples remain after filtering.")


def _evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[Any],
    input_device: Any,
    max_new_tokens: int,
    torch: Any,
) -> dict[str, Any]:
    """Generate held-out answers and score normalized exact matches."""
    results: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for record in records:
            messages = record.messages if hasattr(record, "messages") else record["messages"]
            prompt = next(item["content"] for item in reversed(messages) if item["role"] == "user")
            expected = next(item["content"] for item in reversed(messages) if item["role"] == "assistant")
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            if not hasattr(inputs, "items"):
                raise TypeError("Evaluation chat template must return a BatchEncoding-like mapping")
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            if "input_ids" not in inputs:
                raise ValueError("Evaluation chat template did not return input_ids")
            input_length = inputs["input_ids"].shape[-1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(
                output_ids[0, input_length:], skip_special_tokens=True
            ).strip()
            normalized_expected = " ".join(expected.lower().split())
            normalized_generated = " ".join(generated.lower().split())
            record_metadata = dict(getattr(record, "metadata", {}))
            results.append(
                {
                    "record_id": getattr(record, "record_id", None),
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated,
                    "exact_match": normalized_generated == normalized_expected,
                    "contains_expected": normalized_expected in normalized_generated,
                    "knowledge_unit_ids": list(
                        record_metadata.get("knowledge_unit_ids", [])
                    ),
                    "evidence_ids": list(
                        record_metadata.get("evidence_ids", [])
                    ),
                    "provenance": dict(record_metadata.get("provenance", {})),
                    "metadata": record_metadata,
                }
            )
    if was_training:
        model.train()
    count = len(results)
    suites: dict[str, dict[str, Any]] = {}
    suite_rows: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        role = str(row.get("metadata", {}).get("research_role") or "legacy_evaluation")
        suite_rows.setdefault(role, []).append(row)
    for role, rows in sorted(suite_rows.items()):
        suite_count = len(rows)
        suites[role] = {
            "example_count": suite_count,
            "exact_match_accuracy": sum(item["exact_match"] for item in rows) / suite_count,
            "contains_expected_accuracy": sum(item["contains_expected"] for item in rows) / suite_count,
            "record_ids": [item.get("record_id") for item in rows],
            "evaluation_sets": [
                {"evaluation_set_id": identity[0], "evaluation_set_version": identity[1]}
                for identity in sorted(
                    {
                        (
                            item.get("metadata", {}).get("evaluation_set_id"),
                            item.get("metadata", {}).get("evaluation_set_version"),
                        )
                        for item in rows
                        if item.get("metadata", {}).get("evaluation_set_id")
                    },
                    key=lambda value: (str(value[0]), str(value[1])),
                )
            ],
        }
    return {
        "example_count": count,
        "exact_match_accuracy": (
            sum(item["exact_match"] for item in results) / count if count else None
        ),
        "contains_expected_accuracy": (
            sum(item["contains_expected"] for item in results) / count if count else None
        ),
        "suite_metrics": suites,
        "outputs": results,
    }


class CustomPyTorchTrainerBackend(TrainingBackend):
    """Run a minimal supervised fine-tuning job using Transformers and PEFT."""

    def __init__(
        self,
        config: CustomPyTorchTrainingConfig,
        artifact_registry: ModelArtifactRegistry | None = None,
        storage_runtime: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.artifact_registry = artifact_registry
        self.storage_runtime = storage_runtime
        self._active_publisher: TrainingPublisher | None = None
        self._tracking_session: TrackingSession | None = None
        self._publication_phase = "initialization"

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute training and preserve an immutable failure record when possible."""
        self._active_publisher = None
        self._tracking_session = None
        self._publication_phase = "initialization"
        try:
            return self._train(request)
        except BaseException as exc:
            if self._active_publisher is not None:
                self._active_publisher.publish_failure(
                    exc,
                    phase=self._publication_phase,
                )
            if self._tracking_session is not None:
                try:
                    self._tracking_session.fail(
                        exc,
                        {
                            "phase": self._publication_phase,
                            "failed_at": utc_now(),
                        },
                    )
                except Exception as tracking_exc:
                    exc.add_note(f"Tracking failure reporting also failed: {tracking_exc}")
            raise

    def _train(self, request: TrainingRequest) -> TrainingResult:
        """Execute LoRA/QLoRA supervised fine-tuning.

        Heavy training libraries are imported lazily so configuration, factories,
        and dataset validation remain usable without the ``training`` extra.
        """
        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Training dependencies are missing. Run `uv sync --extra training`."
            ) from exc

        storage_runtime = resolve_storage_runtime(
            storage_runtime=self.storage_runtime,
            storage_config=self.config.storage_config,
            storage_root=self.config.storage_root,
        )
        reader = DataForgeDatasetReader(
            request.dataset.uri,
            storage_runtime=storage_runtime,
            input_mode=self.config.dataset_input_mode,
        )

        model_source_options = {
            "cache_dir": str(self.config.model_cache_dir),
            "local_files_only": self.config.local_files_only,
        }
        if self.config.model_revision is not None:
            model_source_options["revision"] = self.config.model_revision
        tokenizer_source_options = dict(model_source_options)
        if self.config.tokenizer_revision is not None:
            tokenizer_source_options["revision"] = self.config.tokenizer_revision
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            **tokenizer_source_options,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        preflight = preflight_dataset(
            reader,
            tokenizer,
            max_examples=self.config.max_examples,
            max_sequence_length=self.config.max_sequence_length,
            overlength_policy=self.config.overlength_policy,
        )
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            **model_source_options,
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        base_model_identity = {
            "name": self.config.model_name,
            "requested_revision": self.config.model_revision,
            "resolved_revision": getattr(model_config, "_commit_hash", None),
            "tokenizer_revision": (
                getattr(tokenizer, "_commit_hash", None)
                or self.config.tokenizer_revision
            ),
            "chat_template_checksum": (
                hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
                if isinstance(chat_template, str)
                else None
            ),
        }
        variant_identity = canonical_variant_identity(
            self.config,
            preflight.lineage,
            base_model_identity=base_model_identity,
        )
        ids = build_lineage_ids(
            variant_identity,
            requested_experiment_id=self.config.experiment_id,
            requested_run_id=self.config.training_run_id or self.config.run_id,
        )
        publisher: TrainingPublisher | None = None
        if self.config.publication_mode == "storage":
            publisher = TrainingPublisher(
                storage_runtime,
                ids,
                experiment_name=self.config.experiment_name,
                experiment_description=self.config.experiment_description,
                experiment_created_by=self.config.experiment_created_by,
                experiment_tags=self.config.experiment_tags,
            )
            self._active_publisher = publisher
            self._publication_phase = "request-publication"
            publisher.publish_experiment()
            publisher.publish_variant(
                variant_identity,
                dataset_lineage=preflight.lineage,
                base_model_identity=base_model_identity,
            )
            publisher.publish_training_request(
                dataset_lineage=preflight.lineage,
                normalized_request=variant_identity["training"],
                base_model_identity=base_model_identity,
                publication_mode=self.config.publication_mode,
            )
        tracking_started_at = utc_now()
        self._tracking_session = create_tracking_session(
            backend=self.config.tracking_backend,
            tracking_uri=self.config.tracking_uri,
            experiment_name=self.config.tracking_experiment_name,
            run_name=self.config.tracking_run_name,
            parent_run_id=self.config.tracking_parent_run_id,
            failure_policy=self.config.tracking_failure_policy,
            context={
                "identity": ids.to_dict(),
                "parameters": variant_identity,
                "metadata": {
                    "registration_mode": "live",
                    "tracking_started_at": tracking_started_at,
                },
                "idempotency_keys": [ids.training_run_id],
            },
        )
        self._publication_phase = "model-loading"
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if self.config.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("QLoRA 4-bit training requires a CUDA GPU.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        cache_key = (
            self.config.model_name,
            str(self.config.model_cache_dir),
            self.config.local_files_only,
            self.config.load_in_4bit,
        )
        cached = _PERSISTENT_MODEL_CACHE.pop(cache_key, None)
        if cached is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_source_options,
                **model_kwargs,
            )
            if self.config.load_in_4bit:
                model = prepare_model_for_kbit_training(
                    model,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            print("COGNITYX_MODEL_LOADED fresh", flush=True)
        else:
            model, tokenizer = cached
            print("COGNITYX_MODEL_LOADED reused", flush=True)
        original_parameters = parameter_counts(model)
        input_device = model.get_input_embeddings().weight.device
        baseline_evaluation = _evaluate_model(
            model,
            tokenizer,
            reader.iter_evaluation_records(),
            input_device,
            self.config.evaluation_max_new_tokens,
            torch,
        )
        for suite_identity, suite_metrics in _evaluation_tracking_events(
            baseline_evaluation,
            phase="baseline",
        ):
            self._tracking_session.log_evaluation(
                suite_identity,
                suite_metrics,
                step=0,
            )
        model = get_peft_model(
            model,
            LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.target_modules),
                task_type="CAUSAL_LM",
            ),
        )
        final_parameters = parameter_counts(model)

        dataset = _StreamingTrainingDataset(
            reader,
            tokenizer,
            max_examples=self.config.max_examples,
            max_sequence_length=self.config.max_sequence_length,
            overlength_policy=self.config.overlength_policy,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            collate_fn=lambda batch: collate_supervised_batch(
                batch,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        staging_segment = (
            self.config.run_id
            if self.config.publication_mode == "local" and self.config.run_id
            else ids.training_run_id
        )
        output_dir = Path(self.config.output_dir) / staging_segment
        output_dir.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
        )
        input_device = model.get_input_embeddings().weight.device
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def sample_gpu() -> dict[str, Any] | None:
            if not torch.cuda.is_available():
                return None
            device_index = torch.cuda.current_device()
            try:
                gpu = query_nvidia_gpus(self.config.nvidia_smi_path)[device_index]
            except (OSError, subprocess.SubprocessError, IndexError):
                gpu = {
                    "source": "pytorch_cuda",
                    "scope": "framework_process",
                    "device": torch.cuda.get_device_name(device_index),
                    "device_index": device_index,
                    "utilization_percent": None,
                    "memory_used_bytes": torch.cuda.memory_reserved(device_index),
                    "memory_total_bytes": torch.cuda.get_device_properties(
                        device_index
                    ).total_memory,
                }
            gpu["framework_allocated_bytes"] = torch.cuda.memory_allocated(device_index)
            gpu["framework_reserved_bytes"] = torch.cuda.memory_reserved(device_index)
            gpu["sampled_at_monotonic"] = time.monotonic()
            return gpu

        monitor = ResourceMonitor(
            gpu_sampler=sample_gpu,
            host_sampler=lambda: query_host(
                self.config.host_telemetry_source,
                self.config.host_installed_memory_gib,
            ),
            interval_seconds=self.config.resource_sample_interval_seconds,
        )
        print("COGNITYX_TRAINING_STARTED", flush=True)
        self._publication_phase = "training"
        monitor.start()
        started_at = utc_now()
        started = time.perf_counter()
        losses: list[float] = []
        step_times: list[float] = []
        completed_steps = 0
        training_examples_processed = 0
        training_input_tokens_processed = 0
        training_target_tokens_processed = 0
        while completed_steps < self.config.max_steps:
            for batch in loader:
                step_started = time.perf_counter()
                training_examples_processed += int(batch["input_ids"].shape[0])
                training_input_tokens_processed += int(batch["attention_mask"].sum().item())
                training_target_tokens_processed += int((batch["labels"] != -100).sum().item())
                outputs = model(
                    input_ids=batch["input_ids"].to(input_device),
                    attention_mask=batch["attention_mask"].to(input_device),
                    labels=batch["labels"].to(input_device),
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                losses.append(float(outputs.loss.detach().cpu()))
                if len(losses) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed_steps += 1
                    if (
                        completed_steps == 1
                        or completed_steps % self.config.progress_interval_steps == 0
                        or completed_steps == self.config.max_steps
                    ):
                        resource_snapshot = monitor.snapshot()
                        print(
                            format_progress(
                                completed_steps,
                                self.config.max_steps,
                                losses[-1],
                                resource_snapshot,
                            ),
                            flush=True,
                        )
                        self._tracking_session.log_metrics(
                            _live_tracking_metrics(
                                step=completed_steps,
                                loss=losses[-1],
                                examples_processed=training_examples_processed,
                                input_tokens_processed=training_input_tokens_processed,
                                target_tokens_processed=training_target_tokens_processed,
                                elapsed_seconds=time.perf_counter() - started,
                                resource_snapshot=resource_snapshot,
                            ),
                            step=completed_steps,
                        )
                    if completed_steps >= self.config.max_steps:
                        step_times.append(time.perf_counter() - step_started)
                        break
                step_times.append(time.perf_counter() - step_started)
        print(
            "COGNITYX_TRAINING_COMPLETED (optimizer); saving adapter and evaluating...",
            flush=True,
        )
        self._publication_phase = "adapter-staging"
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trained_evaluation = _evaluate_model(
            model,
            tokenizer,
            reader.iter_evaluation_records(),
            input_device,
            self.config.evaluation_max_new_tokens,
            torch,
        )
        for suite_identity, suite_metrics in _evaluation_tracking_events(
            trained_evaluation,
            phase="trained",
        ):
            self._tracking_session.log_evaluation(
                suite_identity,
                suite_metrics,
                step=completed_steps,
            )
        self._publication_phase = "reporting"
        runtime_seconds = time.perf_counter() - started
        resources = monitor.stop()
        finished_at = utc_now()
        metrics = {
            "train_loss": sum(losses) / len(losses),
            "train_steps": completed_steps,
            "train_runtime_seconds": runtime_seconds,
            "training_examples": preflight.statistics.accepted_training_examples,
            "training_examples_processed": training_examples_processed,
            "training_input_tokens_processed": training_input_tokens_processed,
            "training_target_tokens_processed": training_target_tokens_processed,
            "dataset_examples_available": preflight.statistics.total_records,
            "dataset_examples_selected": preflight.statistics.selected_training_candidates,
            "evaluation_examples": preflight.statistics.evaluation_records,
            "evaluation_suite_counts": dict(preflight.statistics.evaluation_suite_counts),
            "micro_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": (
                self.config.per_device_train_batch_size
                * self.config.gradient_accumulation_steps
            ),
        }
        environment = runtime_environment(
            storage_runtime,
            torch_module=torch,
            base_model_identity=base_model_identity,
        )

        artifact = Artifact(
            name=f"{request.dataset.name}-qwen-adapter",
            version="0.1.0",
            uri=output_dir.resolve().as_uri(),
            metadata={
                **ids.to_dict(),
                "base_model": self.config.model_name,
                "model_cache_dir": str(self.config.model_cache_dir),
                "local_files_only": self.config.local_files_only,
                "backend": self.config.backend,
                "qlora": self.config.load_in_4bit,
                "configuration": jsonable_configuration(self.config),
            },
        )
        if (
            self.artifact_registry is not None
            and self.config.publication_mode == "local"
        ):
            artifact = self.artifact_registry.register(artifact)
        adapter_gpu_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.device.type == "cuda"
        )
        report = {
            "schema_version": "1.0",
            "run_id": ids.training_run_id,
            **ids.to_dict(),
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": runtime_seconds,
            "training_type": "qlora" if self.config.load_in_4bit else "lora",
            "model": {
                "name": self.config.model_name,
                "output_uri": (
                    publisher.model_store.uri(publisher.adapter_root)
                    if publisher is not None
                    else artifact.uri
                ),
            },
            "dataset": {
                "name": request.dataset.name,
                "version": request.dataset.version,
                "uri": request.dataset.uri,
                "lineage": {
                    "dataset_id": preflight.lineage.dataset_id,
                    "dataset_name": preflight.lineage.dataset_name,
                    "dataset_version": preflight.lineage.dataset_version,
                    "dataset_variant_id": preflight.lineage.dataset_variant_id,
                    "dataset_manifest_uri": preflight.lineage.dataset_manifest_uri,
                    "dataset_manifest_checksum": preflight.lineage.dataset_manifest_checksum,
                    "records_uri": preflight.lineage.records_uri,
                    "records_checksum": preflight.lineage.records_checksum,
                    "recipe": preflight.lineage.recipe,
                    "source_manifest_uri": preflight.lineage.source_manifest_uri,
                    "source_manifest_checksum": preflight.lineage.source_manifest_checksum,
                    "configuration_checksum": preflight.lineage.configuration_checksum,
                    "split_summary": preflight.statistics.split_summary,
                    "record_counts": {
                        "total": preflight.statistics.total_records,
                        "training": preflight.statistics.training_records,
                        "evaluation": preflight.statistics.evaluation_records,
                        "selected_training_candidates": preflight.statistics.selected_training_candidates,
                        "accepted_training_examples": preflight.statistics.accepted_training_examples,
                        "skipped_overlength": preflight.statistics.skipped_overlength_count,
                    },
                    "evaluation_suite_counts": dict(preflight.statistics.evaluation_suite_counts),
                    "manifest_record_counts": preflight.lineage.record_counts,
                    "research_package_manifest_uri": preflight.lineage.research_package_manifest_uri,
                    "research_package_manifest_checksum": preflight.lineage.research_package_manifest_checksum,
                    "research_package_id": preflight.lineage.research_package_id,
                    "research_package_version": preflight.lineage.research_package_version,
                    "evaluation_sets": list(preflight.lineage.evaluation_sets),
                    "overlength_policy": preflight.lineage.overlength_policy,
                    "skipped_records": list(
                        preflight.statistics.skipped_record_samples
                    ),
                },
            },
            "configuration": jsonable_configuration(self.config),
            "data_order": self.config.data_order,
            "parameter_counts": {
                "original_total": original_parameters["total"],
                "original_trainable": original_parameters["trainable"],
                "final_total": final_parameters["total"],
                "final_trainable": final_parameters["trainable"],
            },
            "system_usage": {
                key: value for key, value in resources.items() if key != "gpu_usage"
            },
            "gpu_usage": resources["gpu_usage"],
            "response_times": [latency_summary("training_step", step_times)],
            "evaluation": {
                "baseline": baseline_evaluation,
                "trained": trained_evaluation,
                **evaluation_changes(
                    baseline_evaluation,
                    trained_evaluation,
                    preflight.statistics.evaluation_records,
                ),
            },
            "adapter_usage": {
                "adapter_type": "qlora" if self.config.load_in_4bit else "lora",
                "parameter_count": final_parameters["trainable"],
                "gpu_resident_bytes": adapter_gpu_bytes,
                "gpu_resident_peak_bytes": adapter_gpu_bytes,
                "serialized_size_bytes": directory_size(output_dir),
                "saved_uri": (
                    publisher.model_store.uri(publisher.adapter_root)
                    if publisher is not None
                    else artifact.uri
                ),
            },
            "metrics": metrics,
            "environment": environment,
            "error": None,
        }
        reuse_enabled = os.environ.get("COGNITYX_REUSE_LOADED_MODEL") == "1"
        cleanup = {
            "location": "gpu" if torch.cuda.is_available() else "cpu",
            "weights_reloaded_from_disk": cached is None,
            "adapter_recreated": True,
            "optimizer_recreated": True,
            "cuda_allocated_before_cleanup_bytes": (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else None
            ),
            "cuda_reserved_before_cleanup_bytes": (
                torch.cuda.memory_reserved() if torch.cuda.is_available() else None
            ),
        }
        if reuse_enabled:
            del optimizer
            base_model = model.unload()
            _PERSISTENT_MODEL_CACHE[cache_key] = (base_model, tokenizer)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            cleanup.update(
                cuda_allocated_after_cleanup_bytes=(
                    torch.cuda.memory_allocated() if torch.cuda.is_available() else None
                ),
                cuda_reserved_after_cleanup_bytes=(
                    torch.cuda.memory_reserved() if torch.cuda.is_available() else None
                ),
                retained_base_parameter_bytes=sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in base_model.parameters()
                ),
            )
        report["base_model_reuse"] = cleanup
        metrics.update(ids.to_dict())
        tracking_references: dict[str, Any]
        tracking_checksums: dict[str, Any]
        if publisher is not None:
            self._publication_phase = "storage-publication"
            decoding = {
                "max_new_tokens": self.config.evaluation_max_new_tokens,
                "do_sample": False,
            }
            publication = publisher.publish_completed_run(
                staging_directory=output_dir,
                dataset_lineage=preflight.lineage,
                base_model_identity=base_model_identity,
                adapter_details={
                    "type": "qlora" if self.config.load_in_4bit else "lora",
                    "format": "peft",
                    "rank": self.config.lora_rank,
                    "alpha": self.config.lora_alpha,
                    "dropout": self.config.lora_dropout,
                    "target_modules": list(self.config.target_modules),
                },
                resolved_config=jsonable_configuration(self.config),
                environment=environment,
                training_report=report,
                metrics=metrics,
                baseline_predictions=prediction_rows(
                    baseline_evaluation,
                    prediction_type="baseline",
                    ids=ids,
                    base_model_identity=base_model_identity,
                    decoding=decoding,
                ),
                trained_predictions=prediction_rows(
                    trained_evaluation,
                    prediction_type="trained",
                    ids=ids,
                    base_model_identity=base_model_identity,
                    decoding=decoding,
                ),
                retain_local_staging=self.config.retain_local_staging,
            )
            artifact = Artifact(
                name=artifact.name,
                version="1",
                uri=publication.adapter_uri,
                metadata={
                    **dict(artifact.metadata),
                    "adapter_manifest_uri": publication.adapter_manifest_uri,
                    "publication_manifest_uri": publication.publication_manifest_uri,
                },
            )
            metrics.update(
                {
                    "adapter_manifest_uri": publication.adapter_manifest_uri,
                    "report_uri": publication.training_report_uri,
                    "baseline_predictions_uri": publication.baseline_predictions_uri,
                    "trained_predictions_uri": publication.trained_predictions_uri,
                    "publication_manifest_uri": publication.publication_manifest_uri,
                }
            )
            report_uri = publication.training_report_uri
            tracking_references = {
                "publication_manifest_uri": publication.publication_manifest_uri,
                "adapter_uri": publication.adapter_uri,
                "adapter_manifest_uri": publication.adapter_manifest_uri,
                "training_report_uri": publication.training_report_uri,
                "baseline_predictions_uri": publication.baseline_predictions_uri,
                "trained_predictions_uri": publication.trained_predictions_uri,
                "dataset_manifest_uri": preflight.lineage.dataset_manifest_uri,
                "research_package_manifest_uri": preflight.lineage.research_package_manifest_uri,
            }
            tracking_checksums = dict(publication.artifact_checksums)
        else:
            report_path = write_training_report(report, output_dir)
            report_uri = report_path.resolve().as_uri()
            metrics["report_uri"] = report_uri
            tracking_references = {
                "training_report_uri": report_uri,
                "adapter_uri": artifact.uri,
                "dataset_manifest_uri": preflight.lineage.dataset_manifest_uri,
                "research_package_manifest_uri": preflight.lineage.research_package_manifest_uri,
            }
            tracking_checksums = {
                "dataset_records": preflight.lineage.records_checksum,
                "research_package_manifest": preflight.lineage.research_package_manifest_checksum,
            }
        self._tracking_session.log_metrics(
            {
                **metrics,
                "resources": {
                    "system_usage": report["system_usage"],
                    "gpu_usage": report["gpu_usage"],
                },
            },
            step=completed_steps,
        )
        self._tracking_session.log_artifact_references(
            tracking_references,
            tracking_checksums,
        )
        tracking_result = self._tracking_session.finish(
            "completed",
            {
                "registration_mode": "live",
                "training_started_at": started_at,
                "training_finished_at": finished_at,
                "training_duration_seconds": runtime_seconds,
                "storage_publication_authoritative": True,
            },
        )
        metrics.update({
            "tracking_backend": tracking_result.backend,
            "tracking_status": tracking_result.status,
            "tracking_external_run_id": tracking_result.external_run_id,
        })
        try:
            return TrainingResult(
                artifact=artifact,
                metrics=metrics,
                report_uri=report_uri,
            )
        except TypeError:
            return TrainingResult(artifact=artifact, metrics=metrics)
