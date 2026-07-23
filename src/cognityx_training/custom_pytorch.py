"""Custom Transformers/PyTorch LoRA and QLoRA training backend."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

from cognityx_core import Artifact, ModelArtifactRegistry, TrainingBackend
from cognityx_core.models import TrainingRequest, TrainingResult

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.dataset_pipeline import (
    evaluation_pair,
    load_jsonl_records,
    partition_records,
    render_ift_examples,
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


_PERSISTENT_MODEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
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
            prompt, expected = evaluation_pair(record)
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(input_device)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(
                output_ids[0, input_ids.shape[-1] :], skip_special_tokens=True
            ).strip()
            normalized_expected = " ".join(expected.lower().split())
            normalized_generated = " ".join(generated.lower().split())
            results.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated,
                    "exact_match": normalized_generated == normalized_expected,
                    "contains_expected": normalized_expected in normalized_generated,
                }
            )
    if was_training:
        model.train()
    count = len(results)
    return {
        "example_count": count,
        "exact_match_accuracy": (
            sum(item["exact_match"] for item in results) / count if count else None
        ),
        "contains_expected_accuracy": (
            sum(item["contains_expected"] for item in results) / count if count else None
        ),
        "outputs": results,
    }


class CustomPyTorchTrainerBackend(TrainingBackend):
    """Run a minimal supervised fine-tuning job using Transformers and PEFT."""

    def __init__(
        self,
        config: CustomPyTorchTrainingConfig,
        artifact_registry: ModelArtifactRegistry | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.artifact_registry = artifact_registry

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute LoRA/QLoRA supervised fine-tuning.

        Heavy training libraries are imported lazily so configuration, factories,
        and dataset validation remain usable without the ``training`` extra.
        """
        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Training dependencies are missing. Run `uv sync --extra training`."
            ) from exc

        all_records = load_jsonl_records(request.dataset.uri)
        records, evaluation_records = partition_records(
            all_records, self.config.max_examples
        )

        model_source_options = {
            "cache_dir": str(self.config.model_cache_dir),
            "local_files_only": self.config.local_files_only,
        }
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
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                **model_source_options,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
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
            evaluation_records,
            input_device,
            self.config.evaluation_max_new_tokens,
            torch,
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

        texts = render_ift_examples(records, tokenizer)
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        dataset = torch.utils.data.TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            labels,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        run_id = self.config.run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ) + f"-{uuid.uuid4().hex[:8]}"
        output_dir = Path(self.config.output_dir) / run_id
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
        monitor.start()
        started_at = utc_now()
        started = time.perf_counter()
        losses: list[float] = []
        step_times: list[float] = []
        completed_steps = 0
        while completed_steps < self.config.max_steps:
            for input_ids, attention_mask, batch_labels in loader:
                step_started = time.perf_counter()
                outputs = model(
                    input_ids=input_ids.to(input_device),
                    attention_mask=attention_mask.to(input_device),
                    labels=batch_labels.to(input_device),
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
                        print(
                            format_progress(
                                completed_steps,
                                self.config.max_steps,
                                losses[-1],
                                monitor.snapshot(),
                            ),
                            flush=True,
                        )
                    if completed_steps >= self.config.max_steps:
                        step_times.append(time.perf_counter() - step_started)
                        break
                step_times.append(time.perf_counter() - step_started)
        print(
            "COGNITYX_TRAINING_COMPLETED (optimizer); saving adapter and evaluating...",
            flush=True,
        )
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trained_evaluation = _evaluate_model(
            model,
            tokenizer,
            evaluation_records,
            input_device,
            self.config.evaluation_max_new_tokens,
            torch,
        )
        runtime_seconds = time.perf_counter() - started
        resources = monitor.stop()
        finished_at = utc_now()
        metrics = {
            "train_loss": sum(losses) / len(losses),
            "train_steps": completed_steps,
            "train_runtime_seconds": runtime_seconds,
            "training_examples": len(dataset),
            "dataset_examples_available": len(all_records),
            "dataset_examples_selected": len(records),
            "evaluation_examples": len(evaluation_records),
            "micro_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": (
                self.config.per_device_train_batch_size
                * self.config.gradient_accumulation_steps
            ),
        }

        artifact = Artifact(
            name=f"{request.dataset.name}-qwen-adapter",
            version="0.1.0",
            uri=output_dir.resolve().as_uri(),
            metadata={
                "base_model": self.config.model_name,
                "model_cache_dir": str(self.config.model_cache_dir),
                "local_files_only": self.config.local_files_only,
                "backend": self.config.backend,
                "qlora": self.config.load_in_4bit,
                "configuration": jsonable_configuration(self.config),
            },
        )
        if self.artifact_registry is not None:
            artifact = self.artifact_registry.register(artifact)
        adapter_gpu_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.device.type == "cuda"
        )
        report = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": runtime_seconds,
            "training_type": "qlora" if self.config.load_in_4bit else "lora",
            "model": {"name": self.config.model_name, "output_uri": artifact.uri},
            "dataset": {
                "name": request.dataset.name,
                "version": request.dataset.version,
                "uri": request.dataset.uri,
            },
            "configuration": jsonable_configuration(self.config),
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
                "exact_match_change": (
                    trained_evaluation["exact_match_accuracy"]
                    - baseline_evaluation["exact_match_accuracy"]
                    if evaluation_records
                    else None
                ),
                "contains_expected_change": (
                    trained_evaluation["contains_expected_accuracy"]
                    - baseline_evaluation["contains_expected_accuracy"]
                    if evaluation_records
                    else None
                ),
            },
            "adapter_usage": {
                "adapter_type": "qlora" if self.config.load_in_4bit else "lora",
                "parameter_count": final_parameters["trainable"],
                "gpu_resident_bytes": adapter_gpu_bytes,
                "gpu_resident_peak_bytes": adapter_gpu_bytes,
                "serialized_size_bytes": directory_size(output_dir),
                "saved_uri": artifact.uri,
            },
            "metrics": metrics,
            "environment": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            },
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
        report_path = write_training_report(report, output_dir)
        metrics["report_uri"] = report_path.resolve().as_uri()
        try:
            return TrainingResult(
                artifact=artifact,
                metrics=metrics,
                report_uri=report_path.resolve().as_uri(),
            )
        except TypeError:
            return TrainingResult(artifact=artifact, metrics=metrics)
