"""Custom Transformers/PyTorch LoRA and QLoRA training backend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from cognityx_core import Artifact, ModelArtifactRegistry, TrainingBackend
from cognityx_core.models import TrainingRequest, TrainingResult

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.dataset_pipeline import load_jsonl_records, render_ift_examples


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

        model_source_options = {
            "cache_dir": str(self.config.model_cache_dir),
            "local_files_only": self.config.local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            **model_source_options,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_source_options,
            **model_kwargs,
        )
        if self.config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
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

        records = load_jsonl_records(request.dataset.uri)
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
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
        )
        input_device = model.get_input_embeddings().weight.device
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses: list[float] = []
        completed_steps = 0
        while completed_steps < self.config.max_steps:
            for input_ids, attention_mask, batch_labels in loader:
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
                    if completed_steps >= self.config.max_steps:
                        break
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        metrics = {
            "train_loss": sum(losses) / len(losses),
            "train_steps": completed_steps,
            "train_runtime_seconds": time.perf_counter() - started,
            "training_examples": len(dataset),
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
                "configuration": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(self.config).items()
                },
            },
        )
        if self.artifact_registry is not None:
            artifact = self.artifact_registry.register(artifact)
        return TrainingResult(artifact=artifact, metrics=metrics)
