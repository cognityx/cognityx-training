"""Custom Transformers/PyTorch LoRA and QLoRA training backend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
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
            from datasets import Dataset as HuggingFaceDataset
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Training dependencies are missing. Run `uv sync --extra training`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
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
        dataset = HuggingFaceDataset.from_dict({"text": texts})

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_sequence_length,
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(output_dir),
                max_steps=self.config.max_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                seed=self.config.seed,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                bf16=torch.cuda.is_available(),
                fp16=False,
                remove_unused_columns=False,
            ),
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        train_output = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        artifact = Artifact(
            name=f"{request.dataset.name}-qwen-adapter",
            version="0.1.0",
            uri=output_dir.resolve().as_uri(),
            metadata={
                "base_model": self.config.model_name,
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
        return TrainingResult(artifact=artifact, metrics=dict(train_output.metrics))
