"""Typed configuration for the custom PyTorch training backend."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from cognityx_core import BackendConfig

DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_HUGGING_FACE_HOME = Path(
    os.environ.get("HF_HOME", "/mnt/d/AI/models/huggingface")
)
DEFAULT_HUGGING_FACE_CACHE = Path(
    os.environ.get("HF_HUB_CACHE", DEFAULT_HUGGING_FACE_HOME / "hub")
)


@dataclass(frozen=True, slots=True)
class CustomPyTorchTrainingConfig(BackendConfig):
    """Configure one small LoRA or QLoRA supervised fine-tuning run."""

    backend: str = "custom-pytorch"
    model_name: str = DEFAULT_MODEL
    model_cache_dir: Path = DEFAULT_HUGGING_FACE_CACHE
    local_files_only: bool = True
    output_dir: Path = Path("/mnt/d/AI/models/cognityx/training/qwen-hello-world")
    run_id: str | None = None
    dataset_input_mode: str = "auto"
    storage_config: Path | None = None
    storage_root: str | Path | None = None
    overlength_policy: str = "error"
    max_sequence_length: int = 512
    max_examples: int | None = None
    max_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    seed: int = 42
    load_in_4bit: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    progress_interval_steps: int = 1
    evaluation_max_new_tokens: int = 32
    resource_sample_interval_seconds: float = 1.0
    host_telemetry_source: str = "auto"
    host_installed_memory_gib: float | None = None
    nvidia_smi_path: str = "nvidia-smi"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CustomPyTorchTrainingConfig":
        """Build validated configuration from a TOML-compatible mapping."""
        normalized = dict(values)
        for path_field in ("model_cache_dir", "output_dir"):
            if path_field in normalized:
                normalized[path_field] = Path(normalized[path_field])
        if "target_modules" in normalized:
            normalized["target_modules"] = tuple(normalized["target_modules"])
        for path_field in ("storage_config",):
            if path_field in normalized and normalized[path_field] is not None:
                normalized[path_field] = Path(normalized[path_field])
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject invalid values before model allocation begins."""
        if self.backend != "custom-pytorch":
            raise ValueError("CustomPyTorchTrainingConfig backend must be 'custom-pytorch'.")
        for name in (
            "max_sequence_length",
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "lora_rank",
            "lora_alpha",
            "progress_interval_steps",
            "evaluation_max_new_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when specified.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.resource_sample_interval_seconds <= 0:
            raise ValueError("resource_sample_interval_seconds must be positive.")
        if self.host_telemetry_source not in {"auto", "windows", "wsl"}:
            raise ValueError("host_telemetry_source must be auto, windows, or wsl.")
        if self.host_installed_memory_gib is not None and self.host_installed_memory_gib <= 0:
            raise ValueError("host_installed_memory_gib must be positive.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be between zero and one.")
        if not self.model_cache_dir.is_dir():
            raise ValueError(
                f"Hugging Face model cache does not exist: {self.model_cache_dir}"
            )
        if self.run_id is not None and (
            not self.run_id or Path(self.run_id).name != self.run_id
        ):
            raise ValueError("run_id must be a non-empty file-name-safe value.")
        if self.dataset_input_mode not in {"auto", "dataforge_manifest", "legacy_jsonl"}:
            raise ValueError("dataset_input_mode must be auto, dataforge_manifest, or legacy_jsonl.")
        if self.overlength_policy not in {"error", "skip"}:
            raise ValueError("overlength_policy must be error or skip.")
        if self.storage_config is not None and not self.storage_config.exists():
            raise ValueError(f"storage_config does not exist: {self.storage_config}")
