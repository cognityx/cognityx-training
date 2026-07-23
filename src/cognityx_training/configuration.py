"""Typed configuration for the custom PyTorch training backend."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cognityx_core import BackendConfig

DEFAULT_MODEL = "Qwen/Qwen3-14B"


@dataclass(frozen=True, slots=True)
class CustomPyTorchTrainingConfig(BackendConfig):
    """Configure one small LoRA or QLoRA supervised fine-tuning run."""

    backend: str = "custom-pytorch"
    model_name: str = DEFAULT_MODEL
    output_dir: Path = Path("outputs/qwen-hello-world")
    max_sequence_length: int = 512
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

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CustomPyTorchTrainingConfig":
        """Build validated configuration from a TOML-compatible mapping."""
        normalized = dict(values)
        if "output_dir" in normalized:
            normalized["output_dir"] = Path(normalized["output_dir"])
        if "target_modules" in normalized:
            normalized["target_modules"] = tuple(normalized["target_modules"])
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
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be between zero and one.")
