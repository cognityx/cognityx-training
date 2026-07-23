from pathlib import Path

import pytest

from cognityx_core import TrainingBackend
from cognityx_training import (
    CustomPyTorchTrainerBackend,
    CustomPyTorchTrainingConfig,
    create_training_backend,
)


def test_default_is_one_step_qwen3_14b_qlora() -> None:
    config = CustomPyTorchTrainingConfig()

    assert config.model_name == "Qwen/Qwen3-14B"
    assert config.model_cache_dir == Path("/mnt/d/AI/models/huggingface/hub")
    assert config.local_files_only is True
    assert config.output_dir == Path(
        "/mnt/d/AI/models/cognityx/training/qwen-hello-world"
    )
    assert config.max_steps == 1
    assert config.max_examples is None
    assert config.load_in_4bit is True


def test_mapping_normalizes_backend_specific_values(tmp_path) -> None:
    config = CustomPyTorchTrainingConfig.from_mapping(
        {
            "model_cache_dir": str(tmp_path),
            "output_dir": "tmp/output",
            "target_modules": ["q_proj", "v_proj"],
        }
    )

    assert config.model_cache_dir == tmp_path
    assert config.output_dir == Path("tmp/output")
    assert config.target_modules == ("q_proj", "v_proj")


def test_factory_creates_custom_backend(tmp_path) -> None:
    backend = create_training_backend(
        CustomPyTorchTrainingConfig(model_cache_dir=tmp_path)
    )

    assert isinstance(backend, TrainingBackend)
    assert isinstance(backend, CustomPyTorchTrainerBackend)


def test_invalid_training_configuration_fails_before_loading_model() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        CustomPyTorchTrainingConfig(max_steps=0).validate()

    with pytest.raises(ValueError, match="max_examples"):
        CustomPyTorchTrainingConfig(max_examples=0).validate()
