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
    assert config.max_steps == 1
    assert config.load_in_4bit is True


def test_mapping_normalizes_backend_specific_values() -> None:
    config = CustomPyTorchTrainingConfig.from_mapping(
        {"output_dir": "tmp/output", "target_modules": ["q_proj", "v_proj"]}
    )

    assert config.output_dir == Path("tmp/output")
    assert config.target_modules == ("q_proj", "v_proj")


def test_factory_creates_custom_backend() -> None:
    backend = create_training_backend(CustomPyTorchTrainingConfig())

    assert isinstance(backend, TrainingBackend)
    assert isinstance(backend, CustomPyTorchTrainerBackend)


def test_invalid_training_configuration_fails_before_loading_model() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        CustomPyTorchTrainingConfig(max_steps=0).validate()
