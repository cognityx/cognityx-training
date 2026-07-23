"""Register and construct training backends from typed configuration."""

from cognityx_core import BackendConfig, BackendFactory, TrainingBackend

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.custom_pytorch import CustomPyTorchTrainerBackend

training_backend_factory: BackendFactory[TrainingBackend] = BackendFactory()


def _build_custom_pytorch(config: BackendConfig) -> TrainingBackend:
    if not isinstance(config, CustomPyTorchTrainingConfig):
        raise TypeError("custom-pytorch requires CustomPyTorchTrainingConfig.")
    return CustomPyTorchTrainerBackend(config)


training_backend_factory.register("custom-pytorch", _build_custom_pytorch)


def create_training_backend(config: BackendConfig) -> TrainingBackend:
    """Create the configured training backend."""
    return training_backend_factory.create(config)
