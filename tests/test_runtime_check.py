from types import SimpleNamespace

from cognityx_training.runtime_check import (
    RUNTIME_CHECK_SCHEMA,
    TRAINING_RUNTIME_PACKAGES,
    check_training_runtime,
)


def test_runtime_check_imports_owner_declared_execution_packages(monkeypatch) -> None:
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)

    def import_package(name: str):
        return SimpleNamespace(cuda=cuda) if name == "torch" else SimpleNamespace()

    monkeypatch.setattr("cognityx_training.runtime_check.import_module", import_package)
    monkeypatch.setattr(
        "cognityx_training.runtime_check.metadata.version", lambda name: f"{name}-1"
    )

    result = check_training_runtime(require_cuda=True)

    assert result["schema"] == RUNTIME_CHECK_SCHEMA
    assert result["passed"] is True
    assert tuple(result["packages"]) == TRAINING_RUNTIME_PACKAGES
    assert result["cuda"] == {
        "required": True,
        "available": True,
        "device_count": 1,
        "error_type": None,
    }


def test_runtime_check_reports_missing_peft_without_model_work(monkeypatch) -> None:
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)

    def import_package(name: str):
        if name == "peft":
            raise ModuleNotFoundError("No module named peft")
        return SimpleNamespace(cuda=cuda) if name == "torch" else SimpleNamespace()

    monkeypatch.setattr("cognityx_training.runtime_check.import_module", import_package)
    monkeypatch.setattr(
        "cognityx_training.runtime_check.metadata.version", lambda name: f"{name}-1"
    )

    result = check_training_runtime(require_cuda=True)

    assert result["passed"] is False
    assert result["missing_packages"] == ["peft"]
    assert result["packages"]["peft"] == {
        "status": "unavailable",
        "version": None,
        "error_type": "ModuleNotFoundError",
    }


def test_runtime_check_requires_cuda_only_when_configuration_does(monkeypatch) -> None:
    cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setattr(
        "cognityx_training.runtime_check.import_module",
        lambda name: SimpleNamespace(cuda=cuda)
        if name == "torch"
        else SimpleNamespace(),
    )
    monkeypatch.setattr(
        "cognityx_training.runtime_check.metadata.version", lambda name: f"{name}-1"
    )

    result = check_training_runtime(require_cuda=False)

    # This test runs in CPU-only CI too; dependency imports, not CUDA, determine
    # success for a non-quantized configuration.
    assert result["cuda"]["required"] is False
    assert result["passed"] is True
