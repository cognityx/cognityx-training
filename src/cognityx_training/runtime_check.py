"""No-model capability check for the production training backend."""

from __future__ import annotations

from contextlib import redirect_stdout
from importlib import import_module, metadata
from io import StringIO
from typing import Any

RUNTIME_CHECK_SCHEMA = "cognityx.training.runtime-check/v1"

# Training owns this list. Orchestrators consume the public CLI result instead of
# copying package knowledge across repository boundaries.
TRAINING_RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "psutil",
)


def check_training_runtime(*, require_cuda: bool) -> dict[str, Any]:
    """Import the real backend dependencies and inspect CUDA without loading a model."""
    packages: dict[str, dict[str, str | None]] = {}
    imported: dict[str, object] = {}

    for package in TRAINING_RUNTIME_PACKAGES:
        try:
            # Some third-party imports print banners to stdout. Keep the CLI's
            # machine-output channel reserved for exactly one JSON document.
            with redirect_stdout(StringIO()):
                imported[package] = import_module(package)
            packages[package] = {
                "status": "available",
                "version": metadata.version(package),
                "error_type": None,
            }
        except Exception as exc:  # noqa: BLE001 - report any capability failure
            packages[package] = {
                "status": "unavailable",
                "version": None,
                "error_type": type(exc).__name__,
            }

    torch_module = imported.get("torch")
    cuda_available = False
    cuda_device_count = 0
    cuda_error_type: str | None = None
    if torch_module is not None:
        try:
            cuda = torch_module.cuda
            cuda_available = bool(cuda.is_available())
            cuda_device_count = int(cuda.device_count())
        except Exception as exc:  # noqa: BLE001 - report any CUDA inspection failure
            cuda_error_type = type(exc).__name__

    missing_packages = [
        package
        for package, result in packages.items()
        if result["status"] != "available"
    ]
    passed = not missing_packages and (cuda_available or not require_cuda)
    return {
        "schema": RUNTIME_CHECK_SCHEMA,
        "backend": "custom-pytorch",
        "passed": passed,
        "packages": packages,
        "missing_packages": missing_packages,
        "cuda": {
            "required": require_cuda,
            "available": cuda_available,
            "device_count": cuda_device_count,
            "error_type": cuda_error_type,
        },
    }
