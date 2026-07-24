"""Capacity autotuner for isolated Cognityx training trials."""

from __future__ import annotations

import argparse
import codecs
import fcntl
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
import json
import itertools
import os
from pathlib import Path
import platform
import pty
import re
import shutil
import struct
import termios
import selectors
import signal
import subprocess
import sys
import time
import tomllib
from typing import Any, Iterable
import uuid

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.final_report import (
    persist_trial_result,
    render_final_safe_combinations,
    update_final_safe_combinations,
)
from cognityx_training.telemetry import (
    WindowsTelemetryProducer,
    query_host,
    query_nvidia_gpus,
    query_windows_bridge,
)


@dataclass(frozen=True, slots=True)
class AutotuneConfig:
    """Control a staged hardware-capacity search."""

    base_training_config: Path
    output_dir: Path
    gpu_memory_limit_percent: float | None = 98.0
    ram_limit_percent: float | None = 95.0
    gpu_temperature_limit_celsius: float | None = 88.0
    gpu_power_limit_watts: float | None = None
    gpu_power_limit_percent: float | None = None
    termination_sustain_seconds: float = 3.0
    model_loading_timeout_seconds: float | None = 1800.0
    trial_timeout_seconds: float | None = 900.0
    no_step_progress_timeout_seconds: float | None = None
    trial_max_steps: int = 2
    cooldown_seconds: float = 3.0
    sequence_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    lora_ranks: tuple[int, ...] = (8, 16, 32, 64)
    model_names: tuple[str, ...] = ()
    strategy: str = "staged"
    axes: tuple[str, ...] = (
        "model_name",
        "max_sequence_length",
        "per_device_train_batch_size",
        "lora_rank",
    )
    candidates: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    max_trials: int = 100
    host_telemetry_source: str = "auto"
    host_installed_memory_gib: float | None = None
    windows_bridge_path: Path | None = None
    windows_bridge_max_age_seconds: float = 5.0
    manage_windows_bridge: bool = False
    windows_bridge_interval_seconds: float = 1.0
    windows_bridge_startup_timeout_seconds: float = 20.0
    nvidia_smi_path: str = "nvidia-smi"
    reuse_loaded_model: bool = False
    restart_worker_after_oom: bool = True
    max_trials_per_worker: int = 25

    def validate(self) -> None:
        if self.gpu_memory_limit_percent is not None and not 0 < self.gpu_memory_limit_percent <= 100:
            raise ValueError("gpu_memory_limit_percent must be in (0, 100].")
        if self.ram_limit_percent is not None and not 0 < self.ram_limit_percent <= 100:
            raise ValueError("ram_limit_percent must be in (0, 100].")
        if self.trial_max_steps <= 0:
            raise ValueError("trial_max_steps must be positive.")
        for name in ("sequence_lengths", "batch_sizes", "lora_ranks"):
            values = getattr(self, name)
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive candidates.")
        if self.strategy not in {"staged", "grid"}:
            raise ValueError("strategy must be staged or grid.")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive.")
        if self.max_trials_per_worker <= 0:
            raise ValueError("max_trials_per_worker must be positive.")
        if self.windows_bridge_max_age_seconds <= 0:
            raise ValueError("windows_bridge_max_age_seconds must be positive.")
        if self.windows_bridge_interval_seconds <= 0:
            raise ValueError("windows_bridge_interval_seconds must be positive.")
        if self.windows_bridge_startup_timeout_seconds <= 0:
            raise ValueError(
                "windows_bridge_startup_timeout_seconds must be positive."
            )
        if self.reuse_loaded_model and not self.restart_worker_after_oom:
            raise ValueError(
                "restart_worker_after_oom must remain true when model reuse is enabled; "
                "interrupted CUDA state cannot be reused safely."
            )
        if self.termination_sustain_seconds < 0:
            raise ValueError("termination_sustain_seconds cannot be negative.")
        for name in (
            "model_loading_timeout_seconds",
            "trial_timeout_seconds",
            "no_step_progress_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured.")
        valid_fields = CustomPyTorchTrainingConfig.__dataclass_fields__
        for axis in self.axes:
            if axis not in valid_fields:
                raise ValueError(f"Unknown training axis: {axis}")
            if axis not in self.candidates:
                raise ValueError(f"No trial candidates configured for axis: {axis}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse capacity-autotuner arguments."""
    parser = argparse.ArgumentParser(
        description="Find a safe Cognityx training configuration for this computer."
    )
    parser.add_argument("--config", type=Path, required=True, help="Autotune TOML file.")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print hardware, software, and the search plan without training.",
    )
    return parser.parse_args(argv)


def load_autotune_config(path: Path) -> tuple[AutotuneConfig, dict[str, Any]]:
    """Load autotune settings and the referenced base training TOML."""
    with path.open("rb") as source:
        values = tomllib.load(source)
    settings = values.get("autotune", {})
    search = values.get("search", {})
    termination = values.get("termination", {})
    telemetry = values.get("telemetry", {})
    execution = values.get("execution", {})
    trial_values = values.get("trials", search.get("candidates", {}))
    base_path = Path(settings["base_training_config"])
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    output_dir = Path(settings.get("output_dir", "outputs/autotune"))
    config = AutotuneConfig(
        base_training_config=base_path,
        output_dir=output_dir,
        gpu_memory_limit_percent=(
            float(termination["gpu_memory_percent"])
            if "gpu_memory_percent" in termination
            else (
                float(settings["gpu_memory_limit_percent"])
                if "gpu_memory_limit_percent" in settings
                else None
            )
        ),
        ram_limit_percent=(
            float(termination["host_ram_percent"])
            if "host_ram_percent" in termination
            else (
                float(settings["ram_limit_percent"])
                if "ram_limit_percent" in settings
                else None
            )
        ),
        gpu_temperature_limit_celsius=(
            float(termination["gpu_temperature_celsius"])
            if "gpu_temperature_celsius" in termination
            else None
        ),
        gpu_power_limit_watts=(
            float(termination["gpu_power_watts"])
            if "gpu_power_watts" in termination
            else None
        ),
        gpu_power_limit_percent=(
            float(termination["gpu_power_percent_of_limit"])
            if "gpu_power_percent_of_limit" in termination
            else None
        ),
        termination_sustain_seconds=float(termination.get("sustain_seconds", 3)),
        model_loading_timeout_seconds=(
            float(termination["model_loading_timeout_seconds"])
            if "model_loading_timeout_seconds" in termination
            else None
        ),
        trial_timeout_seconds=(
            float(termination["trial_timeout_seconds"])
            if "trial_timeout_seconds" in termination
            else None
        ),
        no_step_progress_timeout_seconds=(
            float(termination["no_step_progress_timeout_seconds"])
            if "no_step_progress_timeout_seconds" in termination
            else None
        ),
        trial_max_steps=int(settings.get("trial_max_steps", 2)),
        cooldown_seconds=float(settings.get("cooldown_seconds", 3)),
        sequence_lengths=tuple(
            int(value)
            for value in search.get(
                "sequence_lengths", (128, 256, 512, 1024, 2048)
            )
        ),
        batch_sizes=tuple(
            int(value) for value in search.get("batch_sizes", (1, 2, 4, 8))
        ),
        lora_ranks=tuple(
            int(value) for value in search.get("lora_ranks", (8, 16, 32, 64))
        ),
        model_names=tuple(str(value) for value in search.get("model_names", ())),
        strategy=str(search.get("strategy", "staged")),
        axes=tuple(
            str(value)
            for value in search.get(
                "axes",
                (
                    "model_name",
                    "max_sequence_length",
                    "per_device_train_batch_size",
                    "lora_rank",
                ),
            )
        ),
        candidates={
            str(name): tuple(candidates)
            for name, candidates in trial_values.items()
        },
        max_trials=int(search.get("max_trials", 100)),
        host_telemetry_source=str(telemetry.get("host_source", "auto")),
        host_installed_memory_gib=(
            float(telemetry["installed_memory_gib"])
            if "installed_memory_gib" in telemetry
            else None
        ),
        windows_bridge_path=(
            Path(telemetry["windows_bridge_path"])
            if "windows_bridge_path" in telemetry
            else None
        ),
        windows_bridge_max_age_seconds=float(
            telemetry.get("windows_bridge_max_age_seconds", 5)
        ),
        manage_windows_bridge=bool(
            telemetry.get("manage_windows_bridge", False)
        ),
        windows_bridge_interval_seconds=float(
            telemetry.get("windows_bridge_interval_seconds", 1)
        ),
        windows_bridge_startup_timeout_seconds=float(
            telemetry.get("windows_bridge_startup_timeout_seconds", 20)
        ),
        nvidia_smi_path=str(telemetry.get("nvidia_smi_path", "nvidia-smi")),
        reuse_loaded_model=bool(execution.get("reuse_loaded_model", False)),
        restart_worker_after_oom=bool(execution.get("restart_worker_after_oom", True)),
        max_trials_per_worker=int(execution.get("max_trials_per_worker", 25)),
    )
    with base_path.open("rb") as source:
        base_values = tomllib.load(source)
    base_training = CustomPyTorchTrainingConfig.from_mapping(
        base_values.get("training", {})
    )
    if not config.candidates:
        config = replace(
            config,
            candidates={
                "model_name": config.model_names or (base_training.model_name,),
                "max_sequence_length": config.sequence_lengths,
                "per_device_train_batch_size": config.batch_sizes,
                "lora_rank": config.lora_ranks,
            },
        )
    elif not config.candidates.get("model_name") and "model_name" in config.axes:
        config.candidates["model_name"] = (base_training.model_name,)
    config.validate()
    return config, base_values


def hardware_software_inventory(
    output_dir: Path,
    host_source: str = "auto",
    nvidia_smi_path: str = "nvidia-smi",
    installed_memory_gib: float | None = None,
) -> dict[str, Any]:
    """Collect capacity-relevant host, GPU, driver, and package information."""
    import psutil

    host = query_host(host_source, installed_memory_gib)
    disk_target = output_dir
    while not disk_target.exists() and disk_target != disk_target.parent:
        disk_target = disk_target.parent
    disk = psutil.disk_usage(disk_target)
    cpu_name = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    inventory: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "name": cpu_name,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "source": host["source"],
            "scope": host["scope"],
            "total_kind": host.get("total_kind"),
            "total_bytes": host["total_bytes"],
            "available_bytes": host["available_bytes"],
            "used_bytes": host["used_bytes"],
            "used_percent": host["used_percent"],
            "installed_physical_total_bytes": host.get(
                "installed_physical_total_bytes"
            ),
            "installed_physical_total_source": host.get(
                "installed_physical_total_source"
            ),
        },
        "disk": {
            "path": str(disk_target),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "used_percent": disk.percent,
        },
        "software": {},
        "gpus": [],
    }
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            inventory["software"][package] = version(package)
        except PackageNotFoundError:
            inventory["software"][package] = None
    try:
        import torch

        inventory["software"]["cuda_runtime"] = torch.version.cuda
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            inventory["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    except (ImportError, RuntimeError) as exc:
        inventory["gpu_inventory_error"] = str(exc)
    try:
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=driver_version,pstate,temperature.gpu,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        inventory["nvidia_smi"] = result.stdout.strip().splitlines()
        inventory["nvidia_telemetry"] = query_nvidia_gpus(nvidia_smi_path)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        inventory["nvidia_smi_error"] = str(exc)
    return inventory


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, Path)):
        return json.dumps(str(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def write_trial_config(
    config: CustomPyTorchTrainingConfig,
    dataset_path: Path,
    destination: Path,
) -> None:
    """Write one fully resolved child-training TOML."""
    values = asdict(config)
    lines = ["[training]"]
    for key, value in values.items():
        if value is None or key == "metadata":
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    lines.extend(
        [
            "",
            "[dataset]",
            'name = "autotune-capacity"',
            'version = "1"',
            f"uri = {_toml_value(dataset_path)}",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def generate_capacity_dataset(path: Path, record_count: int, token_target: int) -> None:
    """Generate varied records long enough to exercise sequence truncation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record_index in range(record_count):
            words = " ".join(
                f"capacity_{record_index}_{word_index % 997}"
                for word_index in range(token_target * 2)
            )
            destination.write(
                json.dumps(
                    {
                        "instruction": f"Summarize this capacity input: {words}",
                        "output": "capacity-ok",
                    }
                )
                + "\n"
            )


def _peak_percentages(report: dict[str, Any]) -> tuple[float | None, float | None]:
    gpu_values = [
        float(value["memory_peak_percent"])
        for value in report.get("gpu_usage", [])
        if value.get("memory_peak_percent") is not None
    ]
    ram = report.get("system_usage", {}).get("system_ram_peak_percent")
    return (max(gpu_values) if gpu_values else None, float(ram) if ram is not None else None)


def _live_resources(controller: AutotuneConfig) -> dict[str, Any]:
    """Return labeled host and whole-device GPU measurements."""
    host = query_host(
        controller.host_telemetry_source, controller.host_installed_memory_gib
    )
    try:
        gpus = query_nvidia_gpus(controller.nvidia_smi_path)
    except (OSError, subprocess.SubprocessError):
        gpus = []
    windows = query_windows_bridge(
        controller.windows_bridge_path,
        controller.windows_bridge_max_age_seconds,
    )
    return {"host": host, "gpus": gpus, "windows_bridge": windows}


def _model_storage_info(path: Path, mountinfo_text: str | None = None) -> dict[str, Any]:
    """Resolve the filesystem that contains the configured model cache."""
    resolved = path.resolve()
    if mountinfo_text is None:
        try:
            mountinfo_text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError:
            mountinfo_text = ""
    matches: list[tuple[int, dict[str, Any]]] = []
    for line in mountinfo_text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        trailing = after.split()
        if len(fields) < 5 or len(trailing) < 2:
            continue
        mount_point = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append(
            (
                len(str(mount_point)),
                {
                    "model_cache_path": str(resolved),
                    "mount_point": str(mount_point),
                    "device_major_minor": fields[2],
                    "filesystem_type": trailing[0],
                    "mount_source": trailing[1],
                },
            )
        )
    if not matches:
        return {
            "model_cache_path": str(resolved),
            "mount_point": None,
            "filesystem_type": "unknown",
            "mount_source": None,
            "device_major_minor": None,
            "device_latency_available": False,
        }
    result = max(matches, key=lambda item: item[0])[1]
    major = str(result["device_major_minor"]).split(":", 1)[0]
    result["device_latency_available"] = major != "0"
    result["measurement_scope"] = "loader_process_reads_on_model_cache_filesystem"
    return result


class _LoadingProcessSampler:
    """Keep psutil process CPU state across loading samples."""

    def __init__(self, pid: int, process: Any | None = None) -> None:
        if process is None:
            import psutil

            process = psutil.Process(pid)
        self.process = process
        self.process.cpu_percent(interval=None)

    def sample(self) -> dict[str, float] | None:
        try:
            io = self.process.io_counters()
            return {
                "rss_bytes": float(self.process.memory_info().rss),
                "process_cpu_percent": float(self.process.cpu_percent(interval=None)),
                "read_bytes": float(io.read_bytes),
                "read_chars": float(io.read_chars),
                "read_count": float(io.read_count),
            }
        except Exception:
            return None


def _loading_process_sampler(pid: int) -> _LoadingProcessSampler | None:
    try:
        return _LoadingProcessSampler(pid)
    except Exception:
        return None


def _rate(current: float, previous: float, seconds: float) -> float:
    return max(0.0, current - previous) / max(seconds, 1e-9)


def _format_loading_status(
    elapsed_seconds: float,
    gpu_status: str,
    windows_gpu_status: str,
    storage_status: str,
    cpu_status: str,
    ram_status: str,
) -> str:
    """Order bounded loading telemetry by operational importance."""
    return " | ".join(
        (
            f"LOAD {elapsed_seconds:6.0f}s",
            gpu_status,
            windows_gpu_status,
            storage_status,
            cpu_status,
            ram_status,
        )
    )


class _TwoLineLoadingDisplay:
    """Keep library progress and Cognityx telemetry on two stable TTY rows."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.progress = "Loading weights: starting..."
        self.stats = "LOAD: collecting telemetry..."
        self.active = False
        self.tty = bool(getattr(stream, "isatty", lambda: False)())

    def update_progress(self, value: str) -> None:
        if self.active and value == self.progress:
            return
        self.progress = value
        self._redraw()

    def update_stats(self, value: str) -> None:
        self.stats = value
        self._redraw()

    def _redraw(self) -> None:
        if not self.tty:
            return
        if self.active:
            self.stream.write("\r\x1b[1A\x1b[2K")
        columns = max(20, shutil.get_terminal_size(fallback=(160, 24)).columns - 1)
        progress = self.progress[:columns]
        stats = self.stats[:columns]
        self.stream.write(f"{progress}\n\x1b[2K{stats}")
        self.stream.flush()
        self.active = True

    def finish(self) -> None:
        if self.tty and self.active:
            self.stream.write("\n")
            self.stream.flush()
        elif not self.tty:
            self.stream.write(f"{self.progress}\n{self.stats}\n")
            self.stream.flush()
        self.active = False


class _RawChildOutputParser:
    """Preserve carriage-return records instead of universal-newline conversion."""

    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.buffer = ""

    def feed(self, value: bytes, final: bool = False) -> list[str]:
        self.buffer += self.decoder.decode(value, final=final)
        records: list[str] = []
        start = 0
        for index, character in enumerate(self.buffer):
            if character in {"\r", "\n"}:
                record = self.buffer[start:index]
                if record:
                    records.append(record)
                start = index + 1
        self.buffer = self.buffer[start:]
        if final and self.buffer:
            records.append(self.buffer)
            self.buffer = ""
        return records


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def _strip_terminal_control(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).replace("\x08", "")


class _SingleLineStatusDisplay:
    """Update one bounded terminal row without scrolling."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.active = False
        self.last_value = ""
        self.tty = bool(getattr(stream, "isatty", lambda: False)())

    def update(self, value: str) -> None:
        if value == self.last_value:
            return
        self.last_value = value
        if not self.tty:
            return
        columns = max(20, shutil.get_terminal_size(fallback=(160, 24)).columns - 1)
        self.stream.write("\r\x1b[2K" + value[:columns])
        self.stream.flush()
        self.active = True

    def finish(self) -> None:
        if self.tty and self.active:
            self.stream.write("\n")
            self.stream.flush()
        elif not self.tty and self.last_value:
            self.stream.write(self.last_value + "\n")
            self.stream.flush()
        self.active = False


def _policy_breaches(
    controller: AutotuneConfig,
    resources: dict[str, Any],
    elapsed_seconds: float,
    phase: str = "training",
    no_step_progress_seconds: float | None = None,
) -> list[tuple[str, float, float, str]]:
    """Return configured safety-policy breaches; utilization is informational."""
    breaches: list[tuple[str, float, float, str]] = []
    host = resources["host"]
    gpus = resources["gpus"]
    checks = (
        (
            "gpu_memory_percent",
            max(
                (gpu["memory_used_percent"] for gpu in gpus if gpu["memory_used_percent"] is not None),
                default=None,
            ),
            controller.gpu_memory_limit_percent,
            "whole-device GPU memory",
        ),
        (
            "host_ram_percent",
            float(host["used_percent"]),
            controller.ram_limit_percent,
            f"{host['scope']} RAM",
        ),
        (
            "gpu_temperature_celsius",
            max(
                (gpu["temperature_celsius"] for gpu in gpus if gpu["temperature_celsius"] is not None),
                default=None,
            ),
            controller.gpu_temperature_limit_celsius,
            "GPU temperature",
        ),
        (
            "gpu_power_watts",
            max(
                (gpu["power_draw_watts"] for gpu in gpus if gpu["power_draw_watts"] is not None),
                default=None,
            ),
            controller.gpu_power_limit_watts,
            "GPU power",
        ),
        (
            "gpu_power_percent_of_limit",
            max(
                (gpu["power_percent_of_limit"] for gpu in gpus if gpu["power_percent_of_limit"] is not None),
                default=None,
            ),
            controller.gpu_power_limit_percent,
            "GPU power-limit percentage",
        ),
    )
    for key, value, limit, label in checks:
        if value is not None and limit is not None and value >= limit:
            breaches.append((key, float(value), float(limit), label))
    timeout = {
        "model_loading": controller.model_loading_timeout_seconds,
        "training": controller.trial_timeout_seconds,
        "post_training": None,
    }[phase]
    if timeout is not None and elapsed_seconds >= timeout:
        key = (
            "model_loading_timeout_seconds"
            if phase == "model_loading"
            else "trial_timeout_seconds"
        )
        label = "model loading time" if phase == "model_loading" else "training time"
        breaches.append((key, elapsed_seconds, timeout, label))
    if (
        phase == "training"
        and controller.no_step_progress_timeout_seconds is not None
        and no_step_progress_seconds is not None
        and no_step_progress_seconds >= controller.no_step_progress_timeout_seconds
    ):
        breaches.append(
            (
                "no_step_progress_timeout_seconds",
                no_step_progress_seconds,
                controller.no_step_progress_timeout_seconds,
                "time since last completed training step",
            )
        )
    return breaches


_IMMEDIATE_TIMEOUT_KEYS = frozenset(
    {
        "model_loading_timeout_seconds",
        "trial_timeout_seconds",
        "no_step_progress_timeout_seconds",
    }
)


def _is_immediate_timeout_key(key: str) -> bool:
    return key in _IMMEDIATE_TIMEOUT_KEYS


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """Stop a child trial and escalate only if it ignores SIGINT."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


_PERSISTENT_WORKERS: dict[str, tuple[subprocess.Popen[Any], int]] = {}
_PERSISTENT_OUTPUT_FDS: dict[str, int] = {}


def _spawn_with_pty(
    command: list[str], env: dict[str, str], persistent: bool
) -> tuple[subprocess.Popen[Any], int]:
    """Spawn a child whose output is a real terminal while stdin stays controllable."""
    master_fd, slave_fd = pty.openpty()
    columns, rows = shutil.get_terminal_size(fallback=(160, 24))
    fcntl.ioctl(
        slave_fd,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", rows, columns, 0, 0),
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if persistent else subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            bufsize=0,
            env=env,
            start_new_session=True,
        )
    except Exception:
        os.close(master_fd)
        raise
    finally:
        os.close(slave_fd)
    os.set_blocking(master_fd, False)
    return process, master_fd


def _persistent_worker(model_name: str, controller: AutotuneConfig) -> subprocess.Popen[Any]:
    for other_model in tuple(_PERSISTENT_WORKERS):
        if other_model != model_name:
            _discard_worker(other_model)
    existing = _PERSISTENT_WORKERS.get(model_name)
    if existing is not None:
        process, count = existing
        if process.poll() is None and count < controller.max_trials_per_worker:
            _PERSISTENT_WORKERS[model_name] = (process, count + 1)
            return process
        _stop_process_group(process)
    process, output_fd = _spawn_with_pty(
        [sys.executable, "-m", "cognityx_training.autotune_worker"],
        {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
        persistent=True,
    )
    _PERSISTENT_WORKERS[model_name] = (process, 1)
    _PERSISTENT_OUTPUT_FDS[model_name] = output_fd
    return process


def _discard_worker(model_name: str, reason: str | None = None) -> None:
    entry = _PERSISTENT_WORKERS.pop(model_name, None)
    output_fd = _PERSISTENT_OUTPUT_FDS.pop(model_name, None)
    if entry is not None:
        if reason:
            print(
                f"WORKER RESTART REQUIRED: {reason}. The next {model_name} trial "
                "will reload weights for clean CUDA state.",
                flush=True,
            )
        _stop_process_group(entry[0])
    if output_fd is not None:
        try:
            os.close(output_fd)
        except OSError:
            pass


def _shutdown_workers() -> None:
    for model_name in tuple(_PERSISTENT_WORKERS):
        _discard_worker(model_name)


def _print_inventory(inventory: dict[str, Any]) -> None:
    gib = 1024**3
    print("\nHardware/software inventory")
    print(
        f"CPU: {inventory['cpu']['name']} | physical/logical cores: "
        f"{inventory['cpu']['physical_cores']}/{inventory['cpu']['logical_cores']}"
    )
    print(
        f"RAM ({inventory['ram']['scope']}, {inventory['ram']['source']}): "
        f"{inventory['ram']['total_bytes'] / gib:.1f} GiB total | "
        f"{inventory['ram']['used_bytes'] / gib:.1f} GiB used | "
        f"{inventory['ram']['used_percent']:.1f}% | "
        f"denominator={inventory['ram'].get('total_kind', 'unknown')}"
    )
    if inventory["ram"].get("installed_physical_total_bytes"):
        print(
            "Installed motherboard RAM (configured; Windows live usage unavailable): "
            f"{inventory['ram']['installed_physical_total_bytes'] / gib:.1f} GiB"
        )
    print(f"Disk free: {inventory['disk']['free_bytes'] / gib:.1f} GiB")
    for gpu in inventory["gpus"]:
        print(
            f"GPU {gpu['index']}: {gpu['name']} | "
            f"{gpu['total_memory_bytes'] / gib:.1f} GiB | CC {gpu['compute_capability']}"
        )
    for gpu in inventory.get("nvidia_telemetry", []):
        def display(value: Any, suffix: str = "") -> str:
            return f"{value:.1f}{suffix}" if value is not None else "n/a"

        print(
            f"GPU telemetry ({gpu['scope']}, {gpu['source']}): "
            f"memory {display(gpu['memory_used_percent'], '%')} | "
            f"utilization {display(gpu['utilization_percent'], '%')} | "
            f"power {display(gpu['power_draw_watts'])}/"
            f"{display(gpu['power_limit_watts'])} W | "
            f"temperature {display(gpu['temperature_celsius'], 'C')}"
        )
    if inventory.get("nvidia_smi"):
        print("NVIDIA driver/pstate/temp/power limit: " + " | ".join(inventory["nvidia_smi"]))
    print("Software: " + ", ".join(
        f"{name}={value}" for name, value in inventory["software"].items()
    ))


def _run_trial(
    trial_id: str,
    stage: str,
    training_config: CustomPyTorchTrainingConfig,
    controller: AutotuneConfig,
    dataset_path: Path,
) -> dict[str, Any]:
    config_path = controller.output_dir / "trial-configs" / f"{trial_id}.toml"
    trial_config = replace(
        training_config,
        output_dir=controller.output_dir / "runs",
        run_id=trial_id,
        max_steps=controller.trial_max_steps,
        progress_interval_steps=1,
        host_telemetry_source=controller.host_telemetry_source,
        host_installed_memory_gib=controller.host_installed_memory_gib,
        nvidia_smi_path=controller.nvidia_smi_path,
    )
    write_trial_config(trial_config, dataset_path, config_path)
    printable = {
        "model_name": trial_config.model_name,
        "max_sequence_length": trial_config.max_sequence_length,
        "max_examples": trial_config.max_examples,
        "per_device_train_batch_size": trial_config.per_device_train_batch_size,
        "gradient_accumulation_steps": trial_config.gradient_accumulation_steps,
        "lora_rank": trial_config.lora_rank,
        "load_in_4bit": trial_config.load_in_4bit,
        "max_steps": trial_config.max_steps,
    }
    print(f"\n=== {trial_id} | stage={stage} ===")
    print(json.dumps(printable, indent=2))
    command = [
        sys.executable,
        "-m",
        "cognityx_training.cli",
        "--config",
        str(config_path),
    ]
    started = time.perf_counter()
    training_started: float | None = None
    training_completed: float | None = None
    persistent = controller.reuse_loaded_model
    if persistent:
        process = _persistent_worker(trial_config.model_name, controller)
        assert process.stdin is not None
        process.stdin.write((str(config_path) + "\n").encode())
        process.stdin.flush()
        output_fd = _PERSISTENT_OUTPUT_FDS[trial_config.model_name]
    else:
        process, output_fd = _spawn_with_pty(
            command,
            {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
            persistent=False,
        )
    output_lines: list[str] = []
    selector = selectors.DefaultSelector()
    selector.register(output_fd, selectors.EVENT_READ)
    live_gpu_peak: float | None = None
    live_ram_peak = 0.0
    live_gpu_utilization_peak: float | None = None
    live_gpu_power_peak_watts: float | None = None
    live_gpu_temperature_peak_celsius: float | None = None
    live_energy_joules = 0.0
    previous_power_watts: float | None = None
    previous_sample_time: float | None = None
    live_host_source: str | None = None
    live_host_scope: str | None = None
    live_threshold_reason: str | None = None
    live_threshold_key: str | None = None
    breach_since: dict[str, float] = {}
    next_capacity_check = 0.0
    trial_finished = False
    worker_reported_failure = False
    model_reused = False
    loading_display = _TwoLineLoadingDisplay(sys.stdout)
    training_display = _SingleLineStatusDisplay(sys.stdout)
    output_parser = _RawChildOutputParser()
    loading_process_sampler = _loading_process_sampler(process.pid)
    model_storage = _model_storage_info(trial_config.model_cache_dir)
    completed_step = 0
    total_training_steps = trial_config.max_steps
    last_step_progress_time: float | None = None
    longest_step_seconds = 0.0
    previous_loading_sample: dict[str, float] | None = None
    previous_loading_gpu_bytes: float | None = None
    previous_loading_time: float | None = None
    previous_windows_shared_bytes: float | None = None
    previous_windows_sample_time: float | None = None
    loading_telemetry = {
        "physical_disk_read_peak_bytes_per_second": 0.0,
        "requested_read_peak_bytes_per_second": 0.0,
        "read_operations_peak_per_second": 0.0,
        "vram_growth_peak_bytes_per_second": 0.0,
        "process_ram_peak_bytes": 0.0,
        "process_cpu_peak_percent": 0.0,
        "host_cpu_peak_percent": 0.0,
        "host_ram_peak_percent": 0.0,
        "gpu_memory_peak_bytes": 0.0,
        "gpu_power_peak_watts": 0.0,
        "model_storage": model_storage,
        "windows_shared_gpu_peak_bytes": 0.0,
        "windows_combined_gpu_peak_bytes": 0.0,
        "windows_shared_gpu_growth_peak_bytes_per_second": 0.0,
        "windows_host_memory_peak_bytes": 0.0,
    }

    def handle_output(record: str) -> None:
        nonlocal training_started, training_completed
        nonlocal trial_finished, worker_reported_failure, model_reused
        nonlocal completed_step, total_training_steps
        nonlocal last_step_progress_time, longest_step_seconds
        compact = _strip_terminal_control(record).strip().lstrip("\r")
        if not compact:
            return
        if "Loading weights:" in compact:
            loading_display.update_progress(compact)
            if "100%" in compact:
                if not output_lines or output_lines[-1] != compact + "\n":
                    output_lines.append(compact + "\n")
            return
        if loading_display.active:
            loading_display.finish()
        step_match = re.match(r"step\s+(\d+)/(\d+)\b", compact)
        if (step_match or "COGNITYX_TRAINING_COMPLETED" in compact) and training_display.active:
            training_display.finish()
        print(compact, flush=True)
        output_lines.append(compact + "\n")
        if "COGNITYX_TRAINING_STARTED" in compact and training_started is None:
            training_started = time.perf_counter()
            last_step_progress_time = training_started
        if step_match:
            now_progress = time.perf_counter()
            if last_step_progress_time is not None:
                longest_step_seconds = max(
                    longest_step_seconds, now_progress - last_step_progress_time
                )
            completed_step = int(step_match.group(1))
            total_training_steps = int(step_match.group(2))
            last_step_progress_time = now_progress
        if "COGNITYX_MODEL_LOADED reused" in compact:
            model_reused = True
        if "COGNITYX_TRAINING_COMPLETED" in compact and training_completed is None:
            training_completed = time.perf_counter()
        if "COGNITYX_TRIAL_COMPLETED" in compact:
            trial_finished = True
        if "COGNITYX_TRIAL_FAILED" in compact:
            trial_finished = True
            worker_reported_failure = True

    try:
        while process.poll() is None:
            for _key, _events in selector.select(timeout=0.5):
                try:
                    chunk = os.read(output_fd, 65536)
                except (BlockingIOError, OSError):
                    chunk = b""
                for record in output_parser.feed(chunk):
                    handle_output(record)
            if trial_finished:
                break
            now = time.monotonic()
            if now >= next_capacity_check:
                resources = _live_resources(controller)
                host = resources["host"]
                gpus = resources["gpus"]
                windows = resources["windows_bridge"]
                gpu_percent = max(
                    (
                        gpu["memory_used_percent"]
                        for gpu in gpus
                        if gpu["memory_used_percent"] is not None
                    ),
                    default=None,
                )
                ram_percent = float(host["used_percent"])
                utilization = max(
                    (
                        gpu["utilization_percent"]
                        for gpu in gpus
                        if gpu["utilization_percent"] is not None
                    ),
                    default=None,
                )
                total_power = sum(
                    gpu["power_draw_watts"]
                    for gpu in gpus
                    if gpu["power_draw_watts"] is not None
                )
                temperature = max(
                    (
                        gpu["temperature_celsius"]
                        for gpu in gpus
                        if gpu["temperature_celsius"] is not None
                    ),
                    default=None,
                )
                if gpu_percent is not None:
                    live_gpu_peak = max(live_gpu_peak or 0.0, gpu_percent)
                live_ram_peak = max(live_ram_peak, ram_percent)
                if utilization is not None:
                    live_gpu_utilization_peak = max(
                        live_gpu_utilization_peak or 0.0, utilization
                    )
                if total_power:
                    live_gpu_power_peak_watts = max(
                        live_gpu_power_peak_watts or 0.0, total_power
                    )
                    if previous_power_watts is not None and previous_sample_time is not None:
                        live_energy_joules += previous_power_watts * (
                            now - previous_sample_time
                        )
                    previous_power_watts = total_power
                    previous_sample_time = now
                if temperature is not None:
                    live_gpu_temperature_peak_celsius = max(
                        live_gpu_temperature_peak_celsius or 0.0, temperature
                    )
                live_host_source = host["source"]
                live_host_scope = host["scope"]
                windows_shared_growth_rate = 0.0
                if windows is not None:
                    shared_bytes = float(windows["shared_used_bytes"])
                    if (
                        previous_windows_shared_bytes is not None
                        and previous_windows_sample_time is not None
                    ):
                        windows_shared_growth_rate = _rate(
                            shared_bytes,
                            previous_windows_shared_bytes,
                            now - previous_windows_sample_time,
                        )
                    previous_windows_shared_bytes = shared_bytes
                    previous_windows_sample_time = now
                    loading_telemetry["windows_shared_gpu_peak_bytes"] = max(
                        loading_telemetry["windows_shared_gpu_peak_bytes"],
                        shared_bytes,
                    )
                    loading_telemetry["windows_combined_gpu_peak_bytes"] = max(
                        loading_telemetry["windows_combined_gpu_peak_bytes"],
                        float(windows["combined_used_bytes"]),
                    )
                    loading_telemetry[
                        "windows_shared_gpu_growth_peak_bytes_per_second"
                    ] = max(
                        loading_telemetry[
                            "windows_shared_gpu_growth_peak_bytes_per_second"
                        ],
                        windows_shared_growth_rate,
                    )
                    loading_telemetry["windows_host_memory_peak_bytes"] = max(
                        loading_telemetry["windows_host_memory_peak_bytes"],
                        float(windows["host_memory_used_bytes"]),
                    )
                if training_started is None:
                    phase = "model_loading"
                    phase_started = started
                elif training_completed is None:
                    phase = "training"
                    phase_started = training_started
                else:
                    phase = "post_training"
                    phase_started = training_completed
                process_sample = (
                    loading_process_sampler.sample()
                    if loading_process_sampler is not None
                    else None
                )
                if phase == "model_loading":
                    gpu_used_bytes = float(sum(
                        gpu.get("memory_used_bytes") or 0 for gpu in gpus
                    ))
                    if process_sample is not None:
                        physical_rate = requested_rate = read_operations_rate = 0.0
                        vram_growth_rate = 0.0
                        if previous_loading_sample is not None and previous_loading_time is not None:
                            interval = now - previous_loading_time
                            physical_rate = _rate(
                                process_sample["read_bytes"],
                                previous_loading_sample["read_bytes"],
                                interval,
                            )
                            requested_rate = _rate(
                                process_sample["read_chars"],
                                previous_loading_sample["read_chars"],
                                interval,
                            )
                            read_operations_rate = _rate(
                                process_sample["read_count"],
                                previous_loading_sample["read_count"],
                                interval,
                            )
                            if previous_loading_gpu_bytes is not None:
                                vram_growth_rate = _rate(
                                    gpu_used_bytes, previous_loading_gpu_bytes, interval
                                )
                        loading_telemetry["physical_disk_read_peak_bytes_per_second"] = max(
                            loading_telemetry["physical_disk_read_peak_bytes_per_second"], physical_rate
                        )
                        loading_telemetry["requested_read_peak_bytes_per_second"] = max(
                            loading_telemetry["requested_read_peak_bytes_per_second"], requested_rate
                        )
                        loading_telemetry["read_operations_peak_per_second"] = max(
                            loading_telemetry["read_operations_peak_per_second"],
                            read_operations_rate,
                        )
                        loading_telemetry["vram_growth_peak_bytes_per_second"] = max(
                            loading_telemetry["vram_growth_peak_bytes_per_second"], vram_growth_rate
                        )
                        loading_telemetry["process_ram_peak_bytes"] = max(
                            loading_telemetry["process_ram_peak_bytes"], process_sample["rss_bytes"]
                        )
                        loading_telemetry["process_cpu_peak_percent"] = max(
                            loading_telemetry["process_cpu_peak_percent"],
                            process_sample["process_cpu_percent"],
                        )
                        loading_telemetry["host_cpu_peak_percent"] = max(
                            loading_telemetry["host_cpu_peak_percent"],
                            float(host["cpu_percent"]),
                        )
                        loading_telemetry["host_ram_peak_percent"] = max(
                            loading_telemetry["host_ram_peak_percent"], ram_percent
                        )
                        loading_telemetry["gpu_memory_peak_bytes"] = max(
                            loading_telemetry["gpu_memory_peak_bytes"], gpu_used_bytes
                        )
                        loading_telemetry["gpu_power_peak_watts"] = max(
                            loading_telemetry["gpu_power_peak_watts"], total_power
                        )
                        mib = 1024**2
                        gib = 1024**3
                        windows_memory_text = (
                            f"WinGPU D{windows['dedicated_used_bytes'] / gib:.1f}/"
                            f"{windows['dedicated_total_bytes'] / gib:.1f}G "
                            f"S{windows['shared_used_bytes'] / gib:.1f}/"
                            f"{windows['shared_total_bytes'] / gib:.1f}G "
                            f"C{windows['combined_used_bytes'] / gib:.1f}/"
                            f"{windows['combined_total_bytes'] / gib:.1f}G "
                            f"S+{windows_shared_growth_rate / mib:.0f}M/s | "
                            if windows is not None
                            else "WinGPU shared unavailable | "
                        )
                        loading_line = _format_loading_status(
                            time.perf_counter() - started,
                            (
                                f"VRAM {gpu_used_bytes / gib:4.1f}G "
                                f"+{vram_growth_rate / mib:5.0f}M/s(est) | "
                                f"GPU {utilization or 0:3.0f}% "
                                f"{total_power:5.0f}W {temperature or 0:3.0f}C"
                            ),
                            windows_memory_text.rstrip(" | "),
                            (
                                f"modelFS {model_storage['mount_point'] or '?'} "
                                f"{model_storage['filesystem_type']} | "
                                f"read {physical_rate / mib:5.0f}M/s "
                                f"{read_operations_rate:4.0f}op/s"
                            ),
                            (
                                f"procCPU {process_sample['process_cpu_percent']:4.0f}% "
                                f"{host['scope']}CPU {host['cpu_percent']:3.0f}%"
                            ),
                            (
                                f"procRAM {process_sample['rss_bytes'] / gib:4.1f}G | "
                                f"RAM({host['scope']}) "
                                f"{host['used_bytes'] / gib:4.1f}/"
                                f"{host['total_bytes'] / gib:4.1f}G"
                                + (
                                    " installed "
                                    f"{host['installed_physical_total_bytes'] / gib:.0f}G"
                                    if host.get("installed_physical_total_bytes")
                                    else ""
                                )
                            ),
                        )
                        loading_display.update_stats(loading_line)
                        previous_loading_sample = process_sample
                        previous_loading_gpu_bytes = gpu_used_bytes
                        previous_loading_time = now
                no_step_progress_seconds = (
                    time.perf_counter() - last_step_progress_time
                    if phase == "training" and last_step_progress_time is not None
                    else None
                )
                if phase == "training":
                    gib = 1024**3
                    gpu_used_bytes = float(sum(
                        gpu.get("memory_used_bytes") or 0 for gpu in gpus
                    ))
                    gpu_total_bytes = float(sum(
                        gpu.get("memory_total_bytes") or 0 for gpu in gpus
                    ))
                    process_cpu = (
                        process_sample["process_cpu_percent"]
                        if process_sample is not None
                        else 0.0
                    )
                    training_line = (
                        f"TRAIN {time.perf_counter() - training_started:6.0f}s | "
                        f"step {completed_step}/{total_training_steps} running "
                        f"({no_step_progress_seconds or 0:.0f}s since progress) | "
                        + (
                            f"WinGPU S{windows['shared_used_bytes'] / gib:.1f}/"
                            f"{windows['shared_total_bytes'] / gib:.1f}G C"
                            f"{windows['combined_used_bytes'] / gib:.1f}/"
                            f"{windows['combined_total_bytes'] / gib:.1f}G | "
                            if windows is not None
                            else "WinGPU shared unavailable | "
                        )
                        + f"GPU {utilization or 0:.0f}% "
                        f"{gpu_used_bytes / gib:.1f}/{gpu_total_bytes / gib:.1f}G "
                        f"{total_power:.0f}W {temperature or 0:.0f}C | "
                        f"procCPU {process_cpu:.0f}% {host['scope']}CPU "
                        f"{host['cpu_percent']:.0f}% | RAM "
                        f"{host['used_bytes'] / gib:.1f}/{host['total_bytes'] / gib:.1f}G"
                    )
                    training_display.update(training_line)
                breaches = _policy_breaches(
                    controller,
                    resources,
                    time.perf_counter() - phase_started,
                    phase,
                    no_step_progress_seconds,
                )
                active_keys = {key for key, _value, _limit, _label in breaches}
                for key in tuple(breach_since):
                    if key not in active_keys:
                        del breach_since[key]
                for key, value, limit, label in breaches:
                    if _is_immediate_timeout_key(key):
                        live_threshold_key = key
                        live_threshold_reason = (
                            f"{label} {value:.1f} reached {limit:.1f}"
                        )
                        break
                    breach_since.setdefault(key, now)
                    if now - breach_since[key] >= controller.termination_sustain_seconds:
                        live_threshold_key = key
                        live_threshold_reason = (
                            f"{label} {value:.1f} reached {limit:.1f} for "
                            f"{controller.termination_sustain_seconds:.1f}s"
                        )
                        break
                if live_threshold_reason:
                    training_display.finish()
                    print(
                        f"\nLIVE THRESHOLD: {live_threshold_reason}; stopping trial.",
                        flush=True,
                    )
                    _stop_process_group(process)
                    break
                next_capacity_check = now + 1.0
    except KeyboardInterrupt:
        training_display.finish()
        _stop_process_group(process)
        raise
    if persistent:
        return_code = 1 if worker_reported_failure else (0 if trial_finished else process.poll() or 1)
        if live_threshold_reason or return_code != 0:
            reason = (
                f"resource policy stopped this trial ({live_threshold_reason})"
                if live_threshold_reason
                else "the trial failed or the worker exited"
            )
            _discard_worker(trial_config.model_name, reason)
    else:
        while True:
            try:
                remaining_output = os.read(output_fd, 65536)
            except (BlockingIOError, OSError):
                break
            if not remaining_output:
                break
            for record in output_parser.feed(remaining_output):
                handle_output(record)
        for record in output_parser.feed(b"", final=True):
            handle_output(record)
        return_code = process.wait()
        try:
            os.close(output_fd)
        except OSError:
            pass
    runtime_seconds = time.perf_counter() - started
    model_loading_seconds = (
        training_started - started if training_started is not None else runtime_seconds
    )
    training_seconds = (
        (training_completed or time.perf_counter()) - training_started
        if training_started is not None
        else None
    )
    log_path = controller.output_dir / "logs" / f"{trial_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(output_lines), encoding="utf-8")
    report_path = controller.output_dir / "runs" / trial_id / "training-report.json"
    report = None
    gpu_peak = ram_peak = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gpu_peak, ram_peak = _peak_percentages(report)
    if live_gpu_peak is not None:
        gpu_peak = max(gpu_peak or 0.0, live_gpu_peak)
    ram_peak = max(ram_peak or 0.0, live_ram_peak)
    status = "completed" if return_code == 0 and report is not None else "failed"
    if live_threshold_reason:
        status = (
            "timeout"
            if live_threshold_key is not None
            and _is_immediate_timeout_key(live_threshold_key)
            else "threshold_reached"
        )
    result = {
        "trial_id": trial_id,
        "stage": stage,
        "status": status,
        "return_code": return_code,
        "runtime_seconds": runtime_seconds,
        "model_loading_seconds": model_loading_seconds,
        "training_seconds": training_seconds,
        "completed_steps": completed_step,
        "longest_step_seconds": longest_step_seconds,
        "seconds_since_last_step_at_finish": (
            time.perf_counter() - last_step_progress_time
            if last_step_progress_time is not None
            else None
        ),
        "model_weights_reused": model_reused,
        "loading_telemetry": loading_telemetry,
        "windows_gpu_memory": windows if "windows" in locals() else None,
        "terminal_mode": "pty",
        "live_display": "two_rows" if loading_display.tty else "snapshot_fallback",
        "configuration": printable,
        "gpu_memory_peak_percent": gpu_peak,
        "host_ram_peak_percent": ram_peak,
        "system_ram_peak_percent": ram_peak,
        "host_source": live_host_source,
        "host_scope": live_host_scope,
        "gpu_utilization_peak_percent": live_gpu_utilization_peak,
        "gpu_power_peak_watts": live_gpu_power_peak_watts,
        "gpu_energy_joules": live_energy_joules,
        "gpu_temperature_peak_celsius": live_gpu_temperature_peak_celsius,
        "threshold_reason": live_threshold_reason,
        "timeout_type": (
            "no_step_progress"
            if live_threshold_key == "no_step_progress_timeout_seconds"
            else (
                live_threshold_key.removesuffix("_seconds")
                if status == "timeout" and live_threshold_key is not None
                else None
            )
        ),
        "report_path": str(report_path) if report_path.exists() else None,
        "log_path": str(log_path),
    }
    print(
        f"TRIAL RESULT: {status} | GPU memory peak: {gpu_peak} | "
        f"{live_host_scope} RAM peak: {ram_peak}% | runtime: {runtime_seconds:.1f}s"
    )
    return result


def _ordered_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _capacity_frontier(
    trials: list[dict[str, Any]],
    axes: tuple[str, ...],
    candidates: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    """Summarize maximum successful values and their conditional configs."""
    completed = [trial for trial in trials if trial["status"] == "completed"]
    model_values = _ordered_unique(
        trial["configuration"].get("model_name") for trial in completed
    )
    frontier: dict[str, Any] = {"by_model": {}}
    for model in model_values:
        model_trials = [
            trial for trial in completed
            if trial["configuration"].get("model_name") == model
        ]
        maxima: dict[str, Any] = {}
        for axis in axes:
            ordered = candidates[axis]
            successful_values = {
                trial["configuration"].get(axis) for trial in model_trials
            }
            maximum = next(
                (value for value in reversed(ordered) if value in successful_values),
                None,
            )
            maxima[axis] = {
                "value": maximum,
                "configurations": [
                    trial["configuration"]
                    for trial in model_trials
                    if trial["configuration"].get(axis) == maximum
                ],
            }
        frontier["by_model"][str(model)] = maxima
    return frontier


def _model_execution_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize weight loads, reuse, restarts, and support per model."""
    models = _ordered_unique(
        trial["configuration"].get("model_name") for trial in trials
    )
    summary: dict[str, Any] = {}
    for model in models:
        model_trials = [
            trial
            for trial in trials
            if trial["configuration"].get("model_name") == model
        ]
        fresh_loads = sum(
            not trial.get("model_weights_reused", False) for trial in model_trials
        )
        summary[str(model)] = {
            "trial_count": len(model_trials),
            "completed_trial_count": sum(
                trial["status"] == "completed" for trial in model_trials
            ),
            "model_load_count": fresh_loads,
            "reused_trial_count": sum(
                trial.get("model_weights_reused", False) for trial in model_trials
            ),
            "worker_restart_count": max(0, fresh_loads - 1),
            "supported": any(
                trial["status"] == "completed" for trial in model_trials
            ),
        }
    return summary


def recover_interrupted_session(
    session_dir: Path,
    threshold_trial_id: str,
    observed_gpu_peak_percent: float,
    gpu_limit_percent: float = 90.0,
    ram_limit_percent: float = 95.0,
) -> dict[str, Any]:
    """Build a summary when an older controller was interrupted at a live limit."""
    trials: list[dict[str, Any]] = []
    last_safe_config: dict[str, Any] | None = None
    for config_path in sorted((session_dir / "trial-configs").glob("*.toml")):
        trial_id = config_path.stem
        with config_path.open("rb") as source:
            values = tomllib.load(source)
        training = values["training"]
        printable = {
            key: training.get(key)
            for key in (
                "model_name",
                "max_sequence_length",
                "max_examples",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "lora_rank",
                "load_in_4bit",
                "max_steps",
            )
        }
        report_path = session_dir / "runs" / trial_id / "training-report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            gpu_peak, ram_peak = _peak_percentages(report)
            status = "completed"
            runtime = report.get("duration_seconds")
            last_safe_config = training
        elif trial_id == threshold_trial_id:
            gpu_peak = observed_gpu_peak_percent
            ram_peak = None
            status = "threshold_reached"
            runtime = None
        else:
            continue
        trials.append(
            {
                "trial_id": trial_id,
                "stage": trial_id.split("-", 3)[2] if "-" in trial_id else "unknown",
                "status": status,
                "runtime_seconds": runtime,
                "configuration": printable,
                "gpu_memory_peak_percent": gpu_peak,
                "system_ram_peak_percent": ram_peak,
                "report_path": str(report_path) if report_path.exists() else None,
                "log_path": str(session_dir / "logs" / f"{trial_id}.log"),
            }
        )
    summary = {
        "schema_version": "1.0",
        "recovered_from_interruption": True,
        "inventory": hardware_software_inventory(session_dir),
        "thresholds": {
            "gpu_memory_limit_percent": gpu_limit_percent,
            "ram_limit_percent": ram_limit_percent,
        },
        "trials": trials,
        "stop_reason": (
            f"{threshold_trial_id}: observed GPU memory "
            f"{observed_gpu_peak_percent:.1f}% reached {gpu_limit_percent:.1f}% limit"
        ),
        "recommended_configuration": last_safe_config,
    }
    summary_path = session_dir / "autotune-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n=== RECOVERED AUTOTUNE SUMMARY ===")
    for trial in trials:
        print(
            f"{trial['trial_id']}: {trial['status']} | "
            f"GPU {trial['gpu_memory_peak_percent']}% | "
            f"RAM {trial['system_ram_peak_percent']}%"
        )
    print(f"Stop reason: {summary['stop_reason']}")
    print("Capacity frontier by model:")
    print(json.dumps(summary["capacity_frontier"], indent=2, default=str))
    if summary["pruned_candidates"]:
        print("Pruned candidates:")
        print(json.dumps(summary["pruned_candidates"], indent=2, default=str))
    print("Recommended safe configuration:")
    print(json.dumps(last_safe_config, indent=2, default=str))
    print(f"Full summary: {summary_path.resolve()}")
    return summary


def run_autotune(config_path: Path, plan_only: bool = False) -> dict[str, Any]:
    """Run autotune with an optional session-owned Windows telemetry producer."""
    controller, base_values = load_autotune_config(config_path)
    if not plan_only:
        session_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        session_id += f"-{uuid.uuid4().hex[:8]}"
        controller = replace(
            controller, output_dir=controller.output_dir / "sessions" / session_id
        )
    producer: WindowsTelemetryProducer | None = None
    if not plan_only and controller.manage_windows_bridge:
        bridge_path = controller.output_dir / "windows-host.json"
        controller = replace(controller, windows_bridge_path=bridge_path)
        producer = WindowsTelemetryProducer(
            script_path=(
                Path(__file__).resolve().parent
                / "windows-gpu-telemetry.ps1"
            ),
            output_path=bridge_path,
            interval_seconds=controller.windows_bridge_interval_seconds,
            startup_timeout_seconds=(
                controller.windows_bridge_startup_timeout_seconds
            ),
            max_age_seconds=controller.windows_bridge_max_age_seconds,
        )
        print("Starting managed Windows GPU/host telemetry...")
        producer.start()
        print(f"Windows telemetry ready: {bridge_path}")
    try:
        return _run_autotune(controller, base_values, plan_only)
    except BaseException:
        if not plan_only:
            _shutdown_workers()
            _print_final_safe_combinations(controller.output_dir)
        raise
    finally:
        if producer is not None:
            producer.stop()
            print("Managed Windows telemetry stopped.")


def _record_trial_result(
    trials: list[dict[str, Any]],
    result: dict[str, Any],
    session_dir: Path,
) -> None:
    """Durably record a trial and refresh the latest per-model safe report."""
    trials.append(result)
    try:
        persist_trial_result(session_dir, result)
        update_final_safe_combinations(session_dir)
    except Exception as exc:
        print(
            f"INCREMENTAL REPORT ERROR for {result.get('trial_id')}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _print_final_safe_combinations(session_dir: Path) -> None:
    """Refresh and print the best durable results without masking training errors."""
    try:
        report = update_final_safe_combinations(session_dir)
    except Exception as exc:
        print(f"\nFINAL REPORT ERROR: {exc}", file=sys.stderr, flush=True)
        return
    print("\n" + render_final_safe_combinations(report), flush=True)
    print(
        f"Final JSON: {(session_dir / 'final-safe-combinations.json').resolve()}",
        flush=True,
    )
    print(
        f"Final Markdown: {(session_dir / 'final-safe-combinations.md').resolve()}",
        flush=True,
    )


def _run_autotune(
    controller: AutotuneConfig,
    base_values: dict[str, Any],
    plan_only: bool = False,
) -> dict[str, Any]:
    """Inventory the machine, execute staged trials, and persist a summary."""
    base = CustomPyTorchTrainingConfig.from_mapping(base_values.get("training", {}))
    inventory = hardware_software_inventory(
        controller.output_dir,
        controller.host_telemetry_source,
        controller.nvidia_smi_path,
        controller.host_installed_memory_gib,
    )
    if controller.host_telemetry_source == "auto":
        resolved_host_source = (
            "windows" if inventory["ram"]["scope"] == "windows_host" else "wsl"
        )
        controller = replace(controller, host_telemetry_source=resolved_host_source)
    _print_inventory(inventory)
    plan = {
        "strategy": controller.strategy,
        "axes": list(controller.axes),
        "candidates": {
            name: list(values) for name, values in controller.candidates.items()
        },
        "max_trials": controller.max_trials,
        "telemetry": {
            "host_source": controller.host_telemetry_source,
            "host_scope": inventory["ram"]["scope"],
            "installed_memory_gib": controller.host_installed_memory_gib,
            "nvidia_smi_path": controller.nvidia_smi_path,
            "windows_bridge_path": (
                str(controller.windows_bridge_path)
                if controller.windows_bridge_path is not None
                else None
            ),
            "windows_bridge_max_age_seconds": (
                controller.windows_bridge_max_age_seconds
            ),
            "manage_windows_bridge": controller.manage_windows_bridge,
            "windows_bridge_interval_seconds": (
                controller.windows_bridge_interval_seconds
            ),
            "windows_bridge_startup_timeout_seconds": (
                controller.windows_bridge_startup_timeout_seconds
            ),
        },
        "execution": {
            "reuse_loaded_model": controller.reuse_loaded_model,
            "restart_worker_after_oom": controller.restart_worker_after_oom,
            "max_trials_per_worker": controller.max_trials_per_worker,
        },
        "termination": {
            "gpu_memory_percent": controller.gpu_memory_limit_percent,
            "host_ram_percent": controller.ram_limit_percent,
            "gpu_temperature_celsius": controller.gpu_temperature_limit_celsius,
            "gpu_power_watts": controller.gpu_power_limit_watts,
            "gpu_power_percent_of_limit": controller.gpu_power_limit_percent,
            "model_loading_timeout_seconds": controller.model_loading_timeout_seconds,
            "trial_timeout_seconds": controller.trial_timeout_seconds,
            "no_step_progress_timeout_seconds": (
                controller.no_step_progress_timeout_seconds
            ),
            "sustain_seconds": controller.termination_sustain_seconds,
        },
    }
    print("\nSearch plan")
    print(json.dumps(plan, indent=2))
    if plan_only:
        return {"inventory": inventory, "plan": plan, "trials": []}

    controller.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = controller.output_dir / "capacity-dataset.jsonl"
    batch_candidates = controller.candidates.get(
        "per_device_train_batch_size", (base.per_device_train_batch_size,)
    )
    sequence_candidates = controller.candidates.get(
        "max_sequence_length", (base.max_sequence_length,)
    )
    record_count = max(max(int(value) for value in batch_candidates) * 2, 8)
    generate_capacity_dataset(
        dataset_path, record_count, max(int(value) for value in sequence_candidates)
    )
    trials: list[dict[str, Any]] = []
    last_safe: CustomPyTorchTrainingConfig | None = None
    boundary_reasons: list[str] = []
    pruned_candidates: list[dict[str, Any]] = []
    trial_number = 0
    if controller.strategy == "staged":
        model_candidates = (
            _ordered_unique(controller.candidates["model_name"])
            if "model_name" in controller.axes
            else (base.model_name,)
        )
        non_model_axes = tuple(
            axis for axis in controller.axes if axis != "model_name"
        )
        max_trials_reached = False
        for model_name in model_candidates:
            print(f"\n=== MODEL {model_name} ===")
            print(f"Completing all configured axes before leaving {model_name}.")
            initial_values = {
                axis: controller.candidates[axis][0] for axis in non_model_axes
            }
            current = replace(
                base,
                model_name=model_name,
                max_examples=record_count,
                **initial_values,
            )
            model_safe: CustomPyTorchTrainingConfig | None = None
            axes_to_run = non_model_axes or ("model_name",)
            failed_axis_index: int | None = None
            minimum_failure_was_timeout = False
            for axis_index, field_name in enumerate(axes_to_run):
                field_candidates = _ordered_unique(
                    controller.candidates[field_name]
                    if field_name != "model_name"
                    else (model_name,)
                )
                for candidate_index, candidate in enumerate(field_candidates):
                    if trial_number >= controller.max_trials:
                        boundary_reasons.append("max_trials reached")
                        max_trials_reached = True
                        break
                    if getattr(current, field_name) == candidate and model_safe is not None:
                        continue
                    trial_number += 1
                    candidate_config = replace(current, **{field_name: candidate})
                    safe_model = str(model_name).replace("/", "_")
                    safe_candidate = str(candidate).replace("/", "_")
                    trial_id = (
                        f"trial-{trial_number:03d}-{safe_model}-"
                        f"{field_name}-{safe_candidate}"
                    )
                    result = _run_trial(
                        trial_id,
                        field_name,
                        candidate_config,
                        controller,
                        dataset_path,
                    )
                    _record_trial_result(trials, result, controller.output_dir)
                    if result["status"] == "completed":
                        current = candidate_config
                        model_safe = candidate_config
                        last_safe = candidate_config
                    else:
                        result["boundary_axis"] = field_name
                        result["boundary_value"] = candidate
                        result["last_safe_value"] = (
                            getattr(model_safe, field_name)
                            if model_safe is not None
                            else None
                        )
                        boundary_reasons.append(
                            f"{model_name}/{field_name}: "
                            f"{result['status']} at {candidate}"
                        )
                        if result["status"] == "timeout":
                            for skipped in field_candidates[candidate_index + 1 :]:
                                pruned_candidates.append(
                                    {
                                        "model_name": model_name,
                                        "axis": field_name,
                                        "candidate": skipped,
                                        "boundary_value": candidate,
                                        "last_safe_value": result["last_safe_value"],
                                        "reason": "higher_than_timed_out_boundary",
                                    }
                                )
                        failed_axis_index = axis_index
                        minimum_failure_was_timeout = (
                            model_safe is None and result["status"] == "timeout"
                        )
                        break
                    if controller.cooldown_seconds:
                        print(f"Cooling down for {controller.cooldown_seconds:g}s...")
                        time.sleep(controller.cooldown_seconds)
                if max_trials_reached or model_safe is None:
                    break
            if model_safe is None:
                if failed_axis_index is not None and minimum_failure_was_timeout:
                    for skipped_axis in axes_to_run[failed_axis_index + 1 :]:
                        for skipped in _ordered_unique(
                            controller.candidates[skipped_axis]
                        ):
                            pruned_candidates.append(
                                {
                                    "model_name": model_name,
                                    "axis": skipped_axis,
                                    "candidate": skipped,
                                    "boundary_value": None,
                                    "last_safe_value": None,
                                    "reason": "minimum_configuration_timed_out",
                                }
                            )
                print(
                    f"MODEL UNSUPPORTED: {model_name} failed its minimum "
                    "configuration; skipping remaining axes."
                )
            if max_trials_reached:
                break
    else:
        grid_axes = tuple(
            (["model_name"] if "model_name" in controller.axes else [])
            + [axis for axis in controller.axes if axis != "model_name"]
        )
        candidate_lists = [controller.candidates[axis] for axis in grid_axes]
        combinations = itertools.product(*candidate_lists)
        grid_active_axis = next(
            (axis for axis in reversed(grid_axes) if axis != "model_name"),
            None,
        )
        grid_timeout_boundaries: list[dict[str, Any]] = []
        for values in combinations:
            if trial_number >= controller.max_trials:
                boundary_reasons.append("max_trials reached")
                break
            updates = dict(zip(grid_axes, values))
            matching_boundary = next(
                (
                    boundary
                    for boundary in grid_timeout_boundaries
                    if all(
                        updates[name] == value
                        for name, value in boundary["context"].items()
                    )
                    and controller.candidates[boundary["axis"]].index(
                        updates[boundary["axis"]]
                    )
                    >= boundary["candidate_index"]
                ),
                None,
            )
            if matching_boundary is not None:
                pruned_candidates.append(
                    {
                        **updates,
                        "axis": matching_boundary["axis"],
                        "boundary_value": matching_boundary["boundary_value"],
                        "reason": "dominated_by_timed_out_grid_boundary",
                    }
                )
                continue
            trial_number += 1
            candidate_config = replace(base, max_examples=record_count, **updates)
            label = "_".join(str(value).replace("/", "_") for value in values)
            trial_id = f"trial-{trial_number:03d}-grid-{label}"
            result = _run_trial(
                trial_id, "grid", candidate_config, controller, dataset_path
            )
            _record_trial_result(trials, result, controller.output_dir)
            if result["status"] == "completed":
                last_safe = candidate_config
            else:
                boundary_reasons.append(
                    f"grid: {result['status']} at {updates}"
                )
                if result["status"] == "timeout" and grid_active_axis is not None:
                    result["boundary_axis"] = grid_active_axis
                    result["boundary_value"] = updates[grid_active_axis]
                    result["last_safe_value"] = None
                    grid_timeout_boundaries.append(
                        {
                            "axis": grid_active_axis,
                            "boundary_value": updates[grid_active_axis],
                            "candidate_index": controller.candidates[
                                grid_active_axis
                            ].index(updates[grid_active_axis]),
                            "context": {
                                name: value
                                for name, value in updates.items()
                                if name != grid_active_axis
                            },
                        }
                    )
            if controller.cooldown_seconds:
                print(f"Cooling down for {controller.cooldown_seconds:g}s...")
                time.sleep(controller.cooldown_seconds)

    summary = {
        "schema_version": "1.0",
        "inventory": inventory,
        "plan": plan,
        "trials": trials,
        "capacity_frontier": _capacity_frontier(
            trials, controller.axes, controller.candidates
        ),
        "model_execution": _model_execution_summary(trials),
        "pruned_candidates": pruned_candidates,
        "stop_reason": (
            "; ".join(boundary_reasons)
            if boundary_reasons
            else "candidate_space_exhausted"
        ),
        "recommended_configuration": (
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(last_safe).items()
            }
            if last_safe is not None
            else None
        ),
    }
    summary_path = controller.output_dir / "autotune-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n=== AUTOTUNE SUMMARY ===")
    for trial in trials:
        print(
            f"{trial['trial_id']}: {trial['status']} | "
            f"GPU {trial['gpu_memory_peak_percent']}% | "
            f"RAM {trial['system_ram_peak_percent']}% | "
            f"{trial['runtime_seconds']:.1f}s"
        )
    print(f"Stop reason: {summary['stop_reason']}")
    print("Per-model execution and reuse:")
    print(json.dumps(summary["model_execution"], indent=2, default=str))
    print("Capacity frontier by model:")
    print(json.dumps(summary["capacity_frontier"], indent=2, default=str))
    print("Recommended safe configuration:")
    print(json.dumps(summary["recommended_configuration"], indent=2, default=str))
    print(f"Full summary: {summary_path.resolve()}")
    _shutdown_workers()
    _print_final_safe_combinations(controller.output_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the capacity autotuner."""
    args = parse_args(argv)
    run_autotune(args.config, args.plan)


if __name__ == "__main__":
    main()
