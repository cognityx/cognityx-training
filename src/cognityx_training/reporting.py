"""Runtime measurement and JSON reporting for concrete training jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable


def utc_now() -> str:
    """Return the current time as an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def directory_size(path: Path) -> int:
    """Return the total serialized size of regular files below ``path``."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parameter_counts(model: Any) -> dict[str, int]:
    """Count all and trainable parameters on a framework model."""
    parameters = tuple(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }


def latency_summary(operation: str, values_seconds: list[float]) -> dict[str, Any]:
    """Summarize measured operation latency in milliseconds."""
    values = sorted(value * 1000 for value in values_seconds)
    if not values:
        return {"operation": operation, "sample_count": 0}

    def percentile(fraction: float) -> float:
        index = max(0, min(len(values) - 1, round((len(values) - 1) * fraction)))
        return values[index]

    return {
        "operation": operation,
        "sample_count": len(values),
        "average_ms": statistics.fmean(values),
        "minimum_ms": values[0],
        "median_ms": statistics.median(values),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "maximum_ms": values[-1],
    }


@dataclass(slots=True)
class ResourceMonitor:
    """Sample process CPU/RAM/disk I/O and optional GPU measurements."""

    gpu_sampler: Callable[[], dict[str, Any] | None] | None = None
    host_sampler: Callable[[], dict[str, Any]] | None = None
    interval_seconds: float = 0.25
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _samples: list[dict[str, Any]] = field(default_factory=list, init=False)
    _initial_io: Any = field(default=None, init=False)
    _process: Any = field(default=None, init=False)

    def start(self) -> None:
        """Begin background sampling."""
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "Resource reporting requires psutil; run `uv sync --extra training`."
            ) from exc
        self._process = psutil.Process(os.getpid())
        self._initial_io = self._safe_io()
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return aggregate measurements."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        final_io = self._safe_io()
        cpu = [sample["cpu_percent"] for sample in self._samples]
        ram = [sample["ram_bytes"] for sample in self._samples]
        host_ram_percent = [sample["host"]["used_percent"] for sample in self._samples]
        host_ram_used = [sample["host"]["used_bytes"] for sample in self._samples]
        host_cpu = [sample["host"]["cpu_percent"] for sample in self._samples]
        host = self._samples[-1]["host"]
        report: dict[str, Any] = {
            "sample_count": len(self._samples),
            "cpu_average_percent": statistics.fmean(cpu) if cpu else None,
            "cpu_peak_percent": max(cpu, default=None),
            "ram_average_bytes": round(statistics.fmean(ram)) if ram else None,
            "ram_peak_bytes": max(ram, default=None),
            "host_source": host["source"],
            "host_scope": host["scope"],
            "host_ram_total_bytes": host["total_bytes"],
            "host_ram_average_used_bytes": round(statistics.fmean(host_ram_used)),
            "host_ram_peak_used_bytes": max(host_ram_used),
            "host_ram_average_percent": statistics.fmean(host_ram_percent),
            "host_ram_peak_percent": max(host_ram_percent),
            "host_cpu_average_percent": statistics.fmean(host_cpu),
            "host_cpu_peak_percent": max(host_cpu),
            # Backward-compatible names; source/scope above disambiguate them.
            "system_ram_total_bytes": host["total_bytes"],
            "system_ram_average_used_bytes": round(statistics.fmean(host_ram_used)),
            "system_ram_peak_used_bytes": max(host_ram_used),
            "system_ram_average_percent": statistics.fmean(host_ram_percent),
            "system_ram_peak_percent": max(host_ram_percent),
        }
        if self._initial_io is not None and final_io is not None:
            report.update(
                disk_read_bytes=final_io.read_bytes - self._initial_io.read_bytes,
                disk_write_bytes=final_io.write_bytes - self._initial_io.write_bytes,
                disk_read_operations=final_io.read_count - self._initial_io.read_count,
                disk_write_operations=final_io.write_count - self._initial_io.write_count,
            )
        gpu_samples = [sample["gpu"] for sample in self._samples if sample["gpu"]]
        report["gpu_usage"] = self._aggregate_gpu(gpu_samples)
        return report

    def snapshot(self) -> dict[str, Any]:
        """Return the latest live values for terminal progress output."""
        if not self._samples:
            self._sample()
        latest = self._samples[-1]
        gpu_samples = [sample["gpu"] for sample in self._samples if sample["gpu"]]
        gpu_aggregate = self._aggregate_gpu(gpu_samples)
        current_io = self._safe_io()
        snapshot = {
            "cpu_percent": latest["cpu_percent"],
            "ram_bytes": latest["ram_bytes"],
            "host": latest["host"],
            "gpu": latest["gpu"],
            "gpu_aggregate": gpu_aggregate[0] if gpu_aggregate else None,
            "disk_read_bytes": 0,
            "disk_write_bytes": 0,
        }
        if self._initial_io is not None and current_io is not None:
            snapshot["disk_read_bytes"] = (
                current_io.read_bytes - self._initial_io.read_bytes
            )
            snapshot["disk_write_bytes"] = (
                current_io.write_bytes - self._initial_io.write_bytes
            )
        return snapshot

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        gpu = None
        if self.gpu_sampler is not None:
            try:
                gpu = self.gpu_sampler()
            except Exception:
                gpu = None
        if self.host_sampler is not None:
            try:
                host = self.host_sampler()
            except Exception:
                from cognityx_training.telemetry import query_wsl_host

                host = query_wsl_host()
        else:
            from cognityx_training.telemetry import query_wsl_host

            host = query_wsl_host()
        self._samples.append(
            {
                "sampled_at_monotonic": time.monotonic(),
                "cpu_percent": self._process.cpu_percent(interval=None),
                "ram_bytes": self._process.memory_info().rss,
                "host": host,
                "gpu": gpu,
            }
        )

    def _safe_io(self) -> Any:
        try:
            return self._process.io_counters()
        except (AttributeError, OSError):
            return None

    @staticmethod
    def _aggregate_gpu(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not samples:
            return []
        utilization = [
            item["utilization_percent"]
            for item in samples
            if item.get("utilization_percent") is not None
        ]
        memory = [
            item.get("memory_used_bytes", item.get("memory_bytes", 0))
            for item in samples
        ]
        power = [
            item["power_draw_watts"]
            for item in samples
            if item.get("power_draw_watts") is not None
        ]
        temperatures = [
            item["temperature_celsius"]
            for item in samples
            if item.get("temperature_celsius") is not None
        ]
        first = samples[0]
        energy_joules = 0.0
        for previous, current in zip(samples, samples[1:]):
            if previous.get("power_draw_watts") is not None:
                energy_joules += previous["power_draw_watts"] * (
                    current["sampled_at_monotonic"]
                    - previous["sampled_at_monotonic"]
                )
        return [
            {
                "source": first.get("source", "framework"),
                "scope": first.get("scope", "process"),
                "device": first["device"],
                "device_index": first["device_index"],
                "sample_count": len(samples),
                "utilization_average_percent": (
                    statistics.fmean(utilization) if utilization else None
                ),
                "utilization_peak_percent": max(utilization, default=None),
                "memory_average_bytes": round(statistics.fmean(memory)),
                "memory_peak_bytes": max(memory),
                "memory_total_bytes": first.get("memory_total_bytes"),
                "memory_peak_percent": (
                    max(memory) / first["memory_total_bytes"] * 100
                    if first.get("memory_total_bytes")
                    else None
                ),
                "framework_allocated_peak_bytes": max(
                    (
                        item.get("framework_allocated_bytes", 0)
                        for item in samples
                    ),
                    default=None,
                ),
                "framework_reserved_peak_bytes": max(
                    (item.get("framework_reserved_bytes", 0) for item in samples),
                    default=None,
                ),
                "power_average_watts": statistics.fmean(power) if power else None,
                "power_peak_watts": max(power, default=None),
                "power_limit_watts": first.get("power_limit_watts"),
                "energy_consumed_joules": energy_joules if power else None,
                "temperature_average_celsius": (
                    statistics.fmean(temperatures) if temperatures else None
                ),
                "temperature_peak_celsius": max(temperatures, default=None),
            }
        ]


def format_progress(step: int, total_steps: int, loss: float, sample: dict[str, Any]) -> str:
    """Format one compact training/resource progress line."""
    gib = 1024**3
    gpu = sample.get("gpu")
    host = sample["host"]
    gpu_text = "GPU n/a"
    if gpu:
        memory_bytes = gpu.get("memory_used_bytes", gpu.get("memory_bytes", 0))
        utilization = gpu.get("utilization_percent")
        utilization_text = f"{utilization:.0f}%" if utilization is not None else "n/a"
        power_text = (
            f" {gpu['power_draw_watts']:.0f}W"
            if gpu.get("power_draw_watts") is not None
            else ""
        )
        gpu_text = (
            f"GPU {utilization_text} "
            f"{memory_bytes / gib:.2f} GiB{power_text}"
            if gpu.get("utilization_percent") is not None
            else f"GPU {memory_bytes / gib:.2f} GiB{power_text}"
        )
    return (
        f"step {step}/{total_steps} | loss {loss:.4f} | "
        f"CPU {sample['cpu_percent']:.0f}% | process RAM "
        f"{sample['ram_bytes'] / gib:.2f} GiB | {host['scope']} RAM "
        f"{host['used_bytes'] / gib:.1f}/{host['total_bytes'] / gib:.1f} GiB "
        f"({host['used_percent']:.1f}%, {host.get('total_kind', 'unknown_total')}) | "
        f"host CPU {host['cpu_percent']:.0f}% | "
        f"disk R/W {sample['disk_read_bytes'] / gib:.2f}/"
        f"{sample['disk_write_bytes'] / gib:.2f} GiB | {gpu_text}"
    )


def write_training_report(report: dict[str, Any], run_dir: Path) -> Path:
    """Atomically write a training report in the run output directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "training-report.json"
    temporary_path = run_dir / ".training-report.json.tmp"
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(report_path)
    return report_path


def jsonable_configuration(config: Any) -> dict[str, Any]:
    """Convert a dataclass configuration to JSON-compatible values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
