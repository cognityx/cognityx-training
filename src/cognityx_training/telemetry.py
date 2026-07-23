"""Labeled host and NVIDIA telemetry sources for WSL training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from typing import Any


class WindowsTelemetryProducer:
    """Own a Windows telemetry collector started for one autotune session."""

    def __init__(
        self,
        script_path: Path,
        output_path: Path,
        interval_seconds: float = 1.0,
        startup_timeout_seconds: float = 20.0,
        max_age_seconds: float = 5.0,
    ) -> None:
        self.script_path = script_path
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.max_age_seconds = max_age_seconds
        self.process: subprocess.Popen[str] | None = None
        self._log = None
        self.log_path = output_path.with_suffix(".producer.log")

    @staticmethod
    def _windows_path(path: Path) -> str:
        result = subprocess.run(
            ["wslpath", "-w", str(path.resolve())],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    def start(self) -> dict[str, Any]:
        """Start the collector and wait until it writes a fresh valid sample."""
        powershell = Path(
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        if not powershell.exists():
            raise RuntimeError(
                "Cannot start managed Windows GPU telemetry: powershell.exe "
                "is unavailable from WSL."
            )
        if not self.script_path.exists():
            raise RuntimeError(
                f"Windows GPU telemetry script not found: {self.script_path}"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        command = [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            self._windows_path(self.script_path),
            "-OutputPath",
            self._windows_path(self.output_path),
            "-IntervalSeconds",
            str(self.interval_seconds),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + self.startup_timeout_seconds
            while time.monotonic() < deadline:
                sample = query_windows_bridge(
                    self.output_path, self.max_age_seconds
                )
                if sample is not None:
                    return sample
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Managed Windows GPU telemetry exited during startup. "
                        f"See {self.log_path}."
                    )
                time.sleep(min(0.2, self.interval_seconds))
            raise RuntimeError(
                "Managed Windows GPU telemetry did not produce a fresh sample "
                f"within {self.startup_timeout_seconds:g}s. See {self.log_path}."
            )
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop only the collector process owned by this instance."""
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "WindowsTelemetryProducer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def query_windows_host() -> dict[str, Any] | None:
    """Query Windows host RAM and CPU through WSL interop when available."""
    executable = Path(
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    if not executable.exists():
        return None
    command = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();"
        "$os=Get-CimInstance Win32_OperatingSystem;"
        "$computer=Get-CimInstance Win32_ComputerSystem;"
        "$cpu=(Get-CimInstance Win32_Processor | Measure-Object "
        "-Property LoadPercentage -Average).Average;"
        "[pscustomobject]@{"
        "total_bytes=[int64]$computer.TotalPhysicalMemory;"
        "available_bytes=([int64]$os.FreePhysicalMemory*1024);"
        "cpu_percent=[double]$cpu} | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [str(executable), "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        values = json.loads(result.stdout.strip())
        total = int(values["total_bytes"])
        available = int(values["available_bytes"])
        return {
            "source": "windows_powershell_cim",
            "scope": "windows_host",
            "total_kind": "installed_physical_memory",
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": total - available,
            "used_percent": (total - available) / total * 100,
            "cpu_percent": float(values["cpu_percent"]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        return None


def query_wsl_host() -> dict[str, Any]:
    """Query memory and CPU visible to the WSL virtual machine."""
    import psutil

    memory = psutil.virtual_memory()
    return {
        "source": "psutil",
        "scope": "wsl_vm",
        "total_kind": "wsl_memory_limit",
        "total_bytes": memory.total,
        "available_bytes": memory.available,
        "used_bytes": memory.used,
        "used_percent": float(memory.percent),
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
    }


def query_host(
    source: str = "auto", installed_memory_gib: float | None = None
) -> dict[str, Any]:
    """Query the requested host scope, falling back explicitly when allowed."""
    if source not in {"auto", "windows", "wsl"}:
        raise ValueError("host telemetry source must be auto, windows, or wsl.")
    if source in {"auto", "windows"}:
        windows = query_windows_host()
        if windows is not None:
            return windows
        if source == "windows":
            raise RuntimeError("Windows host telemetry is unavailable from WSL.")
    host = query_wsl_host()
    if installed_memory_gib is not None:
        host["installed_physical_total_bytes"] = round(installed_memory_gib * 1024**3)
        host["installed_physical_total_source"] = "configured"
    return host


def query_windows_bridge(
    path: Path | str | None,
    max_age_seconds: float = 5.0,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Read a fresh sample written by the Windows performance-counter bridge."""
    if path is None:
        return None
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        captured = datetime.fromisoformat(
            str(values["captured_at_utc"]).replace("Z", "+00:00")
        )
        current = now or datetime.now(timezone.utc)
        age = max(0.0, (current - captured.astimezone(timezone.utc)).total_seconds())
        if age > max_age_seconds:
            return None
        required = (
            "dedicated_used_bytes",
            "dedicated_total_bytes",
            "shared_used_bytes",
            "shared_total_bytes",
            "combined_used_bytes",
            "combined_total_bytes",
            "host_memory_used_bytes",
            "host_memory_total_bytes",
            "host_cpu_percent",
        )
        sample = {name: int(values[name]) for name in required[:-1]}
        sample["host_cpu_percent"] = float(values["host_cpu_percent"])
        sample.update(
            source=str(values.get("source", "windows_performance_counters")),
            scope="windows_host",
            captured_at_utc=captured.isoformat(),
            age_seconds=age,
        )
        return sample
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def query_nvidia_gpus(executable: str = "nvidia-smi") -> list[dict[str, Any]]:
    """Query whole-device NVIDIA memory, utilization, power, and temperature."""
    fields = (
        "index,name,memory.used,memory.total,utilization.gpu,"
        "power.draw,power.limit,temperature.gpu"
    )
    result = subprocess.run(
        [
            executable,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        memory_used = _number(parts[2])
        memory_total = _number(parts[3])
        power_draw = _number(parts[5])
        power_limit = _number(parts[6])
        gpus.append(
            {
                "source": "nvidia-smi",
                "scope": "whole_device",
                "device_index": int(parts[0]),
                "device": parts[1],
                "memory_used_bytes": (
                    round(memory_used * 1024**2) if memory_used is not None else None
                ),
                "memory_total_bytes": (
                    round(memory_total * 1024**2) if memory_total is not None else None
                ),
                "memory_used_percent": (
                    memory_used / memory_total * 100
                    if memory_used is not None and memory_total
                    else None
                ),
                "utilization_percent": _number(parts[4]),
                "power_draw_watts": power_draw,
                "power_limit_watts": power_limit,
                "power_percent_of_limit": (
                    power_draw / power_limit * 100
                    if power_draw is not None and power_limit
                    else None
                ),
                "temperature_celsius": _number(parts[7]),
            }
        )
    return gpus
