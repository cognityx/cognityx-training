from subprocess import CompletedProcess
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from cognityx_training import telemetry


def test_nvidia_telemetry_reports_whole_device_memory_and_power(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return CompletedProcess(
            args=args,
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 5090, 11200, 32607, 93, 450, 575, 72\n",
        )

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)

    gpu = telemetry.query_nvidia_gpus()[0]

    assert gpu["scope"] == "whole_device"
    assert gpu["memory_used_bytes"] == 11200 * 1024**2
    assert round(gpu["memory_used_percent"], 1) == 34.3
    assert gpu["power_draw_watts"] == 450
    assert round(gpu["power_percent_of_limit"], 1) == 78.3


def test_windows_host_uses_installed_physical_memory(monkeypatch) -> None:
    monkeypatch.setattr(telemetry.Path, "exists", lambda _path: True)
    monkeypatch.setattr(
        telemetry.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                '{"total_bytes":137438953472,'
                '"available_bytes":68719476736,"cpu_percent":38}'
            ),
        ),
    )

    host = telemetry.query_windows_host()

    assert host is not None
    assert host["scope"] == "windows_host"
    assert host["total_kind"] == "installed_physical_memory"
    assert host["total_bytes"] == 128 * 1024**3
    assert host["used_bytes"] == 64 * 1024**3


def test_windows_bridge_accepts_fresh_sample_and_rejects_stale(tmp_path) -> None:
    captured = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "windows-host.json"
    path.write_text(
        json.dumps(
            {
                "captured_at_utc": captured.isoformat(),
                "source": "windows_performance_counters",
                "dedicated_used_bytes": 31 * 1024**3,
                "dedicated_total_bytes": 32 * 1024**3,
                "shared_used_bytes": 45 * 1024**3,
                "shared_total_bytes": 64 * 1024**3,
                "combined_used_bytes": 76 * 1024**3,
                "combined_total_bytes": 96 * 1024**3,
                "host_memory_used_bytes": 65 * 1024**3,
                "host_memory_total_bytes": 128 * 1024**3,
                "host_cpu_percent": 38,
            }
        ),
        encoding="utf-8",
    )

    fresh = telemetry.query_windows_bridge(
        path, max_age_seconds=5, now=captured + timedelta(seconds=2)
    )
    stale = telemetry.query_windows_bridge(
        path, max_age_seconds=5, now=captured + timedelta(seconds=10)
    )

    assert fresh is not None
    assert fresh["shared_used_bytes"] == 45 * 1024**3
    assert fresh["combined_used_bytes"] == 76 * 1024**3
    assert fresh["host_memory_total_bytes"] == 128 * 1024**3
    assert fresh["age_seconds"] == 2
    assert stale is None


def test_windows_collector_uses_64_bit_safe_gpu_counter_conversion() -> None:
    script = Path("scripts/windows-gpu-telemetry.ps1").read_text(encoding="utf-8")

    assert "ConvertTo-NonnegativeInt64" in script
    assert "[Math]::Max([double]0, [double]$Value)" in script
    assert "[Math]::Max(0, $dedicatedUsed)" not in script
    # The reported failure is greater than Int32.MaxValue but valid as Int64.
    assert 33_497_915_392 > 2_147_483_647
