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
    script = Path(
        "src/cognityx_training/windows-gpu-telemetry.ps1"
    ).read_text(encoding="utf-8")

    assert "ConvertTo-NonnegativeInt64" in script
    assert "[Math]::Max([double]0, [double]$Value)" in script
    assert "[Math]::Max(0, $dedicatedUsed)" not in script
    # The reported failure is greater than Int32.MaxValue but valid as Int64.
    assert 33_497_915_392 > 2_147_483_647


def test_managed_windows_producer_starts_waits_and_stops_owned_process(
    tmp_path, monkeypatch
) -> None:
    script_path = tmp_path / "collector.ps1"
    script_path.write_text("# test", encoding="utf-8")
    output_path = tmp_path / "session" / "windows-host.json"

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            return 0

    process = FakeProcess()
    commands = []
    monkeypatch.setattr(telemetry.Path, "exists", lambda _path: True)
    monkeypatch.setattr(
        telemetry.WindowsTelemetryProducer,
        "_windows_path",
        staticmethod(lambda path: f"WIN:{path}"),
    )
    monkeypatch.setattr(
        telemetry.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )
    monkeypatch.setattr(
        telemetry,
        "query_windows_bridge",
        lambda path, max_age: {"shared_used_bytes": 1},
    )
    producer = telemetry.WindowsTelemetryProducer(script_path, output_path)

    sample = producer.start()
    producer.stop()

    assert sample["shared_used_bytes"] == 1
    assert "-ExecutionPolicy" in commands[0]
    assert "Bypass" in commands[0]
    assert f"WIN:{output_path}" in commands[0]
    assert process.terminated is True
    assert producer.process is None


def test_managed_windows_producer_cleans_up_after_startup_failure(
    tmp_path, monkeypatch
) -> None:
    script_path = tmp_path / "collector.ps1"
    script_path.write_text("# test", encoding="utf-8")
    output_path = tmp_path / "windows-host.json"

    class FailedProcess:
        def poll(self):
            return 1

    monkeypatch.setattr(telemetry.Path, "exists", lambda _path: True)
    monkeypatch.setattr(
        telemetry.WindowsTelemetryProducer,
        "_windows_path",
        staticmethod(lambda path: str(path)),
    )
    monkeypatch.setattr(
        telemetry.subprocess, "Popen", lambda *args, **kwargs: FailedProcess()
    )
    monkeypatch.setattr(
        telemetry, "query_windows_bridge", lambda path, max_age: None
    )
    producer = telemetry.WindowsTelemetryProducer(script_path, output_path)

    try:
        producer.start()
    except RuntimeError as exc:
        assert "exited during startup" in str(exc)
    else:
        raise AssertionError("failed telemetry producer was accepted")

    assert producer.process is None
    assert producer._log is None
