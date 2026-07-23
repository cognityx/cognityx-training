import json
from pathlib import Path

from cognityx_training.cli import parse_args
from cognityx_training.reporting import (
    ResourceMonitor,
    latency_summary,
    write_training_report,
)


def test_cli_accepts_inspection_and_run_overrides() -> None:
    args = parse_args(
        [
            "--config",
            "training.toml",
            "--output-dir",
            "outputs",
            "--run-id",
            "smoke-run",
            "--print-config",
            "--dry-run",
        ]
    )

    assert args.config == Path("training.toml")
    assert args.output_dir == Path("outputs")
    assert args.run_id == "smoke-run"
    assert args.print_config is True
    assert args.dry_run is True


def test_report_writer_and_latency_summary(tmp_path) -> None:
    report = {
        "run_id": "run-1",
        "response_times": [latency_summary("training_step", [0.1, 0.2, 0.3])],
    }

    path = write_training_report(report, tmp_path / "run-1")
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "run-1" / "training-report.json"
    assert saved["response_times"][0]["average_ms"] == 200.0
    assert saved["response_times"][0]["maximum_ms"] == 300.0


def test_resource_monitor_owns_sampling_methods() -> None:
    monitor = ResourceMonitor()

    assert callable(monitor._run)
    assert callable(monitor._sample)
    assert callable(monitor._safe_io)
