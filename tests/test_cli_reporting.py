import json
from pathlib import Path

import pytest

from cognityx_training.cli import parse_args, resolve_training_config
from cognityx_training.custom_pytorch import evaluation_changes
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
            "--experiment-id",
            "exp-smoke",
            "--dataset-uri",
            "storage://local-main/datasets/research/package.json",
            "--seed",
            "29",
            "--parent-run-id",
            "parent-observation-1",
            "--print-config",
            "--dry-run",
        ]
    )

    assert args.config == Path("training.toml")
    assert args.output_dir == Path("outputs")
    assert args.run_id == "smoke-run"
    assert args.experiment_id == "exp-smoke"
    assert args.dataset_uri == "storage://local-main/datasets/research/package.json"
    assert args.seed == 29
    assert args.parent_run_id == "parent-observation-1"
    assert args.print_config is True
    assert args.dry_run is True


def _config_values(tmp_path: Path, *, experiment_id: str, seed: int = 11):
    return {
        "training": {
            "model_cache_dir": str(tmp_path),
            "output_dir": "toml-output",
            "seed": seed,
            "storage_root": "toml-storage",
            "dataset_input_mode": "legacy_jsonl",
        },
        "experiment": {"id": experiment_id},
        "dataset": {"name": "fixture", "uri": "fixture.jsonl"},
    }


def test_cli_override_replaces_invalid_toml_experiment_before_validation(
    tmp_path: Path,
) -> None:
    args = parse_args(["--config", "training.toml", "--experiment-id", "exp-valid123"])

    config = resolve_training_config(
        _config_values(tmp_path, experiment_id="EXP-SYS-E2E-001"), args
    )

    assert config.experiment_id == "exp-valid123"


def test_invalid_toml_experiment_without_override_still_fails(tmp_path: Path) -> None:
    args = parse_args(["--config", "training.toml"])

    with pytest.raises(ValueError, match="exp-"):
        resolve_training_config(
            _config_values(tmp_path, experiment_id="EXP-SYS-E2E-001"), args
        )


def test_invalid_cli_experiment_override_still_fails(tmp_path: Path) -> None:
    args = parse_args(["--config", "training.toml", "--experiment-id", "invalid"])

    with pytest.raises(ValueError, match="exp-"):
        resolve_training_config(
            _config_values(tmp_path, experiment_id="exp-valid-toml"), args
        )


def test_cli_seed_override_wins_over_toml(tmp_path: Path) -> None:
    args = parse_args(["--config", "training.toml", "--seed", "29"])

    config = resolve_training_config(
        _config_values(tmp_path, experiment_id="exp-valid-toml", seed=11), args
    )

    assert config.seed == 29


def test_run_storage_and_dataset_mode_overrides_remain_functional(
    tmp_path: Path,
) -> None:
    storage_config = tmp_path / "storage.toml"
    storage_config.write_text("[storage]\n", encoding="utf-8")
    args = parse_args(
        [
            "--config",
            "training.toml",
            "--output-dir",
            str(tmp_path / "cli-output"),
            "--run-id",
            "trun-cli",
            "--storage-config",
            str(storage_config),
            "--storage-root",
            str(tmp_path / "cli-storage"),
            "--dataset-input-mode",
            "dataforge_manifest",
            "--parent-run-id",
            "parent-cli",
        ]
    )

    config = resolve_training_config(
        _config_values(tmp_path, experiment_id="exp-valid-toml"), args
    )

    assert config.output_dir == tmp_path / "cli-output"
    assert config.run_id is None
    assert config.training_run_id == "trun-cli"
    assert config.storage_config == storage_config
    assert config.storage_root == str(tmp_path / "cli-storage")
    assert config.dataset_input_mode == "dataforge_manifest"
    assert config.tracking_parent_run_id == "parent-cli"


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


def test_evaluation_changes_handle_zero_and_nonzero_counts() -> None:
    baseline = {"exact_match_accuracy": 0.25, "contains_expected_accuracy": 0.5}
    trained = {"exact_match_accuracy": 0.75, "contains_expected_accuracy": 0.8}

    assert evaluation_changes(baseline, trained, 3) == {
        "exact_match_change": 0.5,
        "contains_expected_change": 0.30000000000000004,
    }
    assert evaluation_changes(baseline, trained, 0) == {
        "exact_match_change": None,
        "contains_expected_change": None,
    }
