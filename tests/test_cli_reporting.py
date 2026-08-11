import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cognityx_core import Artifact

from cognityx_training.cli import (
    CLI_RESULT_SCHEMA,
    main,
    parse_args,
    resolve_training_config,
)
from cognityx_training.custom_pytorch import _status, evaluation_changes
from cognityx_training.dataset_pipeline import DatasetLineage, DatasetStatistics
from cognityx_training.reporting import (
    ResourceMonitor,
    latency_summary,
    write_training_report,
)
from cognityx_training.runtime_check import RUNTIME_CHECK_SCHEMA


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
    assert args.output_format == "human"


def _cli_config(tmp_path: Path) -> Path:
    config = tmp_path / "training.toml"
    config.write_text(
        "[training]\n"
        'model_name = ""\n'
        f'model_cache_dir = "{tmp_path}"\n'
        f'output_dir = "{tmp_path / "outputs"}"\n'
        'dataset_input_mode = "legacy_jsonl"\n'
        "max_steps = 1\n"
        "per_device_train_batch_size = 1\n"
        "gradient_accumulation_steps = 1\n"
        "[experiment]\n"
        'id = "exp-cli-contract"\n'
        "[dataset]\n"
        'name = "safe-fixture"\n'
        'version = "1"\n'
        'uri = "fixture.jsonl"\n',
        encoding="utf-8",
    )
    return config


def _preflight_result():
    return SimpleNamespace(
        lineage=DatasetLineage(
            dataset_id="dataset-safe",
            dataset_name="safe-fixture",
            dataset_version="1",
            dataset_variant_id="variant-safe",
            dataset_manifest_uri=None,
            dataset_manifest_checksum="manifest-checksum",
            records_uri=None,
            records_checksum="records-checksum",
            recipe="fixture",
            source_manifest_uri=None,
            source_manifest_checksum="source-checksum",
            configuration_checksum="config-checksum",
            research_package_manifest_checksum="package-checksum",
            research_package_id="package-safe",
            research_package_version="1",
        ),
        statistics=DatasetStatistics(
            total_records=3,
            training_records=1,
            evaluation_records=2,
            selected_training_candidates=1,
            accepted_training_examples=1,
            skipped_overlength_count=0,
            maximum_observed_token_length=None,
            maximum_accepted_token_length=None,
            configured_max_sequence_length=512,
            split_summary={"train": 1, "evaluation": 2},
        ),
    )


def test_human_dry_run_output_remains_backward_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "cognityx_training.dataset_pipeline.preflight_dataset",
        lambda *args, **kwargs: _preflight_result(),
    )

    main(["--config", str(_cli_config(tmp_path)), "--dry-run"])

    captured = capsys.readouterr()
    assert captured.out.startswith("Dataset lineage: ")
    assert "Dataset records: 3 total, 1 training, 2 evaluation." in captured.out


def test_json_dry_run_is_one_directly_parseable_safe_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "cognityx_training.dataset_pipeline.preflight_dataset",
        lambda *args, **kwargs: _preflight_result(),
    )

    main(
        [
            "--config",
            str(_cli_config(tmp_path)),
            "--run-id",
            "trun-json-dry-run",
            "--dry-run",
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert result == {
        "schema": CLI_RESULT_SCHEMA,
        "mode": "dry_run",
        "experiment_id": "exp-cli-contract",
        "training_run_id": "trun-json-dry-run",
        "total_records": 3,
        "accepted_training_examples": 1,
        "evaluation_records": 2,
        "micro_batch_size": 1,
        "effective_batch_size": 1,
        "optimizer_steps": 1,
        "dataset": {
            "dataset_id": "dataset-safe",
            "dataset_version": "1",
            "dataset_manifest_checksum": "manifest-checksum",
            "records_checksum": "records-checksum",
            "research_package_id": "package-safe",
            "research_package_version": "1",
            "research_package_manifest_checksum": "package-checksum",
        },
    }
    assert captured.err == ""


def test_json_completed_result_is_one_directly_parseable_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = SimpleNamespace(
        artifact=Artifact(
            name="fixture-adapter",
            version="1",
            uri="storage://local-main/models/adapters/adp-safe/1",
        ),
        report_uri="storage://local-main/artifacts/runs/trun-safe/report.json",
        metrics={
            "experiment_id": "exp-cli-contract",
            "training_variant_id": "tvar-safe",
            "training_run_id": "trun-safe",
            "adapter_id": "adp-safe",
            "adapter_manifest_uri": (
                "storage://local-main/models/adapters/adp-safe/1/adapter-manifest.json"
            ),
            "publication_manifest_uri": (
                "storage://local-main/artifacts/runs/trun-safe/publication-manifest.json"
            ),
        },
    )
    monkeypatch.setattr(
        "cognityx_training.cli.create_training_backend",
        lambda config: SimpleNamespace(train=lambda request: result),
    )

    main(
        [
            "--config",
            str(_cli_config(tmp_path)),
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["schema"] == CLI_RESULT_SCHEMA
    assert payload["mode"] == "completed"
    for field in (
        "experiment_id",
        "training_variant_id",
        "training_run_id",
        "adapter_id",
        "adapter_manifest_uri",
        "training_report_uri",
        "publication_manifest_uri",
        "artifact_uri",
    ):
        assert payload[field]


def test_training_status_is_written_only_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _status("progress-safe")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "progress-safe\n"


def test_json_output_rejects_ambiguous_print_config() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(
            [
                "--config",
                "training.toml",
                "--print-config",
                "--output-format",
                "json",
            ]
        )
    assert exc.value.code == 2


def test_runtime_check_is_mutually_exclusive_with_dry_run() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(
            [
                "--config",
                "training.toml",
                "--dry-run",
                "--check-runtime",
            ]
        )
    assert exc.value.code == 2


def test_json_runtime_check_is_one_object_and_does_not_read_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {
        "schema": RUNTIME_CHECK_SCHEMA,
        "backend": "custom-pytorch",
        "passed": True,
        "packages": {"peft": {"status": "available", "version": "1"}},
        "missing_packages": [],
        "cuda": {
            "required": True,
            "available": True,
            "device_count": 1,
            "error_type": None,
        },
    }
    monkeypatch.setattr(
        "cognityx_training.cli.check_training_runtime", lambda **kwargs: result
    )
    monkeypatch.setattr(
        "cognityx_training.cli.Dataset",
        lambda **kwargs: pytest.fail("runtime check must not construct a dataset"),
    )

    main(
        [
            "--config",
            str(_cli_config(tmp_path)),
            "--check-runtime",
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert captured.err == ""


def test_failed_runtime_check_exits_nonzero_after_machine_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {
        "schema": RUNTIME_CHECK_SCHEMA,
        "backend": "custom-pytorch",
        "passed": False,
        "packages": {"peft": {"status": "unavailable", "version": None}},
        "missing_packages": ["peft"],
        "cuda": {
            "required": True,
            "available": True,
            "device_count": 1,
            "error_type": None,
        },
    }
    monkeypatch.setattr(
        "cognityx_training.cli.check_training_runtime", lambda **kwargs: result
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--config",
                str(_cli_config(tmp_path)),
                "--check-runtime",
                "--output-format",
                "json",
            ]
        )

    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["missing_packages"] == ["peft"]


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
