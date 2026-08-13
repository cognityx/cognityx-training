import json
from types import SimpleNamespace

import pytest

from cognityx_training import cli, evaluation_cli, tracking_cli
from cognityx_training.human import render_human


def test_human_renderer_preserves_full_values_without_ansi() -> None:
    assert render_human([]) == "No records."
    uri = "storage://local-main/models/adapters/full/manifest.json"
    checksum = "d" * 64
    output = render_human({"manifest_uri": uri, "checksum": checksum})
    assert uri in output
    assert checksum in output
    assert "\x1b" not in output


def test_static_config_human_calls_resolver_once(monkeypatch, capsys) -> None:
    calls = 0
    payload = {
        "component": "training",
        "valid": True,
        "master_config": {"path": "/tmp/training.toml", "sha256": "e" * 64},
        "config_layers": [],
        "overrides": [],
        "effective": {"seed": 29},
        "warnings": [],
        "errors": [],
    }

    def resolve(path, args):
        nonlocal calls
        calls += 1
        return SimpleNamespace(to_dict=lambda: payload)

    monkeypatch.setattr(cli, "resolve_training_configuration", resolve)
    cli.main(["config", "show", "--config", "training.toml", "--human"])
    assert calls == 1
    output = capsys.readouterr().out
    assert "Component: training" in output
    assert "e" * 64 in output
    assert not output.lstrip().startswith("{")


def test_human_alias_conflicts_with_explicit_json_before_resolution(
    monkeypatch,
) -> None:
    called = False

    def resolve(path, args):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "resolve_training_configuration", resolve)
    with pytest.raises(SystemExit) as captured:
        cli.main(
            [
                "--config",
                "training.toml",
                "--human",
                "--output-format",
                "json",
            ]
        )

    assert captured.value.code == 2
    assert called is False


def test_evaluation_human_invokes_pipeline_once(monkeypatch, capsys) -> None:
    calls = 0

    class Pipeline:
        def __init__(self, config):
            return None

        def plan(self):
            nonlocal calls
            calls += 1
            return {"status": "planned", "evaluation_id": "evaluation-full-id"}

    monkeypatch.setattr(evaluation_cli, "load_config", lambda path: object())
    monkeypatch.setattr(evaluation_cli, "EvaluationPipeline", Pipeline)
    evaluation_cli.main(["plan", "--config", "evaluation.toml", "--human"])
    assert calls == 1
    assert "Evaluation id: evaluation-full-id" in capsys.readouterr().out


def test_tracking_human_invokes_write_once(monkeypatch, capsys) -> None:
    calls = 0
    monkeypatch.setattr(
        tracking_cli, "resolve_storage_runtime", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        tracking_cli, "payload_from_publication", lambda *args: object()
    )
    monkeypatch.setattr(tracking_cli, "create_tracker", lambda **kwargs: object())

    def track(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            status="recorded", backend="none", external_run_id="external-full-id"
        )

    monkeypatch.setattr(tracking_cli, "track_with_policy", track)
    tracking_cli.main(
        [
            "storage://local-main/models/publication/manifest.json",
            "--experiment-name",
            "example",
            "--human",
        ]
    )
    assert calls == 1
    assert "External run id: external-full-id" in capsys.readouterr().out


def test_evaluation_default_remains_json(monkeypatch, capsys) -> None:
    class Pipeline:
        def __init__(self, config):
            return None

        def plan(self):
            return {"status": "planned"}

    monkeypatch.setattr(evaluation_cli, "load_config", lambda path: object())
    monkeypatch.setattr(evaluation_cli, "EvaluationPipeline", Pipeline)
    evaluation_cli.main(["plan", "--config", "evaluation.toml"])
    assert json.loads(capsys.readouterr().out) == {"status": "planned"}
