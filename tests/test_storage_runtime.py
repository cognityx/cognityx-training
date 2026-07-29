from pathlib import Path

from cognityx_training.storage_runtime import resolve_storage_runtime


def test_injected_runtime_has_highest_precedence(tmp_path: Path) -> None:
    marker = object()
    assert (
        resolve_storage_runtime(
            storage_runtime=marker,
            storage_config=tmp_path / "ignored.toml",
            storage_root=tmp_path / "ignored",
        )
        is marker
    )


def test_explicit_config_uses_runtime_load(monkeypatch, tmp_path: Path) -> None:
    import cognityx_storage

    marker = object()
    calls = []

    def fake_load(*, config_file=None, **kwargs):
        calls.append(config_file)
        return marker

    monkeypatch.setattr(cognityx_storage.StorageRuntime, "load", fake_load)
    config_file = tmp_path / "storage.toml"
    assert (
        resolve_storage_runtime(
            storage_config=config_file,
            storage_root=tmp_path / "ignored",
        )
        is marker
    )
    assert calls == [config_file]


def test_explicit_root_uses_built_in_runtime(monkeypatch, tmp_path: Path) -> None:
    import cognityx_storage

    marker = object()
    captured = []

    def fake_from_config(config, **kwargs):
        captured.append(config)
        return marker

    monkeypatch.setattr(
        cognityx_storage.StorageRuntime,
        "from_config",
        fake_from_config,
    )
    assert resolve_storage_runtime(storage_root=tmp_path) is marker
    assert captured[0].profiles["local-main"].options["root"] == str(tmp_path)


def test_default_uses_runtime_load(monkeypatch) -> None:
    import cognityx_storage

    marker = object()
    calls = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        return marker

    monkeypatch.setattr(cognityx_storage.StorageRuntime, "load", fake_load)
    assert resolve_storage_runtime() is marker
    assert calls == [{}]
