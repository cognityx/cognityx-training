from pathlib import Path
from io import StringIO
import os
import sys
import time

from cognityx_training.autotune import (
    AutotuneConfig,
    _capacity_frontier,
    _peak_percentages,
    _policy_breaches,
    parse_args,
    load_autotune_config,
    _PERSISTENT_WORKERS,
    _persistent_worker,
    _rate,
    _TwoLineLoadingDisplay,
    _RawChildOutputParser,
    _spawn_with_pty,
    _model_storage_info,
    _LoadingProcessSampler,
    _SingleLineStatusDisplay,
    run_autotune,
    _format_loading_status,
    _is_immediate_timeout_key,
)
import cognityx_training.autotune as autotune_module


def test_autotune_cli_supports_plan_mode() -> None:
    args = parse_args(["--config", "autotune.toml", "--plan"])

    assert args.config == Path("autotune.toml")
    assert args.plan is True


def test_autotune_threshold_measurements_use_peak_percentages() -> None:
    gpu, ram = _peak_percentages(
        {
            "gpu_usage": [
                {"memory_peak_percent": 75.0},
                {"memory_peak_percent": 82.5},
            ],
            "system_usage": {"system_ram_peak_percent": 64.0},
        }
    )

    assert gpu == 82.5
    assert ram == 64.0


def test_autotune_rejects_invalid_thresholds() -> None:
    config = AutotuneConfig(
        base_training_config=Path("base.toml"),
        output_dir=Path("outputs"),
        gpu_memory_limit_percent=0,
    )

    try:
        config.validate()
    except ValueError as exc:
        assert "gpu_memory_limit_percent" in str(exc)
    else:
        raise AssertionError("invalid threshold was accepted")


def test_model_reuse_requires_clean_restart_after_oom() -> None:
    config = AutotuneConfig(
        base_training_config=Path("base.toml"),
        output_dir=Path("outputs"),
        reuse_loaded_model=True,
        restart_worker_after_oom=False,
        candidates={"model_name": ("model-a",)},
        axes=("model_name",),
    )

    try:
        config.validate()
    except ValueError as exc:
        assert "restart_worker_after_oom" in str(exc)
    else:
        raise AssertionError("unsafe CUDA worker reuse was accepted")


def test_policy_uses_configured_capacity_limits_not_cpu_or_gpu_utilization() -> None:
    config = AutotuneConfig(
        base_training_config=Path("base.toml"),
        output_dir=Path("outputs"),
        gpu_memory_limit_percent=98,
        ram_limit_percent=95,
        gpu_power_limit_watts=500,
        candidates={"model_name": ("model-a",)},
        axes=("model_name",),
    )
    resources = {
        "host": {"used_percent": 65, "cpu_percent": 100, "scope": "windows_host"},
        "gpus": [{
            "memory_used_percent": 50,
            "utilization_percent": 100,
            "temperature_celsius": 70,
            "power_draw_watts": 510,
            "power_percent_of_limit": 90,
        }],
    }

    breaches = _policy_breaches(config, resources, elapsed_seconds=1)

    assert [breach[0] for breach in breaches] == ["gpu_power_watts"]


def test_loading_and_training_timeouts_are_independent() -> None:
    config = AutotuneConfig(
        base_training_config=Path("base.toml"),
        output_dir=Path("outputs"),
        model_loading_timeout_seconds=100,
        trial_timeout_seconds=10,
        gpu_memory_limit_percent=None,
        ram_limit_percent=None,
        gpu_temperature_limit_celsius=None,
        candidates={"model_name": ("model-a",)},
        axes=("model_name",),
    )
    resources = {
        "host": {"used_percent": 50, "scope": "windows_host"},
        "gpus": [],
    }

    assert _policy_breaches(config, resources, 20, "model_loading") == []
    assert _policy_breaches(config, resources, 20, "training")[0][0] == "trial_timeout_seconds"
    assert _policy_breaches(config, resources, 100, "model_loading")[0][0] == "model_loading_timeout_seconds"


def test_no_step_progress_timeout_is_optional_and_independent() -> None:
    config = AutotuneConfig(
        base_training_config=Path("base.toml"),
        output_dir=Path("outputs"),
        trial_timeout_seconds=1000,
        no_step_progress_timeout_seconds=30,
        gpu_memory_limit_percent=None,
        ram_limit_percent=None,
        gpu_temperature_limit_celsius=None,
        candidates={"model_name": ("model-a",)},
        axes=("model_name",),
    )
    resources = {
        "host": {"used_percent": 50, "scope": "wsl_vm"},
        "gpus": [],
    }

    breaches = _policy_breaches(config, resources, 40, "training", 31)

    assert [breach[0] for breach in breaches] == [
        "no_step_progress_timeout_seconds"
    ]
    assert _is_immediate_timeout_key("no_step_progress_timeout_seconds")
    assert not _is_immediate_timeout_key("gpu_memory_percent")


def test_capacity_frontier_reports_maximum_success_per_model() -> None:
    trials = [
        {"status": "completed", "configuration": {"model_name": "8B", "per_device_train_batch_size": 8}},
        {"status": "completed", "configuration": {"model_name": "14B", "per_device_train_batch_size": 2}},
        {"status": "threshold_reached", "configuration": {"model_name": "14B", "per_device_train_batch_size": 4}},
    ]

    frontier = _capacity_frontier(
        trials,
        ("model_name", "per_device_train_batch_size"),
        {"model_name": ("8B", "14B"), "per_device_train_batch_size": (1, 2, 4, 8)},
    )

    assert frontier["by_model"]["8B"]["per_device_train_batch_size"]["value"] == 8
    assert frontier["by_model"]["14B"]["per_device_train_batch_size"]["value"] == 2


def test_example_enables_bounded_model_reuse() -> None:
    config, _values = load_autotune_config(
        Path("examples/autotune-5090/config.toml")
    )

    assert config.reuse_loaded_model is True
    assert config.restart_worker_after_oom is True
    assert config.max_trials_per_worker == 25
    assert config.manage_windows_bridge is True
    assert config.windows_bridge_interval_seconds == 1
    assert config.windows_bridge_startup_timeout_seconds == 20


def test_run_autotune_stops_managed_telemetry_after_failure(
    tmp_path, monkeypatch
) -> None:
    controller = AutotuneConfig(
        base_training_config=tmp_path / "base.toml",
        output_dir=tmp_path / "outputs",
        manage_windows_bridge=True,
        candidates={"model_name": ("model-a",)},
        axes=("model_name",),
    )
    events = []

    class FakeProducer:
        def __init__(self, **kwargs) -> None:
            events.append(("created", kwargs["output_path"]))

        def start(self) -> None:
            events.append(("started", None))

        def stop(self) -> None:
            events.append(("stopped", None))

    monkeypatch.setattr(
        autotune_module,
        "load_autotune_config",
        lambda path: (controller, {"training": {}}),
    )
    monkeypatch.setattr(
        autotune_module, "WindowsTelemetryProducer", FakeProducer
    )
    monkeypatch.setattr(
        autotune_module,
        "_run_autotune",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trial failed")),
    )

    try:
        run_autotune(tmp_path / "autotune.toml")
    except RuntimeError as exc:
        assert str(exc) == "trial failed"
    else:
        raise AssertionError("autotune failure was swallowed")

    assert [event[0] for event in events] == ["created", "started", "stopped"]
    assert "sessions" in events[0][1].parts
    assert events[0][1].name == "windows-host.json"


def test_plan_mode_does_not_start_managed_windows_telemetry(
    tmp_path, monkeypatch
) -> None:
    controller = AutotuneConfig(
        base_training_config=tmp_path / "base.toml",
        output_dir=tmp_path / "outputs",
        manage_windows_bridge=True,
        candidates={"model_name": ("model-a",)},
        axes=("model_name",),
    )
    monkeypatch.setattr(
        autotune_module,
        "load_autotune_config",
        lambda path: (controller, {"training": {}}),
    )
    monkeypatch.setattr(
        autotune_module,
        "WindowsTelemetryProducer",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("producer started in plan mode")
        ),
    )
    monkeypatch.setattr(
        autotune_module,
        "_run_autotune",
        lambda received, values, plan_only: {"plan_only": plan_only},
    )

    result = run_autotune(tmp_path / "autotune.toml", plan_only=True)

    assert result == {"plan_only": True}


def test_switching_models_discards_the_previous_worker(monkeypatch) -> None:
    class FakeProcess:
        def poll(self):
            return None

    old = FakeProcess()
    new = FakeProcess()
    _PERSISTENT_WORKERS.clear()
    _PERSISTENT_WORKERS["8B"] = (old, 1)
    monkeypatch.setattr("cognityx_training.autotune._stop_process_group", lambda process: None)
    monkeypatch.setattr("cognityx_training.autotune.subprocess.Popen", lambda *args, **kwargs: new)
    config = AutotuneConfig(
        base_training_config=Path("base.toml"),
        output_dir=Path("outputs"),
        candidates={"model_name": ("14B",)},
        axes=("model_name",),
    )

    assert _persistent_worker("14B", config) is new
    assert "8B" not in _PERSISTENT_WORKERS
    _PERSISTENT_WORKERS.clear()


def test_loading_rate_does_not_report_negative_or_divide_by_zero() -> None:
    assert _rate(300, 100, 2) == 100
    assert _rate(100, 300, 2) == 0
    assert _rate(200, 100, 0) > 0


def test_loading_status_prioritizes_vram_and_gpu_health() -> None:
    status = _format_loading_status(
        19,
        "VRAM 7.7G +0M/s(est) | GPU 17% 45W 52C",
        "WinGPU shared unavailable",
        "modelFS /mnt/d 9p | read 0M/s",
        "procCPU 31% wsl_vmCPU 6%",
        "RAM(wsl_vm) 4.5/62.7G",
    )

    assert status.startswith(
        "LOAD     19s | VRAM 7.7G +0M/s(est) | GPU 17% 45W 52C"
    )
    assert status.index("VRAM") < status.index("WinGPU")
    assert status.index("GPU 17%") < status.index("modelFS")
    assert status.index("modelFS") < status.index("procCPU")


def test_two_line_loading_display_updates_in_place_on_tty() -> None:
    class TtyBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    stream = TtyBuffer()
    display = _TwoLineLoadingDisplay(stream)

    display.update_progress("Loading weights: 10%")
    display.update_stats("LOAD 10s | GPU 4 GiB")
    display.update_progress("Loading weights: 20%")
    display.finish()

    output = stream.getvalue()
    assert "Loading weights: 10%\n" in output
    assert "LOAD 10s | GPU 4 GiB" in output
    assert "\x1b[1A" in output
    assert output.endswith("\n")


def test_training_heartbeat_updates_one_row_without_scrolling() -> None:
    class TtyBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    stream = TtyBuffer()
    display = _SingleLineStatusDisplay(stream)

    display.update("TRAIN 10s | step 0/2 running")
    display.update("TRAIN 20s | step 0/2 running")
    display.update("TRAIN 30s | step 0/2 running")
    display.finish()

    output = stream.getvalue()
    assert output.count("\x1b[2K") == 3
    assert "TRAIN 30s | step 0/2 running" in output
    assert output.count("\n") == 1


def test_raw_loading_progress_preserves_carriage_returns_and_latest_value() -> None:
    parser = _RawChildOutputParser()
    raw = (
        b"Loading weights: 42%|166/399\r"
        b"Loading weights: 42%|166/399\r"
        b"Loading weights: 42%|167/399\r"
        b"Loading weights: 42%|167/399\r"
        b"Loading weights: 42%|169/399\r"
    )

    records = parser.feed(raw)

    assert records == [
        "Loading weights: 42%|166/399",
        "Loading weights: 42%|166/399",
        "Loading weights: 42%|167/399",
        "Loading weights: 42%|167/399",
        "Loading weights: 42%|169/399",
    ]

    class TtyBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    stream = TtyBuffer()
    display = _TwoLineLoadingDisplay(stream)
    for record in records:
        display.update_progress(record)
    display.update_stats("LOAD latest performance")

    assert display.progress.endswith("169/399")
    assert display.stats == "LOAD latest performance"
    # Three unique progress states plus one telemetry redraw.
    assert stream.getvalue().count("\x1b[1A") == 3


def test_pty_child_sees_terminal_and_preserves_progress_updates() -> None:
    process, output_fd = _spawn_with_pty(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(f'TTY={sys.stdout.isatty()}', flush=True); "
                "sys.stdout.write('Loading weights: 10%\\r'); sys.stdout.flush(); "
                "sys.stdout.write('Loading weights: 20%\\r'); sys.stdout.flush()"
            ),
        ],
        dict(os.environ),
        persistent=False,
    )
    chunks = bytearray()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            chunk = os.read(output_fd, 4096)
        except BlockingIOError:
            if process.poll() is None:
                time.sleep(0.01)
                continue
            break
        except OSError:
            break
        if chunk:
            chunks.extend(chunk)
        elif process.poll() is not None:
            break
    process.wait(timeout=5)
    os.close(output_fd)

    parser = _RawChildOutputParser()
    records = parser.feed(bytes(chunks), final=True)
    assert "TTY=True" in records
    assert "Loading weights: 10%" in records
    assert "Loading weights: 20%" in records


def test_model_storage_uses_longest_mount_containing_cache() -> None:
    mountinfo = (
        "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
        "2 1 0:99 / /mnt/d rw - 9p D:\\ rw\n"
    )

    storage = _model_storage_info(
        Path("/mnt/d/AI/models/huggingface"), mountinfo
    )

    assert storage["mount_point"] == "/mnt/d"
    assert storage["filesystem_type"] == "9p"
    assert storage["mount_source"] == "D:\\"
    assert storage["device_latency_available"] is False


def test_loading_cpu_sampler_retains_process_interval_state() -> None:
    class Io:
        read_bytes = 10
        read_chars = 20
        read_count = 3

    class Memory:
        rss = 100

    class Process:
        def __init__(self):
            self.values = iter((0.0, 375.0))

        def cpu_percent(self, interval=None):
            return next(self.values)

        def io_counters(self):
            return Io()

        def memory_info(self):
            return Memory()

    sampler = _LoadingProcessSampler(123, Process())

    sample = sampler.sample()

    assert sample is not None
    assert sample["process_cpu_percent"] == 375.0


def _write_scheduler_config(tmp_path: Path, strategy: str, axes: list[str]) -> Path:
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    base_path = tmp_path / "base.toml"
    base_path.write_text(
        f"""
[training]
model_name = "base-model"
model_cache_dir = "{model_cache}"
output_dir = "{tmp_path / 'training'}"
max_sequence_length = 128
max_steps = 1
per_device_train_batch_size = 1
lora_rank = 8
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "autotune.toml"
    quoted_axes = ", ".join(f'"{axis}"' for axis in axes)
    config_path.write_text(
        f"""
[autotune]
base_training_config = "base.toml"
output_dir = "{tmp_path / 'outputs'}"
trial_max_steps = 1
cooldown_seconds = 0

[telemetry]
host_source = "wsl"

[search]
strategy = "{strategy}"
axes = [{quoted_axes}]
max_trials = 100

[trials]
model_name = ["model-8B", "model-14B"]
max_sequence_length = [128, 256]
per_device_train_batch_size = [1, 2]
""",
        encoding="utf-8",
    )
    return config_path


def _mock_scheduler_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        autotune_module,
        "hardware_software_inventory",
        lambda *args, **kwargs: {"ram": {"scope": "wsl_vm"}},
    )
    monkeypatch.setattr(autotune_module, "_print_inventory", lambda inventory: None)
    monkeypatch.setattr(autotune_module, "_shutdown_workers", lambda: None)
    seen: dict[str, int] = {}

    def fake_trial(trial_id, stage, config, controller, dataset_path):
        count = seen.get(config.model_name, 0)
        seen[config.model_name] = count + 1
        return {
            "trial_id": trial_id,
            "stage": stage,
            "status": "completed",
            "runtime_seconds": 1.0,
            "model_weights_reused": count > 0,
            "configuration": {
                "model_name": config.model_name,
                "max_sequence_length": config.max_sequence_length,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "lora_rank": config.lora_rank,
            },
            "gpu_memory_peak_percent": 50.0,
            "system_ram_peak_percent": 20.0,
        }

    monkeypatch.setattr(autotune_module, "_run_trial", fake_trial)


def test_staged_search_completes_all_axes_before_switching_model(
    tmp_path, monkeypatch
) -> None:
    config_path = _write_scheduler_config(
        tmp_path,
        "staged",
        ["model_name", "max_sequence_length", "per_device_train_batch_size"],
    )
    _mock_scheduler_runtime(monkeypatch)

    summary = run_autotune(config_path)

    models = [trial["configuration"]["model_name"] for trial in summary["trials"]]
    assert models == ["model-8B"] * 3 + ["model-14B"] * 3
    assert summary["model_execution"]["model-8B"]["model_load_count"] == 1
    assert summary["model_execution"]["model-8B"]["reused_trial_count"] == 2


def test_grid_search_is_model_major_even_when_model_axis_is_not_first(
    tmp_path, monkeypatch
) -> None:
    config_path = _write_scheduler_config(
        tmp_path,
        "grid",
        ["per_device_train_batch_size", "model_name", "max_sequence_length"],
    )
    _mock_scheduler_runtime(monkeypatch)

    summary = run_autotune(config_path)

    models = [trial["configuration"]["model_name"] for trial in summary["trials"]]
    assert models == ["model-8B"] * 4 + ["model-14B"] * 4


def test_staged_step_timeout_prunes_higher_axis_and_uses_last_safe_for_next_axis(
    tmp_path, monkeypatch
) -> None:
    config_path = _write_scheduler_config(
        tmp_path,
        "staged",
        [
            "model_name",
            "max_sequence_length",
            "per_device_train_batch_size",
            "lora_rank",
        ],
    )
    text = config_path.read_text(encoding="utf-8").replace(
        "per_device_train_batch_size = [1, 2]",
        "per_device_train_batch_size = [1, 2, 4, 8]",
    )
    config_path.write_text(text + "\nlora_rank = [8, 16]\n", encoding="utf-8")
    _mock_scheduler_runtime(monkeypatch)
    seen: dict[str, int] = {}

    def timed_trial(trial_id, stage, config, controller, dataset_path):
        count = seen.get(config.model_name, 0)
        seen[config.model_name] = count + 1
        timed_out = (
            stage == "per_device_train_batch_size"
            and config.per_device_train_batch_size == 2
        )
        return {
            "trial_id": trial_id,
            "stage": stage,
            "status": "timeout" if timed_out else "completed",
            "timeout_type": "no_step_progress" if timed_out else None,
            "runtime_seconds": 1.0,
            "model_weights_reused": count > 0,
            "configuration": {
                "model_name": config.model_name,
                "max_sequence_length": config.max_sequence_length,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "lora_rank": config.lora_rank,
            },
            "gpu_memory_peak_percent": 50.0,
            "system_ram_peak_percent": 20.0,
        }

    monkeypatch.setattr(autotune_module, "_run_trial", timed_trial)

    summary = run_autotune(config_path)

    model_trials = [
        trial
        for trial in summary["trials"]
        if trial["configuration"]["model_name"] == "model-8B"
    ]
    batch_trials = [
        trial
        for trial in model_trials
        if trial["stage"] == "per_device_train_batch_size"
    ]
    rank_trials = [trial for trial in model_trials if trial["stage"] == "lora_rank"]
    assert [trial["configuration"]["per_device_train_batch_size"] for trial in batch_trials] == [2]
    assert rank_trials[-1]["configuration"]["per_device_train_batch_size"] == 1
    assert {
        item["candidate"]
        for item in summary["pruned_candidates"]
        if item["model_name"] == "model-8B"
        and item["axis"] == "per_device_train_batch_size"
    } == {4, 8}
