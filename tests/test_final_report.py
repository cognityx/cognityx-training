import json

from cognityx_training.final_report import (
    persist_trial_result,
    render_final_safe_combinations,
    update_final_safe_combinations,
)


def _result(
    trial_number,
    model_name,
    *,
    status="completed",
    sequence_length=2048,
    batch_size=1,
    lora_rank=8,
):
    return {
        "trial_id": f"trial-{trial_number:03d}-{model_name.replace('/', '_')}",
        "status": status,
        "configuration": {
            "model_name": model_name,
            "max_sequence_length": sequence_length,
            "per_device_train_batch_size": batch_size,
            "lora_rank": lora_rank,
        },
        "training_seconds": 685.5214,
        "completed_steps": 2,
        "gpu_power_peak_watts": 414.67,
        "gpu_utilization_peak_percent": 100.0,
        "longest_step_seconds": 370.432,
    }


def test_incremental_report_tracks_latest_completed_trial_per_model(
    tmp_path,
) -> None:
    first = _result(1, "Qwen/Qwen3-8B")
    timed_out = _result(
        2,
        "Qwen/Qwen3-8B",
        status="timeout",
        sequence_length=4096,
        batch_size=4,
    )
    latest = _result(
        3,
        "Qwen/Qwen3-8B",
        sequence_length=4096,
        batch_size=2,
        lora_rank=64,
    )
    other_model = _result(
        4,
        "Qwen/Qwen3-14B",
        sequence_length=4096,
        batch_size=1,
        lora_rank=32,
    )
    for result in (first, timed_out, latest, other_model):
        persist_trial_result(tmp_path, result)
        report = update_final_safe_combinations(tmp_path)

    eight_b = report["models"]["Qwen/Qwen3-8B"]
    assert eight_b["matched_trial"] == latest["trial_id"]
    assert eight_b["safe_combination"] == {
        "max_sequence_length": 4096,
        "per_device_train_batch_size": 2,
        "lora_rank": 64,
    }
    assert report["models"]["Qwen/Qwen3-14B"]["matched_trial"] == (
        other_model["trial_id"]
    )


def test_final_report_writes_json_and_requested_markdown_format(tmp_path) -> None:
    result = _result(
        11,
        "Qwen/Qwen3-8B",
        sequence_length=4096,
        batch_size=2,
        lora_rank=64,
    )
    persist_trial_result(tmp_path, result)

    report = update_final_safe_combinations(tmp_path)
    markdown = render_final_safe_combinations(report)
    stored = json.loads(
        (tmp_path / "final-safe-combinations.json").read_text(encoding="utf-8")
    )

    assert stored == report
    assert "## Qwen/Qwen3-8B" in markdown
    assert "max_sequence_length = 4096" in markdown
    assert "per_device_train_batch_size = 2" in markdown
    assert "lora_rank = 64" in markdown
    assert f"Matched trial: {result['trial_id']}" in markdown
    assert "training_seconds = 685.5214" in markdown
    assert "gpu_power_peak_watts = 414.67" in markdown
    assert "gpu_utilization_peak_percent = 100.0" in markdown
    assert "longest_step_seconds = 370.4320" in markdown
    assert (
        tmp_path / "final-safe-combinations.md"
    ).read_text(encoding="utf-8") == markdown
    assert not list(tmp_path.glob(".*.tmp"))


def test_empty_partial_session_still_has_a_valid_final_report(tmp_path) -> None:
    report = update_final_safe_combinations(tmp_path)
    markdown = render_final_safe_combinations(report)

    assert report["models"] == {}
    assert "No completed safe trials are available." in markdown
