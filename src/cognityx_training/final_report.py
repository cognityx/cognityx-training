"""Incremental final safe-combination reporting for autotune sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


def _atomic_write(path: Path, content: str) -> None:
    """Replace a report atomically so interruption cannot leave partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def persist_trial_result(session_dir: Path, result: dict[str, Any]) -> Path:
    """Persist one controller-level result immediately after its trial."""
    trial_id = str(result["trial_id"])
    path = session_dir / "trial-results" / f"{trial_id}.json"
    _atomic_write(
        path,
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
    )
    return path


def _trial_number(trial_id: str) -> int:
    match = re.match(r"trial-(\d+)", trial_id)
    return int(match.group(1)) if match else -1


def load_completed_trial_results(session_dir: Path) -> list[dict[str, Any]]:
    """Load valid completed controller results in trial order."""
    results = []
    for path in (session_dir / "trial-results").glob("trial-*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if result.get("status") != "completed":
            continue
        configuration = result.get("configuration")
        if not isinstance(configuration, dict) or not configuration.get("model_name"):
            continue
        results.append(result)
    return sorted(
        results, key=lambda item: _trial_number(str(item.get("trial_id", "")))
    )


def build_final_safe_combinations(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the latest successful combined configuration for every model."""
    latest_by_model: dict[str, dict[str, Any]] = {}
    for result in results:
        configuration = result["configuration"]
        model_name = str(configuration["model_name"])
        latest_by_model[model_name] = {
            "model_name": model_name,
            "safe_combination": {
                "max_sequence_length": configuration.get("max_sequence_length"),
                "per_device_train_batch_size": configuration.get(
                    "per_device_train_batch_size"
                ),
                "lora_rank": configuration.get("lora_rank"),
            },
            "matched_trial": result["trial_id"],
            "training_metrics": {
                "training_seconds": result.get("training_seconds"),
                "completed_steps": result.get("completed_steps"),
            },
            "gpu_metrics": {
                "gpu_power_peak_watts": result.get("gpu_power_peak_watts"),
                "gpu_utilization_peak_percent": result.get(
                    "gpu_utilization_peak_percent"
                ),
                "longest_step_seconds": result.get("longest_step_seconds"),
            },
        }
    return {
        "schema_version": "1.0",
        "source": "incremental_controller_trial_results",
        "models": latest_by_model,
    }


def _format_metric(value: Any, decimal_places: int = 4) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{decimal_places}f}"
    return str(value)


def render_final_safe_combinations(report: dict[str, Any]) -> str:
    """Render the concise human-readable final report."""
    lines = ["# Final safe combinations from autotune-summary.json", ""]
    models = report.get("models", {})
    if not models:
        lines.extend(["No completed safe trials are available.", ""])
        return "\n".join(lines)
    for model_name, model in models.items():
        combination = model["safe_combination"]
        training = model["training_metrics"]
        gpu = model["gpu_metrics"]
        lines.extend(
            [
                f"## {model_name}",
                "",
                "Safe combination:",
                f"max_sequence_length = {combination['max_sequence_length']}",
                (
                    "per_device_train_batch_size = "
                    f"{combination['per_device_train_batch_size']}"
                ),
                f"lora_rank = {combination['lora_rank']}",
                f"Matched trial: {model['matched_trial']}",
                "",
                "Training metrics:",
                (
                    "training_seconds = "
                    f"{_format_metric(training['training_seconds'])}"
                ),
                f"completed_steps = {_format_metric(training['completed_steps'])}",
                "",
                "GPU metrics:",
                (
                    "gpu_power_peak_watts = "
                    f"{_format_metric(gpu['gpu_power_peak_watts'], 2)}"
                ),
                (
                    "gpu_utilization_peak_percent = "
                    f"{_format_metric(gpu['gpu_utilization_peak_percent'], 1)}"
                ),
                (
                    "longest_step_seconds = "
                    f"{_format_metric(gpu['longest_step_seconds'])}"
                ),
                "",
            ]
        )
    return "\n".join(lines)


def update_final_safe_combinations(session_dir: Path) -> dict[str, Any]:
    """Regenerate JSON and Markdown reports from durable completed trials."""
    report = build_final_safe_combinations(
        load_completed_trial_results(session_dir)
    )
    _atomic_write(
        session_dir / "final-safe-combinations.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        session_dir / "final-safe-combinations.md",
        render_final_safe_combinations(report),
    )
    return report
