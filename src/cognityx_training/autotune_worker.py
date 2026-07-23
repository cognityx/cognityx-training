"""Persistent autotune child that reuses one loaded base model."""

from __future__ import annotations

import os
from pathlib import Path
import traceback

from cognityx_training.cli import main as training_main


def main() -> None:
    os.environ["COGNITYX_REUSE_LOADED_MODEL"] = "1"
    os.environ["COGNITYX_AUTOTUNE_WORKER"] = "1"
    print("COGNITYX_WORKER_READY", flush=True)
    for value in iter(input, ""):
        config_path = value.strip()
        if not config_path:
            continue
        try:
            training_main(["--config", str(Path(config_path))])
        except Exception:
            traceback.print_exc()
            print("COGNITYX_TRIAL_FAILED", flush=True)
        else:
            print("COGNITYX_TRIAL_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
