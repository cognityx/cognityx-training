"""Command-line entry point for configuration-driven training."""

import argparse
import tomllib
from pathlib import Path

from cognityx_core import Dataset, TrainingRequest

from cognityx_training.configuration import CustomPyTorchTrainingConfig
from cognityx_training.factory import create_training_backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training configuration path."""
    parser = argparse.ArgumentParser(description="Run a Cognityx training backend.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load TOML configuration and run the selected backend."""
    args = parse_args(argv)
    with args.config.open("rb") as source:
        values = tomllib.load(source)
    training = values.get("training", {})
    dataset_values = values.get("dataset", {})
    config = CustomPyTorchTrainingConfig.from_mapping(training)
    dataset = Dataset(
        name=dataset_values["name"],
        version=str(dataset_values.get("version", "1")),
        uri=dataset_values["uri"],
    )
    result = create_training_backend(config).train(TrainingRequest(dataset=dataset))
    print(f"Artifact: {result.artifact.uri}")
    print(f"Metrics: {dict(result.metrics)}")


if __name__ == "__main__":
    main()
