# cognityx-training

Training backends and dataset-to-instruction-fine-tuning pipelines for Cognityx.

The first vertical slice provides `CustomPyTorchTrainerBackend`, a one-step
Transformers/PEFT LoRA or QLoRA supervised fine-tuning run. The default model is
`Qwen/Qwen3-14B`; override `model_name` for smaller smoke tests.

## Setup

```bash
uv sync --extra training
```

`cognityx-core` is resolved as a normal private Git dependency. No Git
submodules are used.

## Run the hello-world QLoRA example

```bash
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml
```

The default QLoRA path requires a CUDA GPU, enough memory for the chosen model,
Hugging Face access, and a working BitsAndBytes installation.

## Development

```bash
uv sync --dev
uv run pytest
uv run mkdocs build --strict
uv run mkdocs serve
```
