# cognityx-training

Training backends and dataset-to-instruction-fine-tuning pipelines for Cognityx.

The first vertical slice provides `CustomPyTorchTrainerBackend`, a one-step
PyTorch `DataLoader`/`AdamW` loop using Transformers for Qwen loading and PEFT
for LoRA or QLoRA adapters. The default model is `Qwen/Qwen3-14B`; override
`model_name` for smaller smoke tests.

The checked-in example loads models only from the existing Windows cache:

```text
D:\AI\models\huggingface\hub
WSL: /mnt/d/AI/models/huggingface/hub
```

`local_files_only = true` prevents an accidental duplicate download. Change it
explicitly only when you intend to add a model to that cache.

## Setup

```bash
uv sync --extra training
```

For Hugging Face tools outside this application, select the same cache root:

```bash
export HF_HOME=/mnt/d/AI/models/huggingface
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
