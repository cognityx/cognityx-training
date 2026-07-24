# Qwen hello-world fine-tuning

Install the training extra and run:

```bash
uv sync --extra training
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml
```

Inspect all command options:

```bash
uv run --extra training cognityx-train --help
```

Validate the configuration and dataset without allocating the model:

```bash
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml \
  --print-config --dry-run
```

Use `--output-dir PATH` or `--run-id NAME` to override those values for one
run. Every execution writes its adapter and `training-report.json` beneath
`<output-dir>/<run-id>/`.

## Understanding terminal progress

This command uses the direct `cognityx-train` path. While model weights are
loading, it displays the Transformers `Loading weights ...` progress row only.
The direct trainer starts its resource monitor after model loading, baseline
evaluation, LoRA adapter setup, dataset tokenization, and optimizer
preparation. Loss and resource progress then appear when optimizer steps
complete.

This differs from `cognityx-autotune`, whose parent controller monitors the
child process and maintains a second `LOAD ...` row during weight loading. Use
the [automatic capacity tuner](autotune.md) when loading-time VRAM, GPU,
storage, CPU, and RAM telemetry is required.

The checked-in configuration performs one optimizer step with batch size one.
It uses NF4 4-bit loading, BF16 computation, double quantization, and LoRA
adapters on Qwen attention projections.

It is the phase-one capacity baseline. `max_examples = 1` selects one record,
`per_device_train_batch_size = 1` processes one record per micro batch, and
`max_steps = 1` performs one optimizer update. See
[Capacity and benefit validation](two-phase-validation.md) for the staged plan.

It explicitly uses `/mnt/d/AI/models/huggingface/hub`, corresponding to
`D:\AI\models\huggingface\hub`, and enables `local_files_only`. This reuses the
existing Qwen snapshot and fails clearly instead of downloading a second copy.
The generated LoRA adapter is saved in a unique run directory under
`/mnt/d/AI/models/cognityx/training/qwen3-14b-hello`.

The report captures the model, dataset and resolved configuration; parameter
counts before and after PEFT wrapping; CPU, main RAM and disk I/O; GPU
utilization and memory; step response-time percentiles; and both the GPU and
serialized-on-disk LoRA adapter sizes.

For a lower-cost smoke test, copy the TOML and change `model_name` to a smaller
Qwen causal language model. The dataset pipeline accepts either a `messages`
array or `instruction`/`output` fields in JSONL.

## Documentation server

```bash
uv run mkdocs serve
```

The training documentation uses <http://127.0.0.1:8125/> and can run beside
the Core documentation on port 8124 and the benchmark documentation server.
