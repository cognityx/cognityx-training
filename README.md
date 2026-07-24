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

Trained adapters are also written under `D:\AI\models`, using:

```text
/mnt/d/AI/models/cognityx/training/
```

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

Show CLI help or validate a run without loading the model:

```bash
uv run --extra training cognityx-train --help
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml --print-config --dry-run
```

Each execution stores the adapter and `training-report.json` in a unique
`<output-dir>/<run-id>/` directory. The report includes model, dataset,
configuration, parameter counts, CPU, RAM, disk I/O, GPU usage, step latency,
and LoRA GPU/disk size measurements.

Training workload is controlled independently with `max_examples`,
`per_device_train_batch_size`, `gradient_accumulation_steps`, `max_steps`, and
`max_sequence_length`. Live progress includes loss, CPU, main RAM, disk I/O,
and GPU utilization/memory.

### Direct training versus autotune output

| Command | Weight-loading display | When resource telemetry begins |
| --- | --- | --- |
| `cognityx-train` | Transformers `Loading weights ...` progress row | After model loading, baseline evaluation, adapter setup, and optimizer preparation; progress is then printed at completed optimizer steps. |
| `cognityx-autotune` | Transformers progress plus a second controller-managed `LOAD ...` telemetry row | During model loading and throughout the controlled trial. |

Therefore, seeing only the Transformers progress row while a direct
`cognityx-train` run loads model weights is expected. Use
`cognityx-autotune` when live VRAM, GPU, storage, CPU, and RAM measurements are
needed during loading.

For an evidence-based workflow, first run the one-step hello-world capacity
baseline, then run the private-fact benefit experiment:

```bash
uv run --extra training cognityx-train \
  --config examples/private-fact-benefit/config.toml \
  --run-id private-fact-benefit
```

The benefit experiment compares held-out answers from the untrained base model
and trained adapter. See [the two-phase validation guide](docs/two-phase-validation.md).

## Automatic capacity search

Inventory the computer and preview the search without training:

```bash
uv run --extra training cognityx-autotune \
  --config examples/autotune-5090/config.toml --plan
```

Run isolated trials across configured axes until sustained resource policies,
trial failure/OOM, `max_trials`, or the candidate space ends:

```bash
uv run --extra training cognityx-autotune \
  --config examples/autotune-5090/config.toml
```

Each invocation stores configs, streamed logs, per-trial training reports, the
hardware/software inventory, and a final `autotune-summary.json` in a unique
session directory. See [automatic capacity tuning](docs/autotune.md).

Telemetry is source- and scope-labeled so Windows-host RAM is not confused with
WSL-VM RAM, or PyTorch allocator VRAM with whole-device `nvidia-smi` VRAM.
Reports include GPU utilization, power, sampled energy, and temperature. Trial
axes and candidates support economical staged searches or capped Cartesian
grids for conditional model/sequence/batch/rank capacity comparisons.

The default QLoRA path requires a CUDA GPU, enough memory for the chosen model,
Hugging Face access, and a working BitsAndBytes installation.

## Development

```bash
uv sync --dev
uv run pytest
uv run mkdocs build --strict
uv run mkdocs serve
```

The documentation server uses `http://127.0.0.1:8125/` to avoid the benchmark
documentation server's port.

The documentation preview is served at <http://127.0.0.1:8125/>.
