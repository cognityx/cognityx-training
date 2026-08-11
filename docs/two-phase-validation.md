# Capacity and benefit validation

Use two separate runs. The first answers “what can this computer sustain?” The
second answers “did fine-tuning teach a behavior the base model did not have?”

## Training-size controls

The TOML `[training]` section controls the workload:

| Setting | Meaning |
| --- | --- |
| `max_examples` | Maximum training records selected after held-out evaluation records are removed. |
| `per_device_train_batch_size` | Examples processed in each forward/backward pass. |
| `gradient_accumulation_steps` | Micro batches accumulated before one optimizer update. |
| `max_steps` | Number of optimizer updates. The data loader repeats as necessary. |
| `max_sequence_length` | Maximum tokens per selected example. This strongly affects GPU memory. |
| `progress_interval_steps` | Optimizer-step interval for live progress/resource output. |

For this single-process backend:

```text
effective batch size = per_device_train_batch_size × gradient_accumulation_steps
```

The dry run shows total, selected-training, and held-out-evaluation record
counts without allocating the model:

```bash
uv run cognityx-train \
  --config examples/qwen3-14b-hello/config.toml \
  --print-config --dry-run
```

The dry run proves that the configuration and dataset are usable. It does not
prove that optional execution libraries were installed. Check that boundary
separately before either validation phase:

```bash
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml \
  --check-runtime --output-format json
```

The JSON result lists the Training-owned execution packages, their versions,
and CUDA visibility. A missing package such as PEFT (the library that attaches
the small trainable adapter) or a missing CUDA device for four-bit training
returns a failed result and a non-zero process status, without loading a model.

## Phase 1: establish the computer limit

Start with the one-example, one-step QLoRA configuration:

```bash
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml \
  --run-id capacity-baseline
```

During training, each configured progress interval prints:

```text
step 1/1 | loss ... | CPU ... | RAM ... GiB | disk R/W ... GiB | GPU ...% ... GiB
```

Review `training-report.json`, especially source-labeled whole-device and
framework GPU memory, Windows-host or WSL-VM RAM, power/energy, runtime, step
latency, and disk reads/writes. Then copy the configuration and
change only one dimension at a time in this order:

1. Increase `max_sequence_length` until it matches the intended data.
2. Increase `per_device_train_batch_size` until memory is close to the safe
   limit, then return to the previous value.
3. Increase `gradient_accumulation_steps` to obtain the desired effective batch
   without increasing peak GPU memory as much.
4. Increase `max_examples` and `max_steps`; these primarily increase runtime.

Keep roughly 10–15% GPU memory headroom. An out-of-memory attempt is not a
useful final configuration; the last stable run is the capacity result.

## Phase 2: measure training benefit

The checked-in benefit dataset teaches a deliberately fictional private fact:

```text
Aster-7 calibration code → VIOLET-ORBIT-314
```

The base model cannot have learned this repository-specific association during
pretraining. Eight paraphrases are training records and three different
paraphrases are marked `"split": "evaluation"`; evaluation rows never enter
the optimizer data loader.

Run:

```bash
uv run --extra training cognityx-train \
  --config examples/private-fact-benefit/config.toml \
  --run-id private-fact-benefit
```

Before LoRA is attached, the trainer asks the base model all held-out prompts.
After training, it asks the same prompts again with deterministic generation.
The JSON report preserves every expected/generated answer and compares:

- normalized exact-match accuracy;
- expected-answer containment accuracy;
- baseline-to-trained change for both scores.

Success means the baseline does not produce `VIOLET-ORBIT-314`, while the
trained adapter does on held-out paraphrases. If capacity is stable but the
benefit remains zero, increase `max_steps` gradually and inspect training loss;
do not change the held-out evaluation rows into training examples.
