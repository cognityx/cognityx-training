# Automatic hardware-capacity tuning

`cognityx-autotune` measures the current machine and runs isolated training
trials until it reaches a safety threshold or a trial fails. Each child process
releases its model and CUDA state before the next configuration begins, so an
out-of-memory result does not terminate the controller.

## Inspect the plan without training

```bash
uv run --extra training cognityx-autotune \
  --config examples/autotune-5090/config.toml \
  --plan
```

The inventory includes timestamps, source, and scope. This matters under WSL:

- `windows_host` RAM/CPU comes from Windows PowerShell/CIM and is comparable to
  Windows Task Manager. Its denominator is `Win32_ComputerSystem`
  `TotalPhysicalMemory`, meaning installed motherboard RAM;
- `wsl_vm` RAM/CPU comes from `psutil` inside the Linux VM and is not the same
  denominator as physical Windows RAM;
- `whole_device` GPU measurements come from `nvidia-smi`;
- `framework_process` GPU allocation comes from PyTorch.

Task Manager and a JSON report are comparable only when their timestamps and
scopes match. GPU memory can also fall immediately after a child process exits.

The RTX 5090 example uses `host_source = "auto"` plus
`installed_memory_gib = 128`. When Windows interop works, terminal progress can
show `windows_host RAM 65.0/128.0 GiB`. In this workspace Windows executable
interop may be unavailable temporarily; in that case live usage is explicitly
labeled `wsl_vm ... (wsl_memory_limit)` and motherboard capacity is reported
separately as configured 128 GiB. The program does not fabricate a
Windows-used value.

### Windows shared GPU memory bridge

Windows Task Manager's combined GPU memory is dedicated VRAM plus host-backed
shared GPU memory. `nvidia-smi` exposes dedicated VRAM but not the Windows shared
counter through this WSL setup. The RTX 5090 configuration enables a managed
Windows telemetry producer:

```toml
[telemetry]
manage_windows_bridge = true
windows_bridge_interval_seconds = 1
windows_bridge_startup_timeout_seconds = 20
windows_bridge_max_age_seconds = 5
```

`cognityx-autotune` launches PowerShell itself, waits for a fresh sample before
loading any model, and writes telemetry into the current session directory.
The controller retains the child-process handle and stops only that producer
when autotune completes, fails, times out, or is interrupted. `--plan` does not
start the producer.

For compatibility with an externally managed collector, set
`manage_windows_bridge = false` and configure `windows_bridge_path`. Samples
older than `windows_bridge_max_age_seconds` are rejected. Loading and training
rows prioritize:

```text
WinGPU D31.1/31.5G S45.5/64.0G C76.6/95.5G S+820M/s
```

`D`, `S`, and `C` mean dedicated, shared, and combined GPU memory. `S+` is
positive shared-memory growth, not exact PCIe bandwidth. When the collector is
absent or stale, the row says `WinGPU shared unavailable`; it never substitutes
a fabricated zero. The bridge is read-only and no shared-memory termination
policy is currently implemented.

The inventory includes:

- CPU identity and physical/logical cores;
- total, available, and currently used main RAM;
- output-disk capacity and free space;
- GPU name, total VRAM, and compute capability;
- NVIDIA whole-device VRAM, utilization, power draw/limit, temperature, and
  driver telemetry when `nvidia-smi` is available;
- Python, PyTorch, CUDA runtime, Transformers, PEFT, BitsAndBytes, and
  Accelerate versions.

## Run the automatic search

```bash
uv run --extra training cognityx-autotune \
  --config examples/autotune-5090/config.toml
```

### Reuse loaded model weights

The checked-in example keeps an identical base model resident across trials:

```toml
[execution]
reuse_loaded_model = true
restart_worker_after_oom = true
max_trials_per_worker = 25
```

The controller keeps one active persistent worker for the current model key.
When the search changes model, it stops that worker before loading the next one,
so multiple base models are never intentionally resident in VRAM. Tokenizer loading,
weight deserialization, 4-bit quantization, and GPU placement happen on the
first trial; later sequence-length, batch-size, and LoRA-rank trials reuse that
prepared base. Every trial receives a newly created LoRA adapter and optimizer,
and the trained adapter is unloaded before the base is cached again. Thus
capacity trials do not inherit adapter training from an earlier trial.

Workers are never shared across model names or quantization modes. A resource
termination, failed trial, or CUDA OOM discards the worker so the next trial
starts with clean CUDA state. `max_trials_per_worker` also bounds reuse to limit
long-lived allocator fragmentation. Set `reuse_loaded_model = false` for the
original fully isolated process-per-trial behavior.

Consequently, a same-model trial immediately following `threshold_reached`
will show a fresh weight load. The controller prints `WORKER RESTART REQUIRED`
with the reason before doing so. This reload is deliberate: the threshold can
interrupt forward/backward execution, and that CUDA/model state is not safe to
reuse. `restart_worker_after_oom` must remain true when reuse is enabled.

Each trial result records `model_weights_reused`. Its captured loading duration
still includes per-trial adapter creation, tokenization, and setup even when the
base weights were reused.

Transformers retains its normal weight-loading progress bar on the first
dynamic terminal row. Cognityx displays `LOAD ...` telemetry on the row directly
below it. ANSI cursor movement redraws both rows in place, so changing values do
not scroll the terminal. The controller reads raw child bytes so Transformers'
carriage returns remain update signals instead of being converted into newline
records; identical shard updates are deduplicated. Workers receive PTY-backed
output with the same width as the parent terminal, and telemetry is compacted
and clipped to prevent row wrapping. Both rows are finalized when loading ends.
`COGNITYX_TRAINING_COMPLETED (optimizer)` means optimizer steps have
finished; adapter saving and evaluation follow before the worker emits its
trial-completion marker and the controller starts the next configuration.

While a fresh model is loading, the telemetry row shows:

- the configured model-cache path's mount point and filesystem type;
- physical loader-process reads and read operations per second;
- persistent loader-process CPU and source-labeled host/WSL CPU;
- loader-process RSS and source-labeled Windows-host or WSL-VM RAM usage;
- whole-device VRAM and its positive growth rate;
- GPU utilization, power draw, and temperature.

The bounded row orders whole-device VRAM growth, GPU utilization, power, and
temperature immediately after elapsed time. Windows shared/combined memory,
model-storage, CPU, and RAM follow. This keeps GPU health visible even when a
narrow terminal clips the end of the row:

```text
LOAD 19s | VRAM 7.7G +0M/s(est) | GPU 17% 45W 52C | WinGPU ... | modelFS ...
```

`VRAM growth` is an estimate of how quickly GPU memory is being populated, not
an exact PCIe host-to-device bandwidth measurement. Exact transfer bandwidth
requires a CUDA/CUPTI or Nsight trace. The program no longer substitutes
unrelated system-wide disk latency for model-storage latency. On `/mnt/d`, WSL's
9p/DrvFs mount does not expose the underlying Windows block-device latency, so
the report says `device_latency_available: false`; loader-process reads remain
available and may be zero when files come from the operating-system cache.
Process CPU can exceed 100% because it counts logical cores (for example, 760%
is approximately 7.6 fully occupied cores); `wsl_vmCPU` or `windows_hostCPU` is
normalized and explicitly scoped.
Peak loading measurements are retained in each trial's `loading_telemetry` JSON.

After every reused trial, Cognityx deletes optimizer references, unloads the
LoRA adapter, runs Python garbage collection, synchronizes CUDA, and releases
unused allocator cache while retaining frozen base weights on their existing
GPU devices. `base_model_reuse` in `training-report.json` records whether disk
loading occurred, reuse location, and CUDA allocated/reserved bytes before and
after cleanup.

Before every trial, the shell prints the complete capacity-relevant
configuration. The child trainer then streams step/loss progress with process
CPU/RAM, labeled Windows-host or WSL-VM CPU/RAM, disk I/O, whole-device GPU
utilization/memory/power, and PyTorch allocated/reserved GPU memory.
While a forward/backward step is still running, the controller maintains one
non-scrolling heartbeat row such as:

```text
TRAIN 734s | step 0/2 running (734s since progress) | GPU 100% 31.1/31.8G 133W 51C | procCPU 97% wsl_vmCPU 8% | RAM 6.2/62.7G
```

The next completed `step N/M` line replaces/finalizes that heartbeat. Trial JSON
records `completed_steps`, `longest_step_seconds`, and
`seconds_since_last_step_at_finish`.
The controller checks configured resources every second. A limit must remain
breached for `sustain_seconds`; this avoids terminating on a transient sample.
Afterward, it prints the
trial status, peak whole-device GPU-memory percentage, peak labeled host-RAM
percentage, and
runtime.

## Termination policies

Resource utilization is not inherently bad. In particular, 100% GPU compute
utilization is normally desirable and CPU utilization is informational. Neither
is a default termination signal.

Only metrics present in `[termination]` can stop a trial:

```toml
[termination]
gpu_memory_percent = 98
host_ram_percent = 95
gpu_temperature_celsius = 88
sustain_seconds = 3
model_loading_timeout_seconds = 3600
trial_timeout_seconds = 3600
no_step_progress_timeout_seconds = 900
```

The two timeouts use separate clocks. `model_loading_timeout_seconds` covers
model/tokenizer loading and training setup. `trial_timeout_seconds` begins only
after the child emits its training-start marker, immediately before optimizer
steps begin, and stops when optimizer training completes; adapter saving and
evaluation are outside both clocks. The summary records `model_loading_seconds`,
`training_seconds`, and total `runtime_seconds` separately. Omit either timeout
to disable it.

`no_step_progress_timeout_seconds` is a third clock that resets after every
completed optimizer step. The checked-in RTX 5090 example terminates a trial
when one step reaches 900 seconds. Time deadlines are enforced immediately on
the next approximately one-second controller sample; the three-second
`sustain_seconds` window applies only to fluctuating resource measurements.

A no-step timeout establishes an ordered boundary for the active staged axis.
For example, if batch 4 times out, batch 8 and 16 are recorded in
`pruned_candidates` and are not executed. The next axis continues from the last
safe batch. If the minimum model configuration times out, all remaining axes
for that model are pruned. Grid pruning is context-specific and skips only
equal-or-higher values on the active grid axis when all other configuration
values match.

Optional `gpu_power_watts` and `gpu_power_percent_of_limit` policies are also
supported. Omit a policy to disable it. Reaching the GPU's normal power limit
is not automatically considered unsafe; configure a power policy only when it
matches your electrical, acoustic, or thermal objective.

The checked-in RTX 5090 example stops when any configured policy remains
breached for three seconds, when the child fails/OOMs, or when the candidate
space is exhausted:

- whole-device GPU memory reaches the configured percentage of physical VRAM;
- labeled Windows-host or WSL-VM RAM reaches 95%;
- one optimizer step reaches 900 seconds without completion;
- the child exits unsuccessfully, including CUDA out-of-memory;
- all configured candidates complete below the thresholds.

Per-trial reports include average/peak power, approximate sampled energy in
joules, temperature, utilization, whole-device VRAM, PyTorch allocated/reserved
VRAM, process RAM/CPU, and labeled Windows-host or WSL-VM RAM/CPU.

## Axes, candidates, and search strategy

Declare the axes and their ordered variations explicitly:

```toml
[search]
strategy = "grid"
axes = [
  "model_name",
  "max_sequence_length",
  "per_device_train_batch_size",
  "lora_rank",
]
max_trials = 100

[trials]
model_name = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
max_sequence_length = [512, 1024, 2048]
per_device_train_batch_size = [1, 2, 4, 8, 16]
lora_rank = [8, 16, 32, 64]
```

Model IDs must be exact and available under the base configuration's cache and
download policy. Qwen3 provides 8B, 14B, and 32B sizes; there is no standard
Qwen3-16B model ID.

`staged` groups by model, then changes each remaining axis in declared order;
this is economical and preserves loaded-weight reuse. `grid` runs the Cartesian
product model-major, capped by `max_trials`, and is the appropriate mode
for questions such as:

- which model allows the largest batch at each sequence length;
- the maximum sequence length for each model and batch;
- which LoRA rank remains safe for a particular model/batch/context tuple.

The JSON `capacity_frontier.by_model` section reports the maximum successful
candidate on every axis and includes the other configuration values that made
that result possible. All raw successful, threshold, OOM, and failed
combinations remain in `trials`.

### Staged search behavior

The search treats `model_name` as the outer grouping dimension. It completes
all other axes for one model before loading the next model:

1. Load the model at the minimum configured sequence, batch, and rank.
2. Find that model's maximum sequence length.
3. From its last safe sequence, find its maximum micro-batch size.
4. From its last safe configuration, find its maximum LoRA rank.
5. Unload the model and repeat for the next model.

The last successful configuration below both limits becomes the recommended
safe configuration. A threshold/failure configuration remains in the results
as evidence but is not recommended. After finding one axis boundary, the
controller continues with the next axis from the last safe configuration; for
example, an unsafe sequence length does not prevent measuring batch-size and
LoRA-rank boundaries at the safe sequence length. A threshold interrupts CUDA
state and therefore requires one same-model reload before the next axis; all
successful trials before that boundary reuse the resident base. If the minimum
configuration fails, only that model is marked unsupported and skipped.

The shell and JSON include `model_execution` with trial, load, reused-trial,
and restart counts for every model. Both staged and grid searches force
model-major execution even if `model_name` appears later in the TOML axis list.

`max_examples` and `max_steps` are automatically controlled for short capacity
trials. Increasing them mostly changes runtime rather than peak memory. The
controller generates enough long synthetic examples to fill the largest
configured sequence and batch candidates; this avoids a misleading result from
padding short hello-world text.

## Fixing and searching values

The base training TOML fixes quantization, learning rate, gradient accumulation,
target modules, telemetry sampling, and other normal training behavior.
Candidate lists in `[trials]` control automatic variation:

```toml
[trials]
max_sequence_length = [128, 256, 512, 1024, 2048, 4096]
per_device_train_batch_size = [1, 2, 4, 8]
lora_rank = [8, 16, 32, 64]
```

Use a one-value list to fix a search dimension. Model names have no meaningful
numeric increment, so multiple models must be explicitly listed in the desired
trial order.

## Stored results

Every invocation creates a unique directory:

```text
<output_dir>/sessions/<timestamp-id>/
├── hardware inventory in autotune-summary.json
├── capacity-dataset.jsonl
├── trial-configs/
├── logs/
├── runs/<trial-id>/training-report.json
└── autotune-summary.json
```

The final shell summary lists every configuration and its outcome, explains
why the search stopped, prints the recommended safe configuration, and gives
the full JSON summary path.

!!! warning
    Autotuning intentionally approaches hardware limits. Save unrelated GPU
    work first. Do not run another model workload on the same GPU during the
    search, because it invalidates the measurements and can cause an earlier
    out-of-memory result.
