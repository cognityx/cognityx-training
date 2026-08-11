# Experiment lineage and adapter publication

Cognityx Training turns a completed supervised fine-tuning run into a durable,
traceable candidate adapter:

```text
DataForge dataset
-> training experiment
-> semantic training variant
-> physical training run
-> candidate adapter
-> baseline and trained predictions
```

## Identity model

| ID | Meaning |
| --- | --- |
| `experiment_id` | One experimental question or comparison group. |
| `training_variant_id` | A deterministic hash of the dataset, base model, and result-changing training settings. |
| `training_run_id` | One physical execution or retry of a variant. |
| `adapter_id` | A globally safe hash of the experiment and physical training run. |

Machine-specific cache paths, output directories, Storage roots, telemetry
executables, and progress intervals do not affect the variant ID. Dataset
checksums, model identity, LoRA settings, sequence length, optimizer settings,
batching, seed, overlength policy, and source data order do.

A DataForge research package also names frozen evaluation sets. Those test-only
sets do not enter the optimizer, so their package ID and checksums do not change
the training variant ID when the training dataset itself is unchanged. Training
still records the complete package lineage on the adapter and completed run so
later evaluation can identify the exact tests. A different evaluation package
is a different evaluation context, not a different trained model.

## Production configuration

DataForge-backed production training uses Storage publication explicitly:

```toml
[training]
backend = "custom-pytorch"
model_name = "Qwen/Qwen3-8B"
dataset_input_mode = "dataforge_manifest"
data_order = "source"

[experiment]
id = "exp-enterprise-kuqa-qwen3-8b"
name = "Enterprise knowledge-unit QA"
description = "Evaluate DataForge training on Qwen3-8B"

[publication]
mode = "storage"
retain_local_staging = false

[dataset]
name = "enterprise-kuqa"
version = "3"
uri = "storage://local-main/datasets/enterprise-kuqa/3/manifest.json"
```

`--experiment-id` overrides the configured experiment ID. `--run-id` remains a
compatibility alias and is normalized to a `trun-*` training run ID.

Storage is resolved once and shared by dataset reading and publication, in this
order:

1. An injected `StorageRuntime`.
2. An explicit `storage_config`.
3. An explicit `storage_root`.
4. `StorageRuntime.load()` discovery and built-in defaults.

Storage publication never falls back to local publication.

## Storage layout

The `artifact` role owns experiment and run metadata:

```text
experiments/<experiment_id>/
  experiment.json
  variants/<training_variant_id>.json
  runs/<training_run_id>/
    training-request.json
    resolved-config.json
    dataset-lineage.json
    environment.json
    training-report.json
    metrics.json
    baseline-predictions.jsonl
    trained-predictions.jsonl
    publication-manifest.json
```

The `model` role owns immutable candidate adapters:

```text
adapters/<adapter_id>/1/
  adapter_model.safetensors
  adapter_config.json
  tokenizer files
  checksums.json
  adapter-manifest.json
```

All public references are provider-neutral `storage://` URIs.

When the input is a research package, future `cognityx.training.adapter/v1`
manifests add a `research_package` object. It records the package ID, version,
manifest URI and checksum, plus each frozen evaluation set's role, ID, version,
manifest and records checksums, and freeze checksum. Dataset-only publications
omit this optional object and remain valid.

The terminal `cognityx.training.publication/v1` manifest repeats that package
lineage and points to the immutable `dataset-lineage.json` artifact with its
checksum. Downstream tools can therefore resolve the complete provenance from
Storage without relying on a local training directory or an MLflow run.

## Atomic success criteria

PEFT output is first written to local staging. Training then rejects unsafe or
incomplete output, hashes every adapter file, atomically publishes the directory
through the model-role Storage store, and verifies the stored files and bundle
checksum.

The run's `publication-manifest.json` is written last. Its verified existence
with `status: completed` is the authoritative success signal. A training or
publication error may create an immutable `failure.json`, but never a completed
terminal manifest. Failed staging is retained for diagnosis; successful staging
is removed only when `retain_local_staging = false`.

Use `verify_published_adapter(adapter_manifest_uri, storage_runtime=runtime)` to
verify a candidate through Storage read streams without loading its base model,
requiring a native backend path, or allocating a GPU.

The base-model identity names the tokenizer source actually passed to
Transformers, together with its resolved revision and chat-template checksum.
This makes the text-to-token rules visible without inventing a separate
tokenizer reference when Training did not load one.

## Evaluation handoff

Baseline and trained outputs are streamed to separate JSONL artifacts with
record, normalized DataForge evidence, decoding, model, and run lineage. The
rows also retain the evaluation role, evaluation-set identity, fact group and
source-record identity. Metrics are reported separately for each role. The
evaluation workflow can judge those saved outputs after the training model is
unloaded. A later Inference workflow can consume the verified adapter manifest.
Training publication does not promote, deploy, serve, or release the adapter.

## Local compatibility mode

Legacy hello-world, private-fact, and autotune configurations declare:

```toml
[publication]
mode = "local"
retain_local_staging = true
```

Local mode preserves the existing adapter directory and
`training-report.json`, while adding generated lineage IDs to the report and CLI
result.
