# DataForge Dataset Training

`cognityx-training` can consume a completed DataForge dataset through its durable manifest instead of a local JSONL file.

## Preferred flow

```text
Ingest manifest
  → DataForge dataset manifest in Storage
  → optional DataForge research package with frozen evaluation sets
  → cognityx-training preflight
  → streaming supervised fine-tuning
```

## Configuration

Use a manifest URI plus storage settings:

```toml
[training]
backend = "custom-pytorch"
dataset_input_mode = "dataforge_manifest"
storage_config = ".cognityx/storage.toml"
overlength_policy = "error"
data_order = "source"

[publication]
mode = "storage"
retain_local_staging = false

[dataset]
name = "example"
version = "1"
uri = "storage://local-main/datasets/example/1/manifest.json"
```

The same `uri` can point to a
`cognityx.dataforge.research-package/v1` manifest. Training verifies the linked
dataset and every evaluation-set checksum before model allocation.

`--storage-config`, `--storage-root`, and `--dataset-input-mode` override TOML values when supplied on the CLI.

## Behavior

- DataForge manifests are validated before model allocation.
- Records stream from Storage instead of being loaded into one list.
- Loss is masked to assistant tokens only.
- Oversized records fail by default, or can be skipped explicitly.
- Training reports include a `dataset.lineage` section so runs remain traceable to the manifest and records artifact.
- Only `split=train` records with `research_role=training` and no explicit
  `training_eligible=false` reach the optimizer.
- Historical `validation` and `test` records remain readable as evaluation
  records. Their original split is preserved as `original_split`.
- Frozen evaluation records use `split=evaluation`, an explicit role such as
  `exact_recall` or `paraphrase_evaluation`, and
  `training_eligible=false`.
- Reports distinguish optimizer counts, total evaluation counts and counts for
  each evaluation role. A known zero is written as `0`; an absent source field
  remains absent rather than being guessed.

## Commands

Dry-run validation:

```bash
uv run --extra training cognityx-train \
  --config examples/dataforge-manifest-smoke/config.toml \
  --dry-run --print-config
```

Normal training:

```bash
uv run --extra training cognityx-train \
  --config examples/dataforge-manifest-smoke/config.toml
```

## Legacy compatibility

Local JSONL examples continue to work in `legacy_jsonl` mode for smoke tests and older examples.
