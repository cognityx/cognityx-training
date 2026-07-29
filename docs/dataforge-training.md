# DataForge Dataset Training

`cognityx-training` can consume a completed DataForge dataset through its durable manifest instead of a local JSONL file.

## Preferred flow

```text
Ingest manifest
  → DataForge dataset manifest in Storage
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
storage_root = ""
overlength_policy = "error"

[dataset]
name = "example"
version = "1"
uri = "storage://local-main/datasets/example/1/manifest.json"
```

`--storage-config`, `--storage-root`, and `--dataset-input-mode` override TOML values when supplied on the CLI.

## Behavior

- DataForge manifests are validated before model allocation.
- Records stream from Storage instead of being loaded into one list.
- Loss is masked to assistant tokens only.
- Oversized records fail by default, or can be skipped explicitly.
- Training reports include a `dataset.lineage` section so runs remain traceable to the manifest and records artifact.

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
