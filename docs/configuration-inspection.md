# Inspect a Training run specification

A Training TOML file is a scientific workload specification. It decides what
model, dataset, optimization settings, evaluation, tracking, and publication
rules one run will use, so its path remains explicit.

Inspect the resolved values without reading a dataset, importing Transformers,
checking CUDA, loading a model, opening Storage, starting tracking, creating an
output directory, training, or publishing:

```bash
cognityx-train config show --config training.toml
cognityx-train config validate --config training.toml
```

Add `--human` for a deterministic labelled view of the same static report.
JSON remains the unchanged default, and inspection still stops before dataset
construction, Transformers/CUDA setup, model loading, Storage, tracking,
training, output-directory creation, or publication.

```bash
cognityx-train config show --config training.toml --human
cognityx-train config validate --config training.toml --seed 29 --human
```

For the established full-run interface, `--human` is only an explicit alias for
the existing `--output-format human`; the run output contract is otherwise
unchanged. Supplying `--human` with `--output-format json` is rejected by
argument parsing before configuration or execution.

The supported run overrides can follow the command, for example `--seed 29` or
`--output-dir PATH`. Only supplied values that actually change the resolved
configuration appear in `overrides`; argparse defaults do not. Output includes
the normalized path, exact file-byte SHA-256, field sources, secret-safe
effective values, and validation status.

The established run form remains unchanged:

```bash
cognityx-train --config training.toml
```

Autotune search configurations, evaluation requests, and publication tracking
operands remain explicit specialized scientific inputs and are not included in
ambient discovery.
