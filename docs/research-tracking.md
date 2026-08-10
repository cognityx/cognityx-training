# Research packages and tracking

This foundation supports a controlled raw-versus-qualified DataForge
comparison without adding a new trainer or experiment database.

```text
DataForge research package
  -> one training dataset
  -> exact-recall evaluation
  -> paraphrase evaluation
  -> optional held-out knowledge evaluation
  -> baseline predictions
  -> fixed-budget training
  -> final predictions
  -> Storage publication
  -> optional live tracker index
```

## Input isolation

Training verifies every records checksum before model allocation. Only the
package dataset can supply optimizer records, and only records marked as the
training role can enter the optimizer. Evaluation-set records are always
test-only. Legacy DataForge `validation` and `test` records remain evaluation
records and retain their original split label.

This means an imported paraphrase cannot silently become a training example,
even if its JSONL line is malformed to say `split=train`: package validation
stops the run.

## Baseline and final measurements

The unchanged base model and trained adapter use the same chat template,
deterministic decoding and token limit. The tokenizer returns a structured
batch of tensors (a `BatchEncoding`), every tensor moves to the model input
device, and generation receives the whole batch. Generated tokens are sliced
after the input length.

For a controlled raw-versus-qualified comparison, keep `max_steps`, micro
batch, gradient accumulation, sequence length and the selected record policy
identical. Reports record the actual input-token and assistant-target-token
counts processed, so runs with unequal realized token budgets are visible and
should not be treated as a clean comparison.

Reports contain overall metrics and separate `suite_metrics` for roles such as
`exact_recall`, `paraphrase_evaluation`, `heldout_knowledge_unit`,
`legacy_validation` and `legacy_test`. Prediction rows retain the role and
evaluation lineage for later forensic evaluation.

## Optional live run index

Training already knows the current optimizer step, loss, processed examples and
tokens, elapsed time, and the resource samples collected for its report. An
optional tracker receives those measurements while the run is active. This
means a later research tool can read a learning curve directly instead of
trying to reconstruct one from terminal text.

The tracker follows a small run lifecycle: start, record measurements, record
named evaluation suites, attach Storage references, and finish or fail. The
default implementation does nothing (a `NoOpTracker`), so tracking cannot
become an accidental requirement.

Measurements are emitted at the existing progress interval. Process and host
measurements name their scope—for example, the Python process versus the whole
WSL virtual machine. GPU utilization, memory, power, and accumulated energy are
sent only when the resource sampler actually provides them. Baseline and final
evaluation events retain the research role and frozen evaluation-set identity.

## MLflow implementation

No tracker is used by default:

```toml
[tracking]
backend = "none"
failure_policy = "warn"
```

Install and enable MLflow when a searchable experiment index is useful:

```toml
[tracking]
backend = "mlflow"
uri = "sqlite:///mlruns.db"
experiment_name = "dataforge-qualification"
run_name = "qualified-paragraph-control"
parent_run_id = "optional-existing-parent-run-id"
failure_policy = "warn"
```

`warn` keeps Training running if the tracker is temporarily unavailable;
`error` makes tracking failure fail the command. The tracker logs IDs,
result-changing parameters, live scalar metrics, named evaluation suites,
resource measurements, and Cognityx Storage URI/checksum references. It does
not upload the adapter, report, dataset, or prediction files again. Storage
publication remains the authoritative result.

## Idempotent backfill

A completed Storage publication can be indexed later:

```bash
uv run --extra tracking cognityx-track-publication \
  <publication-manifest-uri> \
  --experiment-name dataforge-qualification \
  --tracking-uri sqlite:///mlruns.db
```

The completed publication URI is the idempotency key. Repeating the backfill
finds the existing external run instead of creating a duplicate. Only a
`cognityx.training.publication/v1` manifest with `status=completed` is accepted.
Backfill is marked as a later registration and stores the original Training
start time, finish time, and duration as metadata. Its registration time is not
presented as the time when training occurred.

Training-to-Inference adapter loading and serving remain outside this release.
