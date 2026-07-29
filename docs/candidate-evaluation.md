# Sequential candidate evaluation

Candidate evaluation begins only after Training has written a completed
publication manifest:

```text
Training publication
-> saved baseline and trained predictions
-> sequential judge evaluation
-> promotion recommendation
-> later human approval and deployment
```

The evaluator never loads the candidate adapter, the student base model, or
Training. It reads saved predictions and loads only the configured judge through
`CognityxInferenceClient`. This no-dual-loading rule allows a large judge to use
the GPU after the training process has exited.

## Install and configure

Install the optional Inference integration:

```bash
uv sync --extra evaluation --group dev
```

The evaluation extra requires Python 3.12 because the current
`cognityx-inference` package requires Python 3.12. Training without that extra
continues to support the repository's broader Python requirement.

Start from `examples/evaluation/config.toml`. Candidate identities must be
completed `storage://` publication manifests. Local paths are not accepted.

The judge can be local or provider-backed. Local lifecycle configuration may
start a named Inference server profile. The evaluator records whether the judge
was already resident or was loaded by this evaluation. It unloads only a model
load it owns and only when `unload_judge_when_done = true`.

Evidence resolution follows durable references without importing DataForge:

```text
publication manifest
-> dataset lineage
-> DataForge dataset manifest
-> Ingest run manifest
-> evidence JSONL
```

By default, artifacts retain evidence IDs and SHA-256 hashes but not evidence
text. `missing_policy = "reference-only"` explicitly labels judgments made
without complete evidence. Use `unjudgeable` or `error` for stricter workflows.

## Commands

Run complete artifact and endpoint preflight without loading or calling the
judge model:

```bash
uv run cognityx-evaluate plan --config examples/evaluation/config.toml
```

Evaluate all candidate records sequentially with one judge lifecycle:

```bash
uv run cognityx-evaluate run --config examples/evaluation/config.toml
```

Resume a partial run. Successfully committed candidate-record judgments are not
called again:

```bash
uv run cognityx-evaluate resume \
  --evaluation-request storage://local-main/artifacts/experiments/exp-example/evaluations/eval-example/evaluation-request.json
```

Show aggregate metrics and recommendations:

```bash
uv run cognityx-evaluate show \
  --evaluation-manifest storage://local-main/artifacts/experiments/exp-example/evaluations/eval-example/evaluation-manifest.json
```

For a non-default local Storage root, `resume` and `show` also accept
`--storage-root` or `--storage-config`.

## Durable execution

Evaluation uses these ordered stages:

```text
preflight
deterministic-scoring
evidence-resolution
judge-evaluation
aggregation
recommendation
finalization
```

Stage checkpoints and per-request records are immutable. A judge response is
committed before the next candidate-record call begins. Consolidated JSONL
artifacts are streamed through bounded temporary files. The terminal
`evaluation-manifest.json` is written last and is the sole completed-evaluation
signal.

A failure preserves checkpoints, valid judge results, and `failure.json`; it
does not create a terminal manifest or alter source Training artifacts. Resume
retains the original `evaluation_run_id`.

## Recommendations

Configured gates produce one of:

```text
recommended_for_promotion
manual_review_required
not_recommended
insufficient_evidence
```

These are recommendations only. Evaluation never changes adapter status,
creates a release, deploys a model, or bypasses human approval.
