# Cognityx Training

Cognityx Training teaches a base model from records prepared by DataForge and
publishes the resulting adapter, measurements and predictions. It sits here in
the application flow:

```text
Source files -> Ingest -> DataForge -> Training -> saved adapter and evidence
```

Ingest extracts passages. DataForge builds and qualifies records. Training
selects only trainable records, runs the optimizer, measures the unchanged base
model and trained adapter on frozen test suites, and publishes the evidence.
It does not parse source files, design new records, deploy the adapter or serve
model requests.

The initial implementation is intentionally narrow: a custom PyTorch
`DataLoader`/`AdamW` loop with Transformers model loading and PEFT adapters for
a small LoRA or QLoRA supervised fine-tuning run.

Training can consume either a DataForge dataset manifest or a DataForge
research package. A research package links one dataset with test-only exact
recall, paraphrase and held-out-knowledge sets. Optional MLflow tracking stores
compact run indexes; Cognityx Storage remains the authority for adapters,
predictions and reports.
