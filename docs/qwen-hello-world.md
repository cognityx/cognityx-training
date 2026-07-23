# Qwen hello-world fine-tuning

Install the training extra and run:

```bash
uv sync --extra training
uv run --extra training cognityx-train \
  --config examples/qwen3-14b-hello/config.toml
```

The checked-in configuration performs one optimizer step with batch size one.
It uses NF4 4-bit loading, BF16 computation, double quantization, and LoRA
adapters on Qwen attention projections.

For a lower-cost smoke test, copy the TOML and change `model_name` to a smaller
Qwen causal language model. The dataset pipeline accepts either a `messages`
array or `instruction`/`output` fields in JSONL.
