# Cognityx Training

`cognityx-training` owns training implementations and dataset preparation. It
depends on the stable contracts in `cognityx-core`.

The initial implementation is intentionally narrow: a custom PyTorch/
Transformers PEFT backend for a small LoRA or QLoRA supervised fine-tuning run.
