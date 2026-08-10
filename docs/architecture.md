# Architecture

```mermaid
flowchart LR
    TOML[Training TOML] --> Config[CustomPyTorchTrainingConfig]
    Config --> Factory[BackendFactory]
    Factory --> Backend[CustomPyTorchTrainerBackend]
    Package[DataForge dataset or research package] --> Pipeline[Dataset-to-IFT pipeline]
    Pipeline --> Backend
    Backend --> Qwen[Qwen causal LM]
    Backend --> PEFT[PEFT LoRA adapter]
    Backend --> Artifact[Model Artifact]
    Artifact --> Registry[Optional ModelArtifactRegistry]
    Artifact --> Tracker[Optional tracker index]
```

Common contracts and backend selection live in `cognityx-core`. Transformers,
PEFT, BitsAndBytes, CUDA, and Qwen-specific configuration stay in this package.
The tracker interface is a sidecar: it logs identities, scalar measurements and
Storage URI/checksum references but never copies adapter or prediction files.
