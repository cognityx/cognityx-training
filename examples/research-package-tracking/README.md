# Research package comparison

This example shows where Training sits in the qualification experiment. A
DataForge research package supplies one trainable dataset and frozen test-only
suites. Training measures the base model, runs the fixed-budget optimizer,
measures the adapter and publishes both sets of predictions.

```text
DataForge research package -> Training -> Storage publication -> optional MLflow index
```

Replace the example Storage URI and model cache values before running. Tracking
is disabled in the checked-in file so the example never contacts an external
service by default.
