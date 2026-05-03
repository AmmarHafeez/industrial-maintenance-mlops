# Results

No benchmark metrics are committed with the repository.

Training runs write local metrics to:

```text
reports/metrics/training_metrics_<dataset_id>.json
```

The metrics file is generated from the configured local dataset, engine-level train-validation split, window settings, RUL clipping value, failure-risk threshold, model configuration, and random seed. Generated metrics and benchmark outputs are ignored by Git.

Expected metric sections:

- RUL regression: `mae`, `rmse`, `r2`
- Failure-risk classification: `accuracy`, `macro_f1`, `balanced_accuracy`, `precision`, `recall`, `confusion_matrix`
