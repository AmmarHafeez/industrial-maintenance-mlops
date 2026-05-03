# Results

No generated metrics JSON files are committed with the repository.

Training runs write local metrics to:

```text
reports/metrics/training_metrics_<dataset_id>.json
```

The metrics file is generated from the configured local dataset, engine-level train-validation split, window settings, RUL clipping value, failure-risk threshold, model configuration, and random seed. Generated metrics and benchmark outputs are ignored by Git.

Expected metric sections:

- RUL regression: `mae`, `rmse`, `r2`
- Failure-risk classification: `accuracy`, `macro_f1`, `balanced_accuracy`, `precision`, `recall`, `confusion_matrix`

## Local FD001 Validation

Dataset: NASA C-MAPSS FD001, with files placed locally under `data/raw/CMAPSSData/`.

This run used `train_FD001.txt` for an engine-level train-validation split. These are not official held-out test-set benchmark results.

Run configuration:

- Model version: `FD001-win30-stride1-rul125-risk30-logistic-regression-seed42`
- Training engines: `80`
- Validation engines: `20`
- Training windows: `14,241`
- Validation windows: `3,490`
- `window_size=30`
- `stride=1`
- `max_rul=125`
- `risk_threshold=30`
- `random_state=42`

Regression validation metrics:

| Metric | Value |
| --- | ---: |
| MAE | 12.0684 |
| RMSE | 15.9273 |
| R2 | 0.8545 |

Failure-risk validation metrics:

| Metric | Value |
| --- | ---: |
| accuracy | 0.9685 |
| macro_f1 | 0.9464 |
| balanced_accuracy | 0.9492 |
| precision | 0.9048 |
| recall | 0.9194 |

Confusion matrix:

```text
[[2810, 60],
 [50, 570]]
```

The LogisticRegression classifier produced a convergence warning during the run. Future work may add feature scaling or use another classifier setting. Raw data, processed data, models, and metrics JSON files are generated locally and ignored by Git.
