# Results

Generated metrics JSON files are not committed. Local runs write metrics under:

```text
reports/metrics/training_metrics_<dataset_id>.json
reports/metrics/test_metrics_<dataset_id>.json
reports/metrics/drift_report_<dataset_id>.json
```

The sections below document real local FD001 runs. Validation, held-out test-set
evaluation, and drift reporting are separate workflows and should not be
compared as if they measure the same behavior.

## FD001 Engine-Level Validation

Validation uses `train_FD001.txt` with an engine-level train-validation split.
Windows from the same engine are not split across training and validation.

Run configuration:

- Dataset: NASA C-MAPSS FD001
- Model version: `FD001-win30-stride1-rul125-risk30-scaled-logistic-regression-seed42`
- Training engines: `80`
- Validation engines: `20`
- Training windows: `14,241`
- Validation windows: `3,490`
- `window_size=30`
- `stride=1`
- `max_rul=125`
- `risk_threshold=30`
- `random_state=42`
- Classifier: `StandardScaler` followed by `LogisticRegression`
- `classifier_max_iter=2000`

Regression validation metrics:

| Metric | Value |
| --- | ---: |
| MAE | 12.0684 |
| RMSE | 15.9273 |
| R2 | 0.8545 |

Failure-risk validation metrics:

| Metric | Value |
| --- | ---: |
| accuracy | 0.9673 |
| macro_f1 | 0.9442 |
| balanced_accuracy | 0.9447 |
| precision | 0.9068 |
| recall | 0.9097 |

Confusion matrix:

```text
[[2812, 58],
 [56, 564]]
```

## FD001 Held-Out Test Set

Held-out test evaluation uses `test_FD001.txt` and `RUL_FD001.txt`. It scores
the final available `30 x 24` window for each test engine.

Run configuration:

- Dataset: NASA C-MAPSS FD001
- Model version: `FD001-win30-stride1-rul125-risk30-scaled-logistic-regression-seed42`
- Test engines: `100`
- Final test-engine windows: `100`
- `window_size=30`
- `max_rul=125`
- `risk_threshold=30`

Regression test metrics:

| Metric | Value |
| --- | ---: |
| MAE | 12.0360 |
| RMSE | 15.8548 |
| R2 | 0.8435 |

Failure-risk test metrics:

| Metric | Value |
| --- | ---: |
| accuracy | 0.9800 |
| macro_f1 | 0.9733 |
| balanced_accuracy | 0.9733 |
| precision | 0.9600 |
| recall | 0.9600 |

Confusion matrix:

```text
[[74, 1],
 [1, 24]]
```

## FD001 Drift Report

The drift report compares final test-engine windows with training reference
statistics from `models/reference_stats.json`. It is a lightweight statistical
check, not a full production monitoring system.

Run configuration:

- Dataset: NASA C-MAPSS FD001
- Window count: `100`
- Feature count: `24`
- `mean_z_threshold=3.0`
- `std_ratio_threshold=2.0`
- Generated local file: `reports/metrics/drift_report_FD001.json`

Summary:

| Metric | Value |
| --- | ---: |
| flagged_features | 0 |

Zero flagged features means no feature exceeded the configured simple mean
z-shift or standard-deviation ratio thresholds in this local FD001 run.

## Artifact Policy

Raw data, processed data, model bundles, generated metrics JSON files, reports,
figures, and benchmark outputs remain local and are ignored by Git.
