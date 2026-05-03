# Results

No generated metrics JSON files are committed with the repository.

Training and evaluation runs write local metrics to:

```text
reports/metrics/training_metrics_<dataset_id>.json
reports/metrics/test_metrics_<dataset_id>.json
reports/metrics/drift_report_<dataset_id>.json
```

The metrics file is generated from the configured local dataset, engine-level train-validation split, window settings, RUL clipping value, failure-risk threshold, model configuration, and random seed. Generated metrics and benchmark outputs are ignored by Git.

Expected metric sections:

- RUL regression: `mae`, `rmse`, `r2`
- Failure-risk classification: `accuracy`, `macro_f1`, `balanced_accuracy`, `precision`, `recall`, `confusion_matrix`

## Held-Out Test Evaluation

The project supports held-out C-MAPSS test-set evaluation with:

```powershell
python -m industrial_maintenance_mlops.evaluation.evaluate_cmapss `
  --dataset-id FD001 `
  --raw-dir data/raw/CMAPSSData `
  --models-dir models `
  --metrics-dir reports/metrics `
  --window-size 30 `
  --max-rul 125 `
  --risk-threshold 30
```

The command uses `test_FD001.txt` and `RUL_FD001.txt`, extracts the last fixed-length window for each test engine, and writes `reports/metrics/test_metrics_FD001.json`. Generated metrics JSON remains local under `reports/metrics/` and is ignored by Git.

## Drift Reporting

The project supports drift reporting with:

```powershell
python -m industrial_maintenance_mlops.monitoring.drift_report `
  --dataset-id FD001 `
  --raw-dir data/raw/CMAPSSData `
  --reference-stats models/reference_stats.json `
  --output reports/metrics/drift_report_FD001.json `
  --window-size 30 `
  --max-rul 125
```

The command compares final test-engine windows with training reference statistics and writes `reports/metrics/drift_report_FD001.json`.

## FD001 Drift Report

Dataset: NASA C-MAPSS FD001, with files placed locally under `data/raw/CMAPSSData/`.

This drift report compared final test-engine windows against training reference statistics from `models/reference_stats.json`. It is a lightweight statistical drift check, not a full production monitoring system.

Run configuration:

- Window count: `100`
- Feature count: `24`
- `mean_z_threshold=3.0`
- `std_ratio_threshold=2.0`
- Generated local file: `reports/metrics/drift_report_FD001.json`

Summary:

| Metric | Value |
| --- | ---: |
| flagged_features | 0 |

Zero flagged features means no feature exceeded the configured simple mean z-shift or standard-deviation ratio thresholds in this local FD001 run. The generated drift report JSON remains local under `reports/metrics/` and is ignored by Git.

## Held-Out FD001 Test Result

Dataset: NASA C-MAPSS FD001, with files placed locally under `data/raw/CMAPSSData/`.

This evaluation used `test_FD001.txt` and `RUL_FD001.txt`. It scores one final available `30 x 24` window per test engine, so the run used 100 test engines and 100 final test-engine windows.

Run configuration:

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

## Local FD001 Validation

Dataset: NASA C-MAPSS FD001, with files placed locally under `data/raw/CMAPSSData/`.

This run used `train_FD001.txt` for an engine-level train-validation split. These are not official held-out test-set benchmark results.

Run configuration:

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
- `classifier_max_iter=2000`
- Classifier: `StandardScaler` followed by `LogisticRegression`

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

The earlier LogisticRegression convergence warning was addressed by adding `StandardScaler` before `LogisticRegression`. Raw data, processed data, models, and metrics JSON files are generated locally and ignored by Git.
