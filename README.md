# industrial-maintenance-mlops

[![Tests](https://github.com/AmmarHafeez/industrial-maintenance-mlops/actions/workflows/tests.yml/badge.svg)](https://github.com/AmmarHafeez/industrial-maintenance-mlops/actions/workflows/tests.yml)

Predictive maintenance MLOps-style project for NASA C-MAPSS turbofan engine
degradation data. The repository demonstrates a reproducible local workflow for
parsing C-MAPSS text files, training Remaining Useful Life (RUL) and
failure-risk models, evaluating held-out test engines, serving predictions with
FastAPI, and producing lightweight monitoring and drift reports.

## Key Capabilities

- Parse C-MAPSS train, test, and RUL text files.
- Generate clipped RUL targets with default `max_rul=125`.
- Build fixed-length multivariate sensor windows.
- Train baseline `RandomForestRegressor` RUL models.
- Train failure-risk classifiers from RUL thresholds, with scaled
  `LogisticRegression` as the default classifier.
- Split training and validation by engine id to avoid window leakage.
- Evaluate held-out C-MAPSS test engines using the final available window per
  engine.
- Serve local joblib model bundles through FastAPI.
- Expose Prometheus-compatible API metrics.
- Compare final test-engine windows against training reference statistics with a
  lightweight drift report.

## Quickstart

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For direct `python -m` commands with the `src` layout, either install the
package in editable mode as above or set:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

Place local C-MAPSS files under:

```text
data/raw/CMAPSSData/
```

Expected FD001 files:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

## Common Commands

Train FD001 baselines:

```powershell
python -m industrial_maintenance_mlops.training.pipeline `
  --dataset-id FD001 `
  --raw-dir data/raw/CMAPSSData `
  --models-dir models `
  --metrics-dir reports/metrics `
  --window-size 30 `
  --stride 1 `
  --max-rul 125 `
  --risk-threshold 30 `
  --test-size 0.2 `
  --random-state 42
```

Evaluate held-out FD001 test engines:

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

Generate a drift report:

```powershell
python -m industrial_maintenance_mlops.monitoring.drift_report `
  --dataset-id FD001 `
  --raw-dir data/raw/CMAPSSData `
  --reference-stats models/reference_stats.json `
  --output reports/metrics/drift_report_FD001.json `
  --window-size 30 `
  --max-rul 125
```

Start the API after local models exist:

```powershell
uvicorn industrial_maintenance_mlops.api.app:app --host 0.0.0.0 --port 8000
Invoke-RestMethod http://localhost:8000/health
```

Run tests:

```powershell
pytest
```

## Results Summary

The documented FD001 model version is:

```text
FD001-win30-stride1-rul125-risk30-scaled-logistic-regression-seed42
```

Engine-level validation on `train_FD001.txt`:

- 80 training engines, 20 validation engines
- 14,241 training windows, 3,490 validation windows
- RUL regression: MAE `12.0684`, RMSE `15.9273`, R2 `0.8545`
- Failure-risk classification: accuracy `0.9673`, macro F1 `0.9442`,
  balanced accuracy `0.9447`, precision `0.9068`, recall `0.9097`

Held-out FD001 test-set evaluation:

- 100 test engines, 100 final test-engine windows
- RUL regression: MAE `12.0360`, RMSE `15.8548`, R2 `0.8435`
- Failure-risk classification: accuracy `0.9800`, macro F1 `0.9733`,
  balanced accuracy `0.9733`, precision `0.9600`, recall `0.9600`

FD001 drift report:

- Compared 100 final test-engine windows with `models/reference_stats.json`
- `mean_z_threshold=3.0`, `std_ratio_threshold=2.0`
- Flagged features: `0` of 24

Validation, held-out test-set evaluation, and drift reporting answer different
questions. Validation uses an engine-level split from `train_FD001.txt`, test
evaluation uses `test_FD001.txt` plus `RUL_FD001.txt`, and drift reporting is a
lightweight statistical comparison rather than a full production monitoring
system.

## API

The API reads model paths from `configs/api.yaml` by default:

```text
models/rul_regressor.joblib
models/failure_risk_classifier.joblib
models/reference_stats.json
```

It starts even if model artifacts are missing. In that case, `/health` returns
`ready=false`, and prediction endpoints return a clear service error until
models are available or paths are overridden with `RUL_MODEL_PATH`,
`FAILURE_RISK_MODEL_PATH`, and `REFERENCE_STATS_PATH`.

Prediction requests use a numeric `sensor_window` shaped as:

```text
window_size x feature_count
```

For the FD001 configuration above, this is `30 x 24`.

## Artifact Policy

Raw data is not included in the repository. Generated data, trained models,
metrics JSON files, reports, figures, and local run artifacts are produced
locally and ignored by Git.

Ignored local outputs include:

- `data/raw/`
- `data/processed/`
- `models/`
- `reports/metrics/`
- `reports/figures/`
- `reports/artifacts/`

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)
