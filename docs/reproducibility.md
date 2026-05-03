# Reproducibility

## Environment

Use Python 3.12 and install dependencies from `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Because the package uses a `src` layout, run commands after either installing the package in editable mode or setting `PYTHONPATH` in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Data

Raw NASA C-MAPSS / Turbofan Engine Degradation Simulation text files are not committed to this repository. Place the extracted files under:

```text
data/raw/CMAPSSData/
```

For the default `FD001` subset, the training pipeline expects:

```text
data/raw/CMAPSSData/train_FD001.txt
data/raw/CMAPSSData/test_FD001.txt
data/raw/CMAPSSData/RUL_FD001.txt
```

Processed data should be written under `data/processed/`. Raw data, processed data, trained models, metrics, reports, figures, and local artifacts are ignored by Git.

## Configuration

Configuration lives in:

- `configs/data.yaml`
- `configs/model.yaml`
- `configs/training.yaml`
- `configs/api.yaml`

The default RUL clipping value is `125`, and the default failure-risk threshold is `30`. Training uses deterministic `random_state` values where sklearn estimators support them. The default failure-risk classifier is a scaled logistic-regression sklearn pipeline.

## Training Run

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

The pipeline reads `train_FD001.txt`, validates that the matching `test_FD001.txt` and `RUL_FD001.txt` files are present, generates clipped RUL labels, builds fixed-length windows, splits by engine id, trains the baseline regression and classification models, saves model bundles, and writes validation metrics.

Expected local outputs:

- `models/rul_regressor.joblib`
- `models/failure_risk_classifier.joblib`
- `models/reference_stats.json`
- `reports/metrics/training_metrics_FD001.json`

These generated outputs are ignored by Git.

## Held-Out Test Evaluation

After local training has produced model bundles under `models/`, run:

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

The evaluation command reads `test_FD001.txt` and `RUL_FD001.txt`, extracts the last fixed-length window for each test engine, clips true final RUL with `max_rul`, derives high-risk labels from `risk_threshold`, and writes:

```text
reports/metrics/test_metrics_FD001.json
```

Generated evaluation metrics are ignored by Git.

The documented FD001 held-out test-set result used 100 test engines, one final window per engine, `window_size=30`, `max_rul=125`, and `risk_threshold=30`. The model version was `FD001-win30-stride1-rul125-risk30-scaled-logistic-regression-seed42`.

## Drift Report

After local training has produced `models/reference_stats.json`, run:

```powershell
python -m industrial_maintenance_mlops.monitoring.drift_report `
  --dataset-id FD001 `
  --raw-dir data/raw/CMAPSSData `
  --reference-stats models/reference_stats.json `
  --output reports/metrics/drift_report_FD001.json `
  --window-size 30 `
  --max-rul 125
```

The command reads `test_FD001.txt`, extracts the final fixed-length window for each test engine, compares test-window feature statistics with training reference statistics, and writes:

```text
reports/metrics/drift_report_FD001.json
```

Generated drift reports are ignored by Git.

The documented FD001 drift report compared 100 final test-engine windows against training reference statistics from `models/reference_stats.json`. With `mean_z_threshold=3.0` and `std_ratio_threshold=2.0`, it flagged 0 of 24 features.

## API Serving

The API reads model artifact paths from `configs/api.yaml`:

```text
models/rul_regressor.joblib
models/failure_risk_classifier.joblib
models/reference_stats.json
```

Start the service after local training has produced those files:

```powershell
uvicorn industrial_maintenance_mlops.api.app:app --host 0.0.0.0 --port 8000
```

Check readiness:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

The service starts even when model artifacts are missing. In that case, `/health` returns `ready=false`, and prediction endpoints return a service error until models are trained or paths are configured. Override paths with `RUL_MODEL_PATH`, `FAILURE_RISK_MODEL_PATH`, and `REFERENCE_STATS_PATH` when needed.

Prediction requests must provide a numeric `sensor_window` with the model window shape. For the default FD001 training settings, this is `30 x 24`.

## Local FD001 Run

The documented FD001 validation result used:

- Dataset files under `data/raw/CMAPSSData/`
- `train_FD001.txt` for an engine-level train-validation split
- 80 training engines and 20 validation engines
- 14,241 training windows and 3,490 validation windows
- `window_size=30`, `stride=1`, `max_rul=125`, `risk_threshold=30`, `random_state=42`
- `classifier_max_iter=2000`
- Model version `FD001-win30-stride1-rul125-risk30-scaled-logistic-regression-seed42`

The classifier uses `StandardScaler` before `LogisticRegression`. The earlier LogisticRegression convergence warning was addressed by adding this scaling step.

## Test Data

The test suite uses synthetic C-MAPSS-like files created in temporary directories. It does not require the real C-MAPSS dataset.
