# industrial-maintenance-mlops

Production-style Python project for industrial predictive maintenance using NASA C-MAPSS turbofan degradation data. The project includes data parsing, Remaining Useful Life (RUL) labeling, time-series windowing, baseline model training, failure-risk classification, API inference, metrics, and drift statistics.

## Scope

- Parse C-MAPSS train, test, and RUL text files.
- Generate clipped RUL targets with a default maximum RUL of 125 cycles.
- Build fixed-length sensor windows for supervised learning.
- Train baseline `RandomForestRegressor` and failure-risk classifiers.
- Serve predictions through FastAPI.
- Export Prometheus-compatible metrics.
- Compare incoming windows against stored reference mean and standard deviation.
- Package the service with Docker and Docker Compose.

## Repository Layout

```text
configs/                         Runtime and training configuration
docs/                            Architecture, reproducibility, and results notes
src/industrial_maintenance_mlops Python package
tests/                           Synthetic-data test suite
```

Generated data, reports, metrics, models, figures, and benchmark outputs are ignored by Git.

## Data Setup

This repository does not include raw C-MAPSS data. Download the NASA C-MAPSS / Turbofan Engine Degradation Simulation data from an appropriate NASA data source and place the extracted text files under:

```text
data/raw/CMAPSSData/
```

Expected filenames follow the C-MAPSS convention, for example:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Processed datasets are written to `data/processed/` when pipeline steps are run.

## Installation

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For the `src` layout, either install the package in editable mode as shown above or set `PYTHONPATH` before running `python -m` commands:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Training

After placing the raw C-MAPSS files in `data/raw/CMAPSSData/`, run:

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

Training outputs are written locally:

- Models: `models/`
- Metrics: `reports/metrics/`

These outputs are intentionally ignored by Git.

The training workflow uses an engine-level train-validation split, so windows from the same engine are not split across training and validation sets. The validation metrics are written to `reports/metrics/training_metrics_FD001.json` for the command above.

The default failure-risk classifier uses `StandardScaler` followed by `LogisticRegression` inside an sklearn pipeline. Random forest classification remains available through configuration.

## Local FD001 Validation Result

The latest local FD001 run used `train_FD001.txt` for an engine-level train-validation split with 80 training engines and 20 validation engines. It produced 14,241 training windows and 3,490 validation windows with `window_size=30`, `stride=1`, `max_rul=125`, `risk_threshold=30`, and `random_state=42`.

Model version: `FD001-win30-stride1-rul125-risk30-scaled-logistic-regression-seed42`. The classifier used `StandardScaler` followed by `LogisticRegression` with `classifier_max_iter=2000`.

Validation metrics:

- RUL regression: MAE `12.0684`, RMSE `15.9273`, R2 `0.8545`
- Failure-risk classification: accuracy `0.9673`, macro F1 `0.9442`, balanced accuracy `0.9447`, precision `0.9068`, recall `0.9097`

These are engine-level validation results, not official held-out test-set benchmark results. Raw data, processed data, models, and metrics JSON files are generated locally and ignored by Git.

The earlier LogisticRegression convergence warning was addressed by adding `StandardScaler` before `LogisticRegression`.

## API

After training, the API loads local artifacts from `configs/api.yaml` by default:

```text
models/rul_regressor.joblib
models/failure_risk_classifier.joblib
models/reference_stats.json
```

These files are generated locally and ignored by Git. The API still starts if model files are missing; `/health` reports `ready=false`, and prediction endpoints return a service error until model artifacts are available.

Start the FastAPI service after models have been trained:

```powershell
uvicorn industrial_maintenance_mlops.api.app:app --host 0.0.0.0 --port 8000
```

Check readiness:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Endpoints:

- `GET /health`
- `POST /predict/rul`
- `POST /predict/failure-risk`
- `POST /predict/batch`
- `GET /metrics`

Prediction requests use a fixed-length two-dimensional `sensor_window` shaped as:

```text
window_size x feature_count
```

The default API configuration expects `30 x 24` windows when no model metadata is loaded.

Single-window RUL prediction:

```powershell
$payload = @{
  sensor_window = $window
  metadata = @{ unit_number = 1 }
} | ConvertTo-Json -Depth 6

Invoke-RestMethod http://localhost:8000/predict/rul -Method Post -Body $payload -ContentType "application/json"
```

Single-window failure-risk prediction:

```powershell
Invoke-RestMethod http://localhost:8000/predict/failure-risk -Method Post -Body $payload -ContentType "application/json"
```

Batch prediction:

```powershell
$batchPayload = @{
  model_type = "rul"
  sensor_windows = @($window, $window)
} | ConvertTo-Json -Depth 6

Invoke-RestMethod http://localhost:8000/predict/batch -Method Post -Body $batchPayload -ContentType "application/json"
```

Model paths can also be overridden with environment variables such as `RUL_MODEL_PATH`, `FAILURE_RISK_MODEL_PATH`, and `REFERENCE_STATS_PATH`.

## Docker

```powershell
docker compose up --build
```

Mounts are configured for raw data, processed data, local models, and local reports.

## Tests

The test suite uses tiny synthetic C-MAPSS-like data and does not require the real dataset.

```powershell
pytest
```

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)
