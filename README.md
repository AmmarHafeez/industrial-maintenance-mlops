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

## API

Start the FastAPI service after models have been trained:

```powershell
uvicorn industrial_maintenance_mlops.api.app:app --host 0.0.0.0 --port 8000
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
