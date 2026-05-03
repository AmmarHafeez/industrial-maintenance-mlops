# Reproducibility

## Environment

Use Python 3.12 and install dependencies from `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
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

The default RUL clipping value is `125`, and the default failure-risk threshold is `30`. Training uses deterministic `random_state` values where sklearn estimators support them.

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

## Test Data

The test suite uses synthetic C-MAPSS-like files created in temporary directories. It does not require the real C-MAPSS dataset.
