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

Processed data should be written under `data/processed/`. Raw and processed data directories are ignored by Git.

## Configuration

Configuration lives in:

- `configs/data.yaml`
- `configs/model.yaml`
- `configs/training.yaml`
- `configs/api.yaml`

The default RUL clipping value is `125`. Training uses deterministic `random_state` values where sklearn estimators support them.

## Training Run

```powershell
python -m industrial_maintenance_mlops.training.pipeline --config-dir configs
```

Expected local outputs:

- `models/rul_regressor.joblib`
- `models/failure_risk_classifier.joblib`
- `models/reference_stats.json`
- `reports/metrics/training_metrics.json`

These generated outputs are ignored by Git.

## Test Data

The test suite uses synthetic C-MAPSS-like files created in temporary directories. It does not require the real C-MAPSS dataset.
