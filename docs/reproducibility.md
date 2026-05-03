# Reproducibility

This page gives a reproducible local PowerShell workflow for FD001. Commands use
Python 3.12 and assume the repository root is the current directory.

## 1. Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

The project uses a `src` layout. Editable installation is the simplest way to
make `python -m` commands work. As an alternative, set `PYTHONPATH` before
running commands:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## 3. Place C-MAPSS Files

Raw NASA C-MAPSS / Turbofan Engine Degradation Simulation files are not
committed to this repository. Place the extracted files under:

```text
data/raw/CMAPSSData/
```

For FD001, the workflow expects:

```text
data/raw/CMAPSSData/train_FD001.txt
data/raw/CMAPSSData/test_FD001.txt
data/raw/CMAPSSData/RUL_FD001.txt
```

## 4. Set PYTHONPATH If Needed

Skip this step if `pip install -e .` completed successfully.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## 5. Run Tests

The test suite uses synthetic C-MAPSS-like data and does not require the real
dataset.

```powershell
pytest
```

## 6. Train Baseline Models

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

Training reads `train_FD001.txt`, creates clipped RUL labels, builds windows,
splits by engine id, trains the RUL regressor and failure-risk classifier, and
writes local artifacts:

```text
models/rul_regressor.joblib
models/failure_risk_classifier.joblib
models/reference_stats.json
reports/metrics/training_metrics_FD001.json
```

## 7. Evaluate Held-Out Test Engines

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

Evaluation reads `test_FD001.txt` and `RUL_FD001.txt`, extracts the final
fixed-length window for each test engine, clips true final RUL, and writes:

```text
reports/metrics/test_metrics_FD001.json
```

## 8. Run Drift Report

```powershell
python -m industrial_maintenance_mlops.monitoring.drift_report `
  --dataset-id FD001 `
  --raw-dir data/raw/CMAPSSData `
  --reference-stats models/reference_stats.json `
  --output reports/metrics/drift_report_FD001.json `
  --window-size 30 `
  --max-rul 125
```

The drift report compares final test-engine window statistics with training
reference statistics and writes:

```text
reports/metrics/drift_report_FD001.json
```

This is a lightweight statistical check, not a full production monitoring
system.

## 9. Start the API

```powershell
uvicorn industrial_maintenance_mlops.api.app:app --host 0.0.0.0 --port 8000
```

Check readiness from another PowerShell session:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

The API loads model paths from `configs/api.yaml` by default. Override paths
with `RUL_MODEL_PATH`, `FAILURE_RISK_MODEL_PATH`, and `REFERENCE_STATS_PATH`
when needed.

## Artifact Policy

Raw data, processed data, trained models, metrics JSON files, reports, figures,
and local artifacts are generated locally and ignored by Git.

Relevant ignored outputs include:

- `data/raw/`
- `data/processed/`
- `models/`
- `reports/metrics/`
- `reports/figures/`
- `reports/artifacts/`

## Documented FD001 Runs

Validation uses an engine-level split from `train_FD001.txt`. Held-out test
evaluation uses `test_FD001.txt` and `RUL_FD001.txt`. Drift reporting compares
final test-engine windows with `models/reference_stats.json`.

See [Results](results.md) for the documented FD001 metrics.
