# Architecture

## Overview

The project is a local Python package organized around the C-MAPSS predictive
maintenance workflow: parse turbofan time-series data, build RUL targets and
sensor windows, train baseline models, evaluate held-out test engines, serve
local model bundles, and generate lightweight monitoring outputs.

```mermaid
flowchart LR
    A["C-MAPSS raw text files"] --> B["Parser"]
    B --> C["RUL labels"]
    C --> D["Sensor windows"]
    D --> E["Baseline training"]
    E --> F["Model bundles"]
    E --> G["Reference statistics"]
    F --> H["Held-out evaluation"]
    F --> I["FastAPI inference"]
    G --> J["Drift report"]
    I --> K["Prometheus metrics"]
```

## Package Areas

`industrial_maintenance_mlops.data` reads C-MAPSS train, test, and RUL files and
generates RUL labels.

`industrial_maintenance_mlops.features` selects feature columns, builds
fixed-length windows, flattens windows for sklearn estimators, and derives
failure-risk labels.

`industrial_maintenance_mlops.models` trains sklearn baselines and saves joblib
model bundles with inference metadata.

`industrial_maintenance_mlops.training` runs the FD001-style training workflow,
including engine-level validation split, metrics output, model bundle saving,
and reference-statistics creation.

`industrial_maintenance_mlops.evaluation` evaluates saved model bundles on
held-out C-MAPSS test engines using the final available window per engine.

`industrial_maintenance_mlops.inference` loads saved bundles and validates
incoming windows before prediction.

`industrial_maintenance_mlops.api` exposes FastAPI endpoints for health checks,
RUL prediction, failure-risk prediction, batch prediction, and metrics.

`industrial_maintenance_mlops.monitoring` provides API counters/histograms,
single-window drift checks, and JSON drift reports comparing final test-engine
windows against training reference statistics.

## Local Artifacts

Raw data is expected under `data/raw/CMAPSSData/`. Training and evaluation write
local artifacts under `models/` and `reports/metrics/`. These generated files
are ignored by Git.

Typical generated files:

- `models/rul_regressor.joblib`
- `models/failure_risk_classifier.joblib`
- `models/reference_stats.json`
- `reports/metrics/training_metrics_FD001.json`
- `reports/metrics/test_metrics_FD001.json`
- `reports/metrics/drift_report_FD001.json`
