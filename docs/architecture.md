# Architecture

## Overview

The project is organized as a Python package with separate modules for data ingestion, feature generation, baseline modeling, training orchestration, inference, API serving, monitoring, and evaluation.

```mermaid
flowchart LR
    A["C-MAPSS text files"] --> B["Parser"]
    B --> C["RUL target generation"]
    C --> D["Window builder"]
    D --> E["Baseline training"]
    E --> F["Saved model bundles"]
    F --> G["Prediction service"]
    G --> H["FastAPI endpoints"]
    H --> I["Prometheus metrics"]
    H --> J["Drift statistics"]
```

## Components

`industrial_maintenance_mlops.data` parses C-MAPSS train, test, and RUL files and generates RUL labels for engine trajectories.

`industrial_maintenance_mlops.features` builds fixed-length windows from ordered time-series data and converts RUL labels into failure-risk classification targets.

`industrial_maintenance_mlops.models` contains deterministic sklearn baseline training helpers and model persistence utilities.

`industrial_maintenance_mlops.training` loads configuration, runs the baseline training workflow, writes metrics, saves model bundles, and stores reference statistics for drift checks.

`industrial_maintenance_mlops.inference` loads saved model bundles and validates incoming prediction windows before calling the underlying estimator.

`industrial_maintenance_mlops.api` exposes FastAPI endpoints for health checks, RUL prediction, failure-risk prediction, batch prediction, and Prometheus-compatible metrics.

`industrial_maintenance_mlops.monitoring` tracks request and prediction metrics and compares incoming windows against saved reference mean and standard deviation.

## Data Flow

Raw C-MAPSS files are expected in `data/raw/CMAPSSData/`. Training reads those files, derives labels, builds windows, trains baseline models, and writes local artifacts under `models/` and `reports/metrics/`.

The API loads model bundles from `models/`. Each prediction request must provide a numeric sensor window with the configured shape. The service returns model version, prediction output, request latency, and metadata.
