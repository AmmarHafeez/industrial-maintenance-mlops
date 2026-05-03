from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sensor_window: list[list[float]] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictionRequest(BaseModel):
    sensor_windows: list[list[list[float]]] = Field(..., min_length=1)
    model_type: Literal["rul", "failure-risk"] = "rul"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    model_version: str
    prediction: Any
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
