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


class RulPredictionResponse(BaseModel):
    model_version: str
    predicted_rul: float
    latency_ms: float
    input_metadata: dict[str, Any] = Field(default_factory=dict)


class FailureRiskPredictionResponse(BaseModel):
    model_version: str
    failure_risk_probability: float | None
    predicted_high_risk: bool
    latency_ms: float
    input_metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictionResponse(BaseModel):
    model_version: str
    model_type: Literal["rul", "failure-risk"]
    predictions: list[Any]
    latency_ms: float
    input_metadata: dict[str, Any] = Field(default_factory=dict)
