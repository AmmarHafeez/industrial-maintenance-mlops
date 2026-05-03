from __future__ import annotations

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


@dataclass
class PrometheusMetrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.request_counter = Counter(
            "http_requests_total",
            "Total HTTP requests.",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )
        self.prediction_counter = Counter(
            "prediction_requests_total",
            "Total prediction requests.",
            ["model_type"],
            registry=self.registry,
        )
        self.latency_histogram = Histogram(
            "prediction_latency_seconds",
            "Prediction latency in seconds.",
            ["endpoint"],
            registry=self.registry,
        )
        self.validation_error_counter = Counter(
            "validation_errors_total",
            "Total validation errors.",
            ["endpoint"],
            registry=self.registry,
        )

    def record_request(self, method: str, endpoint: str, status: int) -> None:
        self.request_counter.labels(method=method, endpoint=endpoint, status=str(status)).inc()

    def record_prediction(self, model_type: str) -> None:
        self.prediction_counter.labels(model_type=model_type).inc()

    def record_latency(self, endpoint: str, seconds: float) -> None:
        self.latency_histogram.labels(endpoint=endpoint).observe(seconds)

    def record_validation_error(self, endpoint: str) -> None:
        self.validation_error_counter.labels(endpoint=endpoint).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
