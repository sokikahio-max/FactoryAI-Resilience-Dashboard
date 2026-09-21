from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"


class DataQualityStatus(str, Enum):
    ACCEPTABLE = "acceptable"
    MODERATE = "moderate"
    SEVERE = "severe"


class PlannerAction(str, Enum):
    TRAIN = "train"
    RECONFIGURE = "reconfigure"
    ABSTAIN = "abstain"
    HUMAN_REVIEW = "human_review"


class ControllerState(str, Enum):
    ACCEPT = "accept"
    RECONFIGURE = "reconfigure"
    ABSTAIN = "abstain"
    ESCALATE = "human_escalation"


ALLOWED_H2O_ALGOS = {"GBM", "DRF", "GLM", "StackedEnsemble"}
ALLOWED_FEATURE_STRATEGIES = {
    "none",
    "drop_suspect_features",
    "remove_low_information_features",
}
ALLOWED_IMBALANCE_STRATEGIES = {"none", "balance_classes"}
ALLOWED_METRICS = {"AUCPR", "AUC", "logloss", "mean_per_class_error", "RMSE", "MAE"}

TASK_METRICS = {
    TaskType.BINARY_CLASSIFICATION: {"AUCPR", "AUC", "logloss"},
    TaskType.MULTICLASS_CLASSIFICATION: {"mean_per_class_error", "logloss"},
    TaskType.REGRESSION: {"RMSE", "MAE"},
    TaskType.FORECASTING: {"RMSE", "MAE"},
}


class PlannerDecision(BaseModel):
    """Machine-readable, auditable LLM output. No chain-of-thought is requested."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    primary_metric: Literal["AUCPR", "AUC", "logloss", "mean_per_class_error", "RMSE", "MAE"]
    candidate_algorithms: list[str] = Field(min_length=1, max_length=5)
    class_imbalance_strategy: Literal["none", "balance_classes"]
    feature_strategy: Literal[
        "none", "drop_suspect_features", "remove_low_information_features"
    ]
    drop_features: list[str] = Field(default_factory=list, max_length=8)
    data_quality_status: DataQualityStatus
    recommended_action: PlannerAction
    confidence: float = Field(ge=0.0, le=1.0)
    short_rationale: str = Field(min_length=1, max_length=500)

    @field_validator("candidate_algorithms")
    @classmethod
    def validate_algorithms(cls, value: list[str]) -> list[str]:
        unknown = set(value) - ALLOWED_H2O_ALGOS
        if unknown:
            raise ValueError(f"Unsupported algorithms: {sorted(unknown)}")
        return value

    @model_validator(mode="after")
    def validate_metric_for_task(self):
        allowed = TASK_METRICS[self.task_type]
        if self.primary_metric not in allowed:
            raise ValueError(
                f"Metric {self.primary_metric!r} is not appropriate for task {self.task_type.value!r}; "
                f"allowed metrics are {sorted(allowed)}"
            )
        return self


class ExecutionConfig(BaseModel):
    """Validated configuration that may be sent to an ML backend."""

    model_config = ConfigDict(extra="forbid")

    include_algos: list[str]
    balance_classes: bool
    sort_metric: str
    drop_features: list[str]
    max_models: int = Field(gt=0)
    seed: int
    nfolds: int = Field(ge=2)


class PlannerTelemetry(BaseModel):
    mode: str
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    schema_valid: bool = True
    error: str | None = None
