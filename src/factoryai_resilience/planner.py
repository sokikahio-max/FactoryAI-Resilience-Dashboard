from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .config import PlannerSettings
from .quality import QualityReport
from .schemas import (
    DataQualityStatus,
    PlannerAction,
    PlannerDecision,
    PlannerTelemetry,
    TaskType,
)


@dataclass(frozen=True)
class PlanningContext:
    problem_description: str
    target: str
    predictors: list[str]
    positive_rate: float
    quality: QualityReport
    previous_metrics: dict[str, float] | None = None
    retry_index: int = 0
    target_cardinality: int | None = 2
    target_is_continuous: bool = False
    previous_execution_config: dict[str, object] | None = None


class BasePlanner:
    def plan(self, context: PlanningContext) -> tuple[PlannerDecision, PlannerTelemetry]:
        raise NotImplementedError


class MockPlanner(BasePlanner):
    """Deterministic, token-free planner for code tests and pipeline debugging."""

    def __init__(self, settings: PlannerSettings):
        self.settings = settings

    def plan(self, context: PlanningContext) -> tuple[PlannerDecision, PlannerTelemetry]:
        start = time.perf_counter()
        imbalance = context.positive_rate < 0.15
        quality = context.quality.severity

        text = context.problem_description.lower()
        if "forecast" in text or "next hour" in text or "next shift" in text:
            task_type = TaskType.FORECASTING
        elif context.target_is_continuous:
            task_type = TaskType.REGRESSION
        elif context.target_cardinality is not None and context.target_cardinality > 2:
            task_type = TaskType.MULTICLASS_CLASSIFICATION
        else:
            task_type = TaskType.BINARY_CLASSIFICATION

        if task_type == TaskType.BINARY_CLASSIFICATION:
            primary_metric = "AUCPR"
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            primary_metric = "mean_per_class_error"
        else:
            primary_metric = "RMSE"

        if quality == DataQualityStatus.SEVERE:
            action = PlannerAction.HUMAN_REVIEW
        elif context.retry_index > 0:
            action = PlannerAction.RECONFIGURE
        else:
            # The first pass remains comparable across non-severe conditions.
            # Suspect-feature removal is reserved for a validation-triggered retry.
            action = PlannerAction.TRAIN

        drop = (
            context.quality.suspicious_features[:3]
            if context.retry_index > 0 and quality == DataQualityStatus.MODERATE
            else []
        )
        algorithms = ["GBM", "DRF", "GLM"] if context.retry_index == 0 else ["GBM", "DRF"]
        decision = PlannerDecision(
            task_type=task_type,
            primary_metric=primary_metric,
            candidate_algorithms=algorithms,
            class_imbalance_strategy=(
                "balance_classes" if task_type == TaskType.BINARY_CLASSIFICATION and imbalance else "none"
            ),
            feature_strategy="drop_suspect_features" if drop else "none",
            drop_features=drop,
            data_quality_status=quality,
            recommended_action=action,
            confidence=0.90 if quality != DataQualityStatus.SEVERE else 0.75,
            short_rationale=(
                "Binary failure prediction is imbalanced; prioritize AUCPR and failure detection. "
                "Use only allowlisted model families and remove objectively suspect features when needed."
            ),
        )
        telemetry = PlannerTelemetry(
            mode="mock",
            model="deterministic_mock",
            requested_model="deterministic_mock",
            provider_returned_model="deterministic_mock",
            temperature_sent=None,
            latency_seconds=time.perf_counter() - start,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
        )
        return decision, telemetry


class OpenAIPlanner(BasePlanner):
    """Structured-output planner. Raw industrial rows are never sent to the API."""

    def __init__(self, settings: PlannerSettings):
        self.settings = settings
        from openai import OpenAI  # imported lazily so mock mode needs no API package at runtime

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _prompt(self, context: PlanningContext) -> str:
        q = context.quality
        previous = context.previous_metrics or {}
        return f"""
You are a constrained planning component in an industrial machine-learning research workflow.
Return only the fields in the supplied schema. Do not provide hidden chain-of-thought and do not claim causal root causes.

Experimental task:
- problem: {context.problem_description}
- target: {context.target}
- target cardinality (if known): {context.target_cardinality}
- target is continuous: {context.target_is_continuous}
- predictor names (sanitized metadata only): {context.predictors}
- positive-class prevalence in training data: {context.positive_rate:.6f}

Data-quality summary computed locally:
- status: {q.severity.value}
- maximum missing rate: {q.max_missing_rate:.6f}
- maximum standardized mean shift: {q.max_standardized_mean_shift:.6f}
- OOD rate: {q.ood_rate:.6f}
- suspect features: {q.suspicious_features}

Previous validation metrics (if this is a retry): {previous}
Previous executable configuration (if this is a retry): {context.previous_execution_config}
Retry index: {context.retry_index}

Rules:
1. Infer task_type from the problem and target metadata. Choose a metric appropriate to that task:
   - binary classification: prefer AUCPR when the positive class is imbalanced; AUC or logloss are also allowed.
   - multiclass classification: use mean_per_class_error or logloss.
   - regression: use RMSE or MAE; never use classification metrics such as logloss, AUCPR, or AUC.
   - forecasting of a continuous target: use RMSE or MAE; never use classification metrics.
2. candidate_algorithms must be selected only from GBM, DRF, GLM, StackedEnsemble. The current H2O runtime is Windows, where H2O-3 XGBoost is unavailable.
3. Never invent executable Python or arbitrary hyperparameters.
4. If data quality is severe, prefer human_review rather than blind retraining.
5. At retry index 0, use train unless data quality is severe; do not remove suspect features before the first validation result.
6. At retry index > 0, this is a recovery decision after validation has already shown degradation. Do not return train. Choose reconfigure, abstain, or human_review.
7. If you choose reconfigure, the proposed executable strategy MUST differ meaningfully from the previous executable configuration in at least one auditable field: candidate_algorithms, class_imbalance_strategy, primary_metric, or drop_features. A seed-only rerun is not a valid LLM-guided recovery.
8. If data quality is moderate and suspect features are provided, you may use drop_suspect_features only for those feature names. If no defensible executable change is available, choose abstain or human_review instead of repeating the same configuration.
9. Recommendations are diagnostic/modeling suggestions, not causal findings.
""".strip()

    def plan(self, context: PlanningContext) -> tuple[PlannerDecision, PlannerTelemetry]:
        start = time.perf_counter()
        requested_model = os.getenv("OPENAI_MODEL", self.settings.model)
        # No temperature parameter is sent in this structured Responses API call.
        # This is recorded explicitly rather than inferring a provider default.
        temperature_sent = None
        try:
            response = self.client.responses.parse(
                model=requested_model,
                input=[
                    {
                        "role": "system",
                        "content": "Produce a safe, auditable structured ML planning decision for the supplied industrial metadata.",
                    },
                    {"role": "user", "content": self._prompt(context)},
                ],
                text_format=PlannerDecision,
            )
            decision = response.output_parsed
            if decision is None:
                raise ValueError("Structured output was empty or refused.")

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", None) if usage else None
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
            estimated_cost = None
            if input_tokens is not None and output_tokens is not None:
                estimated_cost = (
                    input_tokens / 1_000_000 * self.settings.input_cost_per_mtok
                    + output_tokens / 1_000_000 * self.settings.output_cost_per_mtok
                )

            provider_model = str(getattr(response, "model", None) or requested_model)
            return decision, PlannerTelemetry(
                mode="openai",
                model=provider_model,
                requested_model=requested_model,
                provider_returned_model=provider_model,
                response_id=str(getattr(response, "id", "")) or None,
                temperature_sent=temperature_sent,
                latency_seconds=time.perf_counter() - start,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost,
                schema_valid=True,
            )
        except Exception as exc:
            telemetry = PlannerTelemetry(
                mode="openai",
                model=requested_model,
                requested_model=requested_model,
                provider_returned_model=None,
                temperature_sent=temperature_sent,
                latency_seconds=time.perf_counter() - start,
                schema_valid=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError(telemetry.model_dump_json()) from exc


def build_planner(settings: PlannerSettings) -> BasePlanner:
    if settings.mode == "mock":
        return MockPlanner(settings)
    if settings.mode == "openai":
        return OpenAIPlanner(settings)
    raise ValueError(f"Unsupported planner mode {settings.mode!r}")
