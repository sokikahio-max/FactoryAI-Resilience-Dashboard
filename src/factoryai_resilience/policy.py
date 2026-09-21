from __future__ import annotations

from dataclasses import dataclass

from .config import RecoveryPolicy
from .metrics import ClassificationMetrics
from .quality import QualityReport
from .schemas import (
    ALLOWED_H2O_ALGOS,
    ControllerState,
    DataQualityStatus,
    ExecutionConfig,
    PlannerAction,
    PlannerDecision,
)


@dataclass(frozen=True)
class ControllerDecision:
    state: ControllerState
    reason: str
    primary_metric_retention: float | None
    cost_inflation: float | None


def decision_to_execution_config(
    decision: PlannerDecision,
    *,
    predictors: list[str],
    max_models: int,
    seed: int,
    nfolds: int,
    exclude_deep_learning: bool = True,
) -> ExecutionConfig:
    """Convert LLM output to an allowlisted executable configuration.

    This is deliberately deterministic. The LLM never emits arbitrary code.
    """
    include = [a for a in decision.candidate_algorithms if a in ALLOWED_H2O_ALGOS]
    if exclude_deep_learning:
        include = [a for a in include if a != "DeepLearning"]
    if not include:
        include = ["GBM", "DRF", "GLM"]

    predictor_set = set(predictors)
    drop_features = [f for f in decision.drop_features if f in predictor_set]

    return ExecutionConfig(
        include_algos=include,
        balance_classes=decision.class_imbalance_strategy == "balance_classes",
        sort_metric=decision.primary_metric,
        drop_features=drop_features,
        max_models=max_models,
        seed=seed,
        nfolds=nfolds,
    )



def meaningful_execution_change(before: ExecutionConfig, after: ExecutionConfig) -> bool:
    """Return True only when recovery changes an auditable modeling decision.

    Random seed, model budget, and fold count are intentionally ignored: a seed-only
    rerun is not evidence of LLM-guided reconfiguration.
    """
    return any(
        [
            before.include_algos != after.include_algos,
            before.balance_classes != after.balance_classes,
            before.sort_metric != after.sort_metric,
            sorted(before.drop_features) != sorted(after.drop_features),
        ]
    )

def standard_automl_config(*, max_models: int, seed: int, nfolds: int) -> ExecutionConfig:
    return ExecutionConfig(
        include_algos=["GBM", "DRF", "GLM", "StackedEnsemble"],
        balance_classes=False,
        sort_metric="AUCPR",
        drop_features=[],
        max_models=max_models,
        seed=seed,
        nfolds=nfolds,
    )


def deterministic_retry_config(
    base: ExecutionConfig,
    quality: QualityReport,
    predictors: list[str],
    max_models: int,
) -> ExecutionConfig:
    """Simple non-LLM retry rule used by ablation B2."""
    drop = [f for f in quality.suspicious_features if f in predictors][:2]
    return ExecutionConfig(
        include_algos=["GBM", "DRF", "GLM"],
        balance_classes=True,
        sort_metric="AUCPR",
        drop_features=drop,
        max_models=max_models,
        seed=base.seed + 10_000,
        nfolds=base.nfolds,
    )


def controller(
    *,
    quality: QualityReport,
    current: ClassificationMetrics,
    clean_reference: ClassificationMetrics | None,
    recovery_policy: RecoveryPolicy,
    retry_index: int,
) -> ControllerDecision:
    if quality.severity == DataQualityStatus.SEVERE:
        return ControllerDecision(
            ControllerState.ESCALATE,
            "Severe data-quality degradation; retraining is not considered a safe default action.",
            None,
            None,
        )

    if clean_reference is None:
        return ControllerDecision(
            ControllerState.ACCEPT,
            "Clean-condition strategy finalized using validation data; final test remains untouched.",
            None,
            None,
        )

    retention = current.pr_auc / clean_reference.pr_auc if clean_reference.pr_auc else 0.0
    cost_inflation = (
        current.expected_cost / clean_reference.expected_cost
        if clean_reference.expected_cost > 1e-12
        else 1.0
    )

    degraded = (
        retention < recovery_policy.min_primary_metric_retention
        or cost_inflation > recovery_policy.max_cost_inflation
    )

    if not degraded:
        return ControllerDecision(
            ControllerState.ACCEPT,
            "Validation performance remains within the pre-specified retention/cost envelope.",
            retention,
            cost_inflation,
        )

    if retry_index < recovery_policy.max_retries:
        return ControllerDecision(
            ControllerState.RECONFIGURE,
            "Validation performance degraded but data quality is not severe; one constrained recovery attempt is allowed.",
            retention,
            cost_inflation,
        )

    if quality.severity == DataQualityStatus.MODERATE:
        return ControllerDecision(
            ControllerState.ESCALATE,
            "Recovery budget exhausted under persistent data-quality warning; human review required.",
            retention,
            cost_inflation,
        )

    return ControllerDecision(
        ControllerState.ABSTAIN,
        "Recovery budget exhausted and predictive performance remains below the validation-derived operating envelope.",
        retention,
        cost_inflation,
    )
