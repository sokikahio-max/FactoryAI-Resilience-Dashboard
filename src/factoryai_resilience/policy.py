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



def cost_inflation_ratio(current_cost: float, reference_cost: float, eps: float = 1e-12) -> float:
    """Return a degradation-sensitive cost ratio, including a zero-reference case.

    If the clean reference cost is zero and the current cost is positive, returning
    infinity ensures that deterioration is not concealed by a neutral ratio.
    """
    if reference_cost > eps:
        return float(current_cost / reference_cost)
    if current_cost <= eps:
        return 1.0
    return float("inf")


def recovery_candidate_is_acceptable(
    *,
    candidate: ClassificationMetrics,
    incumbent: ClassificationMetrics,
    eps: float = 1e-12,
) -> bool:
    """Joint validation gate used for R2 recovery selection.

    A candidate must not reduce PR-AUC, increase asymmetric classification cost,
    or reduce failure-class recall relative to the incumbent model.
    """
    return bool(
        candidate.pr_auc + eps >= incumbent.pr_auc
        and candidate.expected_cost <= incumbent.expected_cost + eps
        and candidate.recall + eps >= incumbent.recall
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
    cost_inflation = cost_inflation_ratio(
        current.expected_cost, clean_reference.expected_cost
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
