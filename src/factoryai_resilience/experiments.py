from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
from typing import Any

import pandas as pd

from .backends import BaseBackend, ModelHandle
from .config import AppConfig, PerturbationSpec
from .data import DataSplit, class_distribution, three_way_stratified_split
from .metrics import ClassificationMetrics, compute_metrics, resilience_metrics
from .perturbations import apply_perturbation
from .planner import BasePlanner, PlanningContext
from .policy import (
    controller,
    decision_to_execution_config,
    deterministic_retry_config,
    cost_inflation_ratio,
    meaningful_execution_change,
    recovery_candidate_is_acceptable,
    standard_automl_config,
)
from .quality import assess_data_quality
from .schemas import ControllerState, PlannerAction


METHODS = {
    "B0_manual_gbm": "Fixed manually configured H2O GBM baseline",
    "B1_standard_h2o_automl": "Standard AutoML without LLM or retry",
    "B2_rule_retry": "AutoML with deterministic threshold-based retry, no LLM",
    "B3_llm_no_retry": "LLM-guided AutoML without retry",
    "P_full_framework": "Structured LLM + failure-aware controller + constrained retry",
}


def _evaluate_validation(
    backend: BaseBackend,
    model: ModelHandle,
    frame: pd.DataFrame,
    target: str,
    config: AppConfig,
) -> ClassificationMetrics:
    prob = backend.predict_proba(model, frame)
    return compute_metrics(
        frame[target].to_numpy(), prob, config.cost_policy, threshold=None, abstention_band=None
    )


def _evaluate_test(
    backend: BaseBackend,
    model: ModelHandle,
    frame: pd.DataFrame,
    target: str,
    config: AppConfig,
    threshold: float,
    abstention_band: float,
) -> ClassificationMetrics:
    prob = backend.predict_proba(model, frame)
    return compute_metrics(
        frame[target].to_numpy(),
        prob,
        config.cost_policy,
        threshold=threshold,
        abstention_band=abstention_band,
    )


def _fit_initial(
    method: str,
    backend: BaseBackend,
    planner: BasePlanner,
    split: DataSplit,
    validation_frame: pd.DataFrame,
    quality,
    config: AppConfig,
    seed: int,
):
    exp = config.experiment
    telemetry = None
    planner_decision = None

    if method == "B0_manual_gbm":
        model = backend.fit_manual_baseline(
            split.train, validation_frame, split.target, split.predictors, seed
        )
        return model, planner_decision, telemetry

    if method == "B1_standard_h2o_automl":
        execution = standard_automl_config(
            max_models=exp.max_models_total, seed=seed, nfolds=exp.nfolds
        )
        model = backend.fit_automl(
            split.train, validation_frame, split.target, split.predictors, execution
        )
        return model, planner_decision, telemetry

    if method == "B2_rule_retry":
        execution = standard_automl_config(
            max_models=exp.initial_models_recovery_methods,
            seed=seed,
            nfolds=exp.nfolds,
        )
        model = backend.fit_automl(
            split.train, validation_frame, split.target, split.predictors, execution
        )
        return model, planner_decision, telemetry

    if method in {"B3_llm_no_retry", "P_full_framework"}:
        distribution = class_distribution(split.train, split.target)
        planner_decision, telemetry = planner.plan(
            PlanningContext(
                problem_description=(
                    "Predict machine-failure events from process and sensor measurements for "
                    "maintenance decision support, with emphasis on detecting the minority failure class."
                ),
                target=split.target,
                predictors=split.predictors,
                positive_rate=float(distribution["positive_rate"]),
                quality=quality,
                retry_index=0,
            )
        )
        if method == "P_full_framework" and planner_decision.recommended_action in {
            PlannerAction.ABSTAIN,
            PlannerAction.HUMAN_REVIEW,
        }:
            return None, planner_decision, telemetry

        budget = (
            exp.max_models_total
            if method == "B3_llm_no_retry"
            else exp.initial_models_recovery_methods
        )
        execution = decision_to_execution_config(
            planner_decision,
            predictors=split.predictors,
            max_models=budget,
            seed=seed,
            nfolds=exp.nfolds,
            exclude_deep_learning=exp.exclude_deep_learning_for_reproducibility,
        )
        model = backend.fit_automl(
            split.train, validation_frame, split.target, split.predictors, execution
        )
        return model, planner_decision, telemetry

    raise KeyError(method)


def run_method_condition(
    *,
    method: str,
    split: DataSplit,
    perturbation: PerturbationSpec,
    backend: BaseBackend,
    planner: BasePlanner,
    config: AppConfig,
    seed: int,
    clean_validation_reference: ClassificationMetrics | None,
    clean_test_reference: ClassificationMetrics | None,
) -> dict[str, Any]:
    start_total = time.perf_counter()

    validation_frame, applied_val = apply_perturbation(
        split.validation,
        perturbation.name,
        perturbation.severity,
        perturbation.magnitude,
        perturbation.features,
        seed=seed + 1_000,
        target=split.target,
        reference_df=split.train,
    )
    test_frame, applied_test = apply_perturbation(
        split.test,
        perturbation.name,
        perturbation.severity,
        perturbation.magnitude,
        perturbation.features,
        seed=seed + 2_000,
        target=split.target,
        reference_df=split.train,
    )

    quality = assess_data_quality(split.train, validation_frame, split.target, config.quality_policy)

    initial_model, initial_plan, initial_telemetry = _fit_initial(
        method,
        backend,
        planner,
        split,
        validation_frame,
        quality,
        config,
        seed,
    )

    row: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "method_description": METHODS[method],
        "condition": perturbation.condition_id,
        "perturbation": perturbation.name,
        "severity": perturbation.severity,
        "magnitude": perturbation.magnitude,
        "quality_severity": quality.severity.value,
        "quality_max_missing_rate": quality.max_missing_rate,
        "quality_max_standardized_mean_shift": quality.max_standardized_mean_shift,
        "quality_ood_rate": quality.ood_rate,
        "quality_suspicious_features": json.dumps(quality.suspicious_features),
        "retry_count": 0,
        "initial_model_id": None,
        "final_model_id": None,
        "initial_training_seconds": None,
        "recovery_training_seconds": 0.0,
        "recovery_model_selected": False,
        "initial_execution_config": None,
        "retry_execution_config": None,
        "retry_execution_changed": None,
        "planner_calls": 1 if initial_plan is not None else 0,
        "planner_latency_seconds": initial_telemetry.latency_seconds if initial_telemetry else 0.0,
        "planner_input_tokens": initial_telemetry.input_tokens if initial_telemetry else 0,
        "planner_output_tokens": initial_telemetry.output_tokens if initial_telemetry else 0,
        "planner_estimated_cost_usd": initial_telemetry.estimated_cost_usd if initial_telemetry else 0.0,
        "initial_planner_requested_model": initial_telemetry.requested_model if initial_telemetry else None,
        "initial_planner_provider_model": initial_telemetry.provider_returned_model if initial_telemetry else None,
        "initial_planner_response_id": initial_telemetry.response_id if initial_telemetry else None,
        "initial_planner_temperature_sent": initial_telemetry.temperature_sent if initial_telemetry else None,
        "initial_planner_action": initial_plan.recommended_action.value if initial_plan else None,
        "initial_planner_algorithms": json.dumps(initial_plan.candidate_algorithms) if initial_plan else None,
        "initial_planner_drop_features": json.dumps(initial_plan.drop_features) if initial_plan else None,
    }

    # Full framework can refuse unsafe automation before model search.
    if initial_model is None:
        final_state = (
            ControllerState.ESCALATE
            if initial_plan and initial_plan.recommended_action == PlannerAction.HUMAN_REVIEW
            else ControllerState.ABSTAIN
        )
        row.update(
            {
                "final_state": final_state.value,
                "controller_reason": "Planner/quality gate stopped automatic modeling before training.",
                "total_runtime_seconds": time.perf_counter() - start_total,
            }
        )
        return row

    row["initial_model_id"] = initial_model.model_id
    row["initial_training_seconds"] = initial_model.training_seconds
    row["initial_execution_config"] = initial_model.execution_config.model_dump_json()
    before_val = _evaluate_validation(backend, initial_model, validation_frame, split.target, config)
    final_model = initial_model
    final_val = before_val
    final_state = ControllerState.ACCEPT
    controller_reason = "Single-pass baseline; no recovery controller."

    is_clean = perturbation.name == "clean"

    if method == "B2_rule_retry" and not is_clean and clean_validation_reference is not None:
        retention = before_val.pr_auc / clean_validation_reference.pr_auc if clean_validation_reference.pr_auc else 0.0
        cost_inflation = cost_inflation_ratio(
            before_val.expected_cost, clean_validation_reference.expected_cost
        )
        needs_retry = (
            retention < config.recovery_policy.min_primary_metric_retention
            or cost_inflation > config.recovery_policy.max_cost_inflation
        )
        if needs_retry:
            second_budget = config.experiment.max_models_total - config.experiment.initial_models_recovery_methods
            execution = deterministic_retry_config(
                initial_model.execution_config, quality, split.predictors, second_budget
            )
            recovery_model = backend.fit_automl(
                split.train, validation_frame, split.target, split.predictors, execution
            )
            candidate_val = _evaluate_validation(backend, recovery_model, validation_frame, split.target, config)
            # Deterministic baseline uses validation PR-AUC to choose between the two fixed attempts.
            row["retry_execution_config"] = recovery_model.execution_config.model_dump_json()
            if candidate_val.pr_auc >= before_val.pr_auc:
                final_model, final_val = recovery_model, candidate_val
                row["recovery_model_selected"] = True
            row["retry_count"] = 1
            row["recovery_training_seconds"] = recovery_model.training_seconds
            controller_reason = "Deterministic validation-triggered retry without LLM."

    elif method == "P_full_framework":
        decision = controller(
            quality=quality,
            current=before_val,
            clean_reference=None if is_clean else clean_validation_reference,
            recovery_policy=config.recovery_policy,
            retry_index=0,
        )
        final_state = decision.state
        controller_reason = decision.reason

        if decision.state == ControllerState.RECONFIGURE:
            distribution = class_distribution(split.train, split.target)
            second_plan, second_telemetry = planner.plan(
                PlanningContext(
                    problem_description=(
                        "Predict machine-failure events from process and sensor measurements for "
                        "maintenance decision support. Revise the constrained modeling strategy using "
                        "the validation feedback and local data-quality summary."
                    ),
                    target=split.target,
                    predictors=split.predictors,
                    positive_rate=float(distribution["positive_rate"]),
                    quality=quality,
                    previous_metrics=before_val.to_dict(),
                    retry_index=1,
                    previous_execution_config=initial_model.execution_config.model_dump(),
                )
            )
            row["planner_calls"] += 1
            row["planner_latency_seconds"] += second_telemetry.latency_seconds
            row["planner_input_tokens"] = (row["planner_input_tokens"] or 0) + (second_telemetry.input_tokens or 0)
            row["planner_output_tokens"] = (row["planner_output_tokens"] or 0) + (second_telemetry.output_tokens or 0)
            row["planner_estimated_cost_usd"] = (row["planner_estimated_cost_usd"] or 0.0) + (second_telemetry.estimated_cost_usd or 0.0)
            row["retry_planner_action"] = second_plan.recommended_action.value
            row["retry_planner_requested_model"] = second_telemetry.requested_model
            row["retry_planner_provider_model"] = second_telemetry.provider_returned_model
            row["retry_planner_response_id"] = second_telemetry.response_id
            row["retry_planner_temperature_sent"] = second_telemetry.temperature_sent
            row["retry_planner_algorithms"] = json.dumps(second_plan.candidate_algorithms)
            row["retry_planner_drop_features"] = json.dumps(second_plan.drop_features)

            if second_plan.recommended_action in {PlannerAction.ABSTAIN, PlannerAction.HUMAN_REVIEW}:
                final_state = (
                    ControllerState.ESCALATE
                    if second_plan.recommended_action == PlannerAction.HUMAN_REVIEW
                    else ControllerState.ABSTAIN
                )
                controller_reason = "Structured planner stopped recovery after receiving validation feedback."
            elif second_plan.recommended_action == PlannerAction.TRAIN:
                # TRAIN is an initial-stage action. A recovery call must either produce an
                # explicit reconfiguration or stop safely. This prevents a seed-only rerun
                # from being misreported as LLM-guided recovery.
                final_state = (
                    ControllerState.ESCALATE
                    if quality.severity.value == "moderate"
                    else ControllerState.ABSTAIN
                )
                controller_reason = "Recovery planner returned TRAIN instead of an explicit recovery action; no retry was executed."
            else:
                second_budget = config.experiment.max_models_total - config.experiment.initial_models_recovery_methods
                execution = decision_to_execution_config(
                    second_plan,
                    predictors=split.predictors,
                    max_models=second_budget,
                    seed=seed + 10_000,
                    nfolds=config.experiment.nfolds,
                    exclude_deep_learning=config.experiment.exclude_deep_learning_for_reproducibility,
                )
                row["retry_execution_config"] = execution.model_dump_json()
                changed = meaningful_execution_change(initial_model.execution_config, execution)
                row["retry_execution_changed"] = changed

                if not changed:
                    final_state = (
                        ControllerState.ESCALATE
                        if quality.severity.value == "moderate"
                        else ControllerState.ABSTAIN
                    )
                    controller_reason = "Planner proposed no meaningful executable configuration change; seed-only recovery was blocked."
                else:
                    recovery_model = backend.fit_automl(
                        split.train, validation_frame, split.target, split.predictors, execution
                    )
                    candidate_val = _evaluate_validation(backend, recovery_model, validation_frame, split.target, config)
                    # Select only with validation data; the test set remains untouched here.
                    row["retry_execution_config"] = recovery_model.execution_config.model_dump_json()
                    if recovery_candidate_is_acceptable(
                        candidate=candidate_val, incumbent=before_val
                    ):
                        final_model, final_val = recovery_model, candidate_val
                        row["recovery_model_selected"] = True
                    row["retry_count"] = 1
                    row["recovery_training_seconds"] = recovery_model.training_seconds

                    post = controller(
                        quality=quality,
                        current=final_val,
                        clean_reference=clean_validation_reference,
                        recovery_policy=config.recovery_policy,
                        retry_index=1,
                    )
                    final_state, controller_reason = post.state, post.reason

    # Test evaluation occurs only after all validation-driven decisions are locked.
    # We may evaluate both pre-recovery and final locked models now for scientific recovery analysis.
    before_test = _evaluate_test(
        backend, initial_model, test_frame, split.target, config,
        threshold=before_val.threshold, abstention_band=before_val.abstention_band
    )
    after_test = _evaluate_test(
        backend, final_model, test_frame, split.target, config,
        threshold=final_val.threshold, abstention_band=final_val.abstention_band
    )

    row["final_state"] = final_state.value
    row["controller_reason"] = controller_reason
    row["final_model_id"] = final_model.model_id

    for prefix, metrics in [
        ("validation_before", before_val),
        ("validation_after", final_val),
        ("test_before", before_test),
        ("test_after", after_test),
    ]:
        for key, value in metrics.to_dict().items():
            row[f"{prefix}_{key}"] = value

    if clean_test_reference is not None and not is_clean:
        r = resilience_metrics(
            clean=clean_test_reference.pr_auc,
            degraded_before=before_test.pr_auc,
            recovered=after_test.pr_auc,
        )
        row.update(r)
    elif is_clean:
        row.update({"performance_retention": 1.0, "recovery_gain": 0.0, "normalized_recovery_ratio": 0.0})

    row["total_runtime_seconds"] = time.perf_counter() - start_total
    return row


def run_experiment_suite(
    df: pd.DataFrame,
    *,
    config: AppConfig,
    backend: BaseBackend,
    planner: BasePlanner,
    output_csv: str | Path,
    resume: bool = False,
) -> pd.DataFrame:
    """Run the full experiment matrix with per-row checkpointing.

    When ``resume=True`` and ``output_csv`` already exists, completed
    (seed, method, condition) rows are preserved and skipped. This makes the
    long H2O/OpenAI experiment restartable after an interruption. Backend
    temporary objects are cleared after every independent run so H2O models
    and frames do not accumulate across hundreds of AutoML jobs.
    """
    output_csv = Path(output_csv)
    rows: list[dict[str, Any]] = []

    if resume and output_csv.exists():
        previous = pd.read_csv(output_csv)
        rows = previous.to_dict(orient="records")
        print(f"Resuming from {output_csv}: {len(rows)} completed rows loaded.")

    def key_of(row: dict[str, Any]) -> tuple[int, str, str]:
        return (int(row["seed"]), str(row["method"]), str(row["condition"]))

    completed = {key_of(row) for row in rows}

    def find_existing(seed: int, method: str, condition: str) -> dict[str, Any] | None:
        target_key = (int(seed), method, condition)
        for row in rows:
            if key_of(row) == target_key:
                return row
        return None

    def checkpoint(row: dict[str, Any]) -> None:
        rows.append(row)
        completed.add(key_of(row))
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(
            f"Checkpoint {len(rows):>3}: seed={int(row['seed'])} "
            f"method={row['method']} condition={row['condition']}"
        )

    def execute_and_cleanup(**kwargs) -> dict[str, Any]:
        try:
            return run_method_condition(**kwargs)
        finally:
            # All scientific outputs needed from the H2O objects are converted
            # into ordinary Python/pandas values inside run_method_condition.
            # It is therefore safe to release server-side models and frames here.
            backend.cleanup()

    clean_spec = next(p for p in config.perturbations if p.name == "clean")
    non_clean = [p for p in config.perturbations if p.name != "clean"]

    for seed in config.experiment.seeds:
        split = three_way_stratified_split(
            df,
            config.experiment.target,
            seed,
            config.experiment.train_fraction,
            config.experiment.validation_fraction,
            config.experiment.test_fraction,
        )

        for method in METHODS:
            clean_key = (int(seed), method, clean_spec.condition_id)
            if clean_key in completed:
                clean_row = find_existing(seed, method, clean_spec.condition_id)
                assert clean_row is not None
                print(
                    f"Skipping completed: seed={seed} method={method} "
                    f"condition={clean_spec.condition_id}"
                )
            else:
                clean_row = execute_and_cleanup(
                    method=method,
                    split=split,
                    perturbation=clean_spec,
                    backend=backend,
                    planner=planner,
                    config=config,
                    seed=seed,
                    clean_validation_reference=None,
                    clean_test_reference=None,
                )
                checkpoint(clean_row)

            if "validation_after_pr_auc" not in clean_row or "test_after_pr_auc" not in clean_row:
                # If the full framework escalates even on clean data, do not fabricate a reference.
                continue

            clean_val = ClassificationMetrics(**{
                k.replace("validation_after_", ""): v
                for k, v in clean_row.items()
                if k.startswith("validation_after_") and pd.notna(v)
            })
            clean_test = ClassificationMetrics(**{
                k.replace("test_after_", ""): v
                for k, v in clean_row.items()
                if k.startswith("test_after_") and pd.notna(v)
            })

            for perturbation in non_clean:
                run_key = (int(seed), method, perturbation.condition_id)
                if run_key in completed:
                    print(
                        f"Skipping completed: seed={seed} method={method} "
                        f"condition={perturbation.condition_id}"
                    )
                    continue

                row = execute_and_cleanup(
                    method=method,
                    split=split,
                    perturbation=perturbation,
                    backend=backend,
                    planner=planner,
                    config=config,
                    seed=seed,
                    clean_validation_reference=clean_val,
                    clean_test_reference=clean_test,
                )
                checkpoint(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    return result

