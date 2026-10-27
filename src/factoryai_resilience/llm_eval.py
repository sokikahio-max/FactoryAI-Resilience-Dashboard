from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd
import yaml

from .config import AppConfig
from .planner import BasePlanner, PlanningContext
from .quality import QualityReport
from .schemas import DataQualityStatus, TaskType, TASK_METRICS


def _neutral_quality() -> QualityReport:
    return QualityReport(
        severity=DataQualityStatus.ACCEPTABLE,
        max_missing_rate=0.0,
        max_standardized_mean_shift=0.0,
        ood_rate=0.01,
        suspicious_features=[],
        missing_by_feature={},
        standardized_mean_shift_by_feature={},
    )


def run_llm_evaluation(
    planner: BasePlanner,
    config: AppConfig,
    cases_path: str | Path,
    repetitions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = yaml.safe_load(Path(cases_path).read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        for rep in range(repetitions):
            try:
                decision, telemetry = planner.plan(
                    PlanningContext(
                        problem_description=case["problem"],
                        target=case["target"],
                        predictors=["sensor_1", "sensor_2", "sensor_3"],
                        positive_rate=0.04 if case["target_cardinality"] == 2 else 0.20,
                        quality=_neutral_quality(),
                        retry_index=0,
                        target_cardinality=case["target_cardinality"],
                        target_is_continuous=bool(case["target_is_continuous"]),
                    )
                )
                rows.append(
                    {
                        "case_id": case["id"],
                        "group": case["group"],
                        "repetition": rep,
                        "expected_task": case["expected_task"],
                        "predicted_task": decision.task_type.value,
                        "task_correct": decision.task_type.value == case["expected_task"],
                        "recommended_action": decision.recommended_action.value,
                        "candidate_algorithms": "|".join(decision.candidate_algorithms),
                        "primary_metric": decision.primary_metric,
                        "metric_correct": decision.primary_metric in set(case.get("allowed_metrics", [])),
                        "schema_valid": telemetry.schema_valid,
                        "latency_seconds": telemetry.latency_seconds,
                        "input_tokens": telemetry.input_tokens,
                        "output_tokens": telemetry.output_tokens,
                        "estimated_cost_usd": telemetry.estimated_cost_usd,
                        "requested_model": telemetry.requested_model,
                        "provider_returned_model": telemetry.provider_returned_model,
                        "response_id": telemetry.response_id,
                        "temperature_sent": telemetry.temperature_sent,
                        "error": telemetry.error,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "case_id": case["id"],
                        "group": case["group"],
                        "repetition": rep,
                        "expected_task": case["expected_task"],
                        "predicted_task": None,
                        "task_correct": False,
                        "recommended_action": None,
                        "candidate_algorithms": None,
                        "primary_metric": None,
                        "metric_correct": False,
                        "schema_valid": False,
                        "latency_seconds": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "estimated_cost_usd": None,
                        "requested_model": None,
                        "provider_returned_model": None,
                        "response_id": None,
                        "temperature_sent": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    raw = pd.DataFrame(rows)
    summary_rows = []
    for case_id, g in raw.groupby("case_id"):
        valid = g[g["schema_valid"] == True]
        action_counts = Counter(valid["recommended_action"].dropna())
        algo_counts = Counter(valid["candidate_algorithms"].dropna())
        metric_counts = Counter(valid["primary_metric"].dropna())
        summary_rows.append(
            {
                "case_id": case_id,
                "group": g["group"].iloc[0],
                "n_runs": len(g),
                "schema_valid_rate": float(g["schema_valid"].mean()),
                "task_accuracy": float(g["task_correct"].mean()),
                "metric_appropriateness": float(g["metric_correct"].mean()),
                "action_consistency": (max(action_counts.values()) / len(valid)) if len(valid) and action_counts else 0.0,
                "algorithm_recommendation_consistency": (max(algo_counts.values()) / len(valid)) if len(valid) and algo_counts else 0.0,
                "metric_consistency": (max(metric_counts.values()) / len(valid)) if len(valid) and metric_counts else 0.0,
                "mean_latency_seconds": float(pd.to_numeric(g["latency_seconds"], errors="coerce").mean()),
                "mean_input_tokens": float(pd.to_numeric(g["input_tokens"], errors="coerce").mean()),
                "mean_output_tokens": float(pd.to_numeric(g["output_tokens"], errors="coerce").mean()),
                "mean_estimated_cost_usd": float(pd.to_numeric(g["estimated_cost_usd"], errors="coerce").mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        group_rows = []
        for group, g in raw.groupby("group"):
            valid = g[g["schema_valid"] == True]
            task_mode = valid["predicted_task"].mode()
            action_mode = valid["recommended_action"].mode()
            group_rows.append(
                {
                    "case_id": f"GROUP::{group}",
                    "group": group,
                    "n_runs": len(g),
                    "schema_valid_rate": float(g["schema_valid"].mean()),
                    "task_accuracy": float(g["task_correct"].mean()),
                    "metric_appropriateness": float(g["metric_correct"].mean()),
                    "action_consistency": float((valid["recommended_action"] == action_mode.iloc[0]).mean()) if len(valid) and len(action_mode) else 0.0,
                    "algorithm_recommendation_consistency": float(valid["candidate_algorithms"].value_counts(normalize=True).iloc[0]) if len(valid) else 0.0,
                    "metric_consistency": float(valid["primary_metric"].value_counts(normalize=True).iloc[0]) if len(valid) else 0.0,
                    "mean_latency_seconds": float(pd.to_numeric(g["latency_seconds"], errors="coerce").mean()),
                    "mean_input_tokens": float(pd.to_numeric(g["input_tokens"], errors="coerce").mean()),
                    "mean_output_tokens": float(pd.to_numeric(g["output_tokens"], errors="coerce").mean()),
                    "mean_estimated_cost_usd": float(pd.to_numeric(g["estimated_cost_usd"], errors="coerce").mean()),
                }
            )
        summary = pd.concat([summary, pd.DataFrame(group_rows)], ignore_index=True)

    return raw, summary
