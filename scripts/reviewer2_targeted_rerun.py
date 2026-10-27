#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoryai_resilience.backends import build_backend
from factoryai_resilience.config import load_config
from factoryai_resilience.data import load_ai4i_csv, three_way_stratified_split
from factoryai_resilience.metrics import (
    choose_cost_sensitive_threshold,
    compute_metrics,
    risk_coverage_points,
)
from factoryai_resilience.perturbations import apply_perturbation
from factoryai_resilience.policy import (
    controller,
    deterministic_retry_config,
    recovery_candidate_is_acceptable,
)
from factoryai_resilience.quality import assess_data_quality
from factoryai_resilience.schemas import ControllerState, ExecutionConfig


def evaluate_model(backend, handle, validation, test, target, cost_policy):
    p_val = backend.predict_proba(handle, validation)
    p_test = backend.predict_proba(handle, test)
    val = compute_metrics(validation[target].to_numpy(), p_val, cost_policy)
    testm = compute_metrics(
        test[target].to_numpy(), p_test, cost_policy,
        threshold=val.threshold, abstention_band=val.abstention_band,
    )
    return p_val, p_test, val, testm


def metric_row(prefix: str, m) -> dict:
    return {f"{prefix}_{k}": v for k, v in m.to_dict().items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Reviewer-2 targeted rerun: fixed-initial-model recovery isolation and "
            "decision-boundary risk-coverage evaluation. No new LLM calls are made; "
            "archived executable configurations are reused."
        )
    )
    ap.add_argument("--data", default="data/ai4i2020.csv")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--archive", default="results/manuscript/final_10seed_runs_v7.csv")
    ap.add_argument("--output-dir", default="results/reviewer2_targeted")
    ap.add_argument("--backend", default="h2o", choices=["h2o", "sklearn_smoke"])
    ap.add_argument("--seeds", default="", help="Optional comma-separated subset, e.g. 11,22")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_ai4i_csv(args.data)
    archive = pd.read_csv(args.archive)
    p_rows = archive[archive.method == "P_full_framework"].copy()
    if args.seeds:
        seeds = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
        p_rows = p_rows[p_rows.seed.isin(seeds)]

    specs = {p.condition_id: p for p in cfg.perturbations}
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(
        args.backend,
        h2o_port=cfg.backend.h2o_port,
        h2o_max_mem_size=cfg.backend.h2o_max_mem_size,
    )

    isolation_rows: list[dict] = []
    risk_rows: list[dict] = []
    final_rows: list[dict] = []

    for seed in sorted(p_rows.seed.unique()):
        split = three_way_stratified_split(
            df, cfg.experiment.target, int(seed),
            cfg.experiment.train_fraction,
            cfg.experiment.validation_fraction,
            cfg.experiment.test_fraction,
        )

        seed_rows = p_rows[p_rows.seed == seed]
        clean_arch = seed_rows[seed_rows.condition == "clean"].iloc[0]
        clean_val_ref = None

        # Process clean first to establish the reproduced validation reference.
        ordered = ["clean"] + [c for c in specs if c != "clean"]
        for condition in ordered:
            hit = seed_rows[seed_rows.condition == condition]
            if hit.empty:
                continue
            row = hit.iloc[0]
            spec = specs[condition]

            val_frame, _ = apply_perturbation(
                split.validation, spec.name, spec.severity, spec.magnitude,
                spec.features, seed=int(seed) + 1000, target=split.target,
                reference_df=split.train,
            )
            test_frame, _ = apply_perturbation(
                split.test, spec.name, spec.severity, spec.magnitude,
                spec.features, seed=int(seed) + 2000, target=split.target,
                reference_df=split.train,
            )
            quality = assess_data_quality(split.train, val_frame, split.target, cfg.quality_policy)

            # Severe missingness was intentionally stopped before training in v7.
            if pd.isna(row.initial_execution_config):
                final_rows.append({
                    "seed": seed, "condition": condition,
                    "quality_severity": quality.severity.value,
                    "system_state": "human_escalation",
                    "model_available": False,
                    "effective_operational_coverage_090": 0.0,
                })
                continue

            try:
                initial_cfg = ExecutionConfig.model_validate_json(row.initial_execution_config)
                initial = backend.fit_automl(
                    split.train, val_frame, split.target, split.predictors, initial_cfg
                )
                pvi, pti, val_initial, test_initial = evaluate_model(
                    backend, initial, val_frame, test_frame, split.target, cfg.cost_policy
                )

                if condition == "clean":
                    clean_val_ref = val_initial
                    final_handle = initial
                    final_val = val_initial
                    final_test = test_initial
                    pvf, ptf = pvi, pti
                    state = ControllerState.ACCEPT
                    selected_mechanism = "initial"
                else:
                    if clean_val_ref is None:
                        raise RuntimeError("Clean validation reference was not reproduced before degraded conditions.")

                    first_decision = controller(
                        quality=quality,
                        current=val_initial,
                        clean_reference=clean_val_ref,
                        recovery_policy=cfg.recovery_policy,
                        retry_index=0,
                    )
                    final_handle = initial
                    final_val = val_initial
                    final_test = test_initial
                    pvf, ptf = pvi, pti
                    state = first_decision.state
                    selected_mechanism = "initial"

                    if first_decision.state == ControllerState.RECONFIGURE:
                        # Branch 1: no recovery is the common initial model above.
                        iso_base = {
                            "seed": seed,
                            "condition": condition,
                            "quality_severity": quality.severity.value,
                            "initial_model_id": initial.model_id,
                            "initial_execution_config": initial.execution_config.model_dump_json(),
                            **metric_row("initial_validation", val_initial),
                            **metric_row("initial_test", test_initial),
                        }

                        # Branch 2: deterministic recovery from exactly the same initial config/model budget.
                        det_budget = cfg.experiment.max_models_total - cfg.experiment.initial_models_recovery_methods
                        det_cfg = deterministic_retry_config(
                            initial.execution_config, quality, split.predictors, det_budget
                        )
                        det = backend.fit_automl(
                            split.train, val_frame, split.target, split.predictors, det_cfg
                        )
                        pvd, ptd, val_det, test_det = evaluate_model(
                            backend, det, val_frame, test_frame, split.target, cfg.cost_policy
                        )
                        det_select = recovery_candidate_is_acceptable(
                            candidate=val_det, incumbent=val_initial
                        )
                        isolation_rows.append({
                            **iso_base,
                            "mechanism": "deterministic_recovery",
                            "candidate_selected_joint_gate": det_select,
                            "candidate_model_id": det.model_id,
                            "candidate_execution_config": det.execution_config.model_dump_json(),
                            **metric_row("candidate_validation", val_det),
                            **metric_row("candidate_test", test_det),
                            "final_test_pr_auc_if_governed": test_det.pr_auc if det_select else test_initial.pr_auc,
                            "final_test_recall_if_governed": test_det.recall if det_select else test_initial.recall,
                            "final_test_cost_if_governed": test_det.expected_cost if det_select else test_initial.expected_cost,
                        })

                        # Branch 3: archived LLM recovery config from the same initial model.
                        if pd.notna(row.retry_execution_config):
                            llm_cfg = ExecutionConfig.model_validate_json(row.retry_execution_config)
                            llm = backend.fit_automl(
                                split.train, val_frame, split.target, split.predictors, llm_cfg
                            )
                            pvl, ptl, val_llm, test_llm = evaluate_model(
                                backend, llm, val_frame, test_frame, split.target, cfg.cost_policy
                            )
                            llm_select = recovery_candidate_is_acceptable(
                                candidate=val_llm, incumbent=val_initial
                            )
                            isolation_rows.append({
                                **iso_base,
                                "mechanism": "llm_recovery_archived_config",
                                "candidate_selected_joint_gate": llm_select,
                                "candidate_model_id": llm.model_id,
                                "candidate_execution_config": llm.execution_config.model_dump_json(),
                                **metric_row("candidate_validation", val_llm),
                                **metric_row("candidate_test", test_llm),
                                "final_test_pr_auc_if_governed": test_llm.pr_auc if llm_select else test_initial.pr_auc,
                                "final_test_recall_if_governed": test_llm.recall if llm_select else test_initial.recall,
                                "final_test_cost_if_governed": test_llm.expected_cost if llm_select else test_initial.expected_cost,
                            })
                            if llm_select:
                                final_handle, final_val, final_test = llm, val_llm, test_llm
                                pvf, ptf = pvl, ptl
                                selected_mechanism = "llm_recovery_archived_config"

                        post = controller(
                            quality=quality,
                            current=final_val,
                            clean_reference=clean_val_ref,
                            recovery_policy=cfg.recovery_policy,
                            retry_index=1,
                        )
                        state = post.state

                # Risk-coverage points are diagnostic model outputs. Effective operational
                # coverage additionally applies the system-level state gate.
                for pt in risk_coverage_points(
                    test_frame[split.target].to_numpy(), ptf,
                    y_prob_validation=pvf,
                    threshold=final_val.threshold,
                    policy=cfg.cost_policy,
                    target_coverages=(0.70, 0.80, 0.90, 0.95, 1.00),
                ):
                    op_cov = float(pt["coverage"]) if state == ControllerState.ACCEPT else 0.0
                    risk_rows.append({
                        "seed": seed,
                        "condition": condition,
                        "system_state": state.value,
                        "selected_mechanism": selected_mechanism,
                        **pt,
                        "effective_operational_coverage": op_cov,
                    })

                final_rows.append({
                    "seed": seed, "condition": condition,
                    "quality_severity": quality.severity.value,
                    "system_state": state.value,
                    "model_available": True,
                    "selected_mechanism": selected_mechanism,
                    **metric_row("final_validation", final_val),
                    **metric_row("final_test", final_test),
                    "effective_operational_coverage_090": (
                        final_test.coverage if state == ControllerState.ACCEPT else 0.0
                    ),
                })
            finally:
                backend.cleanup()

    pd.DataFrame(isolation_rows).to_csv(outdir / "fixed_initial_recovery_isolation.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(outdir / "risk_coverage_points.csv", index=False)
    pd.DataFrame(final_rows).to_csv(outdir / "targeted_final_runs.csv", index=False)

    if isolation_rows:
        iso = pd.DataFrame(isolation_rows)
        summary = (
            iso.groupby(["condition", "mechanism"], dropna=False)
            .agg(
                n=("seed", "size"),
                selected=("candidate_selected_joint_gate", "sum"),
                initial_pr_auc=("initial_test_pr_auc", "mean"),
                candidate_pr_auc=("candidate_test_pr_auc", "mean"),
                governed_final_pr_auc=("final_test_pr_auc_if_governed", "mean"),
                governed_final_recall=("final_test_recall_if_governed", "mean"),
                governed_final_cost=("final_test_cost_if_governed", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(outdir / "fixed_initial_recovery_isolation_summary.csv", index=False)

    if risk_rows:
        rr = pd.DataFrame(risk_rows)
        rsum = (
            rr.groupby(["condition", "target_coverage"], dropna=False)
            .agg(
                diagnostic_coverage=("coverage", "mean"),
                effective_operational_coverage=("effective_operational_coverage", "mean"),
                selective_error=("selective_error", "mean"),
                selective_recall=("selective_recall", "mean"),
                selective_fnr=("selective_false_negative_rate", "mean"),
                selective_cost=("selective_expected_cost", "mean"),
            )
            .reset_index()
        )
        rsum.to_csv(outdir / "risk_coverage_summary.csv", index=False)

    print(f"Reviewer-2 targeted outputs written to: {outdir}")
    print("No LLM API calls were made; archived executable configurations were reused.")


if __name__ == "__main__":
    main()
