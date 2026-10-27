#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        out[idx] = running
    return out


def safe_ratio(current, reference, eps=1e-12):
    if pd.isna(current) or pd.isna(reference):
        return np.nan
    if reference > eps:
        return float(current / reference)
    if current <= eps:
        return 1.0
    return float("inf")


def quality_state(row):
    if (
        row.quality_max_missing_rate >= 0.25
        or row.quality_max_standardized_mean_shift >= 1.00
        or row.quality_ood_rate >= 0.50
    ):
        return "severe"
    if (
        row.quality_max_missing_rate >= 0.10
        or row.quality_max_standardized_mean_shift >= 0.50
        or row.quality_ood_rate >= 0.20
    ):
        return "moderate"
    return "acceptable"


def apply_joint_recovery_gate(original: pd.DataFrame) -> pd.DataFrame:
    out = original.copy()
    selected = (out.method == "P_full_framework") & (out.recovery_model_selected == True)
    joint = (
        (out.validation_after_pr_auc >= out.validation_before_pr_auc - 1e-12)
        & (out.validation_after_expected_cost <= out.validation_before_expected_cost + 1e-12)
        & (out.validation_after_recall >= out.validation_before_recall - 1e-12)
    )
    reject = selected & ~joint
    for phase in ("validation", "test"):
        for col in list(out.columns):
            if col.startswith(f"{phase}_after_"):
                before = col.replace("_after_", "_before_")
                out.loc[reject, col] = out.loc[reject, before]
    out.loc[reject, "recovery_model_selected"] = False
    out.loc[reject, "final_model_id"] = out.loc[reject, "initial_model_id"]

    p = out[out.method == "P_full_framework"]
    refs = p[p.condition == "clean"].set_index("seed")
    for idx, row in p.iterrows():
        if row.condition == "clean":
            state = "accept"
        else:
            q = quality_state(row)
            if q == "severe":
                state = "human_escalation"
            else:
                ref = refs.loc[row.seed]
                retention = safe_ratio(row.validation_after_pr_auc, ref.validation_after_pr_auc)
                inflation = safe_ratio(row.validation_after_expected_cost, ref.validation_after_expected_cost)
                degraded = retention < 0.85 or inflation > 1.25
                state = "accept" if not degraded else ("human_escalation" if q == "moderate" else "abstain")
        out.loc[idx, "final_state"] = state
    return out


def corrected_resampled_tests(df: pd.DataFrame, metric="test_after_pr_auc"):
    ratio_test_train = 0.20 / 0.60
    proposed = "P_full_framework"
    baselines = [m for m in sorted(df.method.unique()) if m != proposed]
    rows = []
    for condition in df.condition.unique():
        p = df[(df.method == proposed) & (df.condition == condition)][["seed", metric]].dropna()
        local = []
        for baseline in baselines:
            b = df[(df.method == baseline) & (df.condition == condition)][["seed", metric]].dropna()
            paired = p.merge(b, on="seed", suffixes=("_p", "_b"))
            if len(paired) < 3:
                continue
            d = paired[f"{metric}_p"] - paired[f"{metric}_b"]
            n = len(d)
            var = float(d.var(ddof=1))
            se = float(np.sqrt((1.0 / n + ratio_test_train) * var))
            t_stat = float(d.mean() / se) if se > 0 else 0.0
            pval = float(2 * stats.t.sf(abs(t_stat), n - 1))
            crit = float(stats.t.ppf(0.975, n - 1))
            local.append({
                "condition": condition,
                "baseline": baseline,
                "n_pairs": n,
                "mean_difference": float(d.mean()),
                "sd_difference": float(d.std(ddof=1)),
                "corrected_se": se,
                "corrected_t": t_stat,
                "p_value": pval,
                "corrected_ci95_low": float(d.mean() - crit * se),
                "corrected_ci95_high": float(d.mean() + crit * se),
            })
        if local:
            adj = holm_adjust([x["p_value"] for x in local])
            for x, a in zip(local, adj):
                x["holm_adjusted_p"] = float(a)
                rows.append(x)
    return pd.DataFrame(rows)


def governed_states(df_method, tau_r=0.85, tau_c=1.25, fn_weight=None):
    x = df_method.copy()
    refs = x[x.condition == "clean"].set_index("seed")
    states = []
    for _, row in x.iterrows():
        if row.condition == "clean":
            states.append("accept")
            continue
        q = quality_state(row)
        if q == "severe":
            states.append("human_escalation")
            continue
        ref = refs.loc[row.seed]
        retention = safe_ratio(row.validation_after_pr_auc, ref.validation_after_pr_auc)
        if fn_weight is None:
            current_cost = row.validation_after_expected_cost
            reference_cost = ref.validation_after_expected_cost
        else:
            current_cost = fn_weight * row.validation_after_fn + row.validation_after_fp
            reference_cost = fn_weight * ref.validation_after_fn + ref.validation_after_fp
        inflation = safe_ratio(current_cost, reference_cost)
        degraded = retention < tau_r or inflation > tau_c
        states.append("accept" if not degraded else ("human_escalation" if q == "moderate" else "abstain"))
    x["audit_state"] = states
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/manuscript/final_10seed_runs_v7.csv")
    ap.add_argument("--output-dir", default="results/reviewer2_posthoc")
    args = ap.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(args.results)
    revised = apply_joint_recovery_gate(original)
    revised.to_csv(outdir / "final_10seed_runs_joint_recovery_gate.csv", index=False)
    corrected_resampled_tests(revised).to_csv(outdir / "corrected_resampled_pr_auc_tests.csv", index=False)

    # Recovery audit for candidates retained under the original PR-AUC-only rule.
    old_sel = original[(original.method == "P_full_framework") & (original.recovery_model_selected == True)].copy()
    old_sel["validation_cost_worsened"] = old_sel.validation_after_expected_cost > old_sel.validation_before_expected_cost + 1e-12
    old_sel["validation_recall_worsened"] = old_sel.validation_after_recall < old_sel.validation_before_recall - 1e-12
    old_sel["joint_rule_retain"] = (~old_sel.validation_cost_worsened) & (~old_sel.validation_recall_worsened)
    old_sel["test_pr_auc_improved"] = old_sel.test_after_pr_auc > old_sel.test_before_pr_auc + 1e-12
    old_sel.to_csv(outdir / "retained_recovery_operational_audit.csv", index=False)

    # Full framework and a non-LLM governed controller using the same state rules.
    p = governed_states(revised[revised.method == "P_full_framework"])
    b2 = governed_states(revised[revised.method == "B2_rule_retry"])
    for name, x in [("full_framework", p), ("nonllm_governed_retry", b2)]:
        x.to_csv(outdir / f"{name}_runlevel.csv", index=False)

    audit_rows = []
    coverage_rows = []
    for label, x in [("Full framework", p), ("Non-LLM governed retry", b2)]:
        clean = x[x.condition == "clean"].set_index("seed")
        for _, row in x.iterrows():
            diag = row.test_after_coverage if pd.notna(row.test_after_coverage) else np.nan
            op_cov = diag if row.audit_state == "accept" else 0.0
            coverage_rows.append({
                "controller": label, "seed": row.seed, "condition": row.condition,
                "state": row.audit_state, "diagnostic_instance_coverage": diag,
                "effective_operational_coverage": op_cov,
                "test_pr_auc": row.test_after_pr_auc, "test_recall": row.test_after_recall,
                "test_selective_accuracy": row.test_after_selective_accuracy,
            })
            if row.condition == "clean":
                continue
            ref = clean.loc[row.seed]
            test_ret = safe_ratio(row.test_after_pr_auc, ref.test_after_pr_auc)
            test_inf = safe_ratio(row.test_after_expected_cost, ref.test_after_expected_cost)
            if not (np.isfinite(test_ret) and np.isfinite(test_inf)):
                category = "not_evaluable"
            else:
                poor = test_ret < 0.85 or test_inf > 1.25
                withheld = row.audit_state != "accept"
                if not withheld and poor: category = "unsafe_acceptance"
                elif withheld and not poor: category = "unnecessary_withholding"
                elif withheld and poor: category = "appropriate_withholding"
                else: category = "appropriate_acceptance"
            audit_rows.append({
                "controller": label, "seed": row.seed, "condition": row.condition,
                "state": row.audit_state, "test_retention": test_ret,
                "test_cost_inflation": test_inf, "decision_audit": category,
                "test_expected_cost": row.test_after_expected_cost,
                "test_pr_auc": row.test_after_pr_auc, "test_recall": row.test_after_recall,
            })
    audit = pd.DataFrame(audit_rows); audit.to_csv(outdir / "controller_decision_audit.csv", index=False)
    cov = pd.DataFrame(coverage_rows); cov.to_csv(outdir / "operational_coverage_runlevel.csv", index=False)
    cov.groupby(["controller", "condition"]).agg(
        diagnostic_instance_coverage=("diagnostic_instance_coverage", "mean"),
        effective_operational_coverage=("effective_operational_coverage", "mean"),
        accepted_runs=("state", lambda s: int((s == "accept").sum())),
    ).reset_index().to_csv(outdir / "operational_coverage_summary.csv", index=False)

    # Sensitivity of system-level states. Cost-weight sensitivity holds the already-selected
    # decision threshold fixed and recomputes the cost ratio from confusion counts.
    sens = []
    psrc = revised[revised.method == "P_full_framework"]
    for tr in (0.80, 0.85, 0.90):
        xx = governed_states(psrc, tau_r=tr, tau_c=1.25)
        for condition, g in xx.groupby("condition"):
            sens.append({"dimension": "retention_threshold", "value": tr, "condition": condition,
                         "accept": int((g.audit_state == "accept").sum()), "abstain": int((g.audit_state == "abstain").sum()),
                         "escalate": int((g.audit_state == "human_escalation").sum())})
    for tc in (1.10, 1.25, 1.50):
        xx = governed_states(psrc, tau_r=0.85, tau_c=tc)
        for condition, g in xx.groupby("condition"):
            sens.append({"dimension": "cost_inflation_threshold", "value": tc, "condition": condition,
                         "accept": int((g.audit_state == "accept").sum()), "abstain": int((g.audit_state == "abstain").sum()),
                         "escalate": int((g.audit_state == "human_escalation").sum())})
    for weight in (3.0, 5.0, 10.0):
        xx = governed_states(psrc, fn_weight=weight)
        for condition, g in xx.groupby("condition"):
            sens.append({"dimension": "false_negative_cost_weight_frozen_threshold", "value": weight, "condition": condition,
                         "accept": int((g.audit_state == "accept").sum()), "abstain": int((g.audit_state == "abstain").sum()),
                         "escalate": int((g.audit_state == "human_escalation").sum())})
    pd.DataFrame(sens).to_csv(outdir / "controller_sensitivity.csv", index=False)

    zeros = []
    for method, g in revised[revised.condition == "clean"].groupby("method"):
        zeros.append({"method": method, "n": len(g),
                      "zero_clean_validation_cost": int((g.validation_after_expected_cost <= 1e-12).sum()),
                      "min_clean_validation_cost": g.validation_after_expected_cost.min(),
                      "max_clean_validation_cost": g.validation_after_expected_cost.max()})
    pd.DataFrame(zeros).to_csv(outdir / "zero_reference_cost_check.csv", index=False)
    print(f"Post-hoc reviewer-2 outputs written to {outdir}")


if __name__ == "__main__":
    main()
