from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    margin = float(stats.t.ppf(0.975, len(values) - 1) * se)
    return mean - margin, mean + margin


def summarize_results(df: pd.DataFrame, metric: str = "test_after_pr_auc") -> pd.DataFrame:
    rows = []
    for (method, condition), group in df.groupby(["method", "condition"], dropna=False):
        values = pd.to_numeric(group.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
        low, high = _ci95(values)
        rows.append(
            {
                "method": method,
                "condition": condition,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "escalation_rate": float(np.mean(group["final_state"] == "human_escalation")),
                "abstention_rate_system": float(np.mean(group["final_state"] == "abstain")),
                "mean_runtime_seconds": float(pd.to_numeric(group["total_runtime_seconds"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def rank_biserial_from_differences(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / len(diff))


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def paired_wilcoxon_against_proposed(
    df: pd.DataFrame,
    metric: str = "test_after_pr_auc",
    proposed: str = "P_full_framework",
) -> pd.DataFrame:
    tests = []
    baselines = [m for m in sorted(df["method"].dropna().unique()) if m != proposed]

    for condition in sorted(df["condition"].dropna().unique()):
        proposed_df = df[(df.method == proposed) & (df.condition == condition)][["seed", metric]].dropna()
        for baseline in baselines:
            base_df = df[(df.method == baseline) & (df.condition == condition)][["seed", metric]].dropna()
            paired = proposed_df.merge(base_df, on="seed", suffixes=("_proposed", "_baseline"))
            if len(paired) < 3:
                continue
            diff = paired[f"{metric}_proposed"].to_numpy(float) - paired[f"{metric}_baseline"].to_numpy(float)
            if np.allclose(diff, 0):
                statistic, p = 0.0, 1.0
            else:
                statistic, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            tests.append(
                {
                    "condition": condition,
                    "baseline": baseline,
                    "proposed": proposed,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_paired_difference": float(np.mean(diff)),
                    "median_paired_difference": float(np.median(diff)),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p),
                    "rank_biserial_effect": rank_biserial_from_differences(diff),
                }
            )

    if tests:
        adjusted = holm_adjust([r["p_value"] for r in tests])
        for row, p_adj in zip(tests, adjusted):
            row["holm_adjusted_p"] = p_adj
    return pd.DataFrame(tests)


def write_analysis_tables(results_csv: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    summarize_results(df).to_csv(output_dir / "summary_pr_auc.csv", index=False)
    summarize_results(df, "test_after_recall").to_csv(output_dir / "summary_recall.csv", index=False)
    summarize_results(df, "test_after_false_negative_rate").to_csv(output_dir / "summary_fnr.csv", index=False)
    paired_wilcoxon_against_proposed(df).to_csv(output_dir / "paired_wilcoxon_pr_auc.csv", index=False)
