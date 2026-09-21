#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plot(summary: pd.DataFrame, metric_name: str, output: Path) -> None:
    subset = summary[summary["metric"] == metric_name].copy()
    if subset.empty:
        return
    conditions = list(dict.fromkeys(subset["condition"].tolist()))
    methods = list(dict.fromkeys(subset["method"].tolist()))
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(conditions))
    for method in methods:
        g = subset[subset.method == method].set_index("condition").reindex(conditions)
        ax.plot(x, g["mean"], marker="o", label=method)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel(metric_name.replace("test_after_", ""))
    ax.set_xlabel("Experimental condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    analysis = Path(args.analysis_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for file, metric, name in [
        ("summary_pr_auc.csv", "test_after_pr_auc", "pr_auc_by_condition.png"),
        ("summary_recall.csv", "test_after_recall", "recall_by_condition.png"),
        ("summary_fnr.csv", "test_after_false_negative_rate", "fnr_by_condition.png"),
    ]:
        path = analysis / file
        if path.exists():
            save_metric_plot(pd.read_csv(path), metric, out / name)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
