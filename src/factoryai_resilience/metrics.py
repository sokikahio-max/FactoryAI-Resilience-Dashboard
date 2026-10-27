from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CostPolicy


@dataclass(frozen=True)
class ClassificationMetrics:
    threshold: float
    abstention_band: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    mcc: float
    roc_auc: float
    pr_auc: float
    false_negative_rate: float
    brier: float
    expected_cost: float
    abstention_rate: float
    coverage: float
    selective_accuracy: float | None
    predictive_entropy_mean: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict:
        return asdict(self)


def choose_cost_sensitive_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: CostPolicy,
) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_cost = float("inf")
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = policy.false_negative_cost * fn + policy.false_positive_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t


def compute_metrics(
    y_true,
    y_prob,
    policy: CostPolicy,
    threshold: float | None = None,
    abstention_band: float | None = None,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    threshold = threshold if threshold is not None else choose_cost_sensitive_threshold(y_true, y_prob, policy)
    pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    expected_cost = (
        policy.false_negative_cost * fn + policy.false_positive_cost * fp
    ) / max(len(y_true), 1)

    # Calibrate uncertainty around the *decision boundary* using validation data.
    # A fixed absolute band (e.g. +/-0.05) is inappropriate when a cost-sensitive
    # threshold is far from 0.5; it can incorrectly abstain on nearly every sample.
    distance = np.abs(y_prob - threshold)
    if abstention_band is None:
        abstain_fraction = float(np.clip(1.0 - policy.target_coverage, 0.0, 1.0))
        abstention_band = float(np.quantile(distance, abstain_fraction)) if len(distance) else 0.0
    uncertain = distance < float(abstention_band)
    coverage = float(np.mean(~uncertain))
    selective_accuracy = None
    if (~uncertain).sum() > 0:
        selective_accuracy = float(accuracy_score(y_true[~uncertain], pred[~uncertain]))

    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return ClassificationMetrics(
        threshold=float(threshold),
        abstention_band=float(abstention_band),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        mcc=float(matthews_corrcoef(y_true, pred)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        false_negative_rate=float(fnr),
        brier=float(brier_score_loss(y_true, y_prob)),
        expected_cost=float(expected_cost),
        abstention_rate=float(np.mean(uncertain)),
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        predictive_entropy_mean=float(np.mean(entropy)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def metric_value(metrics: ClassificationMetrics, name: str) -> float:
    if not hasattr(metrics, name):
        raise KeyError(f"Unknown metric {name!r}")
    return float(getattr(metrics, name))


def resilience_metrics(clean: float, degraded_before: float, recovered: float) -> dict[str, float | None]:
    retention = degraded_before / clean if clean else None
    gain = recovered - degraded_before
    denom = clean - degraded_before
    normalized = gain / denom if abs(denom) > 1e-12 else None
    return {
        "performance_retention": retention,
        "recovery_gain": gain,
        "normalized_recovery_ratio": normalized,
    }



def calibrate_decision_boundary_band(
    y_prob_validation,
    threshold: float,
    target_coverage: float,
) -> float:
    """Calibrate a decision-boundary abstention band to a target coverage.

    This is a coverage-control heuristic, not a calibration of epistemic or
    aleatoric predictive uncertainty.
    """
    prob = np.asarray(y_prob_validation, dtype=float)
    distance = np.abs(prob - float(threshold))
    abstain_fraction = float(np.clip(1.0 - target_coverage, 0.0, 1.0))
    return float(np.quantile(distance, abstain_fraction)) if len(distance) else 0.0


def selective_metrics(
    y_true,
    y_prob,
    *,
    threshold: float,
    abstention_band: float,
    policy: CostPolicy,
) -> dict[str, float | int | None]:
    """Metrics on predictions released outside the abstention band."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = (y_prob >= float(threshold)).astype(int)
    keep = np.abs(y_prob - float(threshold)) >= float(abstention_band)
    n = int(keep.sum())
    coverage = float(np.mean(keep)) if len(keep) else 0.0
    if n == 0:
        return {
            "coverage": coverage,
            "selective_error": None,
            "selective_accuracy": None,
            "selective_recall": None,
            "selective_false_negative_rate": None,
            "selective_expected_cost": None,
            "released_n": 0,
            "released_positive_n": 0,
        }
    yt, yp = y_true[keep], pred[keep]
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    pos = int(fn + tp)
    recall = float(tp / pos) if pos else None
    fnr = float(fn / pos) if pos else None
    cost = float((policy.false_negative_cost * fn + policy.false_positive_cost * fp) / n)
    acc = float(accuracy_score(yt, yp))
    return {
        "coverage": coverage,
        "selective_error": 1.0 - acc,
        "selective_accuracy": acc,
        "selective_recall": recall,
        "selective_false_negative_rate": fnr,
        "selective_expected_cost": cost,
        "released_n": n,
        "released_positive_n": pos,
    }


def risk_coverage_points(
    y_true_test,
    y_prob_test,
    *,
    y_prob_validation,
    threshold: float,
    policy: CostPolicy,
    target_coverages=(0.70, 0.80, 0.90, 0.95, 1.00),
) -> list[dict[str, float | int | None]]:
    """Compute validation-calibrated decision-boundary risk--coverage points."""
    rows = []
    for target in target_coverages:
        band = calibrate_decision_boundary_band(
            y_prob_validation, threshold, float(target)
        )
        row = selective_metrics(
            y_true_test,
            y_prob_test,
            threshold=threshold,
            abstention_band=band,
            policy=policy,
        )
        row.update({"target_coverage": float(target), "abstention_band": band})
        rows.append(row)
    return rows
