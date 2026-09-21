from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .config import QualityPolicy
from .schemas import DataQualityStatus


@dataclass(frozen=True)
class QualityReport:
    severity: DataQualityStatus
    max_missing_rate: float
    max_standardized_mean_shift: float
    ood_rate: float
    suspicious_features: list[str]
    missing_by_feature: dict[str, float]
    standardized_mean_shift_by_feature: dict[str, float]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


def _numeric_columns(train: pd.DataFrame, current: pd.DataFrame, target: str) -> list[str]:
    return [
        c
        for c in train.columns
        if c != target
        and c in current.columns
        and pd.api.types.is_numeric_dtype(train[c])
        and pd.api.types.is_numeric_dtype(current[c])
    ]


def _mahalanobis_ood_rate(
    train: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    quantile: float,
) -> float:
    if not columns:
        return 0.0

    train_num = train[columns].astype(float).copy()
    curr_num = current[columns].astype(float).copy()
    medians = train_num.median()
    train_num = train_num.fillna(medians)
    curr_num = curr_num.fillna(medians)

    mu = train_num.mean(axis=0).to_numpy(dtype=float)
    x_train = train_num.to_numpy(dtype=float)
    x_curr = curr_num.to_numpy(dtype=float)

    cov = np.cov(x_train, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    # Small ridge for numerical stability with correlated industrial variables.
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    inv_cov = np.linalg.pinv(cov)

    def distance(x: np.ndarray) -> np.ndarray:
        centered = x - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", centered, inv_cov, centered))

    train_d = distance(x_train)
    curr_d = distance(x_curr)
    cutoff = float(np.quantile(train_d, quantile))
    return float(np.mean(curr_d > cutoff))


def assess_data_quality(
    train: pd.DataFrame,
    current: pd.DataFrame,
    target: str,
    policy: QualityPolicy,
) -> QualityReport:
    predictors = [c for c in train.columns if c != target and c in current.columns]
    missing = current[predictors].isna().mean().to_dict()
    max_missing = max(missing.values(), default=0.0)

    numeric = _numeric_columns(train, current, target)
    smd: dict[str, float] = {}
    for col in numeric:
        sigma = float(train[col].std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-12:
            sigma = 1.0
        train_mean = float(train[col].mean())
        current_mean = float(current[col].mean())
        if np.isnan(current_mean):
            shift = float("inf")
        else:
            shift = abs(current_mean - train_mean) / sigma
        smd[col] = float(shift)

    max_shift = max(smd.values(), default=0.0)
    ood_rate = _mahalanobis_ood_rate(train, current, numeric, policy.ood_training_quantile)

    severe = (
        max_missing >= policy.severe_missing_rate
        or max_shift >= policy.severe_standardized_mean_shift
        or ood_rate >= policy.severe_ood_rate
    )
    moderate = (
        max_missing >= policy.moderate_missing_rate
        or max_shift >= policy.moderate_standardized_mean_shift
        or ood_rate >= policy.moderate_ood_rate
    )

    severity = (
        DataQualityStatus.SEVERE
        if severe
        else DataQualityStatus.MODERATE
        if moderate
        else DataQualityStatus.ACCEPTABLE
    )

    suspect = sorted(
        {
            *[k for k, v in missing.items() if v >= policy.moderate_missing_rate],
            *[k for k, v in smd.items() if v >= policy.moderate_standardized_mean_shift],
        }
    )

    return QualityReport(
        severity=severity,
        max_missing_rate=float(max_missing),
        max_standardized_mean_shift=float(max_shift),
        ood_rate=float(ood_rate),
        suspicious_features=suspect,
        missing_by_feature={k: float(v) for k, v in missing.items()},
        standardized_mean_shift_by_feature=smd,
    )
