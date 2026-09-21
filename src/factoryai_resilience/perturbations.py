from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AppliedPerturbation:
    name: str
    severity: str
    magnitude: float
    features: list[str]
    seed: int


DEFAULT_SENSOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def _valid_numeric_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]


def _reference_sigma(reference: pd.DataFrame, feature: str) -> float:
    sigma = float(reference[feature].astype(float).std(ddof=0))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return 1.0
    return sigma


def apply_perturbation(
    df: pd.DataFrame,
    name: str,
    severity: str,
    magnitude: float,
    features: list[str] | None,
    seed: int,
    target: str,
    reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, AppliedPerturbation]:
    """Apply a deterministic feature-only perturbation.

    The target column is never modified. Perturbation magnitudes are defined from
    training-reference statistics when ``reference_df`` is supplied, so validation
    and test degradation use the same scientific scale and never derive their own
    perturbation strength from held-out data.
    """
    out = df.copy()
    reference = reference_df if reference_df is not None else df
    rng = np.random.default_rng(seed)
    requested = features or DEFAULT_SENSOR_FEATURES
    selected = _valid_numeric_features(out, [f for f in requested if f != target])
    selected = [f for f in selected if f in reference.columns]

    if name == "clean":
        return out, AppliedPerturbation(name, severity, 0.0, [], seed)

    if not selected:
        raise ValueError(f"No valid numeric features available for perturbation {name!r}")

    if name == "sensor_drift":
        # Apply a gradual bias ordered by the original row index (a reproducible
        # pseudo-temporal ordering for AI4I) and scaled only from training data.
        original_index = np.asarray(out.index)
        order = np.argsort(original_index, kind="stable")
        ramp_sorted = np.linspace(0.0, 1.0, len(out))
        ramp = np.empty(len(out), dtype=float)
        ramp[order] = ramp_sorted
        for feature in selected:
            sigma = _reference_sigma(reference, feature)
            out[feature] = out[feature].astype(float) + ramp * float(magnitude) * sigma

    elif name == "measurement_noise":
        for feature in selected:
            sigma = _reference_sigma(reference, feature)
            noise = rng.normal(0.0, float(magnitude) * sigma, size=len(out))
            out[feature] = out[feature].astype(float) + noise

    elif name == "missingness":
        rate = float(magnitude)
        if not 0.0 <= rate < 1.0:
            raise ValueError("Missingness magnitude must be in [0, 1).")
        # Mask an exact proportion rather than a Bernoulli approximation so each
        # severity level is reproducible and matches the stated experimental rate.
        n_mask = int(round(rate * len(out)))
        for feature in selected:
            if n_mask <= 0:
                continue
            positions = rng.choice(len(out), size=min(n_mask, len(out)), replace=False)
            out.iloc[positions, out.columns.get_loc(feature)] = np.nan

    else:
        raise ValueError(f"Unsupported perturbation {name!r}")

    return out, AppliedPerturbation(name, severity, magnitude, selected, seed)
