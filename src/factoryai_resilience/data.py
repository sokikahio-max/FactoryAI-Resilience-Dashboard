from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# These columns are not legitimate predictors of "Machine failure".
# UDI/Product ID are identifiers. TWF/HDF/PWF/OSF/RNF and Failure Type encode
# failure outcomes or failure-mode information and therefore leak the target.
IDENTIFIER_COLUMNS = {"udi", "uid", "product id", "product_id"}
FAILURE_LEAKAGE_COLUMNS = {
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "failure type",
    "failure_type",
}


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    target: str
    predictors: list[str]


def _canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("_", " "))


def normalize_ai4i_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common AI4I naming variants without changing scientific meaning."""
    rename: dict[str, str] = {}
    for col in df.columns:
        c = _canonical(str(col))
        if c in {"machine failure", "machinefailure"}:
            rename[col] = "Machine failure"
        elif c in {"air temperature", "air temperature [k]", "air temperature k"}:
            rename[col] = "Air temperature [K]"
        elif c in {"process temperature", "process temperature [k]", "process temperature k"}:
            rename[col] = "Process temperature [K]"
        elif c in {"rotational speed", "rotational speed [rpm]", "rotational speed rpm"}:
            rename[col] = "Rotational speed [rpm]"
        elif c in {"torque", "torque [nm]", "torque nm"}:
            rename[col] = "Torque [Nm]"
        elif c in {"tool wear", "tool wear [min]", "tool wear min"}:
            rename[col] = "Tool wear [min]"
    return df.rename(columns=rename)


def load_ai4i_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_ai4i_columns(df)


def select_predictors(df: pd.DataFrame, target: str = "Machine failure") -> list[str]:
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} was not found. Available columns: {list(df.columns)}")

    predictors: list[str] = []
    for col in df.columns:
        c = _canonical(str(col))
        if col == target:
            continue
        if c in IDENTIFIER_COLUMNS or c in FAILURE_LEAKAGE_COLUMNS:
            continue
        predictors.append(col)

    if not predictors:
        raise ValueError("No legitimate predictor columns remain after leakage controls.")
    return predictors


def prepare_ai4i(df: pd.DataFrame, target: str = "Machine failure") -> tuple[pd.DataFrame, list[str]]:
    df = normalize_ai4i_columns(df).copy()
    if target not in df.columns:
        raise KeyError(f"Expected binary target {target!r}.")

    unique = set(pd.Series(df[target]).dropna().astype(int).unique())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{target!r} must be binary 0/1; observed values: {sorted(unique)}")

    df[target] = df[target].astype(int)
    predictors = select_predictors(df, target)
    return df[predictors + [target]].copy(), predictors


def three_way_stratified_split(
    df: pd.DataFrame,
    target: str,
    seed: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> DataSplit:
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    prepared, predictors = prepare_ai4i(df, target)
    train, remainder = train_test_split(
        prepared,
        test_size=1.0 - train_fraction,
        random_state=seed,
        stratify=prepared[target],
    )
    relative_test = test_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed + 1,
        stratify=remainder[target],
    )

    # Explicit non-overlap check. This guards against accidental split reuse.
    if set(train.index) & set(validation.index):
        raise AssertionError("Train/validation overlap detected")
    if set(train.index) & set(test.index):
        raise AssertionError("Train/test overlap detected")
    if set(validation.index) & set(test.index):
        raise AssertionError("Validation/test overlap detected")

    return DataSplit(
        train=train.copy(),
        validation=validation.copy(),
        test=test.copy(),
        target=target,
        predictors=predictors,
    )


def class_distribution(df: pd.DataFrame, target: str) -> dict[str, float | int]:
    counts = df[target].value_counts().sort_index()
    total = len(df)
    return {
        "n": total,
        "negative_n": int(counts.get(0, 0)),
        "positive_n": int(counts.get(1, 0)),
        "positive_rate": float(counts.get(1, 0) / total),
    }
