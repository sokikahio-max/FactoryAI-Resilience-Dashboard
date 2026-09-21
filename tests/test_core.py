from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoryai_resilience.config import QualityPolicy, RecoveryPolicy, CostPolicy
from factoryai_resilience.data import prepare_ai4i, three_way_stratified_split
from factoryai_resilience.metrics import compute_metrics
from factoryai_resilience.perturbations import apply_perturbation
from factoryai_resilience.policy import controller, meaningful_execution_change
from factoryai_resilience.quality import assess_data_quality
from factoryai_resilience.schemas import ControllerState, ExecutionConfig


def make_df(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    torque = rng.normal(40, 10, n)
    wear = rng.integers(0, 250, n)
    failure = ((torque > 58) | (wear > 235)).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"L{i:05d}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 150, n),
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure,
            # These deliberately encode/follow the target and MUST be excluded.
            "TWF": (wear > 235).astype(int),
            "HDF": failure,
            "PWF": np.zeros(n, dtype=int),
            "OSF": failure,
            "RNF": np.zeros(n, dtype=int),
        }
    )



def test_ucimlrepo_unitless_headers_are_normalized():
    raw = make_df().rename(columns={
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear",
    })
    prepared, predictors = prepare_ai4i(raw)
    expected = {
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    }
    assert expected.issubset(set(predictors))
    assert expected.issubset(set(prepared.columns))

def test_leakage_columns_are_excluded():
    df, predictors = prepare_ai4i(make_df())
    for col in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert col not in predictors
    assert "Torque [Nm]" in predictors
    assert "Machine failure" in df.columns


def test_three_way_split_is_disjoint_and_stratified():
    df = make_df()
    split = three_way_stratified_split(df, "Machine failure", seed=11)
    assert len(split.train) == 600
    assert len(split.validation) == 200
    assert len(split.test) == 200
    assert not (set(split.train.index) & set(split.validation.index))
    assert not (set(split.train.index) & set(split.test.index))
    assert not (set(split.validation.index) & set(split.test.index))
    overall = df["Machine failure"].mean()
    assert abs(split.train["Machine failure"].mean() - overall) < 0.02


def test_perturbation_never_modifies_target():
    df, _ = prepare_ai4i(make_df())
    before = df["Machine failure"].copy()
    out, _ = apply_perturbation(
        df,
        name="measurement_noise",
        severity="mild",
        magnitude=0.2,
        features=["Torque [Nm]", "Process temperature [K]"],
        seed=1,
        target="Machine failure",
    )
    assert before.equals(out["Machine failure"])
    assert not np.allclose(df["Torque [Nm]"], out["Torque [Nm]"])


def test_severe_missingness_escalates():
    prepared, _ = prepare_ai4i(make_df())
    split = three_way_stratified_split(prepared, "Machine failure", seed=11)
    degraded, _ = apply_perturbation(
        split.validation,
        name="missingness",
        severity="severe",
        magnitude=0.35,
        features=["Torque [Nm]"],
        seed=2,
        target="Machine failure",
    )
    quality = assess_data_quality(split.train, degraded, "Machine failure", QualityPolicy())
    y = degraded["Machine failure"].to_numpy()
    # Dummy probabilities are enough because severe quality must dominate the controller decision.
    prob = np.clip(0.1 + 0.8 * y, 0.01, 0.99)
    metrics = compute_metrics(y, prob, CostPolicy())
    decision = controller(
        quality=quality,
        current=metrics,
        clean_reference=metrics,
        recovery_policy=RecoveryPolicy(),
        retry_index=0,
    )
    assert decision.state == ControllerState.ESCALATE



def test_seed_only_retry_is_not_meaningful_reconfiguration():
    before = ExecutionConfig(
        include_algos=["GBM", "DRF", "GLM"], balance_classes=True, sort_metric="AUCPR",
        drop_features=[], max_models=5, seed=11, nfolds=5
    )
    seed_only = before.model_copy(update={"seed": 10011})
    actual_change = before.model_copy(update={"include_algos": ["GBM", "DRF"], "seed": 10011})
    assert not meaningful_execution_change(before, seed_only)
    assert meaningful_execution_change(before, actual_change)

if __name__ == "__main__":
    test_ucimlrepo_unitless_headers_are_normalized()
    test_leakage_columns_are_excluded()
    test_three_way_split_is_disjoint_and_stratified()
    test_perturbation_never_modifies_target()
    test_severe_missingness_escalates()
    test_seed_only_retry_is_not_meaningful_reconfiguration()
    print("All core tests passed.")
