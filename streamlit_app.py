from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from factoryai_resilience.backends import build_backend
from factoryai_resilience.config import load_config
from factoryai_resilience.data import class_distribution, load_ai4i_csv, three_way_stratified_split
from factoryai_resilience.metrics import compute_metrics
from factoryai_resilience.planner import PlanningContext, build_planner
from factoryai_resilience.policy import controller, decision_to_execution_config
from factoryai_resilience.quality import assess_data_quality


st.set_page_config(page_title="Failure-Aware LLM-AutoML", layout="wide")
st.title("Failure-Aware LLM–AutoML Decision Support")
st.caption("Research demonstration. The final test set is not used for retry, threshold selection, or model selection.")

uploaded = st.file_uploader("Upload the AI4I 2020 CSV", type="csv")
planner_mode = st.selectbox("Planner", ["mock", "openai"], help="Use mock for a token-free demonstration.")
backend_name = st.selectbox("ML backend", ["sklearn_smoke", "h2o"], help="Use H2O for manuscript experiments.")
problem = st.text_area(
    "Problem description",
    "Predict machine-failure events from process and sensor measurements for maintenance decision support.",
)

if uploaded and st.button("Run leakage-free workflow"):
    config = load_config(ROOT / "configs" / "experiment.yaml")
    config = replace(
        config,
        planner=replace(config.planner, mode=planner_mode),
        backend=replace(config.backend, name=backend_name),
        experiment=replace(config.experiment, seeds=[config.experiment.seeds[0]]),
    )
    df = pd.read_csv(uploaded)
    split = three_way_stratified_split(df, config.experiment.target, config.experiment.seeds[0])
    quality = assess_data_quality(split.train, split.validation, split.target, config.quality_policy)

    st.subheader("1. Leakage controls and split")
    st.write("Predictors used:", split.predictors)
    st.write({"train": len(split.train), "validation": len(split.validation), "untouched_test": len(split.test)})

    st.subheader("2. Local data-quality assessment")
    st.json(quality.to_dict())

    planner = build_planner(config.planner)
    dist = class_distribution(split.train, split.target)
    plan, telemetry = planner.plan(
        PlanningContext(
            problem_description=problem,
            target=split.target,
            predictors=split.predictors,
            positive_rate=float(dist["positive_rate"]),
            quality=quality,
        )
    )
    st.subheader("3. Structured LLM planning output")
    st.json(plan.model_dump())
    st.caption(f"Planner latency: {telemetry.latency_seconds:.3f}s; no raw rows are sent to the LLM.")

    execution = decision_to_execution_config(
        plan,
        predictors=split.predictors,
        max_models=config.experiment.initial_models_recovery_methods,
        seed=config.experiment.seeds[0],
        nfolds=config.experiment.nfolds,
        exclude_deep_learning=config.experiment.exclude_deep_learning_for_reproducibility,
    )
    st.subheader("4. Validated executable configuration")
    st.json(execution.model_dump())

    backend = build_backend(
        backend_name,
        h2o_port=config.backend.h2o_port,
        h2o_max_mem_size=config.backend.h2o_max_mem_size,
    )
    model = backend.fit_automl(split.train, split.validation, split.target, split.predictors, execution)
    val_prob = backend.predict_proba(model, split.validation)
    val_metrics = compute_metrics(split.validation[split.target], val_prob, config.cost_policy)
    state = controller(
        quality=quality,
        current=val_metrics,
        clean_reference=None,
        recovery_policy=config.recovery_policy,
        retry_index=0,
    )

    st.subheader("5. Validation-only decision")
    st.json({**val_metrics.to_dict(), "controller_state": state.state.value, "reason": state.reason})

    # Only now is the untouched test set evaluated.
    test_prob = backend.predict_proba(model, split.test)
    test_metrics = compute_metrics(
        split.test[split.target], test_prob, config.cost_policy, threshold=val_metrics.threshold
    )
    st.subheader("6. Final untouched-test evaluation")
    st.json(test_metrics.to_dict())

    global_exp, local_exp = backend.global_local_explanations(model, split.test.head(100), local_row=0)
    if global_exp is not None:
        st.subheader("7. Explanation")
        st.write("Global contribution/importance summary")
        st.dataframe(global_exp.head(12))
        if local_exp is not None:
            st.write("Local contribution example")
            st.dataframe(local_exp.head(12))
