#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoryai_resilience.backends import build_backend
from factoryai_resilience.config import load_config
from factoryai_resilience.data import load_ai4i_csv, prepare_ai4i
from factoryai_resilience.experiments import run_experiment_suite
from factoryai_resilience.planner import build_planner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free repeated AI4I ablation experiments.")
    parser.add_argument("--data", default="data/ai4i2020.csv")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default="results/experiment_runs.csv")
    parser.add_argument("--backend", choices=["h2o", "sklearn_smoke"], default=None)
    parser.add_argument("--planner", choices=["mock", "openai"], default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Optional number of leading configured seeds for a quick run.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV, skipping completed seed/method/condition rows.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Dataclasses are frozen; use dataclasses.replace for command-line overrides.
    from dataclasses import replace
    if args.backend:
        config = replace(config, backend=replace(config.backend, name=args.backend))
    if args.planner:
        config = replace(config, planner=replace(config.planner, mode=args.planner))
    if args.seeds:
        config = replace(config, experiment=replace(config.experiment, seeds=config.experiment.seeds[: args.seeds]))

    df = load_ai4i_csv(args.data)
    prepared, predictors = prepare_ai4i(df, config.experiment.target)
    print(f"Loaded {len(prepared):,} AI4I rows")
    print(f"Target: {config.experiment.target}")
    print(f"Legitimate predictors ({len(predictors)}): {predictors}")

    backend = build_backend(
        config.backend.name,
        h2o_port=config.backend.h2o_port,
        h2o_max_mem_size=config.backend.h2o_max_mem_size,
    )
    planner = build_planner(config.planner)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_experiment_suite(
        df,
        config=config,
        backend=backend,
        planner=planner,
        output_csv=output,
        resume=args.resume,
    )
    print(f"\nCompleted {len(result):,} condition-method runs. Results: {output}")


if __name__ == "__main__":
    main()
