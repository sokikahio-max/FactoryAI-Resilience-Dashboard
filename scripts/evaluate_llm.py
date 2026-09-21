#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoryai_resilience.config import load_config
from factoryai_resilience.llm_eval import run_llm_evaluation
from factoryai_resilience.planner import build_planner


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly evaluate the structured LLM planner.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--cases", default="configs/llm_eval_cases.yaml")
    parser.add_argument("--planner", choices=["mock", "openai"], default="openai")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-dir", default="results/llm_eval")
    args = parser.parse_args()

    config = load_config(args.config)
    config = replace(config, planner=replace(config.planner, mode=args.planner))
    planner = build_planner(config.planner)
    raw, summary = run_llm_evaluation(planner, config, args.cases, args.repetitions)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "llm_eval_raw.csv", index=False)
    summary.to_csv(out / "llm_eval_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved evaluation outputs to {out}")


if __name__ == "__main__":
    main()
