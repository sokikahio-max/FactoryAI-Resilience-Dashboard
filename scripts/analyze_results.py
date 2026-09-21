#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoryai_resilience.statistics import write_analysis_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment_runs.csv")
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()
    write_analysis_tables(args.results, args.output_dir)
    print(f"Analysis tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
