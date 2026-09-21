#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI AI4I 2020 dataset via ucimlrepo.")
    parser.add_argument("--output", default="data/ai4i2020.csv")
    args = parser.parse_args()

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=601)
    original = getattr(dataset.data, "original", None)
    if original is not None:
        df = original.copy()
    else:
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
