"""Build data/processed/macro_{daily,monthly,quarterly}.parquet.

    uv run python scripts/build_datasets.py              # from raw downloads
    uv run python scripts/build_datasets.py --synthetic  # offline synthetic data
"""

from __future__ import annotations

import argparse

from economic_lab.data.build import build_from_raw, build_synthetic, write_datasets
from economic_lab.data.validation import missing_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    result = build_synthetic() if args.synthetic else build_from_raw()
    for rep in result.reports:
        print(rep)
    for note in result.notes:
        print("NOTE:", note)
    paths = write_datasets(result)
    for freq, path in paths.items():
        df = getattr(result, freq)
        print(f"\n{freq}: {path} ({len(df)} rows, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d})")
        print(missing_report(df).to_string())


if __name__ == "__main__":
    main()
