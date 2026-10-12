"""Print the series ids and descriptions in a manually downloaded RBNZ workbook.

    uv run python scripts/inspect_rbnz.py data/econ_data/rbnz/hm1.xlsx

Use the output to fill ``rbnz_series`` in ``configs/variables.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from economic_lab.data.readers import read_rbnz_xlsx


def main() -> None:
    path = Path(sys.argv[1])
    data, meta = read_rbnz_xlsx(path)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 80):
        print(meta)
    print(f"\n{data.shape[1]} series, {data.index.min():%Y-%m} to {data.index.max():%Y-%m}")


if __name__ == "__main__":
    main()
