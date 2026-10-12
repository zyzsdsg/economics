"""Execute every teaching notebook end-to-end on synthetic data.

Run with ``uv run pytest -m notebook``. Notebooks must not depend on downloaded data,
network access or variables created by other notebooks.
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))


@pytest.mark.notebook
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs_on_synthetic_data(path, monkeypatch):
    monkeypatch.setenv("ECONOMIC_LAB_FORCE_SYNTHETIC", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_DIR)}}
    )
    client.execute()
    kinds = [
        "".join(o.get("text", ""))
        for c in nb.cells
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    ]
    if any("DATA KIND" in k for k in kinds):
        assert any("DATA KIND: SYNTHETIC" in k for k in kinds)
