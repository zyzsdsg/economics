# Economic Modelling Laboratory

A reproducible, educational research codebase on one question:

> **How do oil-price movements, and economically identified oil-price shocks, affect inflation in New Zealand?**

Every estimator is implemented twice: **from first principles** (NumPy/SciPy, with the
derivation in the accompanying notebook) and **with a production library** (statsmodels,
later scikit-learn/PyMC), and the two are checked against each other in the test suite.

## Research design

| Track | Question | Methods | Status |
|---|---|---|---|
| A. Forecasting | Does oil-price information improve inflation forecasts? | AR, ARDL, VAR/BVAR, regularised regression, tree ensembles, mixed frequency | ARDL done; rest planned |
| B. Structural shocks | What do *identified* oil shocks do to inflation, output, rates and the exchange rate? | SVAR, local projections, semi-structural model, DSGE | planned |
| C. Sectoral | How do energy-cost increases propagate across NZ industries? | Input–output (Leontief), CGE | planned |

The project keeps four claims separate: **prediction ≠ explanation ≠ causal identification ≠ structural counterfactual**.
Phase 1 results below are *conditional associations*, not causal effects of oil supply shocks.

## Status

| Phase | Content | Status |
|---|---|---|
| 1. Foundation | Data acquisition, provenance, validation, frequency alignment; OLS, unit roots, ARDL/ECM | ✅ |
| 2. Advanced econometrics | VAR, BVAR, SVAR, local projections | planned |
| 3. Machine learning | Ridge/Lasso/EN, RF, XGBoost, rolling backtests, mixed frequency | planned |
| 4. Semi-structural | Phillips curve, IS curve, Taylor rule, state space | planned |
| 5. DSGE | NK and small open economy with energy; perturbation, Kalman filter, Bayesian estimation | planned |
| 6. CGE | Input–output, SAM, calibrated CGE | planned |
| 7. Integration | Comparison framework, report, presentation | planned |

## Repository layout

```text
economics/
├── configs/variables.yaml        # variable registry: units, sources, construction; RBNZ series ids
├── data/
│   ├── econ_data/                # raw downloads (git-ignored) + manifest.json (committed)
│   └── processed/                # macro_{daily,monthly,quarterly}.parquet (git-ignored, rebuilt)
├── docs/                         # data sources & transformations, tutorials
├── notebooks/                    # 00–05: teaching notebooks (Chinese explanations, English code)
├── scripts/
│   ├── data_download.py          # download automatic sources, register manual ones
│   ├── build_datasets.py         # raw -> validated, frequency-aligned parquet
│   └── inspect_rbnz.py           # list series ids in a manually downloaded RBNZ workbook
├── src/economic_lab/
│   ├── data/                     # sources, downloader, provenance, readers, validation, transforms, synthetic, build
│   ├── econometrics/             # ols (QR/SVD/HAC/Wald/delta method), timeseries (ACF, Ljung-Box, ADF), ardl (ARDL/UECM)
│   ├── visualization/            # shared plotting style
│   └── utils/                    # project paths
└── tests/                        # pytest: transforms, I/O, estimators vs statsmodels, notebook smoke tests
```

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zyzsdsg/economics.git
cd economics
uv sync                          # installs dependencies and economic_lab (editable)
uv sync --extra interactive      # optional: ipywidgets sliders in notebooks
```

If the project lives in a cloud-synced folder (iCloud Drive, Dropbox), keep the virtual
environment outside it, e.g. `export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/economic-lab"`.
Imports from a synced `.venv` can be extremely slow.

## Reproducing

```bash
uv run python scripts/data_download.py     # EIA, FRED, Dallas Fed, World Bank (RBNZ: manual, see below)
uv run python scripts/build_datasets.py    # validation report + data/processed/*.parquet
uv run pytest                              # unit tests
uv run pytest -m notebook                  # execute all notebooks on synthetic data
uv run ruff check src tests scripts
```

Without downloaded data, `load_dataset()` falls back to a **synthetic** dataset with a
known data-generating process, and every notebook prints `DATA KIND: SYNTHETIC`. CI runs
entirely offline this way.

## Data

See [docs/data_sources.md](docs/data_sources.md) for URLs, units, permissions and every
transformation. Highlights:

- Official sources only: U.S. EIA (Brent), Federal Reserve H.10 via FRED (NZD/USD), Dallas Fed (global activity), World Bank Pink Sheet, RBNZ and Stats NZ.
- **RBNZ restricts automated access**; its workbooks (M1, M5, M9, M14, B1, B2) are downloaded manually and registered with a checksum. CPI and its tradable/non-tradable components, GDP and expenditure components, unemployment, inflation expectations, TWI and the OCR come from these files. The OECD CPI on FRED is used only as a fallback and cross-check.
- Each raw file is recorded with URL, timestamp and SHA-256; refreshed files keep the previous vintage.
- Validation caught real issues in official files: zero placeholders in historical CPI, mid-month dates, and EIA daily Brent covering as little as 49% of business days in 1995.

## Example results (Phase 1)

Real data, 1992Q1–2026Q2 (138 quarters); CPI from RBNZ M1; raw files downloaded
2026-10-12/13. HAC (Newey–West, 4 lags) standard errors. GST increases in 1989Q3 and
2010Q4 are controlled for with a dummy.

| Model | Oil coefficient | Interpretation |
|---|---|---|
| Static OLS, q/q CPI inflation on Δ log NZD oil price | 0.0174 (se 0.0030) | A 10% oil price rise is associated with +0.17 pp q/q inflation in the same quarter |
| ARDL(2,1), long-run multiplier | 0.048 (se 0.013) | A permanent 10% rise is associated with a ≈0.5% higher CPI level; 0.03–0.07 across lag specifications |
| UECM in log levels, bounds test | F = 2.18 (p = 0.49 at I(1) bound) | No evidence of a level (cointegrating) relationship between CPI and oil prices |

By CPI component (ARDL(2,1), common sample 1999Q3–2026Q2, n = 106):

| Price index | Long-run θ (se) | 10% permanent oil rise → long-run price level |
|---|---|---|
| Headline CPI | 0.049 (0.012) | +0.5% |
| Tradable CPI | 0.081 (0.016) | +0.8% |
| Non-tradable CPI | 0.009 (0.009) | not distinguishable from 0 |

Splitting the NZD oil price into the USD oil price and NZD/USD shows the exchange-rate
coefficient is larger in magnitude, consistent with the exchange rate affecting all imports.
These numbers are reduced-form associations. Oil price changes mix supply, global-demand
and oil-specific demand shocks, and NZ demand co-moves with global demand. Identification is the subject of Track B.

## Limitations (current)

- Single-equation models; no identification of oil shocks yet.
- Stats NZ CPI is not seasonally adjusted (quarter-dummy F-test p = 0.04 headline, 0.003 tradable and non-tradable); seasonal dummies are an exercise, not yet the baseline.
- The joint quarterly sample with OCR, TWI and tradable/non-tradable CPI is 1999Q2–2026Q2 (109 quarters): small for multivariate models, which motivates shrinkage (BVAR) in Phase 2.
- No output-gap series yet (it has to be estimated).

## License

MIT, see [LICENSE](LICENSE).
