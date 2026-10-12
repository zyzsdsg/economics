# Data sources, permissions and transformations

All URLs were checked on 2026-10-12. The registry in code is
`src/economic_lab/data/sources.py`; variable definitions are in `configs/variables.yaml`.

## Automatic sources

| Key | Provider | Series | Freq. | Units | Notes |
|---|---|---|---|---|---|
| `brent_daily` | U.S. EIA | Europe Brent spot FOB (`RBRTE`) | daily | USD/bbl | Sparse in 1993–2005 (49–88% of business days per year) |
| `brent_monthly` | U.S. EIA | Same, monthly average | monthly | USD/bbl | Dated the 15th; normalised to month start |
| `nzd_usd_daily` | Federal Reserve H.10 via FRED (`DEXUSNZ`) | NZD/USD noon buying rate, New York | daily | USD per NZD | Missing values are blank |
| `nz_cpi_oecd` | OECD MEI via FRED (`NZLCPIALLQINMEI`) | NZ CPI, all items | quarterly | 2015=100, NSA | **Fallback only**: discontinued after 2025Q1; 27 zero placeholders before 1926 |
| `dallasfed_igrea` | Federal Reserve Bank of Dallas | Index of Global Real Economic Activity | monthly | % deviation from trend | Recent rows dated mid-month; normalised |
| `worldbank_pinksheet_monthly` | World Bank | Commodity prices (Pink Sheet) | monthly | nominal USD | URL changes monthly; discovered from the Commodity Markets page |

Run `uv run python scripts/data_download.py`. Existing files are never overwritten;
`--refresh` archives the previous file under `_vintages/` before downloading.

## Manual sources

**RBNZ restricts automated access to its website** (scripted requests return HTTP 403).
Download these files in a browser from the
[data file index page](https://www.rbnz.govt.nz/statistics/series/data-file-index-page)
and save them under `data/econ_data/rbnz/` **with the file names RBNZ uses**. Do not
attempt to bypass the restriction.

| Key | Table | File | Coverage (2026-10 download) |
|---|---|---|---|
| `rbnz_cpi` | M1 Prices | `hm1.xlsx` | 1988Q1– (tradable/non-tradable from 1999Q2) |
| `rbnz_gdp` | M5 GDP | `hm5.xlsx` | 1987Q2– |
| `rbnz_labour` | M9 Labour market | `hm9.xlsx` | 1994Q1– |
| `rbnz_expectations` | M14 Survey of expectations | `hm14.xlsx` | 1987Q3– |
| `rbnz_exchange_rates` | B1 Exchange rates, daily | `hb1-daily.xlsx` | 2018-01– |
| `rbnz_interest_rates` | B2 Wholesale rates, daily close | `hb2-daily-close.xlsx` | 2018-01– |
| `rbnz_interest_rates_hist` | B2 daily close 1985–2017 ([B2 page](https://www.rbnz.govt.nz/statistics/series/exchange-and-interest-rates/wholesale-interest-rates)) | `hb2-daily-close-1985-2017.xlsx` | OCR 1999-03–; bills/bonds 2009-01– |
| `rbnz_exchange_rates_monthly` | B1 monthly averages | `hb1-monthly.xlsx` | 1999-01– |
| `rbnz_interest_rates_monthly` | B2 monthly close (registered only) | `hb2-monthly-close.xlsx` | 2018-01– |

The series taken from each file are listed under `rbnz_series` in
`configs/variables.yaml`, with ids copied from each workbook's *Series Definitions* sheet.

After saving or replacing a file:

1. `uv run python scripts/data_download.py` registers it in `data/econ_data/manifest.json` (SHA-256, `method: manual`).
2. `uv run python scripts/inspect_rbnz.py data/econ_data/rbnz/hm1.xlsx` prints the series ids and descriptions in a file.
3. `uv run python scripts/build_datasets.py` rebuilds the datasets.

Known gaps:

- **TWI** at monthly/quarterly frequency comes from `hb1-monthly.xlsx` (1999–). Before
  splicing, `build.splice_check` confirms that it equals the monthly average of the daily
  file (max gap 0.009% over 2018–2026). `twi_real` (real TWI) is only in the monthly file.
- **OCR and wholesale rates** join `hb2-daily-close-1985-2017.xlsx` (same series ids) onto
  `hb2-daily-close.xlsx` with `build.join_vintages`, which rejects files that disagree on
  overlapping dates and records the change across the seam (OCR 0.00, 90-day bills +0.01,
  10-year bond +0.06 pp: ordinary day-to-day moves, no level break). Quarterly OCR covers
  1999Q2– (the OCR began on 17 March 1999, so 1999Q1 fails the 80% coverage rule);
  bank bills and the 10-year bond cover 2009Q2– because the closing-rate history starts in 2009.
- The common gap-free quarterly sample of oil, IGREA, CPI (headline/tradable/non-tradable),
  real GDP, unemployment, OCR, TWI and 1-year inflation expectations is **1999Q2–2026Q2 (109 quarters)**.
- In `hb2-monthly-close.xlsx` the **OCR is the month-end value**, while bank bills and bond
  yields are **monthly averages** (checked against the daily file). The build uses
  daily averages throughout, so every rate has the same definition.
- RBNZ publishes the CPI index rounded to whole points (2017Q2 = 1000), so q/q inflation
  computed from it can differ from Stats NZ's published rate by about 0.1 pp. Against the
  OECD/FRED CPI the largest q/q difference is 0.16 pp.
- Output gap is not an official series. It must be estimated (e.g. HP filter or a
  production-function or state-space model) and is not stored in the raw-data build.
- CPI includes GST changes (1989Q3: 10%→12.5%; 2010Q4: 12.5%→15%) that cause one-off
  jumps in headline inflation; consider dummies for those quarters.

Stats NZ (detailed CPI, national accounts, input–output tables):
<https://www.stats.govt.nz/large-datasets/csv-files-for-download/> and the Aotearoa Data
Explorer <https://explore.data.stats.govt.nz/> (API key required).
MBIE weekly fuel prices:
<https://www.mbie.govt.nz/building-and-energy/energy-and-natural-resources/energy-statistics-and-modelling/energy-statistics/weekly-fuel-price-monitoring/>.

## Provenance

`data/econ_data/manifest.json` (committed) records, for each raw file: URL, method,
UTC timestamp, SHA-256, size and HTTP `Last-Modified`/`ETag`. When a refreshed file has a
different checksum, the old entry moves to `history`. Each processed dataset has a
`.meta.json` sidecar listing the checksums of the raw inputs used to build it.

## Standardised datasets (`data/processed/`, not committed)

| File | Index | Columns |
|---|---|---|
| `macro_daily.parquet` | business days | `brent_usd`, `nzd_usd`, `oil_nzd`, `twi`, `ocr`, `bank_bill_90d`, `govt_bond_10y` |
| `macro_monthly.parquet` | first day of month | `brent_usd`, `nzd_usd`, `oil_nzd`, `oil_nzd_from_daily`, `igrea`, `twi`, `ocr`, `bank_bill_90d`, `govt_bond_10y` |
| `macro_quarterly.parquet` | first day of quarter | `brent_usd`, `nzd_usd`, `oil_nzd`, `oil_nzd_from_daily`, `igrea`, plus every variable in `rbnz_series` (CPI and its components, GDP and expenditure, unemployment, LCI, inflation expectations, TWI, OCR, interest rates) |

Only **levels** are stored. Rates are computed in model code with
`economic_lab.data.transforms`.

## Transformation rules

| Rule | Implementation |
|---|---|
| Monthly/quarterly series are dated at the **start** of the period | `transforms.to_period_start` |
| NZD oil price: `oil_nzd = oil_usd / nzd_usd` (NZD/USD quoted as USD per NZD) | `transforms.convert_currency` (inner join, no forward fill) |
| Daily → monthly/quarterly averages require ≥ 80% business-day coverage | `transforms.aggregate` |
| Monthly → quarterly requires all three months | `build.to_quarter` |
| Quarterly `brent_usd` = mean of EIA monthly averages; `oil_nzd` = monthly ratio, then quarterly mean | `build.build_from_raw` |
| q/q inflation `100 (P_t/P_{t-1} - 1)`; y/y `100 (P_t/P_{t-4} - 1)` | `transforms.pct_change`, `transforms.yoy` |
| Log change `100 Δ ln x` (log points) | `transforms.log_diff` |
| Percentage-point change `r_t - r_{t-k}` | `transforms.pp_change` |
| Compounding to annual rates `(1 + g/100)^m - 1` | `transforms.annualise` |
| Shifts/growth rates refuse gappy indices | `transforms.require_regular` |
| Zero placeholders in OECD CPI dropped; longest contiguous run kept | `build.build_from_raw` |

## Seasonal adjustment

Stats NZ CPI is **not seasonally adjusted**. F-tests of quarter dummies on q/q inflation
(`validation.seasonality_test`, RBNZ M1, data to 2026Q2) give p = 0.044 for headline CPI
(1992Q1–), p = 0.003 for tradable and p = 0.003 for non-tradable CPI (1999Q3–), so
quarterly models should include seasonal dummies or use y/y rates. GDP, consumption,
investment, trade and unemployment from M5/M9 are already seasonally adjusted (`sa: true`
in `configs/variables.yaml`). In the GST quarters, q/q CPI inflation was 3.5% (1989Q3)
and 2.3% (2010Q4), against a median of 0.5% since 1992.
