"""Registry of raw data sources.

Each entry describes one raw file. ``filename`` is relative to ``data/econ_data/``.

``mode`` controls acquisition:

- ``"auto"``             -- direct file URL, verified to permit scripted download.
- ``"auto_discover"``    -- the file URL changes each release; it is discovered by
                            searching ``landing_page`` for ``discover_pattern``.
- ``"rbnz_permission"``  -- RBNZ statistics. RBNZ restricts automated access to its
                            website (scripted requests receive HTTP 403), so these
                            files must be downloaded manually in a browser and saved
                            under ``filename``. Never bypass this restriction.
- ``"manual"``           -- other sources that need a browser or an API key.

URLs were checked on 2026-10-12. ``url=None`` means only the landing page is known;
the exact file link should be recorded in the manifest when you download it.
"""

from __future__ import annotations

from typing import Any

RBNZ_INDEX_PAGE = "https://www.rbnz.govt.nz/statistics/series/data-file-index-page"
RBNZ_SERIES = "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series"

SOURCES: dict[str, dict[str, Any]] = {
    # --- International: oil and global activity ---------------------------------
    "brent_daily": {
        "url": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",
        "filename": "eia/brent_daily.xls",
        "mode": "auto",
        "provider": "U.S. Energy Information Administration",
        "description": "Europe Brent Spot Price FOB",
        "frequency": "daily",
        "units": "USD per barrel",
        "landing_page": "https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm",
    },
    "brent_monthly": {
        "url": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls",
        "filename": "eia/brent_monthly.xls",
        "mode": "auto",
        "provider": "U.S. Energy Information Administration",
        "description": "Europe Brent Spot Price FOB, monthly average",
        "frequency": "monthly",
        "units": "USD per barrel",
        "landing_page": "https://www.eia.gov/dnav/pet/pet_pri_spt_s1_m.htm",
    },
    "nzd_usd_daily": {
        # FRED redistributes the Federal Reserve Board H.10 noon buying rates.
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSNZ",
        "filename": "fred/DEXUSNZ.csv",
        "mode": "auto",
        "provider": "Board of Governors of the Federal Reserve System (H.10) via FRED",
        "description": "U.S. dollars to one New Zealand dollar, noon buying rate in New York",
        "frequency": "daily",
        "units": "USD per NZD",
        "landing_page": "https://fred.stlouisfed.org/series/DEXUSNZ",
        "notes": "Secondary to RBNZ B1, but freely scriptable. Missing values are '.' or blank.",
    },
    "nz_cpi_oecd": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NZLCPIALLQINMEI",
        "filename": "fred/NZLCPIALLQINMEI.csv",
        "mode": "auto",
        "provider": "OECD Main Economic Indicators via FRED (original source: Stats NZ)",
        "description": "Consumer Price Index: All items for New Zealand",
        "frequency": "quarterly",
        "units": "Index 2015=100, not seasonally adjusted",
        "landing_page": "https://fred.stlouisfed.org/series/NZLCPIALLQINMEI",
        "notes": (
            "Fallback headline CPI only. The OECD MEI series is discontinued (last obs "
            "2025Q1 when checked). Prefer RBNZ M1 / Stats NZ, which also provide "
            "tradable and non-tradable CPI."
        ),
    },
    "dallasfed_igrea": {
        "url": "https://www.dallasfed.org/-/media/Documents/research/igrea/igrea.xlsx",
        "filename": "dallasfed/igrea.xlsx",
        "mode": "auto",
        "provider": "Federal Reserve Bank of Dallas",
        "description": "Index of Global Real Economic Activity (Kilian 2009, updated)",
        "frequency": "monthly",
        "units": "Percent deviation from trend",
        "landing_page": "https://www.dallasfed.org/research/igrea",
    },
    "worldbank_pinksheet_monthly": {
        "url": None,
        "filename": "worldbank/CMO-Historical-Data-Monthly.xlsx",
        "mode": "auto_discover",
        "landing_page": "https://www.worldbank.org/en/research/commodity-markets",
        "discover_pattern": r"https?://[^\"']*CMO-Historical-Data-Monthly\.xlsx",
        "provider": "World Bank Commodity Markets (Pink Sheet)",
        "description": "Monthly commodity prices",
        "frequency": "monthly",
        "units": "Nominal USD (varies by commodity)",
        "notes": "The file URL contains a hash that changes with each monthly release.",
    },
    # --- New Zealand: RBNZ (manual download only) --------------------------------
    # Files keep the names RBNZ publishes them under, so a re-download can simply
    # overwrite them. Series ids used from each file are listed in configs/variables.yaml.
    "rbnz_cpi": {
        "url": f"{RBNZ_SERIES}/m/m1/hm1.xlsx",
        "filename": "rbnz/hm1.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "M1 Prices: CPI, tradable and non-tradable CPI, petrol, PPI",
        "frequency": "quarterly",
        "landing_page": RBNZ_INDEX_PAGE,
    },
    "rbnz_gdp": {
        "url": f"{RBNZ_SERIES}/m/m5/hm5.xlsx",
        "filename": "rbnz/hm5.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "M5 GDP and expenditure components (consumption, investment, trade)",
        "frequency": "quarterly",
        "landing_page": RBNZ_INDEX_PAGE,
    },
    "rbnz_exchange_rates": {
        "url": f"{RBNZ_SERIES}/b/b1/hb1-daily.xlsx",
        "filename": "rbnz/hb1-daily.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "B1 Exchange rates and TWI, daily (this file starts in 2018)",
        "frequency": "daily",
        "landing_page": RBNZ_INDEX_PAGE,
        "notes": "Earlier daily history is in hb1-daily-1999-2017.xlsx; long monthly history in hb1-monthly.xlsx.",
    },
    "rbnz_interest_rates": {
        "url": f"{RBNZ_SERIES}/b/b2/hb2-daily-close.xlsx",
        "filename": "rbnz/hb2-daily-close.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "B2 Wholesale interest rates incl. the OCR, daily close (this file starts in 2018)",
        "frequency": "daily",
        "landing_page": RBNZ_INDEX_PAGE,
        "notes": "Long monthly history is in hb2-monthly-close.xlsx.",
    },
    "rbnz_interest_rates_hist": {
        "url": None,
        "filename": "rbnz/hb2-daily-close-1985-2017.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "B2 Wholesale interest rates, daily close 1985-2017 (OCR from 1999-03, bills/bonds from 2009)",
        "frequency": "daily",
        "landing_page": (
            "https://www.rbnz.govt.nz/statistics/series/exchange-and-interest-rates/wholesale-interest-rates"
        ),
    },
    "rbnz_exchange_rates_monthly": {
        "url": f"{RBNZ_SERIES}/b/b1/hb1-monthly.xlsx",
        "filename": "rbnz/hb1-monthly.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "B1 Exchange rates and TWI, monthly averages (1999-)",
        "frequency": "monthly",
        "landing_page": RBNZ_INDEX_PAGE,
    },
    "rbnz_interest_rates_monthly": {
        "url": f"{RBNZ_SERIES}/b/b2/hb2-monthly-close.xlsx",
        "filename": "rbnz/hb2-monthly-close.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "B2 Wholesale interest rates, monthly (2018-); OCR is month-end, others monthly averages",
        "frequency": "monthly",
        "landing_page": RBNZ_INDEX_PAGE,
        "notes": "Registered for provenance only; pre-2018 history is in the B2 Daily close (1985-2017) table.",
    },
    "rbnz_labour": {
        "url": None,
        "filename": "rbnz/hm9.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "M9 Labour market: unemployment, employment, LCI, wages",
        "frequency": "quarterly",
        "landing_page": RBNZ_INDEX_PAGE,
    },
    "rbnz_expectations": {
        "url": None,
        "filename": "rbnz/hm14.xlsx",
        "mode": "rbnz_permission",
        "provider": "Reserve Bank of New Zealand",
        "description": "M14 Survey of expectations (inflation, GDP, OCR, exchange rate)",
        "frequency": "quarterly",
        "landing_page": RBNZ_INDEX_PAGE,
    },
    # --- New Zealand: other official sources (manual) ----------------------------
    "statsnz_large_datasets": {
        "url": None,
        "filename": "statsnz/",
        "mode": "manual",
        "provider": "Stats NZ",
        "description": "Detailed CPI, national accounts, input-output tables (CSV releases)",
        "frequency": "various",
        "landing_page": "https://www.stats.govt.nz/large-datasets/csv-files-for-download/",
        "notes": "Aotearoa Data Explorer (https://explore.data.stats.govt.nz/) offers an API with a key.",
    },
    "mbie_weekly_fuel": {
        "url": None,
        "filename": "mbie/",
        "mode": "manual",
        "provider": "Ministry of Business, Innovation and Employment",
        "description": "Weekly retail fuel price monitoring",
        "frequency": "weekly",
        "landing_page": (
            "https://www.mbie.govt.nz/building-and-energy/energy-and-natural-resources/"
            "energy-statistics-and-modelling/energy-statistics/weekly-fuel-price-monitoring/"
        ),
    },
}
