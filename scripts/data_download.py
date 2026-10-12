"""Download raw data files listed in ``economic_lab.data.sources.SOURCES``.

Usage
-----
    uv run python scripts/data_download.py              # download missing auto files
    uv run python scripts/data_download.py --refresh    # fetch new vintages (old ones archived)
    uv run python scripts/data_download.py --only brent_daily nzd_usd_daily

Manual sources (RBNZ, Stats NZ, MBIE) are never fetched. Download them in a browser,
save them under the printed path, and rerun this script: it records their checksum
in ``data/econ_data/manifest.json``.
"""

from __future__ import annotations

import argparse

from economic_lab.data.downloader import discover_url, download_data
from economic_lab.data.provenance import load_manifest, record_file
from economic_lab.data.sources import SOURCES
from economic_lab.utils.paths import manifest_path, raw_data_dir

DATA_DIR = raw_data_dir()
MANIFEST = manifest_path()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download existing files")
    parser.add_argument("--only", nargs="*", help="source names to process")
    args = parser.parse_args()

    recorded = load_manifest(MANIFEST)["files"]

    for name, source in SOURCES.items():
        if args.only and name not in args.only:
            continue
        file_path = DATA_DIR / source["filename"]

        if source["mode"] in ("rbnz_permission", "manual"):
            if file_path.is_file():
                if name not in recorded:
                    record_file(MANIFEST, name, file_path, url=source.get("url"), method="manual")
                    print(f"Registered manual file for {name}: {file_path}")
                continue
            print(f"Manual download required: {name} ({source['provider']})")
            print(f"  Landing page: {source['landing_page']}")
            if source.get("url"):
                print(f"  File link:    {source['url']}")
            print(f"  Save as:      {file_path}")
            continue

        try:
            url = source["url"]
            if source["mode"] == "auto_discover":
                url = discover_url(source["landing_page"], source["discover_pattern"])
            existed = file_path.exists()
            download_data(
                url, file_path, overwrite=args.refresh, manifest=MANIFEST, source_name=name
            )
            if existed and not args.refresh and name not in recorded:
                # File was downloaded before provenance tracking existed.
                record_file(MANIFEST, name, file_path, url=url, method="automatic")
        except Exception as e:  # keep going: one failing source should not stop the rest
            print(f"Failed to download {name}: {e}")


if __name__ == "__main__":
    main()
