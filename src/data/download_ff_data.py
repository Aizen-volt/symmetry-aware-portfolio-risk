"""
Download Fama-French industry portfolio daily returns.

Usage:
    python -m src.data.download_ff_data
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

DATASETS = {
    "ff_12_industry_daily.csv": f"{FF_BASE}/12_Industry_Portfolios_daily_CSV.zip",
    "ff_49_industry_daily.csv": f"{FF_BASE}/49_Industry_Portfolios_daily_CSV.zip",
}

OUT_DIR = Path("data/raw")


def download_and_extract(url: str, out_name: str) -> None:
    out_path = OUT_DIR / out_name
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return

    print(f"  Downloading: {url}")
    resp = urlopen(url)
    data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in {url}")

        # Read the CSV, extract only the daily value-weight returns
        # (first table before any blank-line separator)
        raw = zf.read(csv_names[0]).decode("utf-8", errors="replace")

    # Write raw CSV; the loader handles parsing
    out_path.write_text(raw, encoding="utf-8")
    print(f"  Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, url in DATASETS.items():
        print(f"Dataset: {out_name}")
        try:
            download_and_extract(url, out_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Please download manually from {url}")
            print(f"  and place the CSV as {OUT_DIR / out_name}")


if __name__ == "__main__":
    main()
