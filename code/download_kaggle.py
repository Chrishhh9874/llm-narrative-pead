"""Download the Kaggle dataset `tsaustin/us-historical-stock-prices-with-earnings-data`
and print a field/coverage summary of each file.

Usage:
    pip install -r requirements.txt
    python code/download_kaggle.py              # download + summarize
    python code/download_kaggle.py --skip-download   # summarize only

Credentials (either one):
    ~/.kaggle/kaggle.json  ({"username": "...", "key": "..."}, chmod 600)
    or env var KAGGLE_API_TOKEN (new-style token), or KAGGLE_USERNAME / KAGGLE_KEY
"""

import argparse
import os
import sys
from pathlib import Path

DATASET_REF = "tsaustin/us-historical-stock-prices-with-earnings-data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "kaggle_tsaustin"

# Columns whose min/max we report as the coverage period, per file.
DATE_COLUMNS = {
    "stock_prices_latest.csv": "date",
    "earnings_latest.csv": "date",
    "dividends_latest.csv": "date",
}


def has_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").is_file()


def download(out_dir: Path) -> None:
    if not has_credentials():
        sys.exit(
            "Kaggle credentials not found.\n"
            "  1. Kaggle → Settings → API → Create New Token (downloads kaggle.json)\n"
            "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ "
            "&& chmod 600 ~/.kaggle/kaggle.json"
        )
    try:
        # Importing `kaggle` authenticates immediately, so import only after the check above.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Package `kaggle` missing: pip install -r requirements.txt")

    api = KaggleApi()
    api.authenticate()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_REF} (~2.1 GB) → {out_dir}")
    api.dataset_download_files(DATASET_REF, path=str(out_dir), unzip=True, quiet=False)


def summarize(out_dir: Path) -> None:
    import pandas as pd

    csv_files = sorted(out_dir.rglob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files under {out_dir}")

    for path in csv_files:
        size_mb = path.stat().st_size / 1e6
        print(f"\n=== {path.name} ({size_mb:,.1f} MB) ===")
        head = pd.read_csv(path, nrows=5)
        print("columns:", list(head.columns))
        print(head.to_string(index=False))

        date_col = DATE_COLUMNS.get(path.name)
        usecols = [c for c in (date_col, "symbol") if c in head.columns]
        if not usecols:
            continue
        # Stream in chunks: stock_prices_latest.csv is too large to load whole.
        n_rows, symbols, dmin, dmax = 0, set(), None, None
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=2_000_000):
            n_rows += len(chunk)
            if "symbol" in chunk:
                symbols.update(chunk["symbol"].dropna().unique())
            if date_col in chunk:
                d = pd.to_datetime(chunk[date_col], errors="coerce")
                dmin = d.min() if dmin is None else min(dmin, d.min())
                dmax = d.max() if dmax is None else max(dmax, d.max())
        print(f"rows: {n_rows:,}  symbols: {len(symbols):,}  date range: {dmin} → {dmax}")

    earnings = next(out_dir.rglob("earnings_latest.csv"), out_dir / "earnings_latest.csv")
    if earnings.exists():
        df = pd.read_csv(earnings)
        print("\n=== earnings_latest.csv: non-null share per column ===")
        print(df.notna().mean().round(3).to_string())
        if "date" in df:
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
            print("\nevents per year:")
            print(df["year"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download(args.out)
    summarize(args.out)


if __name__ == "__main__":
    main()
