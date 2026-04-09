"""
Download hourly electricity load data from ENTSO-E Transparency Platform
for the Czech Republic, then aggregate to daily min, mean, max.

Requires:
    - An ENTSO-E API key stored in .env as ENTSOE_API_KEY
    - pip install entsoe-py python-dotenv

Output: data/raw/energy_CZ.csv with columns:
    date, load_min, load_mean, load_max
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

# Allow imports from the project root
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE = "2022-07-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()
COUNTRY_CODE = "CZ"
TIMEZONE = "Europe/Prague"
OUTPUT_PATH = RAW_DIR / "energy_CZ.csv"

# ---------------------------------------------------------------------------
# Load API key from .env
# ---------------------------------------------------------------------------
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)
API_KEY = os.getenv("ENTSOE_API_KEY")

if not API_KEY or API_KEY == "your_api_key_here":
    print("ERROR: Please set your ENTSO-E API key in the .env file.")
    print(f"  File: {ENV_PATH}")
    print("  Format: ENTSOE_API_KEY=your_actual_key")
    sys.exit(1)


def main():
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=API_KEY)

    start = pd.Timestamp(START_DATE, tz=TIMEZONE)
    end = pd.Timestamp(END_DATE, tz=TIMEZONE)

    # --- Download actual total load (hourly) --------------------------------
    print(f"Downloading electricity load for {COUNTRY_CODE} "
          f"from {START_DATE} to {END_DATE}...")
    print("This may take a minute...")

    load_series = client.query_load(COUNTRY_CODE, start=start, end=end)

    # Handle case where result might be a DataFrame with one column
    if isinstance(load_series, pd.DataFrame):
        load_series = load_series.iloc[:, 0]

    print(f"Downloaded {len(load_series)} hourly records")
    print(f"  Range: {load_series.index[0]} to {load_series.index[-1]}")
    print(f"  Load range: {load_series.min():.0f} - {load_series.max():.0f} MW")

    # --- Aggregate to daily -------------------------------------------------
    # Convert index to date for grouping
    load_series.index = load_series.index.tz_convert(TIMEZONE)
    daily = load_series.groupby(load_series.index.date).agg(
        load_min="min",
        load_mean="mean",
        load_max="max",
    )
    daily.index.name = "date"
    daily = daily.reset_index()

    # Ensure date column is proper datetime
    daily["date"] = pd.to_datetime(daily["date"])

    # Round to 1 decimal for cleaner output
    daily["load_min"] = daily["load_min"].round(1)
    daily["load_mean"] = daily["load_mean"].round(1)
    daily["load_max"] = daily["load_max"].round(1)

    # --- Save ---------------------------------------------------------------
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_PATH, index=False)

    print("-" * 60)
    print(f"Saved {len(daily)} daily records to {OUTPUT_PATH}")
    print(f"Date range: {daily['date'].iloc[0].date()} to "
          f"{daily['date'].iloc[-1].date()}")
    print(f"\nSample:")
    print(daily.head(10).to_string(index=False))

    # --- Quick data quality check -------------------------------------------
    missing = daily.isnull().sum()
    if missing.any():
        print(f"\nWarning — missing values found:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values.")

    return daily


if __name__ == "__main__":
    main()
