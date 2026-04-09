"""
Merge raw energy and weather CSVs, add temporal and crisis features,
and save the result to data/processed/data.csv.

Inputs:  data/raw/energy_CZ.csv, data/raw/weather_CZ.csv
Output:  data/processed/data.csv
"""

import sys
from pathlib import Path

import holidays
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import (ENERGY_CSV, WEATHER_CSV, DATA_CSV,
                    CRISIS_START, CRISIS_END,
                    CB_POPULATION, NB_POPULATION, SB_POPULATION,
                    SM_POPULATION, NM_POPULATION)

# Population weights per macro-region
REGION_POPULATIONS = {
    "CB": CB_POPULATION,  # 2,864,095 — Praha + Středočeský
    "NB": NB_POPULATION,  # 2,637,437 — Ústecký + Liberecký + Karlovarský + Královéhradecký + Pardubický
    "SB": SB_POPULATION,  # 1,785,514 — Jihočeský + Plzeňský + Vysočina
    "SM": SM_POPULATION,  # 1,808,341 — Jihomoravský + Zlínský
    "NM": NM_POPULATION,  # 1,814,113 — Moravskoslezský + Olomoucký
}
TOTAL_POPULATION = sum(REGION_POPULATIONS.values())  # 10,909,500
REGION_WEIGHTS = {r: p / TOTAL_POPULATION for r, p in REGION_POPULATIONS.items()}


def main():
    energy = pd.read_csv(ENERGY_CSV, parse_dates=["date"])
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["date"])

    print(f"Energy : {len(energy)} rows  "
          f"({energy['date'].min().date()} → {energy['date'].max().date()})")
    print(f"Weather: {len(weather)} rows  "
          f"({weather['date'].min().date()} → {weather['date'].max().date()})")

    # Merge on date (inner join keeps only days present in both)
    df = energy.merge(weather, on="date", how="inner")
    print(f"Merged : {len(df)} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    # Weight weather columns by regional population share
    for region, weight in REGION_WEIGHTS.items():
        region_cols = [c for c in df.columns if c.startswith(f"{region}_")]
        df[region_cols] *= weight
    print(f"Weather weighted by population ({len(REGION_WEIGHTS)} regions, "
          f"total pop: {TOTAL_POPULATION:,})")

    # Temporal features — astronomical season (one-hot)
    md = df["date"].dt.month * 100 + df["date"].dt.day  # e.g. 321 = Mar 21
    df["season_Winter"] = ((md >= 1221) | (md <= 319)).astype(float)
    df["season_Spring"] = ((md >= 320) & (md <= 620)).astype(float)
    df["season_Summer"] = ((md >= 621) & (md <= 922)).astype(float)
    df["season_Fall"]   = ((md >= 923) & (md <= 1220)).astype(float)

    # Day type: W (weekday), A (Saturday), U (Sunday) — based on actual weekday
    dow = df["date"].dt.dayofweek  # 0=Mon .. 6=Sun
    years = df["date"].dt.year.unique().tolist()
    cz_holidays = holidays.Czechia(years=years)
    is_holiday = df["date"].dt.date.isin(cz_holidays)

    df["day_type"] = "W"                        # default: weekday (Mon-Fri)
    df.loc[dow == 5, "day_type"] = "A"          # Saturday
    df.loc[dow == 6, "day_type"] = "U"          # Sunday
    df["is_holiday"] = is_holiday.astype(int)   # separate binary flag
    counts = df["day_type"].value_counts()
    print(f"Day types: W={counts.get('W',0)}, A={counts.get('A',0)}, "
          f"U={counts.get('U',0)} ({is_holiday.sum()} holidays flagged)")

    # Lag features (same weekday last week / two weeks ago)
    df["load_mean_lag7"] = df["load_mean"].shift(7)
    df["load_mean_lag14"] = df["load_mean"].shift(14)
    # Drop the first 14 rows that have NaN lags
    df = df.dropna(subset=["load_mean_lag14"]).reset_index(drop=True)
    print(f"Added lag features, dropped first 14 rows → {len(df)} rows")

    # Energy crisis flag
    df["crisis"] = ((df["date"] >= CRISIS_START)
                    & (df["date"] <= CRISIS_END)).astype(int)

    # Save
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"\nSaved to {DATA_CSV}  ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
