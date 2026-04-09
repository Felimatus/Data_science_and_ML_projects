"""
Download daily weather data from Open-Meteo for 5 Czech macro-regions.

1) Historical data  → data/raw/weather_CZ.csv       (Archive API)
2) 14-day forecast  → data/raw/weather_forecast_CZ.csv  (Forecast API)

Stations:
    CB  - Central Bohemia  (Prague)
    NB  - North Bohemia    (Ústí nad Labem)
    SB  - South Bohemia    (České Budějovice)
    SM  - South Moravia    (Brno)
    NM  - North Moravia    (Ostrava)
"""

import sys
from pathlib import Path

from datetime import date, timedelta

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Allow imports from the project root
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, AHEAD

# --- Configuration ---
START_DATE = "2022-07-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()
TIMEZONE = "Europe/Berlin"
OUTPUT_PATH = RAW_DIR / "weather_CZ.csv"
FORECAST_PATH = RAW_DIR / "weather_forecast_CZ.csv"

STATIONS = {
    "CB": {"name": "Prague",             "lat": 50.0880, "lon": 14.4208},
    "NB": {"name": "Ústí nad Labem",     "lat": 50.6607, "lon": 14.0323},
    "SB": {"name": "České Budějovice",   "lat": 48.9757, "lon": 14.4802},
    "SM": {"name": "Brno",               "lat": 49.1951, "lon": 16.6068},
    "NM": {"name": "Ostrava",            "lat": 49.8209, "lon": 18.2625},
}

DAILY_VARIABLES = [
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "pressure_msl_mean",
    "cloud_cover_max",
    "dew_point_2m_mean",
]

# --- API client setup ---
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _parse_daily_response(response, prefix: str) -> pd.DataFrame:
    """Parse an Open-Meteo daily response into a DataFrame."""
    daily = response.Daily()
    date_range = pd.date_range(
        start=pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )
    data = {"date": date_range}
    for i, var in enumerate(DAILY_VARIABLES):
        data[f"{prefix}_{var}"] = daily.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.date
    return df


def fetch_station(prefix: str, station: dict) -> pd.DataFrame:
    """Fetch historical daily weather data for a single station."""
    print(f"Downloading weather data for {station['name']} ({prefix})...")

    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": DAILY_VARIABLES,
        "timezone": TIMEZONE,
    }

    responses = openmeteo.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]

    print(f"  Coordinates: {response.Latitude():.4f}°N {response.Longitude():.4f}°E")
    print(f"  Elevation: {response.Elevation()} m asl")

    return _parse_daily_response(response, prefix)


def fetch_station_forecast(prefix: str, station: dict) -> pd.DataFrame:
    """Fetch forecast daily weather data for a single station."""
    print(f"Downloading forecast for {station['name']} ({prefix})...")

    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "daily": DAILY_VARIABLES,
        "timezone": TIMEZONE,
        "forecast_days": AHEAD,
    }

    responses = openmeteo.weather_api(FORECAST_URL, params=params)
    return _parse_daily_response(responses[0], prefix)


def main():
    print(f"Fetching weather data from {START_DATE} to {END_DATE}")
    print(f"Stations: {', '.join(f'{k} ({v['name']})' for k, v in STATIONS.items())}")
    print("-" * 60)

    # Fetch all stations and merge on date
    merged = None
    for prefix, station in STATIONS.items():
        df = fetch_station(prefix, station)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="outer")

    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)

    # Save to CSV
    merged.to_csv(OUTPUT_PATH, index=False)
    print("-" * 60)
    print(f"Saved {len(merged)} rows x {len(merged.columns)} columns to {OUTPUT_PATH}")
    print(f"Date range: {merged['date'].iloc[0]} to {merged['date'].iloc[-1]}")
    print(f"\nColumns: {list(merged.columns)}")

    # Quick data quality check
    missing = merged.isnull().sum()
    if missing.any():
        print(f"\nWarning — missing values found:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values.")

    # ------------------------------------------------------------------
    # Forecast (next AHEAD days)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Fetching {AHEAD}-day weather forecast")
    print("-" * 60)

    forecast_merged = None
    for prefix, station in STATIONS.items():
        fdf = fetch_station_forecast(prefix, station)
        if forecast_merged is None:
            forecast_merged = fdf
        else:
            forecast_merged = forecast_merged.merge(fdf, on="date", how="outer")

    forecast_merged = forecast_merged.sort_values("date").reset_index(drop=True)
    forecast_merged.to_csv(FORECAST_PATH, index=False)
    print("-" * 60)
    print(f"Saved {len(forecast_merged)} rows to {FORECAST_PATH}")
    print(f"Forecast range: {forecast_merged['date'].iloc[0]} to "
          f"{forecast_merged['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
