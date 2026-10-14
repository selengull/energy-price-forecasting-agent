"""Load and clean the raw Spanish energy and weather datasets."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOCAL_TZ = "Europe/Madrid"

# Weather columns worth keeping (the rest are categorical descriptions or redundant).
WEATHER_COLS = ["temp", "pressure", "humidity", "wind_speed", "rain_1h", "snow_3h", "clouds_all"]

# Physically plausible ranges; values outside are sensor errors and get interpolated.
WEATHER_VALID_RANGES = {
    "pressure": (900, 1100),  # hPa
    "wind_speed": (0, 50),  # m/s
}


def _to_snake_case(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def load_energy(path: Path = RAW_DIR / "energy_dataset.csv") -> pd.DataFrame:
    """Load the hourly energy dataset, indexed by local (Madrid) time.

    - Timestamps carry +01:00/+02:00 offsets (daylight saving), so we parse
      them as UTC first and then convert to local time.
    - Columns that are entirely empty or zero are dropped.
    - Short gaps (a few missing hours) are filled by linear interpolation.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(LOCAL_TZ)
    df = df.set_index("time").sort_index()

    empty_cols = [c for c in df.columns if df[c].fillna(0).eq(0).all()]
    df = df.drop(columns=empty_cols)

    df.columns = [_to_snake_case(c) for c in df.columns]
    return df.interpolate(method="time", limit=6)


def load_weather(path: Path = RAW_DIR / "weather_features.csv") -> pd.DataFrame:
    """Load weather data and reshape it to one row per hour, one column per city/variable.

    Fixes applied:
    - city names have stray whitespace (" Barcelona")
    - duplicate (city, timestamp) rows are dropped
    - temperatures are converted from Kelvin to Celsius
    - impossible pressure / wind values are treated as missing and interpolated
    """
    df = pd.read_csv(path)
    df["city_name"] = df["city_name"].str.strip()
    df["time"] = pd.to_datetime(df["dt_iso"], utc=True).dt.tz_convert(LOCAL_TZ)
    df = df.drop_duplicates(subset=["city_name", "time"], keep="first")

    df["temp"] = df["temp"] - 273.15
    for col, (low, high) in WEATHER_VALID_RANGES.items():
        df[col] = df[col].where(df[col].between(low, high))

    wide = df.pivot(index="time", columns="city_name", values=WEATHER_COLS)
    wide.columns = [f"{city.lower()}_{var}" for var, city in wide.columns]
    wide = wide.sort_index(axis=1)
    return wide.interpolate(method="time", limit=6)


def load_dataset() -> pd.DataFrame:
    """Return energy and weather data joined on the hourly timestamp."""
    energy = load_energy()
    weather = load_weather()
    return energy.join(weather, how="left")


def save_processed(df: pd.DataFrame, name: str = "energy_weather.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_parquet(out)
    return out


if __name__ == "__main__":
    data = load_dataset()
    path = save_processed(data)
    print(f"Saved {data.shape[0]:,} rows x {data.shape[1]} columns to {path}")
