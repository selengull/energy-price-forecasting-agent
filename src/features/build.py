"""Feature engineering for 24-hours-ahead price forecasting.

Setting: at hour t we predict the price at hour t + 24. Every feature must
therefore be known at least 24 hours before the target hour. In practice:

- calendar features of the target hour (always known in advance)
- day-ahead forecasts published for the target hour (load, wind, solar)
- actual prices / generation / weather lagged by >= 24 hours

Actual generation and weather *at* the target hour are not used, since they
are only observed after the fact (data leakage).
"""

import holidays
import numpy as np
import pandas as pd

TARGET = "price_actual"
OFFICIAL_FORECAST = "price_day_ahead"

CITIES = ["barcelona", "bilbao", "madrid", "seville", "valencia"]

DAY_AHEAD_FORECASTS = ["total_load_forecast", "forecast_wind_onshore_day_ahead", "forecast_solar_day_ahead"]

FOSSIL_COLS = ["generation_fossil_gas", "generation_fossil_hard_coal", "generation_fossil_brown_coal_lignite"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    es_holidays = holidays.Spain(years=range(idx.year.min(), idx.year.max() + 1))

    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = pd.Series(idx.date, index=idx).map(lambda d: d in es_holidays).astype(int)

    # Cyclical encoding: hour 23 and hour 0 are neighbours, which plain integers hide.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out


def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead forecasts for the target hour, plus derived 'residual load'."""
    out = df[DAY_AHEAD_FORECASTS].copy()
    # Demand that wind and solar cannot cover must come from other (often fossil) plants.
    out["residual_load_forecast"] = (
        df["total_load_forecast"] - df["forecast_wind_onshore_day_ahead"] - df["forecast_solar_day_ahead"]
    )
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past values, all lagged by at least 24 hours."""
    price = df[TARGET]
    out = pd.DataFrame(index=df.index)

    for lag in [24, 48, 72, 168]:
        out[f"price_lag_{lag}h"] = price.shift(lag)

    # Rolling statistics over windows that end 24 hours before the target.
    past = price.shift(24)
    out["price_mean_24h"] = past.rolling(24).mean()
    out["price_mean_7d"] = past.rolling(168).mean()
    out["price_std_24h"] = past.rolling(24).std()
    out["price_min_24h"] = past.rolling(24).min()
    out["price_max_24h"] = past.rolling(24).max()

    out["fossil_generation_lag_24h"] = df[FOSSIL_COLS].sum(axis=1).shift(24)
    out["wind_generation_lag_24h"] = df["generation_wind_onshore"].shift(24)
    out["load_actual_lag_24h"] = df["total_load_actual"].shift(24)

    # National averages of yesterday's weather (cities behave similarly).
    out["temp_mean_lag_24h"] = df[[f"{c}_temp" for c in CITIES]].mean(axis=1).shift(24)
    out["wind_speed_mean_lag_24h"] = df[[f"{c}_wind_speed" for c in CITIES]].mean(axis=1).shift(24)
    return out


def build_features(df: pd.DataFrame, include_official_forecast: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). Rows with missing lags at the start of the series are dropped.

    The official day-ahead price is published before the target hour, so it is a
    legitimate feature. It can be switched off to measure how much it helps.
    """
    parts = [add_calendar_features(df), add_forecast_features(df), add_lag_features(df)]
    if include_official_forecast:
        parts.append(df[[OFFICIAL_FORECAST]])

    X = pd.concat(parts, axis=1)
    y = df[TARGET]

    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]
