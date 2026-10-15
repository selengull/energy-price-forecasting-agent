"""Explain model predictions with SHAP.

SHAP splits every single prediction into contributions, one per feature:

    prediction = base value + contribution(feature 1) + contribution(feature 2) + ...

The base value is the model's average prediction. A positive contribution means
"this feature pushed the price up for this hour", a negative one "pushed it down".
"""

import pandas as pd
import shap

# Human-readable names, used in plots and later by the LLM agent.
FEATURE_DESCRIPTIONS = {
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "day_of_year": "Day of year",
    "is_weekend": "Weekend",
    "is_holiday": "Public holiday",
    "hour_sin": "Hour of day (cyclical)",
    "hour_cos": "Hour of day (cyclical)",
    "doy_sin": "Season (cyclical)",
    "doy_cos": "Season (cyclical)",
    "total_load_forecast": "Forecast demand",
    "forecast_wind_onshore_day_ahead": "Forecast wind generation",
    "forecast_solar_day_ahead": "Forecast solar generation",
    "residual_load_forecast": "Forecast residual load (demand minus wind and solar)",
    "price_lag_24h": "Price 24 hours earlier",
    "price_lag_48h": "Price 48 hours earlier",
    "price_lag_72h": "Price 72 hours earlier",
    "price_lag_168h": "Price one week earlier",
    "price_mean_24h": "Average price over the previous day",
    "price_mean_7d": "Average price over the previous week",
    "price_std_24h": "Price volatility over the previous day",
    "price_min_24h": "Lowest price over the previous day",
    "price_max_24h": "Highest price over the previous day",
    "fossil_generation_lag_24h": "Fossil generation 24 hours earlier",
    "wind_generation_lag_24h": "Wind generation 24 hours earlier",
    "load_actual_lag_24h": "Actual demand 24 hours earlier",
    "temp_mean_lag_24h": "Average temperature 24 hours earlier",
    "wind_speed_mean_lag_24h": "Average wind speed 24 hours earlier",
    "price_day_ahead": "Official day-ahead market price",
}


def get_explainer(model) -> shap.TreeExplainer:
    """TreeExplainer computes exact SHAP values quickly for tree models like LightGBM."""
    return shap.TreeExplainer(model)


def explain_prediction(explainer: shap.TreeExplainer, X: pd.DataFrame, timestamp, top_n: int = 5) -> dict:
    """Explain the forecast for one hour: base value, prediction and the top drivers."""
    row = X.loc[[pd.Timestamp(timestamp)]]
    explanation = explainer(row)

    contributions = pd.Series(explanation.values[0], index=X.columns)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_n)

    base = float(explanation.base_values[0])
    return {
        "timestamp": str(row.index[0]),
        "base_value": round(base, 2),
        "prediction": round(base + float(contributions.sum()), 2),
        "top_drivers": [
            {
                "feature": name,
                "description": FEATURE_DESCRIPTIONS.get(name, name),
                "value": round(float(row.iloc[0][name]), 2),
                "contribution": round(float(value), 2),
            }
            for name, value in top.items()
        ],
    }
