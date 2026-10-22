"""Tools the LLM agent can call.

Each tool is a plain Python function plus a JSON schema that tells the LLM
what the tool does and which arguments it takes.
"""

import json
from functools import cached_property

import pandas as pd

from src.agent.database import run_query
from src.data.load import PROCESSED_DIR
from src.features.build import build_features
from src.models.explain import explain_prediction, get_explainer
from src.models.train import TEST_START, load_model, time_split

FORECAST_YEAR = 2018
MAX_ROWS_FOR_LLM = 50  # keep tool results short; small local models have limited context


class ForecastService:
    """Loads the model, test features and SHAP explainer once, on first use."""

    @cached_property
    def _test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_parquet(PROCESSED_DIR / "energy_weather.parquet")
        X, y = build_features(df)
        _, X_test, _, y_test = time_split(X, y, TEST_START)
        return X_test, y_test

    @cached_property
    def model(self):
        return load_model("lightgbm_price_24h")

    @cached_property
    def explainer(self):
        return get_explainer(self.model)

    def to_timestamp(self, when: str) -> pd.Timestamp:
        X_test, _ = self._test_data
        ts = pd.Timestamp(when)
        ts = ts.tz_localize(X_test.index.tz) if ts.tzinfo is None else ts.tz_convert(X_test.index.tz)
        ts = ts.floor("h")
        if ts not in X_test.index:
            raise ValueError(f"Forecasts are only available for hours in {FORECAST_YEAR} (the model's test year).")
        return ts

    def forecast(self, when: str) -> dict:
        X_test, y_test = self._test_data
        ts = self.to_timestamp(when)
        prediction = float(self.model.predict(X_test.loc[[ts]])[0])
        actual = float(y_test.loc[ts])
        return {
            "time": ts.strftime("%Y-%m-%d %H:%M"),
            "forecast_eur_mwh": round(prediction, 2),
            "actual_eur_mwh": round(actual, 2),
            "official_day_ahead_eur_mwh": round(float(X_test.loc[ts, "price_day_ahead"]), 2),
            "absolute_error_eur_mwh": round(abs(prediction - actual), 2),
        }

    def explain(self, when: str, top_n: int = 5) -> dict:
        X_test, _ = self._test_data
        return explain_prediction(self.explainer, X_test, self.to_timestamp(when), top_n=top_n)


service = ForecastService()


# ---- Tool functions ----------------------------------------------------------

def query_database(sql: str) -> str:
    result = run_query(sql)
    truncated = len(result) > MAX_ROWS_FOR_LLM
    text = result.head(MAX_ROWS_FOR_LLM).to_csv(index=False)
    if truncated:
        text += f"\n(showing first {MAX_ROWS_FOR_LLM} of {len(result)} rows)"
    return text


def forecast_price(datetime: str) -> str:
    return json.dumps(service.forecast(datetime))


def explain_forecast(datetime: str) -> str:
    return json.dumps(service.explain(datetime))


TOOL_FUNCTIONS = {
    "query_database": query_database,
    "forecast_price": forecast_price,
    "explain_forecast": explain_forecast,
}


# ---- JSON schemas (what the LLM sees) ----------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run a read-only SQLite SELECT query on the energy market database and return the result as CSV. "
                "Use it for any question about historical prices, demand, generation or weather, "
                "and for aggregate statistics about the model's forecasts. Always aggregate (AVG, MAX, GROUP BY) "
                "instead of selecting thousands of raw rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "A single SQLite SELECT statement."}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_price",
            "description": (
                f"Get the model's 24-hours-ahead price forecast for one specific hour in {FORECAST_YEAR}, "
                "together with the actual price and the official day-ahead price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {"type": "string", "description": "Local Madrid time, format 'YYYY-MM-DD HH:00'."}
                },
                "required": ["datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_forecast",
            "description": (
                f"Explain WHY the model forecast a given price for one specific hour in {FORECAST_YEAR}. "
                "Returns the top drivers with their SHAP contribution in EUR/MWh "
                "(positive = pushed the price up, negative = pushed it down)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {"type": "string", "description": "Local Madrid time, format 'YYYY-MM-DD HH:00'."}
                },
                "required": ["datetime"],
            },
        },
    },
]


def call_tool(name: str, arguments: dict) -> str:
    """Run a tool by name. Errors are returned as text so the LLM can correct itself."""
    if name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{name}'."
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as exc:  # noqa: BLE001 - any failure is reported back to the model
        return f"Error: {exc}"
