"""SQLite database that the LLM agent can query with SQL.

Two tables are built from the processed data:

- hourly:   one row per hour (2015-2018) with prices, demand, generation and weather
- forecasts: one row per hour of 2018 (the test year) with the model's forecast,
             the official day-ahead price and the actual price
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.data.load import PROCESSED_DIR
from src.features.build import CITIES, build_features
from src.models.train import TEST_START, load_model, time_split

DB_PATH = PROCESSED_DIR / "energy.db"

# Plain-English schema, given to the LLM in its instructions.
SCHEMA_DESCRIPTION = """
Table `hourly` (one row per hour, Spain, 2015-01-01 to 2018-12-31, local Madrid time):
- time TEXT, format 'YYYY-MM-DD HH:MM:SS'
- year INTEGER, month INTEGER (1-12), day_of_week INTEGER (0=Monday ... 6=Sunday), hour INTEGER (0-23)
- price REAL: actual electricity price in EUR/MWh
- price_day_ahead REAL: official day-ahead market price in EUR/MWh
- demand_mw REAL: actual electricity demand in MW
- wind_mw, solar_mw, gas_mw, coal_mw, nuclear_mw, hydro_mw REAL: generation by source in MW
- temperature_c REAL: average temperature of 5 large Spanish cities in degrees Celsius
- wind_speed_ms REAL: average wind speed of the same cities in m/s

Table `forecasts` (one row per hour of 2018 only, the model's out-of-sample test year):
- time TEXT, same format as above
- forecast REAL: our LightGBM model's 24-hours-ahead price forecast in EUR/MWh
- actual REAL: actual price in EUR/MWh
- price_day_ahead REAL: official day-ahead market price in EUR/MWh
- abs_error REAL: absolute error of our forecast in EUR/MWh, i.e. ABS(forecast - actual)

Note: the official day-ahead price is also a forecast of the actual price. Its error is ABS(actual - price_day_ahead).
""".strip()


def _hourly_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["year"] = df.index.year
    out["month"] = df.index.month
    out["day_of_week"] = df.index.dayofweek
    out["hour"] = df.index.hour
    out["price"] = df["price_actual"]
    out["price_day_ahead"] = df["price_day_ahead"]
    out["demand_mw"] = df["total_load_actual"]
    out["wind_mw"] = df["generation_wind_onshore"]
    out["solar_mw"] = df["generation_solar"]
    out["gas_mw"] = df["generation_fossil_gas"]
    out["coal_mw"] = df["generation_fossil_hard_coal"] + df["generation_fossil_brown_coal_lignite"]
    out["nuclear_mw"] = df["generation_nuclear"]
    out["hydro_mw"] = df["generation_hydro_run_of_river_and_poundage"] + df["generation_hydro_water_reservoir"]
    out["temperature_c"] = df[[f"{c}_temp" for c in CITIES]].mean(axis=1)
    out["wind_speed_ms"] = df[[f"{c}_wind_speed" for c in CITIES]].mean(axis=1)
    return out.round(2)


def _forecasts_table(df: pd.DataFrame) -> pd.DataFrame:
    X, y = build_features(df)
    _, X_test, _, y_test = time_split(X, y, TEST_START)
    model = load_model("lightgbm_price_24h")

    out = pd.DataFrame(index=X_test.index)
    out["forecast"] = model.predict(X_test)
    out["actual"] = y_test
    out["price_day_ahead"] = X_test["price_day_ahead"]
    out["abs_error"] = (out["forecast"] - out["actual"]).abs()
    return out.round(2)


def _with_text_time(table: pd.DataFrame) -> pd.DataFrame:
    """SQLite has no timezone-aware type, so store local time as plain text."""
    table = table.copy()
    table.insert(0, "time", table.index.strftime("%Y-%m-%d %H:%M:%S"))
    return table.reset_index(drop=True)


def build_database(path: Path = DB_PATH) -> Path:
    df = pd.read_parquet(PROCESSED_DIR / "energy_weather.parquet")
    with sqlite3.connect(path) as conn:
        _with_text_time(_hourly_table(df)).to_sql("hourly", conn, if_exists="replace", index=False)
        _with_text_time(_forecasts_table(df)).to_sql("forecasts", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hourly_time ON hourly(time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_time ON forecasts(time)")
    return path


def run_query(sql: str, max_rows: int = 200) -> pd.DataFrame:
    """Run a read-only SQL query. Only SELECT statements are allowed."""
    statement = sql.strip().rstrip(";")
    if not statement.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed.")
    if ";" in statement:
        raise ValueError("Only a single SQL statement is allowed.")

    # mode=ro opens the database read-only, so even a malicious query cannot change it.
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
        result = pd.read_sql_query(statement, conn)
    return result.head(max_rows)


if __name__ == "__main__":
    print(f"Database written to {build_database()}")
