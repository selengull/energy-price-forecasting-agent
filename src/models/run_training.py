"""Train both forecasting models step by step, printing progress along the way.

Run from the project root:
    python -m src.models.run_training
"""

import lightgbm as lgb
import pandas as pd

from src.data.load import PROCESSED_DIR
from src.features.build import build_features
from src.models.train import (
    LGBM_PARAMS, TEST_START, VALID_START, regression_metrics, save_model, time_split,
)


def train_verbose(name: str, include_official_forecast: bool) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    print("\n[1/6] Building features...")
    df = pd.read_parquet(PROCESSED_DIR / "energy_weather.parquet")
    X, y = build_features(df, include_official_forecast=include_official_forecast)
    print(f"      {X.shape[0]:,} hours x {X.shape[1]} features")

    print("\n[2/6] Splitting by time...")
    X_train, X_test, y_train, y_test = time_split(X, y, TEST_START)
    X_fit, X_val, y_fit, y_val = time_split(X_train, y_train, VALID_START)
    print(f"      fit:        {X_fit.index.min():%Y-%m-%d} -> {X_fit.index.max():%Y-%m-%d}  ({len(X_fit):,} hours)")
    print(f"      validation: {X_val.index.min():%Y-%m-%d} -> {X_val.index.max():%Y-%m-%d}  ({len(X_val):,} hours)")
    print(f"      test:       {X_test.index.min():%Y-%m-%d} -> {X_test.index.max():%Y-%m-%d}  ({len(X_test):,} hours)")

    print("\n[3/6] Training trees one by one; validation MAE every 50 trees (lower is better):")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_fit, y_fit,
        eval_X=X_val, eval_y=y_val, eval_metric="l1",
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(100, verbose=True)],
    )
    best = model.best_iteration_
    print(f"      -> best number of trees: {best}")

    print(f"\n[4/6] Retraining on the full training period with {best} trees...")
    final = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": best}).fit(X_train, y_train)

    print("\n[5/6] Evaluating on the unseen test year 2018...")
    metrics = regression_metrics(y_test, final.predict(X_test))
    naive = regression_metrics(y_test, X_test["price_lag_24h"])
    print(f"      model MAE: {metrics['MAE']:.2f} EUR/MWh   (naive baseline: {naive['MAE']:.2f})")

    print("\n[6/6] Saving...")
    print(f"      {save_model(final, name)}")


if __name__ == "__main__":
    train_verbose("lightgbm_price_24h", include_official_forecast=True)
    train_verbose("lightgbm_price_24h_fundamentals", include_official_forecast=False)
    print("\nDone.")
