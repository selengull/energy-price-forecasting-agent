"""Train, evaluate and save price forecasting models."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

TEST_START = "2018-01-01"
VALID_START = "2017-10-01"  # last quarter of the training period, used for early stopping

LGBM_PARAMS = {
    "objective": "regression_l1",  # optimise MAE directly; robust to price spikes
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 3000,
    "random_state": 42,
    "verbose": -1,
}


def time_split(X: pd.DataFrame, y: pd.Series, start: str) -> tuple:
    """Split chronologically: everything before `start` vs. everything from `start` on."""
    before = X.index < pd.Timestamp(start, tz=X.index.tz)
    return X[before], X[~before], y[before], y[~before]


def regression_metrics(y_true: pd.Series, y_pred) -> dict:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Bias": float(np.mean(err)),
    }


def train_ridge(X_train: pd.DataFrame, y_train: pd.Series):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return model.fit(X_train, y_train)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    """Fit LightGBM, using the last quarter of the training data for early stopping."""
    X_fit, X_val, y_fit, y_val = time_split(X_train, y_train, VALID_START)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_fit,
        y_fit,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # Refit on the full training period with the number of trees found above.
    final = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": model.best_iteration_})
    return final.fit(X_train, y_train)


def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.joblib")
