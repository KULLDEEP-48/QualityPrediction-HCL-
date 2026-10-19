"""
Trains an XGBoost regressor to predict silica concentrate from flotation
process measurements in the mining process dataset.

Usage:
    python train_ml_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import MINING_DATA_PATH, MODEL_PATH

DATE_COLUMN = "date"
TARGET_COLUMN = "% Silica Concentrate"
OUTPUT_COLUMNS = ["% Iron Concentrate", TARGET_COLUMN]


def load_dataset():
    if not os.path.exists(MINING_DATA_PATH):
        raise FileNotFoundError(
            f"Mining process dataset not found at {MINING_DATA_PATH}"
        )

    df = pd.read_csv(
        MINING_DATA_PATH,
        decimal=",",
        na_values=["?", ""],
        low_memory=False,
    )
    required_columns = {DATE_COLUMN, TARGET_COLUMN, "% Iron Concentrate"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Mining process dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Invalid or missing values found in {DATE_COLUMN}")
    df = df.assign(**{DATE_COLUMN: dates}).sort_values(DATE_COLUMN, kind="stable")
    df = df.reset_index(drop=True)
    feature_columns = [
        column
        for column in df.columns
        if column not in {DATE_COLUMN, *OUTPUT_COLUMNS}
    ]
    numeric_columns = [*feature_columns, *OUTPUT_COLUMNS]
    non_numeric_columns = df[numeric_columns].select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    if non_numeric_columns:
        details = ", ".join(
            non_numeric_columns
        )
        raise ValueError(
            "Mining process readings must be numeric: " + details
        )
    invalid_measurements = df[numeric_columns].isna() | ~np.isfinite(
        df[numeric_columns]
    )
    invalid_counts = invalid_measurements.sum()
    invalid_counts = invalid_counts[invalid_counts > 0]
    if not invalid_counts.empty:
        details = ", ".join(
            f"{column}: {count}" for column, count in invalid_counts.items()
        )
        raise ValueError(
            "Mining process dataset contains missing or non-finite measurements; "
            f"no rows were silently discarded ({details})"
        )
    return df


def main():
    df = load_dataset()
    X = df.drop(columns=[DATE_COLUMN, *OUTPUT_COLUMNS])
    y = df[TARGET_COLUMN]

    if X.empty or len(df) < 2:
        raise ValueError("Mining process dataset needs features and at least two rows")
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "Mining process features must be numeric: "
            + ", ".join(non_numeric_columns)
        )

    split_index = int(len(df) * 0.8)
    if split_index == 0 or split_index == len(df):
        raise ValueError("Not enough rows for a chronological train/test split")
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== Chronological Holdout Evaluation ===")
    print(f"Evaluation training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    model.fit(X, y)
    model.target_name_ = TARGET_COLUMN
    model.target_unit_ = "percent"
    model.training_rows_ = len(X)
    model_path_dir = os.path.dirname(MODEL_PATH)
    if model_path_dir:
        os.makedirs(model_path_dir, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(
        f"\n[train] final model fitted on all {len(X):,} rows and saved to "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
