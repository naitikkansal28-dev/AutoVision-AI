"""
AutoVision AI - Multi-Step Future Sales Forecasting Engine
Uses trained machine learning models to iteratively forecast vehicle demand
over a 1 to 6 month forward-looking horizon with confidence intervals.
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import config
from ml.preprocessing import load_sales_data, aggregate_monthly_sales
from ml.train_models import train_and_evaluate_all


def load_champion_artifacts():
    """Loads saved model, feature columns, and training metadata."""
    model_path = config.MODELS_DIR / "best_forecasting_model.joblib"
    cols_path = config.MODELS_DIR / "feature_cols.joblib"
    meta_path = config.MODELS_DIR / "model_metadata.json"

    if not model_path.exists() or not cols_path.exists():
        print("[Predict] Model artifacts not found. Training pipeline will be triggered.")
        train_results = train_and_evaluate_all()
        return (
            joblib.load(model_path),
            joblib.load(cols_path),
            train_results["metadata"]
        )

    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, feature_cols, metadata


def generate_future_forecast(horizon_months=6, source="db", csv_path=None):
    """
    Generates multi-step autoregressive future sales predictions.
    Computes 95% forecast intervals using model residual standard error.
    """
    model, feature_cols, metadata = load_champion_artifacts()
    
    # Get historical monthly series
    raw_df = load_sales_data(source=source, csv_path=csv_path)
    monthly_df = aggregate_monthly_sales(raw_df)
    monthly_df = monthly_df.sort_values("period").reset_index(drop=True)

    last_date = monthly_df["period"].iloc[-1]
    
    # Extract historical error metric (RMSE) for prediction interval estimation
    model_name = metadata.get("champion_model", "")
    metrics_list = metadata.get("metrics", [])
    model_rmse = 3.5  # fallback default
    for m in metrics_list:
        if m.get("Model") == model_name:
            model_rmse = float(m.get("RMSE", 3.5))
            break

    # Maintain rolling lag buffer of historical unit sales
    recent_sales = list(monthly_df["total_units"].values)
    recent_discounts = list(monthly_df["discount_pct"].values)

    forecast_records = []
    
    for step in range(1, horizon_months + 1):
        future_date = last_date + relativedelta(months=step)
        f_year = future_date.year
        f_month = future_date.month
        f_quarter = (f_month - 1) // 3 + 1

        # Cyclical & Seasonal flags
        sin_m = np.sin(2 * np.pi * f_month / 12)
        cos_m = np.cos(2 * np.pi * f_month / 12)
        is_festive = 1 if f_month in [10, 11] else 0
        is_fy = 1 if f_month == 3 else 0
        is_yearend = 1 if f_month == 12 else 0

        # Autoregressive lags from recent buffer
        lag_1 = recent_sales[-1]
        lag_2 = recent_sales[-2]
        lag_3 = recent_sales[-3]

        # Rolling window stats
        roll_mean_3 = np.mean(recent_sales[-3:])
        roll_mean_6 = np.mean(recent_sales[-6:]) if len(recent_sales) >= 6 else roll_mean_3
        roll_std_3 = np.std(recent_sales[-3:])

        disc_lag_1 = recent_discounts[-1] if recent_discounts else 2.5

        # Feature vector matching exact schema
        feat_dict = {
            "year": f_year,
            "month": f_month,
            "quarter": f_quarter,
            "sin_month": sin_m,
            "cos_month": cos_m,
            "is_festive_season": is_festive,
            "is_fy_close": is_fy,
            "is_year_end": is_yearend,
            "units_lag_1": lag_1,
            "units_lag_2": lag_2,
            "units_lag_3": lag_3,
            "units_roll_mean_3": roll_mean_3,
            "units_roll_mean_6": roll_mean_6,
            "units_roll_std_3": roll_std_3,
            "discount_pct_lag_1": disc_lag_1
        }

        X_step = pd.DataFrame([feat_dict])[feature_cols]
        predicted_val = float(model.predict(X_step)[0])
        predicted_val = max(1.0, predicted_val)

        # Confidence interval expands slightly with forecast horizon: sigma_step = RMSE * sqrt(step)
        horizon_sigma = model_rmse * np.sqrt(step * 0.75)
        lower_bound = max(0.0, predicted_val - 1.96 * horizon_sigma)
        upper_bound = predicted_val + 1.96 * horizon_sigma

        forecast_records.append({
            "period": future_date,
            "predicted_units": round(predicted_val, 1),
            "lower_bound": round(lower_bound, 1),
            "upper_bound": round(upper_bound, 1),
            "month_label": future_date.strftime("%b %Y")
        })

        # Append step prediction to buffer for subsequent step lags
        recent_sales.append(predicted_val)
        recent_discounts.append(disc_lag_1)

    forecast_df = pd.DataFrame(forecast_records)
    return forecast_df, monthly_df, metadata
