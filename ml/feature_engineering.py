"""
AutoVision AI - Machine Learning Feature Engineering
Creates time-series lags, cyclical calendar transforms, rolling statistics,
and seasonal festival indicators for car sales forecasting.
"""

import numpy as np
import pandas as pd


def create_forecasting_features(monthly_df, target_col="total_units"):
    """
    Transforms uniform monthly sales series into rich feature-engineered tabular data.
    Strictly avoids data leakage by only using backward-looking indicators (lags & rolling windows).
    """
    df = monthly_df.copy()
    df = df.sort_values("period").reset_index(drop=True)

    # 1. Temporal & Cyclical Features
    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter

    # Cyclical trigonometric encodings (preserves continuity between Dec and Jan)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # 2. Domain Seasonal Flags (Indian Automotive Festival & Financial Cycle)
    df["is_festive_season"] = df["month"].apply(lambda m: 1 if m in [10, 11] else 0)
    df["is_fy_close"] = df["month"].apply(lambda m: 1 if m == 3 else 0)
    df["is_year_end"] = df["month"].apply(lambda m: 1 if m == 12 else 0)

    # 3. Autoregressive Lag Features (Prior Sales Volume)
    df["units_lag_1"] = df[target_col].shift(1)
    df["units_lag_2"] = df[target_col].shift(2)
    df["units_lag_3"] = df[target_col].shift(3)

    # 4. Rolling Window Statistical Features
    df["units_roll_mean_3"] = df[target_col].shift(1).rolling(window=3).mean()
    df["units_roll_mean_6"] = df[target_col].shift(1).rolling(window=6).mean()
    df["units_roll_std_3"] = df[target_col].shift(1).rolling(window=3).std().fillna(0)

    # 5. Pricing & Incentive Dynamics (Lagged to prevent leakage)
    if "discount_pct" in df.columns:
        df["discount_pct_lag_1"] = df["discount_pct"].shift(1)
    else:
        df["discount_pct_lag_1"] = 2.5

    # Target variable
    df["target"] = df[target_col]

    # Drop early warm-up periods where lag_3 / roll_mean_6 are NaN
    features_df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "year", "month", "quarter", "sin_month", "cos_month",
        "is_festive_season", "is_fy_close", "is_year_end",
        "units_lag_1", "units_lag_2", "units_lag_3",
        "units_roll_mean_3", "units_roll_mean_6", "units_roll_std_3",
        "discount_pct_lag_1"
    ]

    return features_df, feature_cols
