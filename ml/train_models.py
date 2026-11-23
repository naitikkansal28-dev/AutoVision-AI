"""
AutoVision AI - Model Training Pipeline
Performs chronological train-test split, trains candidate machine learning algorithms,
evaluates out-of-time test metrics, and persists trained artifacts.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import config
from ml.preprocessing import load_sales_data, aggregate_monthly_sales
from ml.feature_engineering import create_forecasting_features
from ml.evaluate_models import calculate_metrics, evaluate_baseline
from ml.experiment_tracker import log_experiment


def train_and_evaluate_all(test_size_months=6, source="db", csv_path=None):
    """
    Executes the full research training pipeline:
    1. Loads sales data and aggregates monthly time series.
    2. Constructs lag & seasonal features without data leakage.
    3. Chronological train-test split (last `test_size_months` as test set).
    4. Trains candidate regressors.
    5. Computes validation metrics (MAE, RMSE, R2, MAPE).
    6. Identifies champion model and serializes artifacts.
    """
    print("[ML Pipeline] Loading sales records...")
    raw_df = load_sales_data(source=source, csv_path=csv_path)
    monthly_df = aggregate_monthly_sales(raw_df)

    if len(monthly_df) < 12:
        raise ValueError("At least 12 months of sales records are required for training and validation.")

    features_df, feature_cols = create_forecasting_features(monthly_df)

    # Chronological Train-Test Split (strictly temporal to prevent lookahead bias)
    if len(features_df) <= test_size_months + 6:
        test_size_months = max(3, len(features_df) // 4)

    train_df = features_df.iloc[:-test_size_months].copy()
    test_df = features_df.iloc[-test_size_months:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # Candidate Models Dictionary
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=4,
            min_samples_split=2,
            random_state=config.ML_RANDOM_STATE
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=1,
            random_state=config.ML_RANDOM_STATE
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=config.ML_RANDOM_STATE
        )
    }

    # Attempt to load XGBoost if available
    try:
        from xgboost import XGBRegressor
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=config.ML_RANDOM_STATE
        )
    except ImportError:
        pass

    results = []
    trained_models = {}
    test_predictions = {"Actual": y_test.values, "Period": test_df["period"].dt.strftime("%b %Y").values}

    # 1. Evaluate Naive Persistence Baseline
    baseline_metrics = evaluate_baseline(y_train, y_test, test_df["units_lag_1"].values)
    results.append(baseline_metrics)
    test_predictions["Naive Baseline"] = test_df["units_lag_1"].values

    # 2. Train and evaluate each candidate algorithm
    for name, model in models.items():
        print(f"[ML Pipeline] Fitting {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Enforce non-negative sales volume
        y_pred = np.maximum(y_pred, 0)
        
        metrics = calculate_metrics(y_test, y_pred, model_name=name)
        results.append(metrics)
        trained_models[name] = model
        test_predictions[name] = np.round(y_pred, 1)

    results_df = pd.DataFrame(results)

    # Sort candidates by lowest RMSE (excluding baseline)
    ml_candidates = results_df[results_df["Model"] != "Naive Persistence (Baseline)"].copy()
    champion_name = ml_candidates.sort_values("RMSE").iloc[0]["Model"]
    champion_model = trained_models[champion_name]

    print(f"[ML Pipeline] Best performing algorithm: {champion_name}")

    # Feature Importance (for tree-based models)
    feature_importances = {}
    if hasattr(champion_model, "feature_importances_"):
        for f, imp in zip(feature_cols, champion_model.feature_importances_):
            feature_importances[f] = round(float(imp), 4)

    # Save artifacts to models/ directory
    joblib.dump(champion_model, config.MODELS_DIR / "best_forecasting_model.joblib")
    joblib.dump(feature_cols, config.MODELS_DIR / "feature_cols.joblib")
    joblib.dump(trained_models, config.MODELS_DIR / "all_models.joblib")

    metadata = {
        "champion_model": champion_name,
        "feature_cols": feature_cols,
        "test_size_months": test_size_months,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "metrics": results,
        "feature_importances": feature_importances,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(config.MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Log experiment run
    try:
        exp_id = log_experiment(
            test_size_months=test_size_months,
            metrics_list=results,
            champion_name=champion_name,
            train_samples=len(train_df),
            test_samples=len(test_df),
            source=source
        )
        metadata["experiment_id"] = exp_id
    except Exception as ex:
        print(f"[ML Pipeline] Experiment logging notice: {ex}")

    return {
        "results_df": results_df,
        "champion_name": champion_name,
        "test_predictions": pd.DataFrame(test_predictions),
        "feature_importances": feature_importances,
        "metadata": metadata,
        "features_df": features_df,
        "monthly_df": monthly_df
    }


if __name__ == "__main__":
    train_and_evaluate_all()
