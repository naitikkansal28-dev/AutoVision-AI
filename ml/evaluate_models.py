"""
AutoVision AI - Model Evaluation & Performance Metrics
Computes MAE, RMSE, R2, and MAPE on chronological test datasets.
Evaluates candidate algorithms against a simple Naive Persistence baseline.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Computes standard regression evaluation metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # R2 Score (can be negative if model performs worse than mean line)
    r2 = float(r2_score(y_true, y_pred))

    # MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else 0.0

    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2_Score": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    }


def evaluate_baseline(y_train, y_test, test_lags):
    """
    Evaluates a Naive Persistence Baseline: predicts next month's sales = prior month sales (lag 1).
    Essential for academic benchmark rigor!
    """
    y_pred_naive = test_lags
    return calculate_metrics(y_test, y_pred_naive, model_name="Naive Persistence (Baseline)")
