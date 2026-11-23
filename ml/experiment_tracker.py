"""
AutoVision AI - Machine Learning Experiment Tracking System
Logs and monitors model training runs, hyperparameters, and evaluation metrics (MAE, RMSE, R2, MAPE)
over time to prevent model performance regression and track algorithmic drift.
"""

import json
from datetime import datetime
import pandas as pd
import config

LOG_FILE = config.MODELS_DIR / "experiments_log.json"


def init_experiment_log():
    """Ensures experiment log file exists."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)


def log_experiment(test_size_months, metrics_list, champion_name, train_samples, test_samples, source="db"):
    """
    Appends a new training experiment entry to the persistent registry.
    """
    init_experiment_log()

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except Exception:
            history = []

    exp_id = f"EXP-{len(history) + 1:03d}"
    
    # Extract champion metrics
    champ_metrics = {}
    for m in metrics_list:
        if m.get("Model") == champion_name:
            champ_metrics = m
            break

    entry = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "test_size_months": test_size_months,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "champion_model": champion_name,
        "champion_rmse": champ_metrics.get("RMSE", 0.0),
        "champion_mae": champ_metrics.get("MAE", 0.0),
        "champion_r2": champ_metrics.get("R2_Score", 0.0),
        "champion_mape": champ_metrics.get("MAPE (%)", 0.0),
        "all_metrics": metrics_list
    }

    history.append(entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    return exp_id


def get_experiment_history():
    """
    Returns historical experiment records as a Pandas DataFrame.
    """
    init_experiment_log()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except Exception:
            history = []

    if not history:
        return pd.DataFrame()

    return pd.DataFrame(history)
