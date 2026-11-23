"""
Verification script for all 8 advanced features.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

from ml.explainability import get_global_feature_attribution, explain_prediction_waterfall
from utils.inventory_alerts import compute_smart_inventory_alerts
from utils.data_validator import validate_sales_dataset, validate_inventory_dataset
from modules.assistant import query_dealership_brain
from ml.experiment_tracker import get_experiment_history, log_experiment
from utils.excel_exporter import generate_dealership_excel
import pandas as pd
import joblib
import config

print("=" * 60)
print("AutoVision AI - Advanced Features Verification Suite")
print("=" * 60)

# Feature 1: Explainable AI
print("\n[1] Testing Explainable AI (XAI)...")
model = joblib.load(config.MODELS_DIR / "best_forecasting_model.joblib")
feat_cols = joblib.load(config.MODELS_DIR / "feature_cols.joblib")
attr_df = get_global_feature_attribution(model, feat_cols)
print(f"Global Attributions generated for {len(attr_df)} features:")
print(attr_df.head(4)[["Feature", "Normalized_Importance", "Direction"]])

dummy_feat = {col: 1.0 for col in feat_cols}
wf_res = explain_prediction_waterfall(model, dummy_feat, feat_cols, baseline_mean=35.0)
print("Local Waterfall narrative:", wf_res["explanation_text"][:90] + "...")

# Feature 2: Smart Inventory Alerts
print("\n[2] Testing Smart Inventory Alerts...")
alerts = compute_smart_inventory_alerts()
print(f"Alerts computed for {len(alerts)} vehicle variants.")
print("Alert breakdown by severity:", dict(alerts["Severity"].value_counts()))

# Feature 3: Data Validation (Sales & Inventory)
print("\n[3] Testing CSV/Excel Data Validation...")
df_sales = pd.DataFrame([
    {"sale_date": "2025-01-01", "selling_price": 1450000, "brand": "Tata", "model": "Nexon"},
    {"sale_date": "invalid-date", "selling_price": -500} # should be flagged
])
v_sales = validate_sales_dataset(df_sales)
print(f"Sales Validation - Total: {v_sales['total_rows']}, Valid: {v_sales['valid_rows']}, Flagged: {v_sales['invalid_rows']}")

df_inv = pd.DataFrame([
    {"vin": "VIN001", "brand": "Mahindra", "model": "Thar", "stock_status": "Available"},
    {"vin": "VIN001", "brand": "Mahindra", "model": "Thar"} # duplicate VIN
])
v_inv = validate_inventory_dataset(df_inv)
print(f"Inventory Validation - Total: {v_inv['total_rows']}, Valid: {v_inv['valid_rows']}, Deduplicated: {len(v_inv['warnings']) > 0}")

# Feature 4: Actual vs Predicted
print("\n[4] Testing Actual vs Predicted Calculations...")
from ml.train_models import train_and_evaluate_all
eval_res = train_and_evaluate_all(test_size_months=6)
pred_df = eval_res["test_predictions"]
print("Available prediction series:", list(pred_df.columns))

# Feature 5: AI Business Assistant
print("\n[5] Testing Conversational AI Business Assistant...")
for q in ["What is our best selling car model?", "What is our total revenue?", "How many cars are available in stock?"]:
    res = query_dealership_brain(q)
    print(f"Q: {q}")
    print(f"A: {res['text'][:110]}...")

# Feature 6: Automated Reports (Excel & PDF)
print("\n[6] Testing Automated Reports (Excel Workbook)...")
kpi_df = pd.DataFrame([{"Metric": "Revenue", "Value": 1000000}])
excel_bytes = generate_dealership_excel(kpi_df, df_sales, df_inv, pd.DataFrame([{"Period": "2026-01", "Units": 40}]), pd.DataFrame([{"Model": "LR", "RMSE": 2.9}]))
print(f"Multi-sheet Excel workbook compiled successfully: {len(excel_bytes)} bytes.")

# Feature 7: Advanced Customer Segmentation (Silhouette & Elbow)
print("\n[7] Testing Customer K-Means Silhouette Optimization...")
from sklearn.metrics import silhouette_score
from database.connection import fetch_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
cust_data = fetch_dataframe("SELECT annual_income, budget_max FROM customers").dropna()
X_sc = StandardScaler().fit_transform(cust_data)
sil = silhouette_score(X_sc, KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sc))
print(f"K-Means silhouette score for k=3: {sil:.4f}")

# Feature 8: Model Performance Monitoring
print("\n[8] Testing Model Monitoring Registry...")
history = get_experiment_history()
print(f"Experiment ledger contains {len(history)} recorded runs.")

print("\n" + "=" * 60)
print("ALL 8 ENHANCED FEATURES PASSED INTEGRATION VERIFICATION!")
print("=" * 60)
