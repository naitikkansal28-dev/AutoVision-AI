import sys
import os
import glob
import py_compile
import traceback
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

errors = []
warnings = []

print("=" * 60)
print("AutoVision AI - Comprehensive System Diagnostics & Error Audit")
print("=" * 60)

# 1. Syntax & Bytecode Compilation
print("\n[1] Compiling all Python source files...")
py_files = [f for f in glob.glob("**/*.py", recursive=True) if "venv" not in f and ".git" not in f]
for f in py_files:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        errors.append(f"Syntax/Compile Error in {f}: {e}")
        print(f"  ❌ {f}: {e}")
if not errors:
    print(f"  ✅ All {len(py_files)} Python source files compiled with 0 syntax errors.")

# 2. Database Connection & Basic Queries
print("\n[2] Testing Database Connection Layer...")
try:
    from database.connection import get_connection, fetch_dataframe, execute_query, get_backend
    backend = get_backend()
    print(f"  Active backend: {backend.upper()}")
    
    # Test queries on each table
    tables = ["users", "vehicles", "sales", "inventory", "customers", "enquiries", "bookings", "expenses"]
    for tbl in tables:
        df = fetch_dataframe(f"SELECT * FROM {tbl} LIMIT 5;")
        print(f"  ✅ Table '{tbl}': {len(df)} sample rows retrieved.")
except Exception as e:
    errors.append(f"Database Query Error: {traceback.format_exc()}")
    print(f"  ❌ Database error: {e}")

# 3. Authentication & User Logic
print("\n[3] Testing Authentication & Passwords...")
try:
    from utils.authentication import login, verify_password, hash_password
    # Test hashing
    h, s = hash_password("testpass123")
    if not verify_password("testpass123", h, s):
        errors.append("Password hashing / verification mismatch!")
    
    # Test DB authentication for admin & manager
    admin_ok = login("admin", "admin123")
    mgr_ok = login("manager", "manager123")
    print(f"  ✅ Admin login: {admin_ok}, Manager login: {mgr_ok}")
    if not (admin_ok and mgr_ok):
        errors.append("Default admin or manager credentials failed login!")
except Exception as e:
    errors.append(f"Authentication Error: {traceback.format_exc()}")
    print(f"  ❌ Auth error: {e}")

# 4. Machine Learning & Forecasting
print("\n[4] Testing ML Artifacts & Forecasting Engine...")
try:
    from ml.predict import load_champion_artifacts, generate_future_forecast
    model, feature_cols, metadata = load_champion_artifacts()
    print(f"  ✅ Champion model loaded: {metadata.get('champion_model', 'N/A')}")
    print(f"  ✅ Feature count: {len(feature_cols)}")
    
    forecast_df, monthly_df, meta = generate_future_forecast(horizon_months=3)
    print(f"  ✅ Multi-step forecast generated: {len(forecast_df)} months.")
    print(forecast_df[["period", "predicted_units", "lower_bound", "upper_bound"]])
except Exception as e:
    errors.append(f"ML Forecasting Error: {traceback.format_exc()}")
    print(f"  ❌ ML error: {e}")

# 5. Explainable AI (XAI)
print("\n[5] Testing Explainable AI (XAI)...")
try:
    from ml.explainability import get_global_feature_attribution, explain_prediction_waterfall
    attr_df = get_global_feature_attribution(model, feature_cols)
    print(f"  ✅ Global attribution computed: {len(attr_df)} features.")
    
    dummy_feat = {col: 1.0 for col in feature_cols}
    wf = explain_prediction_waterfall(model, dummy_feat, feature_cols, baseline_mean=35.0)
    print(f"  ✅ Local waterfall explanation computed: {len(wf['contributions'])} contributions.")
except Exception as e:
    errors.append(f"XAI Error: {traceback.format_exc()}")
    print(f"  ❌ XAI error: {e}")

# 6. Smart Inventory Alerts
print("\n[6] Testing Smart Inventory Alerts Engine...")
try:
    from utils.inventory_alerts import compute_smart_inventory_alerts
    alerts_df = compute_smart_inventory_alerts()
    print(f"  ✅ Smart alerts generated: {len(alerts_df)} vehicle variants analyzed.")
except Exception as e:
    errors.append(f"Inventory Alerts Error: {traceback.format_exc()}")
    print(f"  ❌ Inventory alerts error: {e}")

# 7. AI Conversational Assistant
print("\n[7] Testing AI Business Assistant...")
try:
    from modules.assistant import query_dealership_brain
    q_test = query_dealership_brain("What is our total revenue?")
    print(f"  ✅ Assistant query response: {q_test['text'][:80]}...")
except Exception as e:
    errors.append(f"Assistant Error: {traceback.format_exc()}")
    print(f"  ❌ Assistant error: {e}")

# 8. Automated Document Exporters (Excel & PDF)
print("\n[8] Testing Excel and PDF Generation...")
try:
    from utils.excel_exporter import generate_dealership_excel
    from utils.pdf_exporter import generate_dealership_pdf
    
    kpi_df = pd.DataFrame([{"Metric": "Revenue", "Value": 1000000}])
    sales_df = pd.DataFrame([{"sale_date": "2025-01-01", "selling_price": 1200000}])
    inv_df = pd.DataFrame([{"vin": "V123", "brand": "Tata", "model": "Nexon"}])
    fc_df = pd.DataFrame([{"Period": "2026-01", "Forecasted_Sales": 45}])
    eval_df = pd.DataFrame([{"Model": "Random Forest", "RMSE": 2.96}])
    
    xl = generate_dealership_excel(kpi_df, sales_df, inv_df, fc_df, eval_df)
    print(f"  ✅ Excel export generated: {len(xl)} bytes.")
    
    kpis_dict = {"total_revenue": 15000000, "total_cars_sold": 1100, "inventory_count": 65, "active_bookings": 35}
    pdf = generate_dealership_pdf(kpis_dict, sales_df, fc_df, eval_df)
    print(f"  ✅ PDF export generated: {len(pdf)} bytes.")
except Exception as e:
    errors.append(f"Exporter Error: {traceback.format_exc()}")
    print(f"  ❌ Exporter error: {e}")

# 9. Test Ingestion Validator
print("\n[9] Testing Ingestion Validator...")
try:
    from utils.data_validator import validate_sales_dataset, validate_inventory_dataset
    v_res = validate_sales_dataset(sales_df)
    print(f"  ✅ Sales validator: {v_res['total_rows']} rows checked.")
except Exception as e:
    errors.append(f"Data Validator Error: {traceback.format_exc()}")
    print(f"  ❌ Data validator error: {e}")

# 10. Test All UI Modules Import
print("\n[10] Testing Import of All 14 UI Modules...")
modules_list = [
    "dashboard", "sales", "forecasting", "actual_vs_predicted",
    "inventory", "customers", "bookings", "finance", "assistant",
    "insights", "model_monitoring", "upload", "research", "reports"
]
for mod_name in modules_list:
    try:
        mod = __import__(f"modules.{mod_name}", fromlist=["render"])
        if hasattr(mod, "render"):
            print(f"  ✅ modules.{mod_name} has render()")
        else:
            errors.append(f"modules.{mod_name} has no render() function")
            print(f"  ❌ modules.{mod_name}: missing render()")
    except Exception as e:
        errors.append(f"Module modules.{mod_name} import error: {traceback.format_exc()}")
        print(f"  ❌ modules.{mod_name}: {e}")

print("\n" + "=" * 60)
print(f"AUDIT SUMMARY: {len(errors)} Errors, {len(warnings)} Warnings.")
if errors:
    print("\nFAILED CHECKS:")
    for err in errors:
        print(f" - {err}")
    sys.exit(1)
else:
    print("ALL MODULES, PIPELINES, DATABASE & UTILITIES VERIFIED ERROR-FREE!")
    sys.exit(0)
