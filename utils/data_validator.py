"""
AutoVision AI - Dataset Validation & Ingestion Engine
Validates user-uploaded CSV and Excel files for sales transactions and showroom inventory.
Performs schema checking, date/numeric validation, missing value detection, and type conversion.
"""

import pandas as pd
import numpy as np


def read_uploaded_file(uploaded_file):
    """
    Reads an uploaded Streamlit file (CSV or Excel) into a Pandas DataFrame.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")


def validate_sales_dataset(df):
    """
    Validates a sales transaction dataset against schema and business integrity rules.
    """
    results = {
        "is_valid": True,
        "total_rows": len(df),
        "valid_rows": 0,
        "invalid_rows": 0,
        "errors": [],
        "warnings": [],
        "missing_summary": {},
        "cleaned_df": None
    }

    # Normalize column names
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    norm_df = df.rename(columns=col_map).copy()

    # Required and Optional fields
    required_fields = ["sale_date", "selling_price"]
    optional_fields = ["brand", "model", "cost_price", "discount_amount", "gross_profit", "customer_name"]

    # Check for missing required columns
    missing_required = [f for f in required_fields if f not in norm_df.columns]
    if missing_required:
        results["is_valid"] = False
        results["errors"].append(f"Missing mandatory column(s): {', '.join(missing_required)}")
        return results

    # 1. Date Validation
    try:
        norm_df["sale_date"] = pd.to_datetime(norm_df["sale_date"], errors="coerce")
        bad_dates = norm_df["sale_date"].isna().sum()
        if bad_dates > 0:
            results["warnings"].append(f"{bad_dates} row(s) contain invalid or unparseable dates.")
    except Exception as e:
        results["is_valid"] = False
        results["errors"].append(f"Date column conversion failed: {str(e)}")
        return results

    # 2. Numeric Validation
    try:
        norm_df["selling_price"] = pd.to_numeric(norm_df["selling_price"], errors="coerce")
        bad_prices = (norm_df["selling_price"].isna() | (norm_df["selling_price"] <= 0)).sum()
        if bad_prices > 0:
            results["warnings"].append(f"{bad_prices} row(s) have missing or non-positive selling prices.")
    except Exception as e:
        results["is_valid"] = False
        results["errors"].append(f"Selling price numeric conversion failed: {str(e)}")
        return results

    # 3. Missing values summary
    results["missing_summary"] = {col: int(norm_df[col].isna().sum()) for col in norm_df.columns}

    # Filter out corrupted rows
    clean_df = norm_df.dropna(subset=["sale_date", "selling_price"]).copy()
    clean_df = clean_df[clean_df["selling_price"] > 0]

    # Impute standard fields if missing
    if "cost_price" not in clean_df.columns:
        clean_df["cost_price"] = clean_df["selling_price"] * 0.85
        results["warnings"].append("Defaulted 'cost_price' to 85% of selling price.")
    if "gross_profit" not in clean_df.columns:
        clean_df["gross_profit"] = clean_df["selling_price"] - clean_df["cost_price"]
    if "discount_amount" not in clean_df.columns:
        clean_df["discount_amount"] = 0.0
    if "brand" not in clean_df.columns:
        clean_df["brand"] = "Dealership Catalog"
    if "model" not in clean_df.columns:
        clean_df["model"] = "Vehicle Unit"

    results["valid_rows"] = len(clean_df)
    results["invalid_rows"] = results["total_rows"] - results["valid_rows"]
    results["cleaned_df"] = clean_df
    results["is_valid"] = (results["valid_rows"] > 0)

    return results


def validate_inventory_dataset(df):
    """
    Validates a showroom physical inventory dataset against schema and uniqueness rules.
    """
    results = {
        "is_valid": True,
        "total_rows": len(df),
        "valid_rows": 0,
        "invalid_rows": 0,
        "errors": [],
        "warnings": [],
        "missing_summary": {},
        "cleaned_df": None
    }

    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    norm_df = df.rename(columns=col_map).copy()

    required_fields = ["vin", "brand", "model"]
    missing_required = [f for f in required_fields if f not in norm_df.columns]
    if missing_required:
        results["is_valid"] = False
        results["errors"].append(f"Missing mandatory column(s): {', '.join(missing_required)}")
        return results

    # Check for duplicate VINs
    dup_vins = norm_df["vin"].duplicated().sum()
    if dup_vins > 0:
        results["warnings"].append(f"Detected {dup_vins} duplicate VIN records. Duplicate rows will be deduplicated.")
        norm_df = norm_df.drop_duplicates(subset=["vin"])

    # Arrival date check
    if "arrival_date" in norm_df.columns:
        norm_df["arrival_date"] = pd.to_datetime(norm_df["arrival_date"], errors="coerce")
    else:
        norm_df["arrival_date"] = pd.Timestamp.now()
        results["warnings"].append("Defaulted 'arrival_date' to current date.")

    # Status check
    if "stock_status" in norm_df.columns:
        norm_df["stock_status"] = norm_df["stock_status"].fillna("Available")
    else:
        norm_df["stock_status"] = "Available"

    # Days in stock
    if "days_in_stock" in norm_df.columns:
        norm_df["days_in_stock"] = pd.to_numeric(norm_df["days_in_stock"], errors="coerce").fillna(0).astype(int)
    else:
        now = pd.Timestamp.now()
        norm_df["days_in_stock"] = (now - norm_df["arrival_date"]).dt.days.clip(lower=0)

    clean_df = norm_df.dropna(subset=["vin", "brand", "model"]).copy()

    results["valid_rows"] = len(clean_df)
    results["invalid_rows"] = results["total_rows"] - results["valid_rows"]
    results["cleaned_df"] = clean_df
    results["is_valid"] = (results["valid_rows"] > 0)
    results["missing_summary"] = {col: int(norm_df[col].isna().sum()) for col in norm_df.columns}

    return results
