"""
AutoVision AI - Machine Learning Data Preprocessing
Prepares monthly aggregated automotive sales time series from database transactions or CSV.
Ensures zero calendar gaps and formats consistent tabular features.
"""

import pandas as pd
import numpy as np
from database.connection import fetch_dataframe


def load_sales_data(source="db", csv_path=None):
    """
    Loads raw transaction sales records either from database or uploaded/local CSV.
    """
    if source == "db":
        sql = """
            SELECT s.id, s.sale_date, s.selling_price, s.cost_price, s.discount_amount,
                   s.accessories_amount, s.insurance_amount, s.gross_profit,
                   v.brand, v.model, v.fuel_type, v.body_type, v.transmission
            FROM sales s
            JOIN vehicles v ON s.vehicle_id = v.id
            ORDER BY s.sale_date ASC
        """
        df = fetch_dataframe(sql)
    else:
        if csv_path is None:
            raise ValueError("CSV path must be provided when source='csv'")
        df = pd.read_csv(csv_path)

    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def aggregate_monthly_sales(raw_sales_df, group_by_brand=False):
    """
    Aggregates daily sales transactions into uniform monthly time series intervals.
    """
    df = raw_sales_df.copy()
    df["period"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()

    if group_by_brand and "brand" in df.columns:
        grouped = df.groupby(["period", "brand"]).agg(
            total_units=("id", "count"),
            total_revenue=("selling_price", "sum"),
            avg_selling_price=("selling_price", "mean"),
            avg_discount=("discount_amount", "mean"),
            total_gross_profit=("gross_profit", "sum")
        ).reset_index()
    else:
        grouped = df.groupby("period").agg(
            total_units=("id", "count"),
            total_revenue=("selling_price", "sum"),
            avg_selling_price=("selling_price", "mean"),
            avg_discount=("discount_amount", "mean"),
            total_gross_profit=("gross_profit", "sum")
        ).reset_index()

    grouped = grouped.sort_values("period").reset_index(drop=True)

    # Calculate average discount percentage
    grouped["discount_pct"] = (grouped["avg_discount"] / grouped["avg_selling_price"]) * 100
    grouped["discount_pct"] = grouped["discount_pct"].fillna(0)

    return grouped
