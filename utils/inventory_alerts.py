"""
AutoVision AI - Smart Inventory Alerts & Demand Deficit Engine
Correlates real-time showroom stock with ML forecasted demand to identify
critical shortages, dwell-time aging, and proactive order replenishment recommendations.
"""

import pandas as pd
import numpy as np
from database.connection import fetch_dataframe
from ml.predict import generate_future_forecast


def compute_smart_inventory_alerts():
    """
    Computes comprehensive inventory risk alerts:
    - Critical Shortages (Days of Stock < 15 days based on forecasted demand)
    - Low-Stock Warnings (Days of Stock 15 - 30 days)
    - Critical Ageing Stock (> 90 days in bay)
    - Moderate Ageing Stock (60 - 89 days in bay)
    - Healthy Stock Equilibrium
    """
    # 1. Fetch available inventory by model
    sql_inv = """
        SELECT
            v.id as vehicle_id, v.brand, v.model, v.variant, v.body_type,
            v.fuel_type, v.cost_price, v.ex_showroom_price,
            COUNT(CASE WHEN i.stock_status = 'Available' THEN 1 END) as available_units,
            COUNT(CASE WHEN i.stock_status = 'Booked' THEN 1 END) as booked_units,
            COALESCE(AVG(CASE WHEN i.stock_status = 'Available' THEN i.days_in_stock END), 0) as avg_days_in_stock,
            COUNT(CASE WHEN i.stock_status = 'Available' AND i.days_in_stock >= 90 THEN 1 END) as critical_aging_units,
            COUNT(CASE WHEN i.stock_status = 'Available' AND i.days_in_stock >= 60 AND i.days_in_stock < 90 THEN 1 END) as warning_aging_units
        FROM vehicles v
        LEFT JOIN inventory i ON v.id = i.vehicle_id
        GROUP BY v.id
    """
    inv_df = fetch_dataframe(sql_inv)

    # 2. Fetch historical sales velocity (past 90 days)
    sql_sales_rate = """
        SELECT vehicle_id, COUNT(id) as sales_90d
        FROM sales
        WHERE sale_date >= (SELECT DATE(MAX(sale_date), '-90 day') FROM sales)
        GROUP BY vehicle_id
    """
    sales_df = fetch_dataframe(sql_sales_rate)
    merged = inv_df.merge(sales_df, on="vehicle_id", how="left").fillna({"sales_90d": 0})

    # Historical monthly sales rate
    merged["monthly_sales_rate"] = merged["sales_90d"] / 3.0

    # 3. Integrate next-month ML forecasted demand
    try:
        fc_df, _, _ = generate_future_forecast(horizon_months=1)
        next_month_total_forecast = float(fc_df["predicted_units"].iloc[0])
    except Exception:
        next_month_total_forecast = 40.0

    total_past_sales = merged["monthly_sales_rate"].sum()
    if total_past_sales > 0:
        merged["forecasted_model_demand"] = (merged["monthly_sales_rate"] / total_past_sales) * next_month_total_forecast
    else:
        merged["forecasted_model_demand"] = next_month_total_forecast / len(merged)

    # 4. Calculate Days of Supply (DOS)
    # DOS = (Available Stock / Monthly Demand) * 30 days
    merged["days_of_supply"] = np.where(
        merged["forecasted_model_demand"] > 0,
        (merged["available_units"] / merged["forecasted_model_demand"]) * 30.0,
        np.where(merged["available_units"] > 0, 999.0, 0.0)
    )

    # 5. Classify Severity & Suggested Order Quantity
    alerts = []
    for _, row in merged.iterrows():
        car_name = f"{row['brand']} {row['model']}"
        avail = int(row["available_units"])
        dos = float(row["days_of_supply"])
        demand = float(row["forecasted_model_demand"])
        avg_age = float(row["avg_days_in_stock"])
        crit_age = int(row["critical_aging_units"])
        warn_age = int(row["warning_aging_units"])

        # Desired buffer = 45 days of demand
        target_buffer = int(np.ceil((demand / 30.0) * 45.0))
        suggested_order = max(0, target_buffer - avail)

        # Alert Logic
        if avail == 0 and demand >= 0.5:
            severity = "🚨 Critical Shortage"
            reason = f"Zero units in showroom! Demand expects ~{demand:.1f} deliveries next month."
            action = f"Immediate purchase order required: {max(2, suggested_order)} units."
        elif dos < 15 and demand >= 1.0:
            severity = "⚡ Imminent Stockout"
            reason = f"Only {avail} unit(s) left (~{dos:.0f} days supply vs forecasted demand of {demand:.1f} units)."
            action = f"Order replenishment: {suggested_order} units."
        elif crit_age > 0:
            severity = "⏳ Critical Aging Alert"
            reason = f"{crit_age} unit(s) have exceeded 90 days in bay (Average dwell: {avg_age:.0f} days)."
            action = "Implement targeted clearance discounts or manager incentives."
        elif warn_age > 0:
            severity = "⚠️ Aging Warning"
            reason = f"{warn_age} unit(s) between 60-89 days in stock."
            action = "Review showroom floor rotation and test drive promotions."
        elif dos < 30 and demand >= 0.8:
            severity = "ℹ️ Low Stock Buffer"
            reason = f"Stock level is adequate for only ~{dos:.0f} days."
            action = f"Plan replenishment cycle for {suggested_order} units."
        else:
            severity = "✅ Healthy Equilibrium"
            reason = f"Healthy stock buffer of {avail} unit(s) ({dos:.0f} days coverage)."
            action = "Maintain regular monitoring."

        alerts.append({
            "Vehicle": car_name,
            "Variant": row["variant"],
            "Body_Type": row["body_type"],
            "Fuel": row["fuel_type"],
            "Available_Units": avail,
            "Booked_Units": int(row["booked_units"]),
            "Days_of_Supply": round(dos, 1),
            "Forecasted_Demand": round(demand, 1),
            "Avg_Days_In_Stock": round(avg_age, 0),
            "Critical_Aging": crit_age,
            "Severity": severity,
            "Reason": reason,
            "Suggested_Order_Qty": suggested_order,
            "Recommended_Action": action
        })

    alerts_df = pd.DataFrame(alerts)
    return alerts_df
