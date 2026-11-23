"""
AutoVision AI - Explainable AI Business Recommendations
Generates transparent, rule-based decision support insights by cross-referencing
live showroom inventory, machine learning sales forecasts, and CRM pipeline activity.
"""

import textwrap
import streamlit as st
import pandas as pd
from database.connection import fetch_dataframe
from ml.predict import generate_future_forecast
from utils.helpers import format_currency, render_hero_banner, render_badge


def generate_recommendations():
    """
    Computes explainable, rule-grounded recommendations across 4 operational pillars:
    1. Inventory Holding & Aging Risk
    2. Stock Depletion & Reorder Urgency
    3. Discount Policy & Margin Preservation
    4. CRM Sales Pipeline Acceleration
    """
    recs = []

    # 1. Aging Inventory Inspection (>60 days and >90 days)
    sql_aging = """
        SELECT i.vin, v.brand, v.model, v.variant, i.days_in_stock, v.cost_price, v.ex_showroom_price
        FROM inventory i
        JOIN vehicles v ON i.vehicle_id = v.id
        WHERE i.stock_status = 'Available' AND i.days_in_stock >= 60
        ORDER BY i.days_in_stock DESC
    """
    aging_df = fetch_dataframe(sql_aging)
    if not aging_df.empty:
        critical_aging = aging_df[aging_df["days_in_stock"] >= 90]
        recs.append({
            "category": "🚨 Critical Inventory Risk",
            "title": f"{len(aging_df)} Units Over 60 Days in Bay ({len(critical_aging)} Critical >90d)",
            "impact": "High Holding & Capital Cost",
            "detail": (
                f"Available inventory contains {len(aging_df)} vehicles with high dwell times, locking "
                f"{format_currency(aging_df['cost_price'].sum())} in working capital. "
                f"Longest dwelling unit: {aging_df.iloc[0]['brand']} {aging_df.iloc[0]['model']} ({aging_df.iloc[0]['days_in_stock']} days)."
            ),
            "action": "Authorize a targeted 3-5% dealer discount or bundled accessories promotion for slow-moving units."
        })

    # 2. Demand vs Stock Deficit (Stockout Risk)
    sql_low = """
        SELECT v.brand, v.model,
               COUNT(CASE WHEN i.stock_status = 'Available' THEN 1 END) as in_stock,
               COUNT(s.id) as past_sales
        FROM vehicles v
        LEFT JOIN inventory i ON v.id = i.vehicle_id
        LEFT JOIN sales s ON v.id = s.vehicle_id AND s.sale_date >= DATE('now', '-90 day')
        GROUP BY v.brand, v.model
        HAVING past_sales >= 10 AND in_stock <= 1
    """
    deficit_df = fetch_dataframe(sql_low)
    if not deficit_df.empty:
        deficit_models = [f"{r['brand']} {r['model']} ({r['in_stock']} in stock)" for _, r in deficit_df.iterrows()]
        recs.append({
            "category": "⚡ Imminent Stockout Warning",
            "title": f"High Velocity Models Facing Stockout: {', '.join(deficit_models[:2])}",
            "impact": "Potential Lost Sales & Extended Lead Times",
            "detail": (
                f"High-demand models ({len(deficit_df)} variants) have less than 2 units remaining in stock "
                f"despite strong quarterly sales velocity."
            ),
            "action": "Place priority allocation orders with OEM dispatch for the flagged models to prevent booking drop-offs."
        })

    # 3. Forecast Growth Uplift
    try:
        fc_df, _, _ = generate_future_forecast(3)
        next_month_units = fc_df["predicted_units"].iloc[0]
        recs.append({
            "category": "📈 Predictive Sales Forecast",
            "title": f"Projected Sales for Next Month: ~{int(next_month_units)} Units",
            "impact": "Capacity & Logistics Planning",
            "detail": (
                f"Our supervised ML regression engine projects next month sales between "
                f"{fc_df['lower_bound'].iloc[0]:.0f} and {fc_df['upper_bound'].iloc[0]:.0f} units (95% CI)."
            ),
            "action": "Align PDI (Pre-Delivery Inspection) bay schedules and schedule sales consultant rosters to handle delivery surges."
        })
    except Exception:
        pass

    # 4. Pipeline Conversion Bottlenecks
    sql_funnel = """
        SELECT
            COUNT(*) as total_enq,
            SUM(test_drive_conducted) as test_drives,
            (SELECT COUNT(*) FROM bookings WHERE status = 'Confirmed') as confirmed_books
        FROM enquiries
    """
    f_df = fetch_dataframe(sql_funnel)
    if not f_df.empty:
        t_enq = f_df["total_enq"].iloc[0]
        t_td = f_df["test_drives"].iloc[0]
        c_rate = (t_td / max(1, t_enq)) * 100
        if c_rate < 50:
            recs.append({
                "category": "🎯 Sales Pipeline Optimization",
                "title": f"Test Drive Conversion currently at {c_rate:.1f}%",
                "impact": "Customer Drop-off at Top of Funnel",
                "detail": f"Out of {t_enq} active enquiries, only {t_td} have progressed to a completed or scheduled test drive.",
                "action": "Incentivize doorstep test drive offerings for customers in distant suburban zones to drive conversion into booking."
            })

    return recs


def render():
    render_hero_banner(
        title="Autonomous Dealership Business Recommendations",
        subtitle="Explainable, rule-based decision support generated from inventory velocity, ML forecasting, and CRM metrics",
        badge_text="Explainable AI"
    )

    st.markdown("""
    > [!NOTE]
    > **Algorithmic Transparency Disclosure:** These business recommendations are dynamically generated 
    > by evaluating explainable heuristic business logic against live database records and supervised ML forecast outputs.
    """)

    recommendations = generate_recommendations()

    for idx, rec in enumerate(recommendations):
        st.markdown(textwrap.dedent(f"""
        <div class="content-card" style="border-left: 4px solid var(--primary-cyan); margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #00F5D4; font-size: 0.95rem;">{rec['category']}</span>
                <span class="badge badge-amber">{rec['impact']}</span>
            </div>
            <h3 style="margin: 0 0 10px 0; font-size: 1.15rem; color: #FFFFFF;">{rec['title']}</h3>
            <p style="color: #94A3B8; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.5;">{rec['detail']}</p>
            <div style="background: rgba(0, 245, 212, 0.08); padding: 10px 14px; border-radius: 6px; border-left: 3px solid #00F5D4;">
                <strong style="color: #00F5D4;">💡 Recommended Action:</strong> <span style="color: #E2E8F0;">{rec['action']}</span>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
