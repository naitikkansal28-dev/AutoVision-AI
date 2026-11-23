"""
AutoVision AI - Inventory Management & Demand Replenishment Module
Tracks physical vehicle stock units, ageing inventory (>60 and >90 days),
stock movement, low-stock triggers, and ML-assisted replenishment decision support.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from database.connection import fetch_dataframe, execute_query
from utils.inventory_alerts import compute_smart_inventory_alerts
from utils.helpers import format_currency, format_number, render_hero_banner, render_kpi_card, render_badge
from utils.charts import plot_bar, plot_donut


def render():
    render_hero_banner(
        title="Showroom Inventory & Supply Replenishment",
        subtitle="Physical vehicle tracking, aging analysis, stock turn velocity, and ML-assisted order planning",
        badge_text="Inventory Intelligence"
    )

    tab_alerts, tab_overview, tab_aging, tab_replenish, tab_add = st.tabs([
        "⚡ Smart Inventory Alerts",
        "📦 Stock Overview",
        "⏳ Ageing Inventory & Risk",
        "💡 Replenishment Decision Support",
        "➕ Add New Stock Unit"
    ])

    # 1. Base Inventory Query
    sql_inv = """
        SELECT
            i.id, i.vin, i.stock_status, i.arrival_date, i.showroom_location, i.days_in_stock,
            v.id as vehicle_id, v.brand, v.model, v.variant, v.body_type, v.fuel_type,
            v.transmission, v.cost_price, v.ex_showroom_price
        FROM inventory i
        JOIN vehicles v ON i.vehicle_id = v.id
        ORDER BY i.days_in_stock DESC
    """
    df_inv = fetch_dataframe(sql_inv)

    # Key Metrics
    total_stock = len(df_inv)
    avail_df = df_inv[df_inv["stock_status"] == "Available"]
    booked_df = df_inv[df_inv["stock_status"] == "Booked"]
    sold_df = df_inv[df_inv["stock_status"] == "Sold"]

    avail_count = len(avail_df)
    booked_count = len(booked_df)
    sold_count = len(sold_df)

    # Capital locked in available inventory
    locked_capital = avail_df["cost_price"].sum()
    avg_age = avail_df["days_in_stock"].mean() if not avail_df.empty else 0

    # Aging counters (>60 and >90 days)
    warning_aging = len(avail_df[(avail_df["days_in_stock"] >= 60) & (avail_df["days_in_stock"] < 90)])
    critical_aging = len(avail_df[avail_df["days_in_stock"] >= 90])

    # =========================================================================
    # TAB 1: Smart Inventory Alerts (Integrated with ML Forecast)
    # =========================================================================
    with tab_alerts:
        st.markdown("### Smart Inventory Alert Hub & Stockout Risk Management")
        st.markdown("""
        Proactively monitors inventory levels against **ML forecasted demand** to prevent stockouts 
        and flags units experiencing extended showroom dwell times.
        """)

        alerts_df = compute_smart_inventory_alerts()

        # Severity Counts
        crit_shortage = len(alerts_df[alerts_df["Severity"].str.contains("Critical Shortage|Imminent Stockout", na=False)])
        crit_aging_count = len(alerts_df[alerts_df["Severity"].str.contains("Critical Aging", na=False)])
        warn_aging_count = len(alerts_df[alerts_df["Severity"].str.contains("Aging Warning", na=False)])
        low_buf = len(alerts_df[alerts_df["Severity"].str.contains("Low Stock", na=False)])
        healthy_count = len(alerts_df[alerts_df["Severity"].str.contains("Healthy", na=False)])

        a1, a2, a3, a4, a5 = st.columns(5)
        with a1:
            st.markdown(render_kpi_card("Stockout Threats", f"{crit_shortage}", "< 15 Days Supply", "negative" if crit_shortage > 0 else "positive", "red"), unsafe_allow_html=True)
        with a2:
            st.markdown(render_kpi_card("Critical Aging (>90d)", f"{crit_aging_count}", "Holding Cost Risk", "negative" if crit_aging_count > 0 else "positive", "amber"), unsafe_allow_html=True)
        with a3:
            st.markdown(render_kpi_card("Aging Warning (60-89d)", f"{warn_aging_count}", "Dwell Time Warning", "neutral", "purple"), unsafe_allow_html=True)
        with a4:
            st.markdown(render_kpi_card("Low Buffer (15-30d)", f"{low_buf}", "Reorder Planning", "neutral", "blue"), unsafe_allow_html=True)
        with a5:
            st.markdown(render_kpi_card("Well-Stocked", f"{healthy_count}", "Optimal Coverage", "positive", "green"), unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Filter by Severity
        severities = ["All Alerts"] + sorted(alerts_df["Severity"].unique().tolist())
        sel_sev = st.selectbox("Filter Alerts by Severity Level", severities, index=0)

        filtered_alerts = alerts_df if sel_sev == "All Alerts" else alerts_df[alerts_df["Severity"] == sel_sev]

        st.dataframe(
            filtered_alerts[[
                "Severity", "Vehicle", "Variant", "Available_Units", "Days_of_Supply",
                "Forecasted_Demand", "Suggested_Order_Qty", "Reason", "Recommended_Action"
            ]].rename(
                columns={
                    "Available_Units": "Available",
                    "Days_of_Supply": "Days of Supply (DOS)",
                    "Forecasted_Demand": "Next Mo. Demand",
                    "Suggested_Order_Qty": "Order Qty",
                    "Recommended_Action": "Prescriptive Action"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 2: Stock Overview
    # =========================================================================
    with tab_overview:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(render_kpi_card("Total Units Tracked", f"{total_stock}", "Showroom + Yard", "neutral", "blue"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_kpi_card("Available for Sale", f"{avail_count}", "Ready for Delivery", "positive", "green"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_kpi_card("Units Booked", f"{booked_count}", "Deposit Received", "positive", "purple"), unsafe_allow_html=True)
        with c4:
            st.markdown(render_kpi_card("Locked Working Capital", format_currency(locked_capital), "At Dealer Cost", "neutral", "cyan"), unsafe_allow_html=True)
        with c5:
            st.markdown(render_kpi_card("Average Age in Bay", f"{avg_age:.0f} Days", f"{critical_aging} critical (>90d)", "negative" if critical_aging > 0 else "positive", "amber"), unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        col_st1, col_st2 = st.columns([1.6, 1.4])
        with col_st1:
            st.markdown("#### Vehicle-Wise Available Stock Levels")
            model_stock = avail_df.groupby(["brand", "model"]).size().reset_index(name="Units In Stock")
            model_stock["Vehicle"] = model_stock["brand"] + " " + model_stock["model"]
            model_stock = model_stock.sort_values("Units In Stock", ascending=False)

            fig_stock = plot_bar(
                model_stock,
                x="Units In Stock",
                y="Vehicle",
                orientation="h",
                title="Current Available Units by Vehicle Model",
                height=380
            )
            fig_stock.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_stock, use_container_width=True)

        with col_st2:
            st.markdown("#### Stock Status Distribution")
            status_df = df_inv.groupby("stock_status").size().reset_index(name="Count")
            fig_stat = plot_donut(status_df, names="stock_status", values="Count", title="Physical Stock Status Split", height=340)
            st.plotly_chart(fig_stat, use_container_width=True)

        # Inventory Table with search filter
        st.markdown("#### Live Physical Units Directory")
        search_query = st.text_input("🔍 Quick Search by VIN, Model, or Location:", "")
        display_inv = df_inv.copy()
        if search_query:
            display_inv = display_inv[
                display_inv["vin"].str.contains(search_query, case=False, na=False) |
                display_inv["model"].str.contains(search_query, case=False, na=False) |
                display_inv["showroom_location"].str.contains(search_query, case=False, na=False)
            ]

        display_inv["Ex-Showroom"] = display_inv["ex_showroom_price"].apply(lambda p: format_currency(p))
        st.dataframe(
            display_inv[["vin", "brand", "model", "variant", "fuel_type", "stock_status", "days_in_stock", "showroom_location", "Ex-Showroom"]].rename(
                columns={"vin": "VIN", "stock_status": "Status", "days_in_stock": "Days in Stock", "showroom_location": "Location"}
            ),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 2: Ageing Inventory & Risk
    # =========================================================================
    with tab_aging:
        st.markdown("### Ageing Vehicle Depreciation & Carrying Cost Risk")
        st.markdown("""
        Vehicles standing in the showroom bay or holding yard for extended periods incur insurance holding costs, 
        battery degradation, and risk margin compression during model facelifts.
        """)

        aging_col1, aging_col2 = st.columns(2)
        with aging_col1:
            st.warning(f"⚠️ **Moderate Ageing (60 - 89 Days):** {warning_aging} Units currently in bay.")
        with aging_col2:
            st.error(f"🚨 **Critical Ageing (90+ Days):** {critical_aging} Units requiring immediate clearance promotion.")

        aging_table = avail_df[avail_df["days_in_stock"] >= 60].copy()
        if not aging_table.empty:
            aging_table["Car"] = aging_table["brand"] + " " + aging_table["model"] + " " + aging_table["variant"]
            aging_table["Locked Cost"] = aging_table["cost_price"].apply(lambda c: format_currency(c))
            aging_table["Risk Level"] = aging_table["days_in_stock"].apply(
                lambda d: "🚨 Critical (>90d)" if d >= 90 else "⚠️ Warning (60-89d)"
            )

            st.dataframe(
                aging_table[["vin", "Car", "days_in_stock", "Risk Level", "Locked Cost", "showroom_location"]].rename(
                    columns={"vin": "VIN", "days_in_stock": "Days in Stock", "showroom_location": "Current Bay"}
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("Excellent! No available vehicles currently exceed the 60-day aging threshold.")

    # =========================================================================
    # TAB 3: Replenishment Decision Support
    # =========================================================================
    with tab_replenish:
        st.markdown("### Intelligent Replenishment & Order Quantity Suggestions")
        st.markdown("""
        *Note: These recommendations represent algorithmic decision-support suggestions for showroom managers 
        to balance inventory holding costs against projected customer demand, rather than automatic purchase orders.*
        """)

        # Compute model-level current stock vs sales velocity
        sql_sales_3m = """
            SELECT v.brand, v.model, COUNT(s.id) as recent_sales_count
            FROM sales s
            JOIN vehicles v ON s.vehicle_id = v.id
            WHERE s.sale_date >= DATE('now', '-90 day')
               OR s.sale_date >= (SELECT DATE(MAX(sale_date), '-90 day') FROM sales)
            GROUP BY v.brand, v.model
        """
        recent_sales_df = fetch_dataframe(sql_sales_3m)
        recent_sales_df["monthly_run_rate"] = recent_sales_df["recent_sales_count"] / 3.0

        avail_by_model = avail_df.groupby(["brand", "model"]).size().reset_index(name="current_available_stock")

        replenish_merged = recent_sales_df.merge(avail_by_model, on=["brand", "model"], how="outer").fillna(0)
        replenish_merged["Vehicle"] = replenish_merged["brand"] + " " + replenish_merged["model"]

        # Target 1.5 months of buffer inventory
        replenish_merged["suggested_stock_level"] = np.ceil(replenish_merged["monthly_run_rate"] * 1.5)
        replenish_merged["suggested_reorder_units"] = np.maximum(
            0, replenish_merged["suggested_stock_level"] - replenish_merged["current_available_stock"]
        ).astype(int)

        replenish_merged["Status"] = replenish_merged.apply(
            lambda r: "🚨 Urgent Reorder" if r["current_available_stock"] == 0 and r["monthly_run_rate"] > 1
            else ("⚠️ Low Buffer" if r["suggested_reorder_units"] > 2 else "✅ Well Stocked"),
            axis=1
        )

        display_rep = replenish_merged.sort_values(["suggested_reorder_units", "monthly_run_rate"], ascending=[False, False])
        
        st.dataframe(
            display_rep[["Vehicle", "current_available_stock", "monthly_run_rate", "suggested_stock_level", "suggested_reorder_units", "Status"]].rename(
                columns={
                    "current_available_stock": "Current In Stock",
                    "monthly_run_rate": "Monthly Sales Rate",
                    "suggested_stock_level": "Optimal Target Buffer",
                    "suggested_reorder_units": "Recommended Order Qty"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 4: Add New Stock Unit
    # =========================================================================
    with tab_add:
        st.markdown("### Receive & Register New Vehicle In Showroom Yard")
        
        vehicles_all = fetch_dataframe("SELECT id, brand || ' ' || model || ' - ' || variant as name FROM vehicles")
        v_dict = dict(zip(vehicles_all["name"], vehicles_all["id"]))

        with st.form("form_add_inventory"):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                new_vin = st.text_input("Vehicle VIN / Chassis Number", value=f"AVIND{datetime.now().strftime('%y%m%d%H%M')}")
                chosen_veh = st.selectbox("Vehicle Model & Variant", list(v_dict.keys()))
                location = st.selectbox("Showroom Allocation", ["Main Showroom Floor", "Bay A (Ground)", "Bay B (First Floor)", "Holding Yard"])
            with fcol2:
                arrival_date = st.date_input("Arrival Date", value=datetime.now())
                stock_status = st.selectbox("Initial Stock Status", ["Available", "Booked"])

            submit_stock = st.form_submit_button("📥 Register In Inventory", type="primary")

            if submit_stock:
                try:
                    veh_id = v_dict[chosen_veh]
                    sql_ins = """
                        INSERT INTO inventory (vin, vehicle_id, stock_status, arrival_date, showroom_location, days_in_stock)
                        VALUES (%s, %s, %s, %s, %s, 0)
                    """
                    execute_query(sql_ins, (new_vin, veh_id, stock_status, arrival_date.strftime("%Y-%m-%d"), location))
                    st.success(f"Vehicle {new_vin} successfully registered in inventory!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to add vehicle: {str(ex)}")
