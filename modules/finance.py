"""
AutoVision AI - Revenue & Profit Analytics Module
Comprehensive financial intelligence distinguishing Sales Revenue, Vehicle Acquisition Cost,
Gross Profit, Operating Expenses, and Net Dealership Profit.
"""

import streamlit as st
import pandas as pd
from database.connection import fetch_dataframe
from utils.helpers import format_currency, format_number, render_hero_banner, render_kpi_card
from utils.charts import plot_bar, plot_line, plot_donut, plot_grouped_bar


def render():
    render_hero_banner(
        title="Revenue & Profitability Intelligence",
        subtitle="Financial auditing, gross margin decomposition, operational expense accounting, and net profit margins",
        badge_text="Financial Audit"
    )

    # 1. Fetch Aggregated Financials
    sql_rev = """
        SELECT
            COUNT(s.id) as units_sold,
            COALESCE(SUM(s.selling_price), 0) as total_revenue,
            COALESCE(SUM(s.cost_price), 0) as total_acquisition_cost,
            COALESCE(SUM(s.discount_amount), 0) as total_discounts,
            COALESCE(SUM(s.accessories_amount), 0) as accessories_rev,
            COALESCE(SUM(s.insurance_amount), 0) as insurance_rev,
            COALESCE(SUM(s.gross_profit), 0) as gross_profit
        FROM sales s
    """
    fin_df = fetch_dataframe(sql_rev)
    total_rev = float(fin_df["total_revenue"].iloc[0])
    total_cost = float(fin_df["total_acquisition_cost"].iloc[0])
    total_discount = float(fin_df["total_discounts"].iloc[0])
    acc_rev = float(fin_df["accessories_rev"].iloc[0])
    ins_rev = float(fin_df["insurance_rev"].iloc[0])
    gross_profit = float(fin_df["gross_profit"].iloc[0])

    # Fetch Total Operating Expenses
    sql_exp = "SELECT COALESCE(SUM(amount), 0) as total_expenses FROM expenses"
    operating_expenses = float(fetch_dataframe(sql_exp)["total_expenses"].iloc[0])

    # Net Profit = Gross Profit - Operating Expenses
    net_profit = gross_profit - operating_expenses

    gross_margin_pct = (gross_profit / total_rev * 100) if total_rev > 0 else 0
    net_margin_pct = (net_profit / total_rev * 100) if total_rev > 0 else 0

    # 2. Financial KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(render_kpi_card("Sales Revenue", format_currency(total_rev), "Vehicle Turnover", "positive", "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Acquisition Cost", format_currency(total_cost), "OEM Supply Invoices", "neutral", "cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Gross Profit", format_currency(gross_profit), f"Gross Margin {gross_margin_pct:.1f}%", "positive", "green"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Operating Expenses", format_currency(operating_expenses), "Rent, Payroll, Utilities", "negative", "amber"), unsafe_allow_html=True)
    with c5:
        st.markdown(render_kpi_card("Net Dealership Profit", format_currency(net_profit), f"Net Margin {net_margin_pct:.1f}%", "positive" if net_profit >= 0 else "negative", "purple"), unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    tab_breakdown, tab_expenses, tab_models_profit = st.tabs([
        "💰 Revenue & Margin Decomposition",
        "🏢 Operating Cost Breakdown",
        "🚗 Profitability by Vehicle Model"
    ])

    # =========================================================================
    # TAB 1: Revenue & Margin Decomposition
    # =========================================================================
    with tab_breakdown:
        st.markdown("### Financial Bridge: From Gross Revenue to Net Profit")
        
        waterfall_df = pd.DataFrame([
            {"Component": "Total Vehicle Revenue", "Amount": total_rev, "Type": "Revenue"},
            {"Component": "Less: Acquisition Cost", "Amount": -total_cost, "Type": "Cost"},
            {"Component": "Add: Accessories & Insurance Margins", "Amount": (acc_rev * 0.35 + ins_rev * 0.15), "Type": "Addon"},
            {"Component": "Gross Dealership Profit", "Amount": gross_profit, "Type": "Subtotal"},
            {"Component": "Less: Showroom Operating Costs", "Amount": -operating_expenses, "Type": "Expense"},
            {"Component": "Net Operational Profit", "Amount": net_profit, "Type": "Final"}
        ])

        col_w1, col_w2 = st.columns([1.6, 1.4])
        with col_w1:
            # Monthly Revenue vs Cost Trajectory
            sql_m_fin = """
                SELECT
                    SUBSTR(sale_date, 1, 7) as period,
                    SUM(selling_price) as monthly_revenue,
                    SUM(cost_price) as monthly_cost,
                    SUM(gross_profit) as monthly_gross_profit
                FROM sales
                GROUP BY SUBSTR(sale_date, 1, 7)
                ORDER BY period ASC
            """
            m_fin_df = fetch_dataframe(sql_m_fin)
            fig_m_fin = plot_line(
                m_fin_df,
                x="period",
                y="monthly_revenue",
                title="Monthly Revenue Velocity Over Time",
                height=360
            )
            st.plotly_chart(fig_m_fin, use_container_width=True)

        with col_w2:
            st.markdown("#### Profit Waterfall Summary")
            st.dataframe(waterfall_df, use_container_width=True, hide_index=True)
            st.info(
                f"**Audit Note:** Gross Profit is **{format_currency(gross_profit)}** ({gross_margin_pct:.1f}% margin). "
                f"After accounting for **{format_currency(operating_expenses)}** in operational overheads, "
                f"the dealership achieves a healthy Net Profit of **{format_currency(net_profit)}** ({net_margin_pct:.1f}% net margin)."
            )

    # =========================================================================
    # TAB 2: Operating Cost Breakdown
    # =========================================================================
    with tab_expenses:
        st.markdown("### Showroom Operating Overhead Expenses")
        
        sql_exp_cat = """
            SELECT category, SUM(amount) as total_cat_amount
            FROM expenses
            GROUP BY category
            ORDER BY total_cat_amount DESC
        """
        exp_cat_df = fetch_dataframe(sql_exp_cat)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            fig_exp = plot_donut(exp_cat_df, names="category", values="total_cat_amount", title="Operating Expense Category Split", height=350)
            st.plotly_chart(fig_exp, use_container_width=True)

        with c_exp2:
            exp_display = exp_cat_df.copy()
            exp_display["Total Incurred"] = exp_display["total_cat_amount"].apply(lambda a: format_currency(a))
            exp_display["Share of Total"] = (exp_display["total_cat_amount"] / operating_expenses * 100).apply(lambda s: f"{s:.1f}%")

            st.dataframe(
                exp_display[["category", "Total Incurred", "Share of Total"]].rename(
                    columns={"category": "Expense Category"}
                ),
                use_container_width=True,
                hide_index=True
            )

    # =========================================================================
    # TAB 3: Profitability by Vehicle Model
    # =========================================================================
    with tab_models_profit:
        st.markdown("### Unit Economics & Margin Yield by Car Model")

        sql_mod_profit = """
            SELECT
                v.brand || ' ' || v.model as vehicle,
                COUNT(s.id) as units_sold,
                SUM(s.selling_price) as revenue,
                SUM(s.gross_profit) as total_profit,
                AVG(s.gross_profit) as profit_per_unit,
                (SUM(s.gross_profit) / SUM(s.selling_price)) * 100 as margin_pct
            FROM sales s
            JOIN vehicles v ON s.vehicle_id = v.id
            GROUP BY v.brand, v.model
            ORDER BY total_profit DESC
        """
        mod_profit_df = fetch_dataframe(sql_mod_profit)

        fig_profit_bar = plot_bar(
            mod_profit_df.head(10),
            x="total_profit",
            y="vehicle",
            orientation="h",
            title="Top 10 Profit-Generating Vehicle Models (Gross ₹)",
            height=380
        )
        fig_profit_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_profit_bar, use_container_width=True)

        mod_profit_display = mod_profit_df.copy()
        mod_profit_display["Total Gross Profit"] = mod_profit_display["total_profit"].apply(lambda p: format_currency(p))
        mod_profit_display["Profit / Unit"] = mod_profit_display["profit_per_unit"].apply(lambda u: format_currency(u))
        mod_profit_display["Margin (%)"] = mod_profit_display["margin_pct"].apply(lambda m: f"{m:.2f}%")

        st.dataframe(
            mod_profit_display[["vehicle", "units_sold", "Total Gross Profit", "Profit / Unit", "Margin (%)"]].rename(
                columns={"vehicle": "Model", "units_sold": "Units Sold"}
            ),
            use_container_width=True,
            hide_index=True
        )
