"""
AutoVision AI - Sales Analytics Module
Detailed automotive transaction analytics, model breakdowns, fuel/transmission shares,
sales representative performance, discount elasticity, and interactive multi-dimensional filtering.
"""

import streamlit as st
import pandas as pd
from database.connection import fetch_dataframe
from utils.helpers import format_currency, format_number, render_hero_banner, render_kpi_card
from utils.charts import plot_bar, plot_line, plot_donut, plot_grouped_bar, plot_scatter


def render():
    render_hero_banner(
        title="Sales Analytics & Performance Intelligence",
        subtitle="Multi-dimensional breakdown of sales volume, revenue drivers, dealer margins, and team performance",
        badge_text="Sales Deep Dive"
    )

    # 1. Load Sales Base with Joins
    sql_base = """
        SELECT
            s.id, s.invoice_number, s.sale_date, s.selling_price, s.cost_price,
            s.discount_amount, s.accessories_amount, s.insurance_amount, s.gross_profit,
            s.payment_method, s.finance_partner,
            v.brand, v.model, v.variant, v.body_type, v.fuel_type, v.transmission,
            c.city as customer_city,
            e.name as sales_rep, e.monthly_sales_target, e.commission_rate
        FROM sales s
        JOIN vehicles v ON s.vehicle_id = v.id
        JOIN customers c ON s.customer_id = c.id
        JOIN employees e ON s.sales_rep_id = e.id
    """
    df = fetch_dataframe(sql_base)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["year"] = df["sale_date"].dt.year
    df["month_year"] = df["sale_date"].dt.strftime("%Y-%m")

    # 2. Interactive Filter Sidebar / Header Expander
    with st.expander("Interactive Filter Controls", expanded=True):
        fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
        
        with fcol1:
            years = sorted(df["year"].unique())
            selected_years = st.multiselect("Select Year(s)", years, default=years)
            
        with fcol2:
            brands = sorted(df["brand"].unique())
            selected_brands = st.multiselect("Select Brand(s)", brands, default=brands)
            
        with fcol3:
            fuels = sorted(df["fuel_type"].unique())
            selected_fuels = st.multiselect("Select Fuel Type", fuels, default=fuels)
            
        with fcol4:
            transmissions = sorted(df["transmission"].unique())
            selected_trans = st.multiselect("Transmission", transmissions, default=transmissions)
            
        with fcol5:
            payment_opts = sorted(df["payment_method"].unique())
            selected_payments = st.multiselect("Payment Mode", payment_opts, default=payment_opts)

    # Apply Filters
    filtered_df = df[
        (df["year"].isin(selected_years)) &
        (df["brand"].isin(selected_brands)) &
        (df["fuel_type"].isin(selected_fuels)) &
        (df["transmission"].isin(selected_trans)) &
        (df["payment_method"].isin(selected_payments))
    ].copy()

    if filtered_df.empty:
        st.warning("No sales records match the selected filter criteria. Please broaden your selection.")
        return

    # Filtered KPIs
    f_units = len(filtered_df)
    f_rev = filtered_df["selling_price"].sum()
    f_profit = filtered_df["gross_profit"].sum()
    f_discount = filtered_df["discount_amount"].sum()
    f_avg_price = f_rev / max(1, f_units)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(render_kpi_card("Units Sold", f"{f_units:,}", "Filtered Scope", "positive", "cyan"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("Sales Revenue", format_currency(f_rev), f"Avg {format_currency(f_avg_price)}", "positive", "blue"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card("Gross Profit", format_currency(f_profit), f"Margin {(f_profit/f_rev*100):.1f}%", "positive", "green"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card("Total Discounts", format_currency(f_discount), f"{(f_discount/f_rev*100):.1f}% of Rev", "negative", "amber"), unsafe_allow_html=True)
    with k5:
        st.markdown(render_kpi_card("Add-on Revenue", format_currency(filtered_df["accessories_amount"].sum() + filtered_df["insurance_amount"].sum()), "Accessories + Insurance", "positive", "purple"), unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 3. Monthly & YoY Sales Trends
    st.markdown("### 1. Monthly Sales Trajectory & Yearly Comparisons")
    trend_col1, trend_col2 = st.columns([1.5, 1.5])

    with trend_col1:
        monthly_trend = filtered_df.groupby("month_year").agg(
            units=("id", "count"),
            revenue=("selling_price", "sum"),
            gross_profit=("gross_profit", "sum")
        ).reset_index()

        fig_monthly = plot_line(
            monthly_trend,
            x="month_year",
            y="units",
            title="Sales Volume Trend (Units per Month)",
            height=360
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with trend_col2:
        yoy_df = filtered_df.groupby(["year", filtered_df["sale_date"].dt.month_name()]).size().reset_index(name="Units")
        yoy_df.rename(columns={"sale_date": "Month"}, inplace=True)
        # Order months properly
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        yoy_df["Month"] = pd.Categorical(yoy_df["Month"], categories=month_order, ordered=True)
        yoy_df = yoy_df.sort_values("Month")

        fig_yoy = plot_grouped_bar(
            yoy_df,
            x="Month",
            y="Units",
            group_col="year",
            title="Year-over-Year Seasonal Comparison (Units)",
            height=360
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    # 4. Brand & Vehicle Model Performance
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### 2. Vehicle Model Performance Matrix")
    model_col1, model_col2 = st.columns(2)

    with model_col1:
        model_perf = filtered_df.groupby(["brand", "model"]).agg(
            units=("id", "count"),
            revenue=("selling_price", "sum"),
            avg_margin=("gross_profit", lambda x: (x.sum() / filtered_df.loc[x.index, "selling_price"].sum()) * 100)
        ).reset_index().sort_values("units", ascending=False)
        model_perf["car_name"] = model_perf["brand"] + " " + model_perf["model"]

        fig_top_bar = plot_bar(
            model_perf.head(8),
            x="units",
            y="car_name",
            orientation="h",
            title="Top 8 Best-Selling Car Models (Volume)",
            height=380
        )
        fig_top_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top_bar, use_container_width=True)

    with model_col2:
        least_models = model_perf.tail(8).sort_values("units", ascending=True)
        fig_least_bar = plot_bar(
            least_models,
            x="units",
            y="car_name",
            orientation="h",
            title="Lowest Moving Models (Potential Overstock Risk)",
            height=380
        )
        fig_least_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_least_bar, use_container_width=True)

    # 5. Powertrain & Transmission Analysis
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### 3. Powertrain & Transmission Dynamics")
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        fuel_grouped = filtered_df.groupby("fuel_type").agg(
            units=("id", "count"),
            revenue=("selling_price", "sum")
        ).reset_index()
        fig_p1 = plot_donut(fuel_grouped, names="fuel_type", values="units", title="Powertrain Volume Share", height=320)
        st.plotly_chart(fig_p1, use_container_width=True)

    with p_col2:
        trans_grouped = filtered_df.groupby("transmission").agg(
            units=("id", "count"),
            revenue=("selling_price", "sum")
        ).reset_index()
        fig_p2 = plot_donut(trans_grouped, names="transmission", values="units", title="Transmission Preference (Manual vs Automatic)", height=320)
        st.plotly_chart(fig_p2, use_container_width=True)

    # 6. Sales Executive Performance & Targets
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### 4. Sales Executive Leaderboard & Commission Tracker")
    rep_summary = filtered_df.groupby("sales_rep").agg(
        units_sold=("id", "count"),
        total_revenue=("selling_price", "sum"),
        total_gross_profit=("gross_profit", "sum"),
        commission_rate=("commission_rate", "mean")
    ).reset_index()

    rep_summary["commission_earned"] = rep_summary["total_revenue"] * (rep_summary["commission_rate"] / 100)
    rep_summary = rep_summary.sort_values("units_sold", ascending=False)

    col_rep_chart, col_rep_tbl = st.columns([1.4, 1.6])
    with col_rep_chart:
        fig_rep = plot_bar(
            rep_summary,
            x="sales_rep",
            y="units_sold",
            title="Consultant Sales Volume (Delivered Units)",
            height=340
        )
        st.plotly_chart(fig_rep, use_container_width=True)

    with col_rep_tbl:
        display_rep = rep_summary.copy()
        display_rep["Total Revenue"] = display_rep["total_revenue"].apply(lambda r: format_currency(r))
        display_rep["Gross Profit"] = display_rep["total_gross_profit"].apply(lambda g: format_currency(g))
        display_rep["Est. Commission"] = display_rep["commission_earned"].apply(lambda c: format_currency(c))

        st.dataframe(
            display_rep[["sales_rep", "units_sold", "Total Revenue", "Gross Profit", "Est. Commission"]].rename(
                columns={"sales_rep": "Sales Consultant", "units_sold": "Units"}
            ),
            use_container_width=True,
            hide_index=True
        )

    # 7. Discount Elasticity Analysis
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("### 5. Pricing & Discount Elasticity")
    disc_summary = filtered_df.groupby("model").agg(
        avg_discount=("discount_amount", "mean"),
        units_sold=("id", "count"),
        avg_selling_price=("selling_price", "mean"),
        brand=("brand", "first")
    ).reset_index()

    fig_disc = plot_scatter(
        disc_summary,
        x="avg_discount",
        y="units_sold",
        color="brand",
        hover_name="model",
        size="avg_selling_price",
        title="Average Discount (₹) vs Sales Volume by Vehicle Model",
        height=380
    )
    st.plotly_chart(fig_disc, use_container_width=True)
