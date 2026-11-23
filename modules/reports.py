"""
AutoVision AI - Reports & Export Center
Generates downloadable CSV datasets and formatted executive PDF reports
for sales transactions, inventory audits, ML forecasts, and algorithm benchmarks.
"""

import streamlit as st
import pandas as pd
from database.connection import fetch_dataframe
from ml.predict import generate_future_forecast, load_champion_artifacts
from utils.helpers import format_currency, render_hero_banner, render_badge
from utils.pdf_exporter import generate_dealership_pdf
from utils.excel_exporter import generate_dealership_excel


def render():
    render_hero_banner(
        title="Reports & Executive Export Center",
        subtitle="Download verified CSV data exports and generate executive PDF briefing documents",
        badge_text="Data & PDF Exporter"
    )

    tab_pdf, tab_excel, tab_csv = st.tabs([
        "📄 Export Executive Dealership PDF",
        "📗 Export Multi-Sheet Excel (.xlsx)",
        "📊 Download CSV Data Tables"
    ])

    # =========================================================================
    # TAB 1: PDF Generation
    # =========================================================================
    with tab_pdf:
        st.markdown("### Executive Showroom Summary Briefing (PDF)")
        st.markdown("""
        Generate a print-ready PDF briefing summarizing current showroom KPIs, 
        top vehicle models, multi-step ML sales forecasts, and model evaluation metrics.
        """)

        if st.button("Generate & Download Executive PDF Report", type="primary"):
            with st.spinner("Compiling dealership audit and rendering PDF via ReportLab..."):
                try:
                    # 1. KPIs
                    kpi_raw = fetch_dataframe("""
                        SELECT
                            COUNT(s.id) as total_sold,
                            SUM(s.selling_price) as total_revenue,
                            SUM(s.gross_profit) as gross_profit
                        FROM sales s
                    """)
                    avail_stock = fetch_dataframe("SELECT COUNT(*) as c FROM inventory WHERE stock_status = 'Available'")["c"].iloc[0]
                    active_books = fetch_dataframe("SELECT COUNT(*) as c FROM bookings WHERE status = 'Confirmed'")["c"].iloc[0]
                    t_enq = fetch_dataframe("SELECT COUNT(*) as c FROM enquiries")["c"].iloc[0]
                    t_td = fetch_dataframe("SELECT COUNT(*) as c FROM enquiries WHERE test_drive_conducted = 1")["c"].iloc[0]

                    kpis = {
                        "total_sold": f"{kpi_raw['total_sold'].iloc[0]:,}",
                        "total_revenue": format_currency(kpi_raw['total_revenue'].iloc[0]),
                        "gross_profit": format_currency(kpi_raw['gross_profit'].iloc[0]),
                        "available_stock": f"{avail_stock} units",
                        "active_bookings": f"{active_books} bookings",
                        "conversion_rate": f"{(t_td / max(1, t_enq) * 100):.1f}%"
                    }

                    # 2. Top models
                    top_df = fetch_dataframe("""
                        SELECT v.brand, v.model, v.fuel_type, COUNT(s.id) as units_sold, SUM(s.selling_price) as total_rev
                        FROM sales s
                        JOIN vehicles v ON s.vehicle_id = v.id
                        GROUP BY v.brand, v.model, v.fuel_type
                        ORDER BY units_sold DESC
                        LIMIT 5
                    """)
                    top_df["formatted_revenue"] = top_df["total_rev"].apply(lambda r: format_currency(r))

                    # 3. Forecast DF
                    fc_df, _, metadata = generate_future_forecast(6)

                    # 4. Metrics DF
                    metrics_df = pd.DataFrame(metadata.get("metrics", []))

                    # Generate PDF bytes
                    pdf_bytes = generate_dealership_pdf(kpis, top_df, fc_df, metrics_df)

                    st.success("PDF generated successfully!")
                    st.download_button(
                        label="📥 Download Dealership Report PDF",
                        data=pdf_bytes,
                        file_name="AutoVision_Executive_Report.pdf",
                        mime="application/pdf"
                    )

                except Exception as ex:
                    st.error(f"Failed to generate PDF: {str(ex)}")

    # =========================================================================
    # TAB 2: Multi-Sheet Excel (.xlsx) Generation
    # =========================================================================
    with tab_excel:
        st.markdown("### Executive Showroom Workbook (Excel .xlsx)")
        st.markdown("""
        Generate a multi-sheet Microsoft Excel workbook containing:
        - **Executive_Summary:** Dealership high-level KPIs and timestamp
        - **Sales_Register:** Detailed invoice-level sales transaction records
        - **Inventory_Audit:** Complete yard and floor vehicle stock levels
        - **ML_Forecasts:** Multi-step forward-looking sales demand horizon
        - **Model_Evaluation:** Comparative algorithm benchmark scores
        """)

        if st.button("Generate & Download Excel Workbook (.xlsx)", type="primary"):
            with st.spinner("Compiling multi-sheet Excel workbook..."):
                try:
                    # 1. KPIs
                    kpi_raw = fetch_dataframe("""
                        SELECT
                            COUNT(s.id) as total_sold,
                            SUM(s.selling_price) as total_revenue,
                            SUM(s.gross_profit) as gross_profit
                        FROM sales s
                    """)
                    avail_stock = fetch_dataframe("SELECT COUNT(*) as c FROM inventory WHERE stock_status = 'Available'")["c"].iloc[0]
                    active_books = fetch_dataframe("SELECT COUNT(*) as c FROM bookings WHERE status = 'Confirmed'")["c"].iloc[0]

                    kpis_summary_df = pd.DataFrame([
                        {"Metric": "Report Title", "Value": "AutoVision AI Executive Dealership Audit"},
                        {"Metric": "Generated Date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                        {"Metric": "Total Vehicles Sold", "Value": int(kpi_raw["total_sold"].iloc[0])},
                        {"Metric": "Total Sales Revenue (INR)", "Value": float(kpi_raw["total_revenue"].iloc[0])},
                        {"Metric": "Gross Profit Realized (INR)", "Value": float(kpi_raw["gross_profit"].iloc[0])},
                        {"Metric": "Available Showroom Inventory Units", "Value": int(avail_stock)},
                        {"Metric": "Active Customer Bookings", "Value": int(active_books)}
                    ])

                    # 2. Sales
                    sales_reg_df = fetch_dataframe("""
                        SELECT s.invoice_number, s.sale_date, v.brand, v.model, v.variant, v.fuel_type,
                               s.selling_price, s.cost_price, s.discount_amount, s.gross_profit,
                               s.payment_method, c.city as customer_city, e.name as sales_rep
                        FROM sales s
                        JOIN vehicles v ON s.vehicle_id = v.id
                        JOIN customers c ON s.customer_id = c.id
                        JOIN employees e ON s.sales_rep_id = e.id
                        ORDER BY s.sale_date DESC
                    """)

                    # 3. Inventory
                    inv_audit_df = fetch_dataframe("""
                        SELECT i.vin, v.brand, v.model, v.variant, v.fuel_type, v.transmission,
                               i.stock_status, i.arrival_date, i.days_in_stock, i.showroom_location,
                               v.cost_price, v.ex_showroom_price
                        FROM inventory i
                        JOIN vehicles v ON i.vehicle_id = v.id
                        ORDER BY i.days_in_stock DESC
                    """)

                    # 4. Forecast
                    fc_df, _, metadata = generate_future_forecast(6)
                    forecast_export_df = fc_df[["month_label", "predicted_units", "lower_bound", "upper_bound"]].rename(
                        columns={
                            "month_label": "Forecast Period",
                            "predicted_units": "Predicted Sales (Units)",
                            "lower_bound": "Lower Bound (95% CI)",
                            "upper_bound": "Upper Bound (95% CI)"
                        }
                    )

                    # 5. Metrics
                    metrics_export_df = pd.DataFrame(metadata.get("metrics", []))

                    excel_bytes = generate_dealership_excel(
                        kpis_df=kpis_summary_df,
                        sales_df=sales_reg_df,
                        inv_df=inv_audit_df,
                        forecast_df=forecast_export_df,
                        metrics_df=metrics_export_df
                    )

                    st.success("Excel workbook generated successfully!")
                    st.download_button(
                        label="📥 Download Dealership Workbook (.xlsx)",
                        data=excel_bytes,
                        file_name="AutoVision_Executive_Workbook.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as ex:
                    st.error(f"Failed to generate Excel report: {str(ex)}")

    # =========================================================================
    # TAB 3: CSV Downloads
    # =========================================================================
    with tab_csv:
        st.markdown("### Export Structured Tabular Data (CSV)")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 1. Full Sales Transaction Register")
            sales_csv_df = fetch_dataframe("""
                SELECT s.invoice_number, s.sale_date, v.brand, v.model, v.variant, v.fuel_type,
                       s.selling_price, s.cost_price, s.discount_amount, s.gross_profit,
                       s.payment_method, c.city as customer_city, e.name as sales_rep
                FROM sales s
                JOIN vehicles v ON s.vehicle_id = v.id
                JOIN customers c ON s.customer_id = c.id
                JOIN employees e ON s.sales_rep_id = e.id
                ORDER BY s.sale_date DESC
            """)
            st.download_button(
                label="📥 Download Sales Transactions (CSV)",
                data=sales_csv_df.to_csv(index=False).encode("utf-8"),
                file_name="autovision_sales_records.csv",
                mime="text/csv"
            )

            st.markdown("#### 2. Physical Inventory Audit")
            inv_csv_df = fetch_dataframe("""
                SELECT i.vin, v.brand, v.model, v.variant, v.fuel_type, v.transmission,
                       i.stock_status, i.arrival_date, i.days_in_stock, i.showroom_location,
                       v.cost_price, v.ex_showroom_price
                FROM inventory i
                JOIN vehicles v ON i.vehicle_id = v.id
                ORDER BY i.days_in_stock DESC
            """)
            st.download_button(
                label="📥 Download Inventory Audit (CSV)",
                data=inv_csv_df.to_csv(index=False).encode("utf-8"),
                file_name="autovision_inventory_audit.csv",
                mime="text/csv"
            )

        with c2:
            st.markdown("#### 3. Machine Learning Forecast Horizon")
            try:
                fc_df, _, _ = generate_future_forecast(6)
                st.download_button(
                    label="📥 Download Sales Forecast (CSV)",
                    data=fc_df.to_csv(index=False).encode("utf-8"),
                    file_name="autovision_ml_forecast_horizon.csv",
                    mime="text/csv"
                )
            except Exception:
                st.info("Train ML models to download forecast CSV.")

            st.markdown("#### 4. Model Evaluation Benchmark Scores")
            try:
                _, _, metadata = load_champion_artifacts()
                met_csv_df = pd.DataFrame(metadata.get("metrics", []))
                st.download_button(
                    label="📥 Download Algorithm Benchmarks (CSV)",
                    data=met_csv_df.to_csv(index=False).encode("utf-8"),
                    file_name="autovision_algorithm_comparison.csv",
                    mime="text/csv"
                )
            except Exception:
                st.info("Train ML models to download benchmarks CSV.")
