"""
AutoVision AI - Dataset Upload & Validation Center
Allows users to upload custom CSV and Excel (.xlsx) files for sales transactions
and showroom inventory with automated schema validation and integrity verification.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_validator import read_uploaded_file, validate_sales_dataset, validate_inventory_dataset
from database.connection import execute_query, fetch_dataframe
from ml.train_models import train_and_evaluate_all
import config
from utils.helpers import render_hero_banner, render_kpi_card, render_badge


def render():
    render_hero_banner(
        title="Dataset Ingestion & Validation Gateway",
        subtitle="Upload custom dealership sales records and showroom inventory files in CSV or Excel format with automated integrity auditing",
        badge_text="Data Validation Hub"
    )

    tab_sales, tab_inventory, tab_templates = st.tabs([
        "📊 Upload Sales Transactions",
        "📦 Upload Showroom Inventory",
        "📥 Download Standard Templates"
    ])

    # =========================================================================
    # TAB 1: Upload Sales Transactions
    # =========================================================================
    with tab_sales:
        st.markdown("### Upload Dealership Sales Dataset (CSV / Excel)")
        st.markdown("Upload historical sales transactions to run ad-hoc ML forecasting or permanently append into the showroom database.")

        uploaded_sales = st.file_uploader("Select Sales File", type=["csv", "xlsx", "xls"], key="upload_sales_file")

        if uploaded_sales is not None:
            try:
                raw_df = read_uploaded_file(uploaded_sales)
                st.markdown(f"**Loaded {len(raw_df)} raw records from `{uploaded_sales.name}`**")

                # Run Validation
                val_res = validate_sales_dataset(raw_df)

                # Validation Scorecard
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                with col_v1:
                    st.markdown(render_kpi_card("Total Rows", f"{val_res['total_rows']}", "Raw Upload", "neutral", "blue"), unsafe_allow_html=True)
                with col_v2:
                    st.markdown(render_kpi_card("Valid Records", f"{val_res['valid_rows']}", "Passed Schema", "positive", "green"), unsafe_allow_html=True)
                with col_v3:
                    st.markdown(render_kpi_card("Flagged Rows", f"{val_res['invalid_rows']}", "Corrupted / Missing", "negative" if val_res['invalid_rows'] > 0 else "positive", "amber"), unsafe_allow_html=True)
                with col_v4:
                    status_text = "PASSED" if val_res["is_valid"] else "FAILED"
                    st.markdown(render_kpi_card("Validation Status", status_text, "Schema Compliance", "positive" if val_res["is_valid"] else "negative", "cyan"), unsafe_allow_html=True)

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

                if val_res["errors"]:
                    for err in val_res["errors"]:
                        st.error(f"❌ **Schema Error:** {err}")

                if val_res["warnings"]:
                    for warn in val_res["warnings"]:
                        st.warning(f"⚠️ **Integrity Notice:** {warn}")

                if val_res["is_valid"] and val_res["cleaned_df"] is not None:
                    clean_df = val_res["cleaned_df"]
                    st.markdown("#### Cleaned Dataset Preview")
                    st.dataframe(clean_df.head(6), use_container_width=True)

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("🚀 Train ML Models on this Uploaded Sales Data", type="primary", use_container_width=True):
                            temp_csv = config.DATA_DIR / "uploaded_sales_training.csv"
                            clean_df.to_csv(temp_csv, index=False)
                            with st.spinner("Training ML candidate algorithms on uploaded dataset..."):
                                results = train_and_evaluate_all(source="csv", csv_path=temp_csv)
                                st.success(f"Models successfully trained! Best Algorithm: **{results['champion_name']}**")
                                st.dataframe(results["results_df"], use_container_width=True, hide_index=True)

                    with btn_col2:
                        st.info("Uploaded data is verified and ready for forecasting and analytical processing.")

            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")

    # =========================================================================
    # TAB 2: Upload Showroom Inventory
    # =========================================================================
    with tab_inventory:
        st.markdown("### Upload Showroom Physical Inventory (CSV / Excel)")
        st.markdown("Register a bulk shipment batch into showroom inventory tracking.")

        uploaded_inv = st.file_uploader("Select Inventory File", type=["csv", "xlsx", "xls"], key="upload_inv_file")

        if uploaded_inv is not None:
            try:
                raw_inv_df = read_uploaded_file(uploaded_inv)
                st.markdown(f"**Loaded {len(raw_inv_df)} raw records from `{uploaded_inv.name}`**")

                val_inv = validate_inventory_dataset(raw_inv_df)

                col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                with col_i1:
                    st.markdown(render_kpi_card("Total Units", f"{val_inv['total_rows']}", "File Rows", "neutral", "blue"), unsafe_allow_html=True)
                with col_i2:
                    st.markdown(render_kpi_card("Valid VINs", f"{val_inv['valid_rows']}", "Ready to Ingest", "positive", "green"), unsafe_allow_html=True)
                with col_i3:
                    st.markdown(render_kpi_card("Discarded", f"{val_inv['invalid_rows']}", "Duplicates / Incomplete", "negative" if val_inv['invalid_rows'] > 0 else "positive", "amber"), unsafe_allow_html=True)
                with col_i4:
                    st.markdown(render_kpi_card("Schema Integrity", "VALID" if val_inv['is_valid'] else "INVALID", "Integrity Check", "positive" if val_inv['is_valid'] else "negative", "cyan"), unsafe_allow_html=True)

                if val_inv["warnings"]:
                    for w in val_inv["warnings"]:
                        st.warning(f"⚠️ {w}")

                if val_inv["is_valid"] and val_inv["cleaned_df"] is not None:
                    st.markdown("#### Cleaned Inventory Preview")
                    st.dataframe(val_inv["cleaned_df"].head(6), use_container_width=True)

            except Exception as e:
                st.error(f"Error validating inventory file: {str(e)}")

    # =========================================================================
    # TAB 3: Download Templates
    # =========================================================================
    with tab_templates:
        st.markdown("### Standard Dealership Template Files")
        st.markdown("Download standardized pre-formatted template files to ensure 100% schema compliance before uploading.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### Sales Transactions Template")
            sample_sales_template = pd.DataFrame([
                {
                    "sale_date": "2025-01-15",
                    "selling_price": 1450000,
                    "cost_price": 1200000,
                    "brand": "Tata",
                    "model": "Nexon",
                    "variant": "Creative Plus",
                    "fuel_type": "Petrol",
                    "discount_amount": 25000,
                    "gross_profit": 275000
                },
                {
                    "sale_date": "2025-01-18",
                    "selling_price": 2200000,
                    "cost_price": 1800000,
                    "brand": "Mahindra",
                    "model": "XUV700",
                    "variant": "AX7 Diesel",
                    "fuel_type": "Diesel",
                    "discount_amount": 35000,
                    "gross_profit": 435000
                }
            ])
            st.download_button(
                label="📥 Download Sales Template (CSV)",
                data=sample_sales_template.to_csv(index=False).encode("utf-8"),
                file_name="autovision_sales_template.csv",
                mime="text/csv"
            )

        with col_t2:
            st.markdown("#### Showroom Inventory Template")
            sample_inv_template = pd.DataFrame([
                {
                    "vin": "AVTATA50101",
                    "brand": "Tata",
                    "model": "Nexon.ev",
                    "variant": "Empowered Plus LR",
                    "stock_status": "Available",
                    "arrival_date": "2025-02-01",
                    "showroom_location": "Main Showroom Floor",
                    "cost_price": 1550000,
                    "ex_showroom_price": 1920000
                },
                {
                    "vin": "AVMAH50102",
                    "brand": "Mahindra",
                    "model": "Thar",
                    "variant": "LX 4x4 Hard Top",
                    "stock_status": "Available",
                    "arrival_date": "2025-01-10",
                    "showroom_location": "Bay A (Ground)",
                    "cost_price": 1350000,
                    "ex_showroom_price": 1720000
                }
            ])
            st.download_button(
                label="📥 Download Inventory Template (CSV)",
                data=sample_inv_template.to_csv(index=False).encode("utf-8"),
                file_name="autovision_inventory_template.csv",
                mime="text/csv"
            )
