"""
AutoVision AI - Bookings & Customer Enquiries Management
Interactive CRM pipeline for customer enquiries, test drives, vehicle bookings,
cancellation management, delivery completion, and conversion funnel analytics.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from database.connection import fetch_dataframe, execute_query
from utils.helpers import format_currency, format_number, render_hero_banner, render_kpi_card, render_badge
from utils.charts import plot_bar, plot_donut


def render():
    render_hero_banner(
        title="Enquiry, Test Drive & Booking Management",
        subtitle="Manage dealership sales pipeline, schedule customer test drives, track bookings, and calculate conversion velocity",
        badge_text="Dealership CRM"
    )

    tab_funnel, tab_enquiries, tab_bookings, tab_new_enq, tab_new_book = st.tabs([
        "🌪️ Conversion Funnel",
        "📋 Live Enquiries Pipeline",
        "📑 Active Bookings",
        "➕ Log New Enquiry",
        "📝 Create Vehicle Booking"
    ])

    # Base Queries
    total_enquiries = fetch_dataframe("SELECT COUNT(*) as c FROM enquiries")["c"].iloc[0]
    test_drives = fetch_dataframe("SELECT COUNT(*) as c FROM enquiries WHERE test_drive_conducted = 1")["c"].iloc[0]
    total_bookings = fetch_dataframe("SELECT COUNT(*) as c FROM bookings")["c"].iloc[0]
    confirmed_bookings = fetch_dataframe("SELECT COUNT(*) as c FROM bookings WHERE status = 'Confirmed'")["c"].iloc[0]
    cancelled_bookings = fetch_dataframe("SELECT COUNT(*) as c FROM bookings WHERE status = 'Cancelled'")["c"].iloc[0]
    total_sales = fetch_dataframe("SELECT COUNT(*) as c FROM sales")["c"].iloc[0]

    # Rates
    enquiry_to_test = (test_drives / max(1, total_enquiries)) * 100
    test_to_booking = (total_bookings / max(1, test_drives)) * 100
    booking_to_sale = ((total_bookings - cancelled_bookings) / max(1, total_bookings)) * 100

    # =========================================================================
    # TAB 1: Conversion Funnel
    # =========================================================================
    with tab_funnel:
        st.markdown("### Dealership Sales Conversion Funnel")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(render_kpi_card("Total Enquiries", f"{total_enquiries}", "Pipeline Entry", "neutral", "blue"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_kpi_card("Test Drives", f"{test_drives}", f"{enquiry_to_test:.1f}% Enq-to-Test", "positive", "cyan"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_kpi_card("Confirmed Bookings", f"{confirmed_bookings}", f"{test_to_booking:.1f}% Test-to-Book", "positive", "green"), unsafe_allow_html=True)
        with c4:
            st.markdown(render_kpi_card("Cancelled Bookings", f"{cancelled_bookings}", f"{(cancelled_bookings/max(1, total_bookings)*100):.1f}% Fallout", "negative", "amber"), unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        funnel_df = pd.DataFrame([
            {"Stage": "1. Customer Enquiries", "Count": total_enquiries},
            {"Stage": "2. Test Drives Taken", "Count": test_drives},
            {"Stage": "3. Vehicle Bookings", "Count": total_bookings},
            {"Stage": "4. Final Vehicle Deliveries", "Count": total_bookings - cancelled_bookings}
        ])

        fig_funnel = plot_bar(
            funnel_df,
            x="Count",
            y="Stage",
            orientation="h",
            title="Dealership Sales Pipeline Progression Funnel",
            text_auto=True,
            height=360
        )
        fig_funnel.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_funnel, use_container_width=True)

    # =========================================================================
    # TAB 2: Live Enquiries Pipeline
    # =========================================================================
    with tab_enquiries:
        st.markdown("### Customer Enquiries & Test Drive Records")
        sql_enq_list = """
            SELECT
                eq.id, eq.enquiry_date,
                c.full_name as customer_name, c.phone, c.city,
                v.brand || ' ' || v.model as vehicle,
                e.name as sales_rep,
                eq.status, eq.test_drive_conducted, eq.notes
            FROM enquiries eq
            JOIN customers c ON eq.customer_id = c.id
            JOIN vehicles v ON eq.vehicle_id = v.id
            JOIN employees e ON eq.sales_rep_id = e.id
            ORDER BY eq.enquiry_date DESC
        """
        enq_list_df = fetch_dataframe(sql_enq_list)

        enq_filter = st.selectbox("Filter by Status", ["All"] + list(enq_list_df["status"].unique()))
        if enq_filter != "All":
            enq_list_df = enq_list_df[enq_list_df["status"] == enq_filter]

        enq_list_df["Test Drive"] = enq_list_df["test_drive_conducted"].apply(lambda t: "✅ Yes" if t == 1 else "❌ No")
        st.dataframe(
            enq_list_df[["id", "enquiry_date", "customer_name", "phone", "vehicle", "sales_rep", "status", "Test Drive", "notes"]].rename(
                columns={"id": "ID", "enquiry_date": "Date", "customer_name": "Customer", "phone": "Contact", "vehicle": "Car Model", "sales_rep": "Assigned Rep", "status": "Pipeline Stage"}
            ),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 3: Active Bookings
    # =========================================================================
    with tab_bookings:
        st.markdown("### Active Vehicle Bookings & Delivery Schedules")
        sql_b_list = """
            SELECT
                b.id, b.booking_reference, b.booking_date, b.booking_amount, b.status, b.delivery_promised_date,
                c.full_name as customer_name, c.phone,
                v.brand || ' ' || v.model || ' ' || v.variant as vehicle,
                e.name as sales_rep,
                b.notes
            FROM bookings b
            JOIN customers c ON b.customer_id = c.id
            JOIN vehicles v ON b.vehicle_id = v.id
            JOIN employees e ON b.sales_rep_id = e.id
            ORDER BY b.booking_date DESC
        """
        b_df = fetch_dataframe(sql_b_list)
        b_df["Deposit Paid"] = b_df["booking_amount"].apply(lambda a: format_currency(a))

        st.dataframe(
            b_df[["booking_reference", "booking_date", "customer_name", "phone", "vehicle", "sales_rep", "Deposit Paid", "status", "delivery_promised_date"]].rename(
                columns={
                    "booking_reference": "Ref #",
                    "booking_date": "Date",
                    "customer_name": "Customer",
                    "phone": "Phone",
                    "vehicle": "Vehicle",
                    "sales_rep": "Consultant",
                    "status": "Booking Status",
                    "delivery_promised_date": "Target Delivery"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        # Quick action: Update booking status
        st.markdown("#### Update Booking Status")
        b_refs = b_df["booking_reference"].tolist() if not b_df.empty else ["No Bookings Available"]
        with st.form("form_update_booking"):
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                b_ref_chosen = st.selectbox("Select Booking Reference", b_refs)
            with u_col2:
                new_status = st.selectbox("Update Status To", ["Confirmed", "Completed", "Cancelled"])

            if st.form_submit_button("Update Status"):
                if not b_df.empty and b_ref_chosen != "No Bookings Available":
                    execute_query(
                        "UPDATE bookings SET status = %s WHERE booking_reference = %s",
                        (new_status, b_ref_chosen)
                    )
                    st.success(f"Booking {b_ref_chosen} updated to {new_status}!")
                    st.rerun()

    # =========================================================================
    # TAB 4: Log New Enquiry
    # =========================================================================
    with tab_new_enq:
        st.markdown("### Register New Walk-in / Digital Enquiry")
        
        cust_list = fetch_dataframe("SELECT id, full_name || ' (' || city || ')' as label FROM customers")
        c_dict = dict(zip(cust_list["label"], cust_list["id"]))

        veh_list = fetch_dataframe("SELECT id, brand || ' ' || model || ' - ' || variant as label FROM vehicles")
        v_dict = dict(zip(veh_list["label"], veh_list["id"]))

        rep_list = fetch_dataframe("SELECT id, name FROM employees WHERE is_active = 1")
        r_dict = dict(zip(rep_list["name"], rep_list["id"]))

        with st.form("form_add_enquiry"):
            col1, col2 = st.columns(2)
            with col1:
                sel_cust = st.selectbox("Select Customer", list(c_dict.keys()))
                sel_veh = st.selectbox("Interested Vehicle Model", list(v_dict.keys()))
                sel_rep = st.selectbox("Assign Sales Consultant", list(r_dict.keys()))
            with col2:
                enq_date = st.date_input("Enquiry Date", value=datetime.now())
                stage = st.selectbox("Initial Pipeline Stage", ["New", "Contacted", "Test Drive Scheduled", "Test Drive Completed", "Negotiating"])
                td_check = st.checkbox("Customer has scheduled / taken test drive?", value=False)
                notes = st.text_area("Customer Requirement & Notes", placeholder="Customer interested in year-end delivery...")

            if st.form_submit_button("Save Enquiry", type="primary"):
                try:
                    execute_query(
                        """
                        INSERT INTO enquiries (customer_id, vehicle_id, sales_rep_id, enquiry_date, status, test_drive_conducted, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (c_dict[sel_cust], v_dict[sel_veh], r_dict[sel_rep], enq_date.strftime("%Y-%m-%d"), stage, 1 if td_check else 0, notes)
                    )
                    st.success("Enquiry logged successfully into CRM pipeline!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error logging enquiry: {str(ex)}")

    # =========================================================================
    # TAB 5: Create Vehicle Booking
    # =========================================================================
    with tab_new_book:
        st.markdown("### Create Confirmed Customer Booking")

        with st.form("form_create_booking"):
            col1, col2 = st.columns(2)
            with col1:
                book_ref = st.text_input("Booking Reference Number", value=f"BK-{datetime.now().strftime('%y%m%d%H%M')}")
                b_cust = st.selectbox("Booking Customer", list(c_dict.keys()), key="b_cust_key")
                b_veh = st.selectbox("Selected Vehicle", list(v_dict.keys()), key="b_veh_key")
                b_rep = st.selectbox("Handling Consultant", list(r_dict.keys()), key="b_rep_key")
            with col2:
                b_date = st.date_input("Booking Date", value=datetime.now())
                promised_date = st.date_input("Promised Delivery Date", value=datetime.now() + timedelta(days=21))
                deposit = st.number_input("Advance Booking Deposit (₹)", min_value=10000.0, value=50000.0, step=5000.0)
                b_notes = st.text_input("Special Delivery Instructions / Finance Details", value="Token amount received.")

            if st.form_submit_button("Confirm Booking", type="primary"):
                try:
                    execute_query(
                        """
                        INSERT INTO bookings (booking_reference, customer_id, vehicle_id, sales_rep_id, booking_date, booking_amount, status, delivery_promised_date, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, 'Confirmed', %s, %s)
                        """,
                        (book_ref, c_dict[b_cust], v_dict[b_veh], r_dict[b_rep], b_date.strftime("%Y-%m-%d"), deposit, promised_date.strftime("%Y-%m-%d"), b_notes)
                    )
                    st.success(f"Booking {book_ref} confirmed successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error creating booking: {str(ex)}")
