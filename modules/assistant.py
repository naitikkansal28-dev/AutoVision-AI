"""
AutoVision AI - Conversational AI Business Assistant
Answers natural language dealership management inquiries by dynamically querying
the relational database and machine-learning forecasting pipelines.
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime

from database.connection import fetch_dataframe
from ml.predict import generate_future_forecast
from utils.inventory_alerts import compute_smart_inventory_alerts
from utils.helpers import format_currency, render_hero_banner


def query_dealership_brain(prompt: str) -> dict:
    """
    Interprets natural language queries using regex semantic routing
    and executes verified database queries or ML pipelines.
    """
    p = prompt.lower().strip()

    # 1. Best Selling / Top Models
    if any(k in p for k in ["top", "best", "most sold", "popular", "highest selling"]) and any(w in p for w in ["model", "car", "vehicle"]):
        sql = """
            SELECT v.brand, v.model, COUNT(s.id) as units_sold, SUM(s.selling_price) as total_rev
            FROM sales s
            JOIN vehicles v ON s.vehicle_id = v.id
            GROUP BY v.brand, v.model
            ORDER BY units_sold DESC
            LIMIT 5
        """
        df = fetch_dataframe(sql)
        top_car = f"{df.iloc[0]['brand']} {df.iloc[0]['model']}"
        top_units = int(df.iloc[0]["units_sold"])
        top_rev = format_currency(df.iloc[0]["total_rev"])

        answer = (
            f"The **best-selling car model** is the **{top_car}** with **{top_units} units sold**, "
            f"generating **{top_rev}** in gross sales revenue. "
            f"The top 5 best-performing models in your dealership catalog are summarized below."
        )
        return {"text": answer, "df": df.rename(columns={"brand": "Brand", "model": "Model", "units_sold": "Units Sold", "total_rev": "Revenue (₹)"})}

    # 2. Revenue and Profit Queries
    elif any(k in p for k in ["revenue", "turnover", "sales amount", "gross profit", "net profit", "margin"]):
        # Check if year is mentioned (e.g. 2023, 2024, 2025)
        year_match = re.search(r"\b(202[0-9])\b", p)
        if year_match:
            year = year_match.group(1)
            sql = f"""
                SELECT COUNT(id) as units, SUM(selling_price) as rev, SUM(gross_profit) as profit
                FROM sales WHERE SUBSTR(sale_date, 1, 4) = '{year}'
            """
            sql_exp = f"SELECT SUM(amount) as exp FROM expenses WHERE SUBSTR(expense_date, 1, 4) = '{year}'"
            df = fetch_dataframe(sql)
            exp_val = float(fetch_dataframe(sql_exp)["exp"].iloc[0] or 0)
            rev = float(df["rev"].iloc[0] or 0)
            gross = float(df["profit"].iloc[0] or 0)
            net = gross - exp_val
            units = int(df["units"].iloc[0] or 0)

            answer = (
                f"In **{year}**, the showroom delivered **{units} vehicles**, generating **{format_currency(rev)}** in sales revenue. "
                f"Gross profit reached **{format_currency(gross)}** ({(gross/rev*100):.1f}% margin), and after deducting "
                f"**{format_currency(exp_val)}** in operating overheads, realized net profit was **{format_currency(net)}** ({(net/rev*100):.1f}% net margin)."
            )
            return {"text": answer}
        else:
            sql_all = "SELECT SUM(selling_price) as rev, SUM(gross_profit) as gross FROM sales"
            df_all = fetch_dataframe(sql_all)
            rev = float(df_all["rev"].iloc[0])
            gross = float(df_all["gross"].iloc[0])
            exp_total = float(fetch_dataframe("SELECT SUM(amount) as e FROM expenses")["e"].iloc[0])
            net = gross - exp_total

            answer = (
                f"Across the full 36-month operational history, cumulative sales revenue stands at **{format_currency(rev)}**. "
                f"Total Gross Profit is **{format_currency(gross)}** ({gross/rev*100:.1f}% gross margin), and "
                f"Net Profit after operating expenses ({format_currency(exp_total)}) is **{format_currency(net)}** ({net/rev*100:.1f}% net margin)."
            )
            return {"text": answer}

    # 3. Available Inventory / Stock in hand
    elif any(k in p for k in ["inventory", "stock", "available", "units in hand", "yard", "units left"]):
        sql_st = """
            SELECT
                SUM(CASE WHEN stock_status = 'Available' THEN 1 ELSE 0 END) as avail,
                SUM(CASE WHEN stock_status = 'Booked' THEN 1 ELSE 0 END) as booked,
                SUM(CASE WHEN stock_status = 'Available' AND days_in_stock >= 90 THEN 1 ELSE 0 END) as crit_aging,
                SUM(CASE WHEN stock_status = 'Available' AND days_in_stock >= 60 AND days_in_stock < 90 THEN 1 ELSE 0 END) as warn_aging
            FROM inventory
        """
        df = fetch_dataframe(sql_st)
        avail = int(df["avail"].iloc[0])
        booked = int(df["booked"].iloc[0])
        crit = int(df["crit_aging"].iloc[0])
        warn = int(df["warn_aging"].iloc[0])

        answer = (
            f"You currently have **{avail} vehicles available for immediate sale** across showroom bays and the yard. "
            f"An additional **{booked} units are reserved under active customer bookings** awaiting delivery. "
            f"Attention: **{crit} units have exceeded 90 days in stock** and require aging clearance."
        )
        return {"text": answer}

    # 4. Aging Vehicles / Slow Moving
    elif any(k in p for k in ["aging", "ageing", "slow", "old", "dwell", "90 day", "60 day"]):
        sql_aging = """
            SELECT i.vin, v.brand || ' ' || v.model as vehicle, v.variant, i.days_in_stock,
                   v.cost_price, i.showroom_location
            FROM inventory i
            JOIN vehicles v ON i.vehicle_id = v.id
            WHERE i.stock_status = 'Available' AND i.days_in_stock >= 60
            ORDER BY i.days_in_stock DESC
            LIMIT 6
        """
        df = fetch_dataframe(sql_aging)
        if df.empty:
            return {"text": "Great news! There are currently no available vehicles in your inventory exceeding the 60-day aging threshold."}

        crit_count = len(df[df["days_in_stock"] >= 90])
        oldest = df.iloc[0]
        answer = (
            f"There are **{len(df)} available vehicles aging over 60 days** in stock ({crit_count} critically past 90 days). "
            f"The longest-dwelling unit is a **{oldest['vehicle']} {oldest['variant']}** (VIN: `{oldest['vin']}`) "
            f"which has been in the **{oldest['showroom_location']}** for **{oldest['days_in_stock']} days**. "
            f"Action Recommended: Authorize promotional dealer discounts to free up locked working capital."
        )
        return {"text": answer, "df": df.rename(columns={"vin": "VIN", "vehicle": "Vehicle", "variant": "Variant", "days_in_stock": "Days in Bay", "showroom_location": "Location"})}

    # 5. Sales Representatives / Consultants
    elif any(k in p for k in ["consultant", "sales rep", "executive", "employee", "salesperson", "commission"]):
        sql = """
            SELECT e.name, COUNT(s.id) as units_sold, SUM(s.selling_price) as total_rev,
                   SUM(s.selling_price * (e.commission_rate / 100)) as commission_earned
            FROM employees e
            JOIN sales s ON e.id = s.sales_rep_id
            GROUP BY e.id
            ORDER BY units_sold DESC
        """
        df = fetch_dataframe(sql)
        top_rep = df.iloc[0]
        answer = (
            f"Our top-performing sales consultant is **{top_rep['name']}** with **{int(top_rep['units_sold'])} vehicle deliveries**, "
            f"generating **{format_currency(top_rep['total_rev'])}** in revenue and earning **{format_currency(top_rep['commission_earned'])}** in commissions."
        )
        return {"text": answer, "df": df.rename(columns={"name": "Sales Consultant", "units_sold": "Units Delivered", "total_rev": "Revenue (₹)", "commission_earned": "Commission (₹)"})}

    # 6. Future Sales Forecast / ML Predictions
    elif any(k in p for k in ["forecast", "predict", "next month", "future", "expected sales"]):
        try:
            fc_df, _, meta = generate_future_forecast(3)
            next_month = fc_df.iloc[0]
            champ = meta.get("champion_model", "Linear Regression")
            answer = (
                f"Using our champion **{champ}** model, the forecasted vehicle demand for **{next_month['month_label']}** is "
                f"**{next_month['predicted_units']:.1f} units** (95% Confidence Interval: **{next_month['lower_bound']:.0f} to {next_month['upper_bound']:.0f} units**). "
                f"Over the next 3 months, total showroom volume is projected at **{fc_df['predicted_units'].sum():.0f} units**."
            )
            return {"text": answer, "df": fc_df[["month_label", "predicted_units", "lower_bound", "upper_bound"]].rename(columns={"month_label": "Month", "predicted_units": "Projected Units", "lower_bound": "95% Lower Bound", "upper_bound": "95% Upper Bound"})}
        except Exception as e:
            return {"text": f"Error loading forecasting engine: {str(e)}"}

    # 7. Customer Pipeline & Conversion
    elif any(k in p for k in ["conversion", "enquiry", "test drive", "funnel", "customer"]):
        sql = """
            SELECT
                (SELECT COUNT(*) FROM enquiries) as total_enq,
                (SELECT COUNT(*) FROM enquiries WHERE test_drive_conducted = 1) as test_drives,
                (SELECT COUNT(*) FROM bookings) as bookings,
                (SELECT COUNT(*) FROM sales) as total_sales
        """
        row = fetch_dataframe(sql).iloc[0]
        enq = int(row["total_enq"])
        td = int(row["test_drives"])
        bk = int(row["bookings"])

        enq_to_td = (td / max(1, enq)) * 100
        td_to_bk = (bk / max(1, td)) * 100

        answer = (
            f"The dealership customer conversion funnel shows **{enq} total enquiries**, of which **{td} scheduled or completed a test drive** ({enq_to_td:.1f}% enquiry-to-test rate). "
            f"From those test drives, **{bk} customer bookings were confirmed** ({td_to_bk:.1f}% conversion rate)."
        )
        return {"text": answer}

    # Default Fallback
    else:
        return {
            "text": (
                "I can answer dealership intelligence questions on: **Top-selling vehicle models**, "
                "**Monthly and annual sales revenue**, **Available inventory & aging stock (>90 days)**, "
                "**Sales consultant performance & commissions**, **ML sales forecasts for next month**, "
                "and **Enquiry-to-booking conversion rates**. "
                "Please select one of the suggested query chips below or rephrase your question."
            )
        }


def render():
    render_hero_banner(
        title="AI Dealership Business Assistant",
        subtitle="Conversational business intelligence providing instant, verified answers from your relational database and ML models",
        badge_text="Text-to-Insights"
    )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your **AutoVision AI Dealership Assistant**. "
                    "I have live access to your vehicle catalog, sales ledger, showroom bays, and machine-learning forecasting engine. "
                    "Ask me any question about your showroom's performance or click a quick prompt below."
                )
            }
        ]

    # Quick Question Chips
    st.markdown("#### Quick Analytical Questions")
    chip_cols = st.columns(4)
    quick_queries = [
        "What are our top 5 selling car models?",
        "What is our total revenue and gross profit?",
        "Which vehicles are aging past 90 days?",
        "What is our sales forecast for next month?"
    ]

    selected_quick = None
    for idx, (col, q) in enumerate(zip(chip_cols, quick_queries)):
        with col:
            if st.button(q, key=f"chip_{idx}", use_container_width=True):
                selected_quick = q

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Display Chat History
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and msg["df"] is not None:
                st.dataframe(msg["df"], use_container_width=True, hide_index=True)

    # Process Input from user text input or quick chip
    user_input = st.chat_input("Ask a business question (e.g. 'How many cars are available in stock?')")
    prompt_to_process = selected_quick or user_input

    if prompt_to_process:
        # Add user message to state
        st.session_state["chat_messages"].append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Querying showroom database and predictive models..."):
                response_data = query_dealership_brain(prompt_to_process)
                st.markdown(response_data["text"])
                if "df" in response_data and response_data["df"] is not None:
                    st.dataframe(response_data["df"], use_container_width=True, hide_index=True)

        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": response_data["text"],
            "df": response_data.get("df")
        })
