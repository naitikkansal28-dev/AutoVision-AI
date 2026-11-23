"""
AutoVision AI - Customer Analytics & Machine Learning Segmentation Module
Analyzes demographic geographic distribution, budget brackets, brand affinity,
repeat customer behavior, and unsupervised K-Means customer segmentation clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

from database.connection import fetch_dataframe
from utils.helpers import format_currency, format_number, render_hero_banner, render_kpi_card
from utils.charts import plot_bar, plot_donut, plot_scatter


def render():
    render_hero_banner(
        title="Customer Intelligence & Segmentation Analytics",
        subtitle="Demographic analysis, geographic dispersion, budget segmentation, and K-Means behavioural clustering",
        badge_text="Customer 360"
    )

    # 1. Fetch Customers and Purchase History
    sql_cust = """
        SELECT
            c.id, c.full_name, c.city, c.state, c.annual_income, c.budget_max,
            c.preferred_brand, c.preferred_fuel,
            COUNT(s.id) as purchases_count,
            COALESCE(SUM(s.selling_price), 0) as total_spent
        FROM customers c
        LEFT JOIN sales s ON c.id = s.customer_id
        GROUP BY c.id
    """
    cust_df = fetch_dataframe(sql_cust)

    total_customers = len(cust_df)
    active_buyers = len(cust_df[cust_df["purchases_count"] > 0])
    repeat_buyers = len(cust_df[cust_df["purchases_count"] > 1])
    repeat_rate = (repeat_buyers / max(1, active_buyers)) * 100
    avg_income = cust_df["annual_income"].mean()
    avg_budget = cust_df["budget_max"].mean()

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(render_kpi_card("Total Customers", f"{total_customers:,}", "In CRM Database", "neutral", "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Active Car Owners", f"{active_buyers:,}", "Completed Deliveries", "positive", "cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Repeat Buyers", f"{repeat_buyers}", f"{repeat_rate:.1f}% Repeat Rate", "positive", "green"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Mean Annual Income", format_currency(avg_income), "Self-Reported", "neutral", "purple"), unsafe_allow_html=True)
    with c5:
        st.markdown(render_kpi_card("Mean Purchase Budget", format_currency(avg_budget), "Target Ceiling", "positive", "amber"), unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    tab_geo, tab_pref, tab_kmeans, tab_profiles = st.tabs([
        "🏙️ Geographic & Demographic Spread",
        "🎯 Brand & Powertrain Preferences",
        "🤖 K-Means Clustering Segmentation",
        "👥 Customer Directory & History"
    ])

    # =========================================================================
    # TAB 1: Geographic Spread
    # =========================================================================
    with tab_geo:
        st.markdown("### Regional Customer Concentration")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            city_df = cust_df.groupby("city").size().reset_index(name="Customer Count").sort_values("Customer Count", ascending=False)
            fig_city = plot_bar(city_df.head(10), x="Customer Count", y="city", orientation="h", title="Top 10 Metros by Customer Volume", height=380)
            fig_city.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_city, use_container_width=True)

        with col_g2:
            state_df = cust_df.groupby("state").size().reset_index(name="Count")
            fig_state = plot_donut(state_df, names="state", values="Count", title="State-wise Dealership Footprint", height=380)
            st.plotly_chart(fig_state, use_container_width=True)

    # =========================================================================
    # TAB 2: Brand & Powertrain Preferences
    # =========================================================================
    with tab_pref:
        st.markdown("### Customer Intent & Stated Preferences")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            pref_b = cust_df.groupby("preferred_brand").size().reset_index(name="Interested Customers").sort_values("Interested Customers", ascending=False)
            fig_pb = plot_bar(pref_b, x="preferred_brand", y="Interested Customers", title="Stated Brand Preference in CRM", height=360)
            st.plotly_chart(fig_pb, use_container_width=True)

        with col_p2:
            pref_f = cust_df.groupby("preferred_fuel").size().reset_index(name="Interested Customers")
            fig_pf = plot_donut(pref_f, names="preferred_fuel", values="Interested Customers", title="Stated Powertrain Preference", height=360)
            st.plotly_chart(fig_pf, use_container_width=True)

    # =========================================================================
    # TAB 3: Advanced K-Means Clustering & Optimization
    # =========================================================================
    with tab_kmeans:
        st.markdown("### Advanced K-Means Customer Clustering & Mathematical Validation")
        st.markdown("""
        To avoid arbitrary choice of cluster counts, we evaluate cluster quality using the **Elbow Method (Inertia)** 
        and the **Silhouette Coefficient** across candidate $k \\in [2, 6]$.
        """)

        # Preprocessing for K-Means
        cluster_data = cust_df[["annual_income", "budget_max"]].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)

        # Compute Elbow Curve & Silhouette Scores
        k_eval_results = []
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            k_eval_results.append({
                "k": k,
                "Inertia (Within-Cluster Sum of Squares)": round(km.inertia_, 2),
                "Silhouette Score": round(sil, 4)
            })

        k_eval_df = pd.DataFrame(k_eval_results)

        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            # Elbow Curve Plot
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=k_eval_df["k"],
                y=k_eval_df["Inertia (Within-Cluster Sum of Squares)"],
                mode="lines+markers",
                name="Inertia",
                line=dict(color="#00F5D4", width=3),
                marker=dict(size=8, color="#00F5D4")
            ))
            fig_elbow.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131C38",
                plot_bgcolor="#131C38",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(color="#F8FAFC", family="Plus Jakarta Sans"),
                title=dict(text="<b>Elbow Method (Optimal k by Inflection)</b>", font=dict(size=13)),
                xaxis=dict(title="Number of Clusters (k)", gridcolor="#21325E"),
                yaxis=dict(title="Inertia", gridcolor="#21325E")
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

        with opt_col2:
            st.markdown("#### Mathematical Cluster Scorecard")
            st.dataframe(k_eval_df, use_container_width=True, hide_index=True)
            best_k = k_eval_df.sort_values("Silhouette Score", ascending=False).iloc[0]["k"]
            st.caption(f"💡 Highest Silhouette Separation achieved at **k = {best_k}**.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Interactive Slider
        k_clusters = st.slider("Select Operational Clusters for Dealership Campaigns", min_value=2, max_value=6, value=int(best_k), step=1)

        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        cust_df["Cluster_ID"] = kmeans.fit_predict(X_scaled)

        # Dynamic Segment Naming based on average budget
        cluster_means = cust_df.groupby("Cluster_ID")["budget_max"].mean().sort_values()
        rank_to_name = {
            cluster_means.index[0]: "Tier 1: Budget Entry Segment",
            cluster_means.index[1]: "Tier 2: Premium Mid-Range Family" if len(cluster_means) > 1 else "Standard",
        }
        if len(cluster_means) >= 3:
            rank_to_name[cluster_means.index[2]] = "Tier 3: Executive Luxury Segment"
        if len(cluster_means) >= 4:
            rank_to_name[cluster_means.index[3]] = "Tier 4: Ultra HNI & Exotic Fleet"
        if len(cluster_means) >= 5:
            rank_to_name[cluster_means.index[4]] = "Tier 5: Corporate Lease Segment"
        if len(cluster_means) >= 6:
            rank_to_name[cluster_means.index[5]] = "Tier 6: Bespoke Collector VIPs"

        cust_df["Customer Segment"] = cust_df["Cluster_ID"].map(rank_to_name)

        # 2D Cluster Scatter
        fig_cluster = plot_scatter(
            cust_df,
            x="annual_income",
            y="budget_max",
            color="Customer Segment",
            hover_name="city",
            title=f"K-Means Customer Clusters (k = {k_clusters})",
            height=430
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Persona Profile Breakdown
        st.markdown("#### Strategic Segment Profiles & Dealership Targeting")
        seg_summary = cust_df.groupby("Customer Segment").agg(
            customer_count=("id", "count"),
            avg_income=("annual_income", "mean"),
            avg_budget=("budget_max", "mean"),
            total_actual_spend=("total_spent", "sum")
        ).reset_index()

        seg_summary["Avg Annual Income"] = seg_summary["avg_income"].apply(lambda i: format_currency(i))
        seg_summary["Avg Vehicle Budget"] = seg_summary["avg_budget"].apply(lambda b: format_currency(b))
        seg_summary["Realized Dealership Revenue"] = seg_summary["total_actual_spend"].apply(lambda s: format_currency(s))

        st.dataframe(
            seg_summary[["Customer Segment", "customer_count", "Avg Annual Income", "Avg Vehicle Budget", "Realized Dealership Revenue"]].rename(
                columns={"customer_count": "Customer Count"}
            ),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 4: Customer Directory & History
    # =========================================================================
    with tab_profiles:
        st.markdown("### Customer Directory (Privacy Masked)")
        st.caption("Personally Identifiable Information (PII) is masked to comply with enterprise data protection guidelines.")

        display_crm = cust_df.copy()
        # Privacy masking on name (first name + initial)
        display_crm["Masked Customer"] = display_crm["full_name"].apply(
            lambda n: n.split()[0] + " " + n.split()[-1][0] + "." if len(n.split()) > 1 else n
        )
        display_crm["Budget Max"] = display_crm["budget_max"].apply(lambda b: format_currency(b))
        display_crm["Total Spend"] = display_crm["total_spent"].apply(lambda s: format_currency(s))

        st.dataframe(
            display_crm[["Masked Customer", "city", "Budget Max", "preferred_brand", "preferred_fuel", "purchases_count", "Total Spend"]].rename(
                columns={"Masked Customer": "Client", "city": "City", "preferred_brand": "Preferred Brand", "preferred_fuel": "Fuel", "purchases_count": "Cars Purchased"}
            ),
            use_container_width=True,
            hide_index=True
        )
