"""
AutoVision AI - Model Performance Monitoring & Experiment Tracking Module
Continuously tracks regression evaluation metrics (MAE, RMSE, R2, MAPE) across training runs
to monitor model drift, stability, and algorithmic improvements.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ml.experiment_tracker import get_experiment_history
from ml.train_models import train_and_evaluate_all
from utils.helpers import render_hero_banner, render_kpi_card
from utils.charts import apply_dark_theme, plot_line


def render():
    render_hero_banner(
        title="Model Performance Monitoring & Experiment Registry",
        subtitle="Tracking algorithm validation metrics, drift detection, and longitudinal forecasting stability",
        badge_text="MLOps & Governance"
    )

    history_df = get_experiment_history()

    if history_df.empty:
        # Trigger an initial logged experiment
        st.info("No prior experiment records detected. Running baseline benchmark initialization...")
        with st.spinner("Logging baseline experiment run..."):
            train_and_evaluate_all(test_size_months=6)
            history_df = get_experiment_history()

    # 1. Summary Metrics of latest experiment
    latest = history_df.iloc[-1]
    total_experiments = len(history_df)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(render_kpi_card("Total Experiments", f"{total_experiments}", "Persistent Registry", "neutral", "blue"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("Active Champion", str(latest.get("champion_model", "N/A")), "Latest Run", "positive", "cyan"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card("Latest RMSE", f"{latest.get('champion_rmse', 0):.2f}", "Root Mean Sq Error", "positive" if latest.get('champion_rmse', 0) < 3.5 else "neutral", "green"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card("Latest MAE", f"{latest.get('champion_mae', 0):.2f}", "Mean Absolute Error", "positive" if latest.get('champion_mae', 0) < 3.0 else "neutral", "purple"), unsafe_allow_html=True)
    with k5:
        st.markdown(render_kpi_card("Latest R² Score", f"{latest.get('champion_r2', 0):.4f}", "Variance Explained", "positive" if latest.get('champion_r2', 0) > 0.6 else "amber", "amber"), unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    tab_history, tab_drift, tab_inspector = st.tabs([
        "📋 Experiment Log Registry",
        "📈 Longitudinal Metric Tracking",
        "🔍 Single Experiment Deep-Dive"
    ])

    # =========================================================================
    # TAB 1: History Log Table
    # =========================================================================
    with tab_history:
        st.markdown("### Persistent Experiment Ledger")
        
        display_hist = history_df[[
            "experiment_id", "timestamp", "champion_model", "test_size_months",
            "champion_rmse", "champion_mae", "champion_r2", "champion_mape", "source"
        ]].rename(
            columns={
                "experiment_id": "Exp ID",
                "timestamp": "Execution Time",
                "champion_model": "Champion Algorithm",
                "test_size_months": "Test Split (Mo)",
                "champion_rmse": "RMSE",
                "champion_mae": "MAE",
                "champion_r2": "R² Score",
                "champion_mape": "MAPE (%)",
                "source": "Data Source"
            }
        ).sort_values("Exp ID", ascending=False)

        st.dataframe(display_hist, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Longitudinal Metric Tracking
    # =========================================================================
    with tab_drift:
        st.markdown("### Performance Progression Across Experiments")
        st.markdown("Monitor whether changes in train-test horizon or new data batches improve or degrade prediction error.")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Scatter(
                x=history_df["experiment_id"],
                y=history_df["champion_rmse"],
                mode="lines+markers",
                name="RMSE",
                line=dict(color="#EF4444", width=3),
                marker=dict(size=8, color="#EF4444")
            ))
            fig_rmse = apply_dark_theme(fig_rmse, "RMSE Progression (Lower is Better)", height=340)
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col_c2:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(
                x=history_df["experiment_id"],
                y=history_df["champion_r2"],
                mode="lines+markers",
                name="R² Score",
                line=dict(color="#00F5D4", width=3),
                marker=dict(size=8, color="#00F5D4")
            ))
            fig_r2 = apply_dark_theme(fig_r2, "R² Score Progression (Higher is Better)", height=340)
            st.plotly_chart(fig_r2, use_container_width=True)

    # =========================================================================
    # TAB 3: Single Experiment Deep Dive
    # =========================================================================
    with tab_inspector:
        st.markdown("### Detailed Scorecard Inspector")
        chosen_id = st.selectbox("Select Experiment ID to Inspect", history_df["experiment_id"].tolist(), index=len(history_df) - 1)
        selected_entry = history_df[history_df["experiment_id"] == chosen_id].iloc[0]

        st.markdown(f"**Experiment:** `{chosen_id}` | **Ran at:** `{selected_entry['timestamp']}` | **Champion:** **{selected_entry['champion_model']}**")

        all_met = selected_entry.get("all_metrics", [])
        if all_met:
            met_df = pd.DataFrame(all_met)
            st.dataframe(met_df, use_container_width=True, hide_index=True)
