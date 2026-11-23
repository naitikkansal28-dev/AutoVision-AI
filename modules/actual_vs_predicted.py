"""
AutoVision AI - Actual vs Predicted Validation Dashboard
Deep-dive comparative evaluation contrasting ground-truth vehicle sales against
machine-learning model predictions, computing residual distributions, and directional bias.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import config
from ml.predict import load_champion_artifacts
from ml.train_models import train_and_evaluate_all
from utils.helpers import render_hero_banner, render_kpi_card
from utils.charts import apply_dark_theme, plot_bar


def render():
    render_hero_banner(
        title="Actual vs Predicted Sales Evaluation",
        subtitle="Empirical out-of-time model validation, residual analysis, error distributions, and directional bias auditing",
        badge_text="Model Validation Lab"
    )

    # 1. Fetch trained artifacts or train if needed
    try:
        model, feature_cols, metadata = load_champion_artifacts()
    except Exception:
        results = train_and_evaluate_all()
        metadata = results["metadata"]

    all_models_dict = metadata.get("metrics", [])
    
    # Run test evaluation to get ground truth actuals vs candidate predictions
    test_results = train_and_evaluate_all(test_size_months=metadata.get("test_size_months", 6))
    pred_df = test_results["test_predictions"].copy()
    candidate_names = [c for c in pred_df.columns if c not in ["Period", "Actual"]]

    # 2. Model Selector & Filter Controls
    col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])
    with col_ctrl1:
        selected_model = st.selectbox("Select Model to Evaluate Against Ground Truth", candidate_names, index=0)

    # Calculate Error Metrics for Selected Model
    pred_df["Model_Pred"] = pred_df[selected_model]
    pred_df["Residual"] = pred_df["Actual"] - pred_df["Model_Pred"]
    pred_df["Abs_Error"] = np.abs(pred_df["Residual"])
    pred_df["Pct_Error"] = (pred_df["Abs_Error"] / pred_df["Actual"]) * 100

    mae = pred_df["Abs_Error"].mean()
    rmse = np.sqrt((pred_df["Residual"]**2).mean())
    mape = pred_df["Pct_Error"].mean()
    over_forecasts = (pred_df["Residual"] < 0).sum()
    under_forecasts = (pred_df["Residual"] > 0).sum()

    # 3. KPI Scorecard Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(render_kpi_card("Mean Absolute Error (MAE)", f"{mae:.2f} units", "Average deviation", "positive" if mae < 3.5 else "neutral", "cyan"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("Root Mean Sq Error (RMSE)", f"{rmse:.2f} units", "Penalty for outliers", "positive" if rmse < 4.0 else "neutral", "blue"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card("Mean Abs % Error (MAPE)", f"{mape:.1f}%", "Percentage accuracy", "positive" if mape < 10 else "neutral", "green"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card("Over-Forecasted", f"{over_forecasts} Months", "Predicted > Actual", "neutral", "purple"), unsafe_allow_html=True)
    with k5:
        st.markdown(render_kpi_card("Under-Forecasted", f"{under_forecasts} Months", "Predicted < Actual", "neutral", "amber"), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    tab_curves, tab_residuals, tab_multi = st.tabs([
        "📈 Actual vs Predicted Trajectory",
        "🎯 Residual Distribution & Bias",
        "📊 Multi-Model Parallel Comparison"
    ])

    # =========================================================================
    # TAB 1: Curves
    # =========================================================================
    with tab_curves:
        st.markdown(f"### Historical Actual Deliveries vs {selected_model}")

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=pred_df["Period"],
            y=pred_df["Actual"],
            mode="lines+markers",
            name="Ground Truth (Actual Sales)",
            line=dict(color="#48CAE4", width=3),
            marker=dict(size=8, color="#48CAE4")
        ))
        fig_compare.add_trace(go.Scatter(
            x=pred_df["Period"],
            y=pred_df["Model_Pred"],
            mode="lines+markers",
            name=f"Predicted ({selected_model})",
            line=dict(color="#00F5D4", width=3, dash="dash"),
            marker=dict(size=8, color="#00F5D4", symbol="diamond")
        ))

        fig_compare = apply_dark_theme(fig_compare, f"Out-of-Time Test Set: Actual vs {selected_model}", height=420)
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("#### Period-by-Period Error Breakdown")
        display_tbl = pred_df[["Period", "Actual", "Model_Pred", "Residual", "Abs_Error", "Pct_Error"]].rename(
            columns={
                "Actual": "Actual (Units)",
                "Model_Pred": f"Predicted ({selected_model})",
                "Residual": "Residual (e = y - ŷ)",
                "Abs_Error": "Absolute Error",
                "Pct_Error": "Error Percentage (%)"
            }
        )
        display_tbl["Error Percentage (%)"] = display_tbl["Error Percentage (%)"].apply(lambda p: f"{p:.2f}%")
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Residuals & Bias
    # =========================================================================
    with tab_residuals:
        st.markdown("### Model Residual Diagnostic Auditing")
        st.markdown(r"""
        Residuals ($e_t = y_t - \hat{y}_t$) evaluate whether the regression model has extracted 
        all systematic patterns. An ideal model exhibits zero-mean residuals with no autocorrelation.
        """)

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            # Residual Plot
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=pred_df["Period"],
                y=pred_df["Residual"],
                marker_color=pred_df["Residual"].apply(lambda r: "#00F5D4" if r >= 0 else "#EF4444"),
                name="Residual"
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
            fig_res = apply_dark_theme(fig_res, "Residual by Evaluation Period", height=360)
            st.plotly_chart(fig_res, use_container_width=True)

        with r_col2:
            # Error Histogram
            fig_hist = px.histogram(
                pred_df,
                x="Residual",
                nbins=6,
                color_discrete_sequence=["#4CC9F0"],
                title="Residual Error Distribution"
            )
            fig_hist = apply_dark_theme(fig_hist, "Distribution of Prediction Errors", height=360)
            st.plotly_chart(fig_hist, use_container_width=True)

    # =========================================================================
    # TAB 3: Multi-Model Parallel
    # =========================================================================
    with tab_multi:
        st.markdown("### Parallel Algorithm Trajectories vs Ground Truth")

        fig_multi = go.Figure()
        fig_multi.add_trace(go.Scatter(
            x=pred_df["Period"],
            y=pred_df["Actual"],
            mode="lines+markers",
            name="Ground Truth (Actuals)",
            line=dict(color="#FFFFFF", width=4),
            marker=dict(size=9, color="#FFFFFF")
        ))

        colors = ["#00F5D4", "#48CAE4", "#F59E0B", "#F72585", "#7209B7"]
        for idx, m_name in enumerate(candidate_names):
            fig_multi.add_trace(go.Scatter(
                x=pred_df["Period"],
                y=pred_df[m_name],
                mode="lines+markers",
                name=m_name,
                line=dict(color=colors[idx % len(colors)], width=2, dash="dot" if "Baseline" in m_name else "solid"),
                marker=dict(size=6)
            ))

        fig_multi = apply_dark_theme(fig_multi, "Multi-Algorithm Forecast Comparison on Test Set", height=450)
        st.plotly_chart(fig_multi, use_container_width=True)
