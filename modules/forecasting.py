"""
AutoVision AI - Machine Learning Sales Forecasting Module
Interactive interface for model training, algorithm comparison (Linear Regression,
Decision Trees, Random Forests, Gradient Boosting), residual validation,
and multi-step future sales forecasting with 95% confidence bounds.
"""

import streamlit as st
import pandas as pd
import numpy as np

import config
from ml.train_models import train_and_evaluate_all
from ml.predict import generate_future_forecast, load_champion_artifacts
from ml.explainability import get_global_feature_attribution, explain_prediction_waterfall, plot_xai_waterfall
from utils.helpers import render_hero_banner, render_kpi_card, render_badge
from utils.charts import plot_line, plot_bar, plot_forecast, plot_model_comparison


def render():
    render_hero_banner(
        title="ML Vehicle Sales Forecasting & Predictive Intelligence",
        subtitle="Empirical time-series forecasting powered by supervised learning and chronological validation",
        badge_text="Research Core Module"
    )

    tab_forecast, tab_xai, tab_models, tab_retrain, tab_upload = st.tabs([
        "🔮 Future Sales Forecast",
        "💡 Explainable AI (XAI)",
        "📊 Model Comparison & Benchmarks",
        "⚙️ Model Retraining Console",
        "📂 Upload Custom Dealership Data"
    ])

    # =========================================================================
    # TAB 1: Future Sales Forecast
    # =========================================================================
    with tab_forecast:
        st.markdown("### Multi-Step Autoregressive Sales Horizon")
        
        col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])
        with col_ctrl1:
            horizon = st.slider("Select Forecast Horizon (Months Ahead)", min_value=1, max_value=12, value=6, step=1)

        # Generate Forecast
        try:
            forecast_df, monthly_df, metadata = generate_future_forecast(horizon_months=horizon)
            champ_name = metadata.get("champion_model", "Selected Model")
            
            # Forecast Metrics Summary
            total_fc_units = forecast_df["predicted_units"].sum()
            avg_fc_monthly = forecast_df["predicted_units"].mean()
            last_actual = monthly_df["total_units"].iloc[-1]
            growth_pct = ((avg_fc_monthly - last_actual) / last_actual) * 100

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(render_kpi_card("Champion Model", champ_name, "Lowest Test RMSE", "positive", "cyan"), unsafe_allow_html=True)
            with k2:
                st.markdown(render_kpi_card("Forecast Horizon", f"{horizon} Months", f"Through {forecast_df['month_label'].iloc[-1]}", "positive", "blue"), unsafe_allow_html=True)
            with k3:
                st.markdown(render_kpi_card("Projected Total Sales", f"{int(total_fc_units):,} units", f"Avg {avg_fc_monthly:.1f}/month", "positive", "green"), unsafe_allow_html=True)
            with k4:
                st.markdown(render_kpi_card("Demand Shift", f"{growth_pct:+.1f}%", "vs Last Recorded Month", "positive" if growth_pct >= 0 else "negative", "amber"), unsafe_allow_html=True)

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            # Interactive Plot
            hist_plot_df = monthly_df.tail(18).copy()
            fig_fc = plot_forecast(
                historical_df=hist_plot_df,
                forecast_df=forecast_df,
                date_col="period",
                value_col="total_units",
                title=f"Historical Sales vs ML Forecast Horizon ({champ_name})",
                height=450
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Tabular breakdown
            st.markdown("#### Projected Monthly Demand Breakdown")
            display_fc = forecast_df[["month_label", "predicted_units", "lower_bound", "upper_bound"]].rename(
                columns={
                    "month_label": "Forecast Period",
                    "predicted_units": "Expected Sales (Units)",
                    "lower_bound": "Lower Bound (95% CI)",
                    "upper_bound": "Upper Bound (95% CI)"
                }
            )
            st.dataframe(display_fc, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error computing forecast: {str(e)}")

    # =========================================================================
    # TAB 2: Explainable AI (XAI)
    # =========================================================================
    with tab_xai:
        st.markdown("### Explainable AI: Feature Attribution & Prediction Drivers")
        st.markdown("""
        Explainable AI (XAI) provides transparent visibility into **why** the model generates its predictions.
        We evaluate both **Global Feature Attribution** (which factors drive sales overall) and 
        **Local Prediction Decomposition** (what pushed a specific month's forecast up or down).
        """)

        try:
            model, feat_cols, metadata = load_champion_artifacts()
            champ_name = metadata.get("champion_model", "Selected Model")

            # 1. Global Feature Attribution
            st.markdown(f"#### 1. Global Model Drivers ({champ_name})")
            attr_df = get_global_feature_attribution(model, feat_cols)

            col_x1, col_x2 = st.columns([1.5, 1.5])
            with col_x1:
                fig_attr = plot_bar(
                    attr_df.head(8),
                    x="Normalized_Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 8 Features by Relative Predictive Power",
                    height=360
                )
                fig_attr.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_attr, use_container_width=True)

            with col_x2:
                st.dataframe(
                    attr_df.rename(columns={
                        "Feature": "Feature Name",
                        "Raw_Weight": "Model Weight / Coef",
                        "Normalized_Importance": "Relative Power (0-1)",
                        "Direction": "Causal Direction"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            # 2. Local Waterfall Attribution for a Specific Month
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 2. Local Month Attribution (Waterfall Analysis)")

            fc_df, monthly_df, _ = generate_future_forecast(horizon_months=6)
            month_choices = fc_df["month_label"].tolist()
            sel_month = st.selectbox("Select Forecast Period to Explain", month_choices, index=0)

            # Construct feature vector for chosen month
            sel_row = fc_df[fc_df["month_label"] == sel_month].iloc[0]
            step_idx = month_choices.index(sel_month) + 1
            hist_mean = float(monthly_df["total_units"].mean())

            # Sample feature dict
            future_date = sel_row["period"]
            feat_sample = {
                "year": future_date.year,
                "month": future_date.month,
                "quarter": (future_date.month - 1) // 3 + 1,
                "sin_month": np.sin(2 * np.pi * future_date.month / 12),
                "cos_month": np.cos(2 * np.pi * future_date.month / 12),
                "is_festive_season": 1 if future_date.month in [10, 11] else 0,
                "is_fy_close": 1 if future_date.month == 3 else 0,
                "is_year_end": 1 if future_date.month == 12 else 0,
                "units_lag_1": float(monthly_df["total_units"].iloc[-1]),
                "units_lag_2": float(monthly_df["total_units"].iloc[-2]),
                "units_lag_3": float(monthly_df["total_units"].iloc[-3]),
                "units_roll_mean_3": float(monthly_df["total_units"].iloc[-3:].mean()),
                "units_roll_mean_6": float(monthly_df["total_units"].iloc[-6:].mean()),
                "units_roll_std_3": float(monthly_df["total_units"].iloc[-3:].std()),
                "discount_pct_lag_1": float(monthly_df["discount_pct"].iloc[-1])
            }

            wf_data = explain_prediction_waterfall(model, feat_sample, feat_cols, baseline_mean=hist_mean)

            # Render Waterfall Chart
            fig_wf = plot_xai_waterfall(wf_data, month_label=sel_month)
            st.plotly_chart(fig_wf, use_container_width=True)

            # Plain English Explanation Box
            st.info(f"**🤖 Natural Language Causal Narrative:** {wf_data['explanation_text']}")

        except Exception as e:
            st.error(f"Error in Explainable AI module: {str(e)}")

    # =========================================================================
    # TAB 2: Model Comparison & Benchmarks
    # =========================================================================
    with tab_models:
        st.markdown("### Comparative Performance Evaluation of Candidate Algorithms")
        st.markdown("""
        To evaluate model generalization without lookahead bias, all models are evaluated on an 
        **unseen 6-month chronological test partition**. Metrics are computed against actual ground truth 
        and contrasted with a **Naive Persistence Baseline** (predicting prior month's sales).
        """)

        try:
            _, _, metadata = load_champion_artifacts()
            metrics_list = metadata.get("metrics", [])
            metrics_df = pd.DataFrame(metrics_list)

            # Display Benchmark Bar Chart
            fig_comp = plot_model_comparison(metrics_df, height=380)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Metrics Table
            st.markdown("#### Evaluation Metric Scorecard")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Feature Importance
            feat_imp = metadata.get("feature_importances", {})
            if feat_imp:
                st.markdown("#### Feature Importance Analysis")
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v} for k, v in feat_imp.items()
                ]).sort_values("Importance", ascending=False)

                fig_imp = plot_bar(
                    imp_df.head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Predictive Features in Champion Tree Model",
                    height=360
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading evaluation metrics: {str(e)}")

    # =========================================================================
    # TAB 3: Model Retraining Console
    # =========================================================================
    with tab_retrain:
        st.markdown("### Automated Retraining & Validation Console")
        st.markdown("Trigger a full re-computation of the chronological training pipeline using updated records.")

        retrain_col1, retrain_col2 = st.columns([1.5, 2.5])
        with retrain_col1:
            test_months = st.slider("Chronological Test Set Size (Months)", min_value=3, max_value=10, value=6, step=1)
            btn_retrain = st.button("🚀 Retrain All Candidate Models", type="primary")

        if btn_retrain:
            with st.spinner("Training Linear Regression, Decision Tree, Random Forest, and Gradient Boosting..."):
                try:
                    results = train_and_evaluate_all(test_size_months=test_months)
                    st.success(f"Retraining completed successfully! Champion Model: **{results['champion_name']}**")
                    st.dataframe(results["results_df"], use_container_width=True, hide_index=True)
                except Exception as ex:
                    st.error(f"Retraining failed: {str(ex)}")

    # =========================================================================
    # TAB 4: Upload Custom Dealership Data
    # =========================================================================
    with tab_upload:
        st.markdown("### Upload Custom Showroom Dataset for Analysis")
        st.markdown("You can upload permitted real showroom transaction records (CSV) to analyze external dealership data.")

        uploaded_file = st.file_uploader("Select CSV File with sales transactions", type=["csv"])
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                st.markdown(f"**Previewing Uploaded Dataset ({len(custom_df)} records):**")
                st.dataframe(custom_df.head(5), use_container_width=True)

                if "sale_date" in custom_df.columns and "selling_price" in custom_df.columns:
                    st.success("Valid transaction format detected! You can trigger model training on this custom dataset.")
                    if st.button("Train Models on Custom Uploaded Dataset"):
                        temp_path = config.DATA_DIR / "uploaded_custom_sales.csv"
                        custom_df.to_csv(temp_path, index=False)
                        with st.spinner("Training on uploaded custom data..."):
                            res = train_and_evaluate_all(source="csv", csv_path=temp_path)
                            st.success(f"Custom training complete! Best Model: **{res['champion_name']}**")
                            st.dataframe(res["results_df"], use_container_width=True, hide_index=True)
                else:
                    st.warning("CSV must contain at least 'sale_date' and 'selling_price' columns.")
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {str(e)}")
