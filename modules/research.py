"""
AutoVision AI - Academic Research & Empirical Evaluation Section
Dedicated B.Tech final-year research documentation presenting methodology,
chronological validation framework, empirical results, and critical limitations.
"""

import streamlit as st
import pandas as pd
from ml.predict import load_champion_artifacts
from utils.helpers import render_hero_banner, render_badge
from utils.charts import plot_model_comparison, plot_bar


def render():
    render_hero_banner(
        title="Academic Research & Empirical Methodology",
        subtitle="Comparative Evaluation of Machine Learning Algorithms for Automobile Sales Forecasting",
        badge_text="B.Tech Research Defense"
    )

    tab_paper, tab_results, tab_limitations = st.tabs([
        "📄 Research Paper Abstract & Formulation",
        "🔬 Empirical Findings & Validation",
        "⚠️ Limitations & Future Work"
    ])

    # =========================================================================
    # TAB 1: Paper Abstract & Formulation
    # =========================================================================
    with tab_paper:
        st.markdown(r"""
        ### Title of Research
        **"Comparative Evaluation of Machine Learning Algorithms for Automobile Sales Forecasting and Inventory Demand Prediction"**

        ---
        #### 1. Abstract
        Accurate sales forecasting in automotive retail is critical for balancing multi-million dollar inventory holding costs 
        against the opportunity costs of stockouts. This research paper formulates an empirical benchmarking framework 
        comparing supervised machine learning regression paradigms—including ordinary least squares Linear Regression, 
        non-parametric Decision Tree Regressors, Random Forest ensemble bagging, and Gradient Boosting machines—to forecast 
        monthly showroom vehicle delivery volumes. Using chronological out-of-time train-test partitioning to prevent lookahead bias 
        and autoregressive lag feature engineering, models are benchmarked against an empirical Naive Persistence baseline.

        #### 2. Research Problem Statement
        Traditional automobile dealerships rely heavily on intuitive heuristics or static seasonal rules of thumb. 
        Such approaches fail to capture nonlinear macroeconomic demand swings, festival surges, promotional discounting elasticity, 
        and customer powertrain transitions (ICE to Hybrid and EV). This leads to severe inventory mismatches, working capital 
        stagnation in aging stock (>60 days), and lost delivery bookings.

        #### 3. Primary Research Objectives
        1. **Feature Formulation:** Construct informative autoregressive temporal lag features ($y_{t-1}, y_{t-2}, y_{t-3}$), cyclical trigonometric encodings ($\sin(2\pi m/12), \cos(2\pi m/12)$), and rolling volatility windows without target leakage.
        2. **Strict Chronological Validation:** Implement a temporal train-test split reserving the final 6 months of historical transactions as an unseen evaluation horizon.
        3. **Multi-Model Benchmark:** Compare parametric (Linear Regression) vs ensemble tree-based models across standard empirical metrics (MAE, RMSE, $R^2$, and MAPE).
        4. **Inventory Reorder Translation:** Translate econometric volume projections into actionable, unit-level dealer stock replenishment recommendations.

        #### 4. Feature Space & Mathematical Formulation
        Given monthly sales observations $Y = \{y_1, y_2, \dots, y_T\}$, the feature vector for target month $t$ is formulated as:
        """)

        st.latex(r"X_t = \left[ t_{\text{year}}, \sin\left(\frac{2\pi m_t}{12}\right), \cos\left(\frac{2\pi m_t}{12}\right), F_{\text{festive}}, y_{t-1}, y_{t-2}, y_{t-3}, \mu_{3}(y), \sigma_{3}(y), D_{t-1} \right]")

        st.markdown(r"""
        Where:
        - $y_{t-k}$ represents autoregressive lag terms of vehicle sales volume.
        - $\mu_3(y)$ and $\sigma_3(y)$ represent the 3-month backward rolling mean and standard deviation.
        - $F_{\text{festive}} \in \{0, 1\}$ denotes festive demand seasonality (Diwali/Dussehra in Q3/Q4, Fiscal close in March).
        - $D_{t-1}$ represents historical dealer discount concessions granted in the preceding cycle.
        """)

    # =========================================================================
    # TAB 2: Empirical Results & Benchmarking
    # =========================================================================
    with tab_results:
        st.markdown("### Empirical Benchmark Results")
        st.markdown("""
        All models were trained strictly on preceding temporal partitions and evaluated on the final 
        **6 months of out-of-time test data**. Below are the computed evaluation scores:
        """)

        try:
            _, _, metadata = load_champion_artifacts()
            metrics_df = pd.DataFrame(metadata.get("metrics", []))

            fig_comp = plot_model_comparison(metrics_df, height=380)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            champion = metadata.get("champion_model", "Linear Regression")
            st.success(
                f"**Experimental Conclusion:** The **{champion}** model achieved superior generalization on the out-of-time validation horizon, "
                f"demonstrating that regularized linear trends combined with cyclical trigonometric seasonality provide high robustness "
                f"for monthly macroeconomic retail forecasting without overfitting."
            )

            # Feature Importance
            imp_dict = metadata.get("feature_importances", {})
            if imp_dict:
                st.markdown("#### Relative Feature Importance Breakdown")
                imp_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in imp_dict.items()]).sort_values("Importance", ascending=False)
                fig_imp = plot_bar(imp_df.head(8), x="Importance", y="Feature", orientation="h", title="Predictive Influence", height=320)
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading research benchmarks: {str(e)}")

    # =========================================================================
    # TAB 3: Limitations & Future Work
    # =========================================================================
    with tab_limitations:
        st.markdown("### Research Limitations & Roadmap for Future Exploration")
        
        st.markdown("""
        #### 1. Synthetic Data Constraints
        The initial demonstration dataset was synthetically modeled based on automotive market parameters (Tata, Mahindra, Maruti, Hyundai sales patterns). 
        While it accurately models seasonality, random consumer shocks and supply-chain semiconductor halts represent black swan events not fully captured.

        #### 2. Granularity Constraints
        The current forecasting engine operates at the aggregated showroom monthly level and brand segment level. 
        Training individualized daily variant-level models (e.g. Nexon Fearless Petrol Manual vs Automatic) suffers from small sample size sparsity in single dealership environments.

        #### 3. Exogenous Variables
        Future iterations should incorporate external macroeconomic features:
        - RBI repo rate adjustments (affecting auto loan interest EMIs).
        - Brent crude oil & national petrol/diesel fuel pump prices.
        - Competitor showroom proximity promotions.
        """)
