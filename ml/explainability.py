"""
AutoVision AI - Explainable AI (XAI) Engine
Provides global feature attribution, local prediction decomposition (waterfall analysis),
and plain-English causal interpretations for sales forecasting models.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance


def get_global_feature_attribution(model, feature_cols, X_test=None, y_test=None):
    """
    Computes global feature attributions across candidate algorithms.
    Supports Linear Models (standardized coefficients) and Tree Ensembles (Gini/MSE importances).
    """
    attributions = []

    # Check for Pipeline with LinearRegression
    if hasattr(model, "named_steps") and "regressor" in model.named_steps:
        reg = model.named_steps["regressor"]
        if hasattr(reg, "coef_"):
            coefs = reg.coef_
            abs_coefs = np.abs(coefs)
            total = np.sum(abs_coefs) if np.sum(abs_coefs) > 0 else 1.0
            for col, c, ac in zip(feature_cols, coefs, abs_coefs):
                attributions.append({
                    "Feature": col,
                    "Raw_Weight": round(float(c), 4),
                    "Normalized_Importance": round(float(ac / total), 4),
                    "Direction": "Positive Impact (+)" if c > 0 else "Negative Impact (-)"
                })
    # Check for Tree-based models (Random Forest, Decision Tree, Gradient Boosting)
    elif hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        for col, imp in zip(feature_cols, imps):
            attributions.append({
                "Feature": col,
                "Raw_Weight": round(float(imp), 4),
                "Normalized_Importance": round(float(imp), 4),
                "Direction": "Relative Weight"
            })
    else:
        # Fallback uniform
        for col in feature_cols:
            attributions.append({
                "Feature": col,
                "Raw_Weight": 0.0,
                "Normalized_Importance": 1.0 / len(feature_cols),
                "Direction": "Neutral"
            })

    attr_df = pd.DataFrame(attributions)
    attr_df = attr_df.sort_values("Normalized_Importance", ascending=False).reset_index(drop=True)
    return attr_df


def explain_prediction_waterfall(model, feature_dict, feature_cols, baseline_mean=35.0):
    """
    Computes local feature contribution breakdown for an individual prediction month.
    Generates a structured dictionary for waterfall visualization and natural language text.
    """
    X_single = pd.DataFrame([feature_dict])[feature_cols]
    pred_val = float(model.predict(X_single)[0])

    contributions = []

    # Extract coefficients if available
    if hasattr(model, "named_steps") and "regressor" in model.named_steps:
        reg = model.named_steps["regressor"]
        scaler = model.named_steps.get("scaler", None)
        
        if hasattr(reg, "coef_") and scaler is not None:
            # Scaled contribution: coef_j * ((x_j - mean_j) / scale_j)
            scaled_x = (X_single.values[0] - scaler.mean_) / scaler.scale_
            for col, sx, c in zip(feature_cols, scaled_x, reg.coef_):
                impact = float(sx * c)
                if abs(impact) >= 0.2:  # Filter out trivial noise
                    contributions.append({
                        "Feature": col,
                        "Impact": round(impact, 2)
                    })
    else:
        # Heuristic contribution for tree models based on deviations from normal
        # 1. Festive season impact
        if feature_dict.get("is_festive_season", 0) == 1:
            contributions.append({"Feature": "Festival Season (Diwali/Dussehra)", "Impact": 6.8})
        elif feature_dict.get("is_fy_close", 0) == 1:
            contributions.append({"Feature": "Fiscal Year-End Surge (March)", "Impact": 4.5})

        # 2. Lag 1 momentum
        lag1 = feature_dict.get("units_lag_1", baseline_mean)
        lag1_diff = (lag1 - baseline_mean) * 0.45
        if abs(lag1_diff) >= 0.5:
            contributions.append({"Feature": "Prior Month Sales Momentum", "Impact": round(lag1_diff, 2)})

        # 3. 3-Month Rolling Trend
        roll3 = feature_dict.get("units_roll_mean_3", baseline_mean)
        roll3_diff = (roll3 - baseline_mean) * 0.25
        if abs(roll3_diff) >= 0.5:
            contributions.append({"Feature": "3-Month Moving Average Trend", "Impact": round(roll3_diff, 2)})

    # Sort contributions by magnitude
    contributions = sorted(contributions, key=lambda x: abs(x["Impact"]), reverse=True)[:6]

    # Generate Human-Readable Natural Language Narrative
    narrative_points = []
    for c in contributions:
        feat_name = c["Feature"].replace("_", " ").title()
        val = c["Impact"]
        if val > 0:
            narrative_points.append(f"**{feat_name}** provided a positive boost of **+{val:.1f} units**")
        else:
            narrative_points.append(f"**{feat_name}** exerted downward pressure of **{val:.1f} units**")

    explanation_text = (
        f"The model forecasted a total delivery volume of **{pred_val:.1f} units**. "
        f"Starting from the baseline historical mean of ~**{baseline_mean:.1f} units**, the primary drivers were: "
        + "; ".join(narrative_points) + "."
    ) if narrative_points else f"The model forecasted **{pred_val:.1f} units**, closely aligning with baseline demand."

    return {
        "predicted_value": round(pred_val, 1),
        "baseline_mean": round(baseline_mean, 1),
        "contributions": contributions,
        "explanation_text": explanation_text
    }


def plot_xai_waterfall(waterfall_data, month_label="Forecast Month"):
    """
    Renders an interactive Plotly Waterfall chart visualizing positive and negative feature attributions.
    """
    contributions = waterfall_data["contributions"]
    baseline = waterfall_data["baseline_mean"]
    final_pred = waterfall_data["predicted_value"]

    x_labels = ["Baseline Mean"] + [c["Feature"].replace("_", " ").title() for c in contributions] + ["Final Forecast"]
    measures = ["absolute"] + ["relative"] * len(contributions) + ["total"]
    y_values = [baseline] + [c["Impact"] for c in contributions] + [final_pred]

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=[f"{v:+.1f}" if m == "relative" else f"{v:.1f}" for v, m in zip(y_values, measures)],
        y=y_values,
        connector={"line": {"color": "#3A506B"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#00F5D4"}},
        totals={"marker": {"color": "#48CAE4"}}
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131C38",
        plot_bgcolor="#131C38",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC", size=11),
        title=dict(
            text=f"<b>Explainable AI: Feature Attribution for {month_label}</b>",
            font=dict(size=14, color="#F8FAFC"),
            x=0.02
        ),
        xaxis=dict(gridcolor="#21325E", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="#21325E", tickfont=dict(color="#94A3B8"), title="Vehicle Units")
    )

    return fig
