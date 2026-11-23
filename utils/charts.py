"""
AutoVision AI - Interactive Chart Engine
Standardized Plotly visualization components styled to match the dark navy automotive theme.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Unified Color Palette
PALETTE = [
    "#00F5D4",  # Neon Teal / Cyan
    "#48CAE4",  # Sky Blue
    "#7209B7",  # Vivid Violet
    "#F72585",  # Neon Pink
    "#4CC9F0",  # Bright Cyan
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber Yellow
    "#3B82F6",  # Cobalt Blue
    "#EC4899",  # Soft Pink
    "#8B5CF6"   # Soft Violet
]

BG_CARD = "#131C38"
BG_MAIN = "#0B132B"
BORDER_COLOR = "#21325E"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"


def apply_dark_theme(fig, title=None, height=420):
    """Applies standardized dark automotive styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        height=height,
        margin=dict(l=40, r=40, t=55 if title else 25, b=40),
        font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", color=TEXT_COLOR, size=12),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, color=TEXT_COLOR),
            x=0.02,
            xanchor="left"
        ) if title else None,
        legend=dict(
            bgcolor="rgba(19, 28, 56, 0.7)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        ),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=MUTED_TEXT),
            title_font=dict(color=TEXT_COLOR)
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=MUTED_TEXT),
            title_font=dict(color=TEXT_COLOR)
        ),
        hoverlabel=dict(
            bgcolor="#1C2541",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#FFFFFF"
        )
    )
    return fig


def plot_line(df, x, y, title=None, color=None, markers=True, height=400):
    """Generates an interactive styled line chart."""
    fig = px.line(
        df, x=x, y=y, color=color,
        markers=markers,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(line=dict(width=3))
    return apply_dark_theme(fig, title, height)


def plot_bar(df, x, y, title=None, color=None, orientation='v', text_auto=False, height=400):
    """Generates an interactive styled bar chart."""
    fig = px.bar(
        df, x=x, y=y, color=color,
        orientation=orientation,
        text_auto=text_auto,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return apply_dark_theme(fig, title, height)


def plot_donut(df, names, values, title=None, hole=0.52, height=400):
    """Generates a modern donut / pie chart."""
    fig = px.pie(
        df, names=names, values=values,
        hole=hole,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color=BG_CARD, width=2))
    )
    return apply_dark_theme(fig, title, height)


def plot_grouped_bar(df, x, y, group_col, title=None, barmode="group", height=400):
    """Generates a grouped or stacked comparison bar chart."""
    fig = px.bar(
        df, x=x, y=y, color=group_col,
        barmode=barmode,
        color_discrete_sequence=PALETTE
    )
    return apply_dark_theme(fig, title, height)


def plot_scatter(df, x, y, title=None, size=None, color=None, hover_name=None, height=400):
    """Generates a 2D scatter/bubble plot for customer segmentation or pricing analysis."""
    fig = px.scatter(
        df, x=x, y=y, size=size, color=color,
        hover_name=hover_name,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="#FFFFFF")))
    return apply_dark_theme(fig, title, height)


def plot_forecast(historical_df, forecast_df, date_col, value_col, title="Sales Forecast & Historical Trajectory", height=450):
    """
    Renders historical actuals vs ML forecasted future trajectory with distinct styling.
    """
    fig = go.Figure()

    # Historical Actual Line
    fig.add_trace(go.Scatter(
        x=historical_df[date_col],
        y=historical_df[value_col],
        mode="lines+markers",
        name="Historical Actuals",
        line=dict(color="#48CAE4", width=3),
        marker=dict(size=6, color="#48CAE4")
    ))

    # Forecasted Trajectory
    # Connect last historical point to forecast to make a continuous smooth line
    bridge_dates = [historical_df[date_col].iloc[-1], *forecast_df[date_col].tolist()]
    bridge_values = [historical_df[value_col].iloc[-1], *forecast_df[value_col].tolist()]

    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_values,
        mode="lines+markers",
        name="ML Forecast Horizon",
        line=dict(color="#00F5D4", width=3, dash="dash"),
        marker=dict(size=8, color="#00F5D4", symbol="diamond")
    ))

    # If upper/lower bounds exist in forecast_df
    if "upper_bound" in forecast_df.columns and "lower_bound" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col].tolist() + forecast_df[date_col].tolist()[::-1],
            y=forecast_df["upper_bound"].tolist() + forecast_df["lower_bound"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0, 245, 212, 0.15)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

    return apply_dark_theme(fig, title, height)


def plot_model_comparison(metrics_df, height=380):
    """
    Renders bar comparison across candidate algorithms for RMSE, MAE, and R2 score.
    """
    fig = go.Figure()
    
    if "RMSE" in metrics_df.columns:
        fig.add_trace(go.Bar(
            name="RMSE (Lower is Better)",
            x=metrics_df["Model"],
            y=metrics_df["RMSE"],
            marker_color="#EF4444"
        ))
        
    if "MAE" in metrics_df.columns:
        fig.add_trace(go.Bar(
            name="MAE (Lower is Better)",
            x=metrics_df["Model"],
            y=metrics_df["MAE"],
            marker_color="#F59E0B"
        ))
        
    if "R2_Score" in metrics_df.columns:
        fig.add_trace(go.Bar(
            name="R² Score (Higher is Better)",
            x=metrics_df["Model"],
            y=metrics_df["R2_Score"],
            marker_color="#00F5D4"
        ))

    fig.update_layout(barmode="group")
    return apply_dark_theme(fig, "Machine Learning Algorithm Performance Benchmark", height)
