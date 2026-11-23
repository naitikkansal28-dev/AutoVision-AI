"""
AutoVision AI - UI Helper Functions & Formatters
Provides currency formatting, KPI card HTML generators, CSS injectors, and date helpers.
"""

import streamlit as st
import config


def load_custom_css():
    """Injects custom CSS from assets/custom.css into Streamlit."""
    css_file = config.ASSETS_DIR / "custom.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def format_currency(amount: float, currency: str = None, abbreviate: bool = True) -> str:
    """
    Formats numeric amounts into currency strings.
    Supports Indian notation (Crores / Lakhs) or full comma notation.
    """
    if currency is None:
        currency = config.CURRENCY_SYMBOL
        
    if amount is None or amount != amount:  # Check for NaN
        return f"{currency} 0"
        
    amount = float(amount)
    abs_amt = abs(amount)
    sign = "-" if amount < 0 else ""
    
    if abbreviate:
        if abs_amt >= 10_000_000:  # 1 Crore
            return f"{sign}{currency} {abs_amt / 10_000_000:.2f} Cr"
        elif abs_amt >= 100_000:   # 1 Lakh
            return f"{sign}{currency} {abs_amt / 100_000:.2f} L"
        elif abs_amt >= 1_000:
            return f"{sign}{currency} {abs_amt / 1_000:.1f} K"
        else:
            return f"{sign}{currency} {abs_amt:.0f}"
    else:
        # Indian comma formatting or standard
        return f"{sign}{currency} {abs_amt:,.2f}"


def format_number(val: float, precision: int = 1) -> str:
    """Formats standard numbers with optional comma separators."""
    if val is None or val != val:
        return "0"
    if isinstance(val, int) or val.is_integer():
        return f"{int(val):,}"
    return f"{val:,.{precision}f}"


def render_hero_banner(title: str, subtitle: str, badge_text: str = "Automotive Dealership Intelligence"):
    """Renders the top executive header banner."""
    html = f"""
    <div class="autovision-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <span class="badge badge-cyan">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtext: str = "", subtext_type: str = "neutral", style: str = "cyan") -> str:
    """
    Generates HTML string for an interactive glassmorphic KPI card.
    Styles: 'cyan', 'blue', 'green', 'amber', 'purple', 'red'
    subtext_type: 'positive', 'negative', 'neutral'
    """
    color_class = style.lower()
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext"><span class="{subtext_type}">{subtext}</span></div>
    </div>
    """


def render_badge(text: str, style: str = "cyan") -> str:
    """Generates an inline styled badge HTML snippet."""
    return f'<span class="badge badge-{style}">{text}</span>'
