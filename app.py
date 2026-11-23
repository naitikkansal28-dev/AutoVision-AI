"""
AutoVision AI - Main Streamlit Application Orchestrator
An Intelligent Car Showroom Analytics and Sales Forecasting System Using Machine Learning.
"""

import textwrap
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="AutoVision AI - Car Showroom Analytics & Sales Forecasting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

import config
from database.connection import get_backend
from utils.helpers import load_custom_css, render_badge
from utils.authentication import init_session_state, login, logout

# Import application modules
from modules import (
    dashboard,
    sales,
    forecasting,
    actual_vs_predicted,
    inventory,
    customers,
    bookings,
    finance,
    assistant,
    insights,
    model_monitoring,
    upload,
    research,
    reports
)


def render_login_screen():
    """Renders the executive login portal with a luxury automotive side-by-side layout."""
    load_custom_css()
    
    st.markdown("""
<style>
/* Strict single-page lock: zero scrolling */
html, body, .stApp {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background: radial-gradient(circle at 15% 15%, rgba(0, 245, 212, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(67, 97, 238, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(13, 21, 42, 0.6) 0%, transparent 70%),
                #060B18 !important;
}
section[data-testid="stSidebar"],
header[data-testid="stHeader"],
footer {
    display: none !important;
}
.block-container {
    max-width: 900px !important;
    padding-top: 13vh !important;
    padding-bottom: 0 !important;
    margin: auto !important;
}
/* Style right column of the main login block */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    background: linear-gradient(150deg, rgba(19, 28, 56, 0.92) 0%, rgba(10, 16, 34, 0.98) 100%) !important;
    border: 1px solid rgba(72, 202, 228, 0.22) !important;
    border-top: 3px solid #48CAE4 !important;
    border-radius: 20px !important;
    padding: 26px 28px !important;
    min-height: 360px !important;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6), 0 0 30px rgba(72, 202, 228, 0.08) !important;
    backdrop-filter: blur(20px) !important;
}
/* Neutralize nested demo sub-columns so they never get card styles */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stHorizontalBlock"] > div {
    background: transparent !important;
    border: none !important;
    border-top: none !important;
    box-shadow: none !important;
    min-height: auto !important;
    padding: 0 !important;
}
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stTextInput"] {
    margin-bottom: -4px !important;
}
div[data-testid="stTextInput"] label {
    font-size: 0.8rem !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
}
div[data-testid="stTextInput"] input {
    background-color: #0C1428 !important;
    border: 1px solid #21325E !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
    height: 38px !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #00F5D4 !important;
    box-shadow: 0 0 10px rgba(0, 245, 212, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.08, 1.0], gap="large")
    
    with col_left:
        left_card_html = f"""<div class="login-side-left">
<div>
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px;">
<div style="width: 52px; height: 52px; border-radius: 14px; background: linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(67, 97, 238, 0.25) 100%); border: 1px solid rgba(0, 245, 212, 0.5); display: flex; align-items: center; justify-content: center; font-size: 1.85rem; box-shadow: 0 0 20px rgba(0, 245, 212, 0.3);">🏎️</div>
<div>
<h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #FFFFFF 0%, #00F5D4 60%, #48CAE4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">{config.APP_NAME}</h1>
<span style="font-size: 0.72rem; color: #48CAE4; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Dealership Intelligence & ML OS</span>
</div>
</div>
<p style="margin: 0 0 14px 0; font-size: 0.88rem; color: #94A3B8; font-weight: 400; line-height: 1.45;">{config.APP_SUBTITLE}</p>
<div style="height: 1px; background: linear-gradient(90deg, rgba(0, 245, 212, 0.5) 0%, rgba(67, 97, 238, 0.4) 60%, transparent 100%); margin: 10px 0 18px 0;"></div>
</div>
<div style="display: flex; flex-direction: column; gap: 10px; margin-top: 14px;">
<div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 245, 212, 0.1); border: 1px solid rgba(0, 245, 212, 0.4); border-radius: 10px; padding: 8px 14px; width: fit-content;">
<span class="engine-pulse-dot"></span>
<span style="font-size: 0.78rem; font-weight: 700; color: #00F5D4; letter-spacing: 0.04em;">ACTIVE ENGINE: {get_backend().upper()}</span>
</div>
<div style="display: flex; gap: 8px; flex-wrap: wrap;">
<div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(72, 202, 228, 0.1); border: 1px solid rgba(72, 202, 228, 0.4); border-radius: 10px; padding: 7px 14px;">
<span style="font-size: 0.78rem; font-weight: 700; color: #48CAE4;">v{config.APP_VERSION}</span>
</div>
<div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 10px; padding: 7px 14px;">
<span style="font-size: 0.78rem; font-weight: 700; color: #F59E0B;">🎓 B.Tech Capstone</span>
</div>
</div>
</div>
</div>"""
        st.markdown(left_card_html, unsafe_allow_html=True)
        
    with col_right:
        header_right_html = """<div style="margin-bottom: 12px;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #48CAE4; box-shadow: 0 0 8px #48CAE4;"></span>
<h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em;">Sign In</h2>
</div>
<span style="font-size: 0.78rem; color: #94A3B8; margin-left: 16px;">Select demo role or enter credentials</span>
</div>"""
        st.markdown(header_right_html, unsafe_allow_html=True)
        
        with st.form("form_login"):
            username = st.text_input("Username", value="admin", key="login_username_input")
            password = st.text_input("Password", type="password", value="admin123", key="login_password_input")
            submit_btn = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            
        if submit_btn:
            if login(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials.")
                
        divider_html = """<div style="display: flex; align-items: center; text-align: center; margin: 6px 0 8px 0;">
<div style="flex: 1; border-bottom: 1px solid rgba(33, 50, 94, 0.7);"></div>
<span style="padding: 0 8px; font-size: 0.65rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Quick Demo Access</span>
<div style="flex: 1; border-bottom: 1px solid rgba(33, 50, 94, 0.7);"></div>
</div>"""
        st.markdown(divider_html, unsafe_allow_html=True)
        
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button("👑 Admin", key="btn_login_demo_admin", use_container_width=True):
                if login("admin", "admin123"):
                    st.rerun()
        with demo_col2:
            if st.button("💼 Manager", key="btn_login_demo_manager", use_container_width=True):
                if login("manager", "manager123"):
                    st.rerun()


def main():
    init_session_state()

    # Route between Login and Portal
    if not st.session_state.get("authenticated", False):
        render_login_screen()
        return

    # Load custom styling for authenticated session
    load_custom_css()

    user = st.session_state.get("user", {})
    user_role = st.session_state.get("role", "Manager")
    user_name = user.get("full_name", "Dealership User")

    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    with st.sidebar:
        st.markdown(textwrap.dedent(f"""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #21325E; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🚗</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.25rem; color: #00F5D4; font-weight: 800;">{config.APP_NAME}</h2>
                    <span style="font-size: 0.75rem; color: #94A3B8;">Dealership Analytics v{config.APP_VERSION}</span>
                </div>
            </div>
            <div style="margin-top: 12px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #FFFFFF;">{user_name}</div>
                    <span class="badge badge-cyan">{user_role}</span>
                </div>
                <span class="badge badge-blue">{get_backend().upper()}</span>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)

        # Navigation Options
        nav_options = [
            "🏠 Executive Dashboard",
            "📊 Sales Analytics",
            "🔮 ML Sales Forecasting & XAI",
            "🎯 Actual vs Predicted Lab",
            "📦 Inventory & Smart Alerts",
            "👥 Customer Segmentation",
            "📋 Bookings & Enquiries",
            "💰 Revenue & Profit",
            "🤖 AI Business Assistant",
            "💡 AI Business Insights",
            "📈 Model Monitoring & Drift",
            "📂 Dataset Ingestion Gateway",
            "🔬 Academic Research & Paper",
            "📑 Reports & Export Center"
        ]

        selected_page = st.radio("Navigation Menu", nav_options, index=0)

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            logout()

        st.markdown(textwrap.dedent("""
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 20px; line-height: 1.4;">
            <b>B.Tech Final-Year Project</b><br>
            AutoVision Dealership Intelligence<br>
            Supervised Regression & Time-Series
        </div>
        """).strip(), unsafe_allow_html=True)

    # =========================================================================
    # PAGE ROUTER
    # =========================================================================
    if selected_page == "🏠 Executive Dashboard":
        dashboard.render()
    elif selected_page == "📊 Sales Analytics":
        sales.render()
    elif selected_page == "🔮 ML Sales Forecasting & XAI":
        forecasting.render()
    elif selected_page == "🎯 Actual vs Predicted Lab":
        actual_vs_predicted.render()
    elif selected_page == "📦 Inventory & Smart Alerts":
        inventory.render()
    elif selected_page == "👥 Customer Segmentation":
        customers.render()
    elif selected_page == "📋 Bookings & Enquiries":
        bookings.render()
    elif selected_page == "💰 Revenue & Profit":
        finance.render()
    elif selected_page == "🤖 AI Business Assistant":
        assistant.render()
    elif selected_page == "💡 AI Business Insights":
        insights.render()
    elif selected_page == "📈 Model Monitoring & Drift":
        model_monitoring.render()
    elif selected_page == "📂 Dataset Ingestion Gateway":
        upload.render()
    elif selected_page == "🔬 Academic Research & Paper":
        research.render()
    elif selected_page == "📑 Reports & Export Center":
        reports.render()


if __name__ == "__main__":
    main()
