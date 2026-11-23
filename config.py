"""
AutoVision AI - System Configuration
Centralized configuration for database, ML pipelines, UI themes, and file paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
MODULES_DIR = BASE_DIR / "modules"
ML_DIR = BASE_DIR / "ml"
UTILS_DIR = BASE_DIR / "utils"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
for directory in [DATABASE_DIR, DATA_DIR, MODELS_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "autovision_ai")

# SQLite Fallback Path
SQLITE_DB_PATH = DATABASE_DIR / "autovision.db"

# App Settings
APP_NAME = "AutoVision AI"
APP_SUBTITLE = "Intelligent Car Showroom Analytics & Sales Forecasting System"
APP_VERSION = "2.0.0"
CURRENCY_SYMBOL = os.getenv("CURRENCY_SYMBOL", "₹")

# UI Theme Constants (Dealership Dark Navy & Cyan Palette)
THEME = {
    "primary": "#00F5D4",         # Neon Teal / Cyan
    "secondary": "#48CAE4",       # Electric Sky Blue
    "accent": "#7209B7",          # Deep Violet
    "background": "#0B132B",      # Deep Navy Dark
    "card_bg": "#131C38",         # Slate Blue Card
    "card_border": "#21325E",     # Card Border
    "text_primary": "#F8FAFC",    # Bright White
    "text_secondary": "#94A3B8",  # Muted Slate
    "success": "#10B981",         # Emerald Green
    "warning": "#F59E0B",         # Amber Yellow
    "danger": "#EF4444",          # Crimson Red
}

# Machine Learning Settings
ML_RANDOM_STATE = 42
DEFAULT_TEST_SIZE_MONTHS = 6
DEFAULT_FORECAST_HORIZON = int(os.getenv("DEFAULT_FORECAST_HORIZON", "6"))
SUPPORTED_ALGORITHMS = [
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
]

# Demo Credentials for local presentation testing
DEMO_CREDENTIALS = {
    "admin": {
        "password": "admin123",
        "full_name": "Executive Director (Admin)",
        "role": "Admin",
        "email": "admin@autovision.ai"
    },
    "manager": {
        "password": "manager123",
        "full_name": "Showroom General Manager",
        "role": "Manager",
        "email": "manager@autovision.ai"
    }
}
