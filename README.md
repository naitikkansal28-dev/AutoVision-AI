# AutoVision AI 🚗
### An Intelligent Car Showroom Analytics and Sales Forecasting System Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-MySQL%20%7C%20SQLite%20Fallback-00758F.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/Academic%20Project-B.Tech%20Final%20Year-green.svg)]()

---

## 1. Project Overview

**AutoVision AI** is an enterprise-grade automobile showroom management, business intelligence, and predictive analytics application designed for vehicle dealerships. It provides interactive executive dashboards, multidimensional sales analytics, machine-learning-based vehicle demand forecasting, customer segmentation clustering, inventory aging risk management, and explainable business recommendations.

This project was developed as a research-oriented **B.Tech Computer Science and Engineering Final-Year Capstone Project**. It bridges theoretical machine-learning algorithms with real-world dealership operations.

### Academic Research Title:
> *"Comparative Evaluation of Machine Learning Algorithms for Automobile Sales Forecasting and Inventory Demand Prediction"*

---

## 2. System Architecture & Modular Layout

```
AutoVision AI/
│
├── app.py                      # Main Streamlit Orchestrator & Router
├── config.py                   # System Constants, Paths, Theme & Demo Credentials
├── requirements.txt            # Python Dependencies
├── .env.example                # Environment Variable Template
├── .env                        # Local Environment Configuration
├── README.md                   # Comprehensive Project Documentation
│
├── database/                   # Database Layer
│   ├── connection.py           # Dual-Engine Connector (MySQL with auto SQLite Fallback)
│   ├── schema.sql              # Normalized Relational Database Schema (9 Tables)
│   ├── seed_data.py            # 36-Month Synthetic Realistic Dealership Generator
│   └── autovision.db           # SQLite Local Development Database (persistent)
│
├── modules/                    # Application Feature Modules (14 Specialized Workspaces)
│   ├── dashboard.py            # Executive KPI Dashboard & Real-Time Summaries
│   ├── sales.py                # Sales Analytics, Brand/Model Deep Dive & Team Targets
│   ├── forecasting.py          # ML Forecasting Interface, Horizons & Explainable AI (XAI)
│   ├── actual_vs_predicted.py  # Out-of-Time Model Validation Lab & Residual Bias Auditing
│   ├── inventory.py            # Stock Tracking, Aging Risk (>60d, >90d) & Smart Alerts
│   ├── customers.py            # Demographic Analytics & K-Means Customer Clustering
│   ├── bookings.py             # Enquiries, Test Drives & Booking Pipeline CRM
│   ├── finance.py              # Gross Revenue, Margin Decomposition & Net Profit
│   ├── assistant.py            # Natural Language Conversational Dealership AI Assistant
│   ├── insights.py             # Explainable Rule-Based AI Recommendations
│   ├── model_monitoring.py     # Persistent Experiment Registry, Drift & MLOps Tracking
│   ├── upload.py               # Custom Dealership Dataset Ingestion & Validation Gateway
│   ├── research.py             # Academic Paper Defense Section & Research Equations
│   └── reports.py              # Export Center (CSVs, Multi-Sheet Excel & Executive PDF)
│
├── ml/                         # Machine Learning Pipeline
│   ├── preprocessing.py        # Calendar Aggregation & Missing Data Handling
│   ├── feature_engineering.py  # Autoregressive Lags, Rolling Windows & Sine/Cosine Encodings
│   ├── train_models.py         # Chronological Train-Test Split & Algorithm Benchmark
│   ├── evaluate_models.py      # MAE, RMSE, R2, and MAPE Validation Scorecards
│   ├── predict.py              # Multi-Step Autoregressive Future Forecasting Engine
│   ├── explainability.py       # Global Feature Attributions & Prediction Waterfall Breakdown
│   └── experiment_tracker.py   # Persistent Multi-Run Model Performance & Drift Ledger
│
├── utils/                      # Utilities & Core Services
│   ├── authentication.py       # PBKDF2-HMAC-SHA256 Password Hashing & RBAC Sessions
│   ├── charts.py               # Standardized Automotive Dark Plotly Engine
│   ├── helpers.py              # Currency Formatters (₹ Lakhs/Crores), Badges & CSS
│   ├── inventory_alerts.py     # Smart Days-of-Supply (DOS) & Stockout Alert Calculation
│   ├── data_validator.py       # Schema & Data Quality Ingestion Validation Engine
│   ├── excel_exporter.py       # Formatted Multi-Sheet Excel Workbook Compiler (OpenPyXL)
│   └── pdf_exporter.py         # Executive Briefing PDF Compiler (ReportLab)
│
├── data/                       # Standalone Datasets
│   └── sample_sales.csv        # Standalone CSV Benchmark Dataset
│
├── models/                     # Serialized Model Artifacts
│   ├── best_forecasting_model.joblib
│   ├── feature_cols.joblib
│   ├── all_models.joblib
│   └── model_metadata.json
│
└── assets/                     # UI Assets & Styling
    └── custom.css              # Dealership Navy & Neon Cyan Glassmorphic Stylesheet
```

---

## 3. Technology Stack

- **Programming Language:** Python 3.10+ (tested on Python 3.12)
- **Frontend / UI:** Streamlit (Custom HTML5/CSS3 glassmorphic injection)
- **Database:** MySQL 8.0+ / MySQL Workbench (with automatic, zero-configuration SQLite fallback)
- **Data Engineering:** Pandas 2.0+, NumPy 1.24+
- **Machine Learning Algorithms:**
  - Ordinary Least Squares Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor (Ensemble Bagging)
  - Gradient Boosting Regressor (Ensemble Boosting)
  - Naive Persistence Baseline ($y_t = y_{t-1}$)
- **Evaluation Metrics:** MAE, RMSE, $R^2$ Score, MAPE (%)
- **Data Visualization:** Plotly Graph Objects & Express
- **Model Serialization:** Joblib
- **Document & Spreadsheet Generation:** ReportLab 5.0+ (PDF exporter), OpenPyXL 3.1+ (Multi-sheet Excel exporter)
- **Explainable AI (XAI):** Permutation Feature Attribution, Local Prediction Waterfall Breakdown
- **Security:** Salted PBKDF2-HMAC-SHA256 password cryptography

---

## 4. Key Functional Modules & Advanced Features

### 1. Executive Dashboard (`modules/dashboard.py`)
- Real-time dealership metrics: Total Vehicles Sold, Sales Revenue, Estimated Gross Profit, Available Inventory, Active Bookings, and Customer Pipeline.
- Monthly vehicle delivery trends and revenue velocity charts.
- Brand market shares and powertrain distribution (Petrol, Diesel, Hybrid, EV).
- Top 5 best-selling models and live recent transactions.

### 2. Multidimensional Sales Analytics (`modules/sales.py`)
- Multi-filter slicing: filter by year, brand, fuel type, transmission, and payment mode.
- Model performance matrix: top-performing vs least-moving vehicle variants.
- Sales consultant performance leaderboard tracking units delivered, revenue generated, and commission earnings.
- Pricing and discount elasticity analysis.

### 3. ML Sales Forecasting & Explainable AI (`modules/forecasting.py` & `ml/explainability.py`)
- **Multi-Step Horizon:** 1-to-12-month forward-looking sales projections with 95% confidence intervals.
- **Explainable AI (XAI):**
  - **Global Feature Attribution:** Displays normalized importance and causal direction of predictive features.
  - **Local Prediction Waterfall Analysis:** Breaks down any individual forecast month into baseline mean + seasonal boosts + lag momentum + discount effects.
  - **Natural Language Narrative:** Dynamically generates plain-English explanations of model predictions.

### 4. Actual vs Predicted Validation Lab (`modules/actual_vs_predicted.py`)
- Period-by-period ground truth comparison on out-of-time test partitions.
- Computes MAE, RMSE, MAPE, Absolute Error, and Error Percentage per month.
- Residual Diagnostic Auditing: Residual plot over time and error distribution histogram.
- Directional bias auditing (Over-forecasting vs Under-forecasting counters).
- Multi-model synchronized trajectory comparison.

### 5. Smart Inventory Alerts & Reorder Planning (`modules/inventory.py` & `utils/inventory_alerts.py`)
- Computes **Days of Supply (DOS)** by integrating current physical stock with **ML forecasted monthly demand**.
- Automated Alert Severities:
  - 🚨 **Critical Shortage:** Zero units in stock with positive forecasted demand.
  - ⚡ **Imminent Stockout:** DOS $< 15$ days.
  - ⏳ **Critical Aging Alert:** Vehicle dwell time $\ge 90$ days.
  - ⚠️ **Aging Warning:** Dwell time between 60–89 days.
  - ℹ️ **Low Stock Buffer:** DOS between 15–30 days.
  - ✅ **Healthy Equilibrium:** Optimal stock coverage.
- Prescriptive suggested replenishment order quantities.

### 6. Conversational AI Business Assistant (`modules/assistant.py`)
- Interactive text-to-insights chat interface answering dealership queries using live database queries.
- Answers questions on top models, revenues, profit margins, inventory levels, critical aging cars, consultant leaderboards, and conversion rates.
- One-click quick-question chips for rapid viva demonstrations.

### 7. Advanced Customer Segmentation (`modules/customers.py`)
- **Mathematical Cluster Optimization:** Elbow Method curve (Inertia vs $k$) and Silhouette Coefficient analysis ($k \in [2, 6]$).
- Multidimensional clustering across Annual Income, Vehicle Budget Ceiling, and Realized Spend.
- Interactive 2D scatter plots and strategic segment personas for targeted marketing.

### 8. Model Performance Monitoring & Experiment Registry (`modules/model_monitoring.py` & `ml/experiment_tracker.py`)
- Persistent experiment logging in `models/experiments_log.json`.
- Longitudinal tracking of MAE, RMSE, $R^2$, and MAPE across training experiments.
- Visual drift detection and algorithmic stability auditing.

### 9. Dataset Ingestion & Validation Gateway (`modules/upload.py` & `utils/data_validator.py`)
- Upload custom CSV or Excel (`.xlsx`, `.xls`) files for sales transactions and showroom inventory.
- Automated schema validation checking mandatory columns, date parsing, numeric positive values, and duplicate VINs.
- Validation scorecards and one-click training on uploaded datasets.

### 10. Automated Reports & Export Center (`modules/reports.py`, `utils/pdf_exporter.py`, `utils/excel_exporter.py`)
- **Executive PDF Report:** Print-ready briefing compiled with ReportLab.
- **Multi-Sheet Excel Workbook:** Formatted `.xlsx` report featuring Executive Summary, Sales Register, Inventory Audit, ML Forecasts, and Model Evaluations.
- Raw CSV exports for all datasets.
- Distinct separation of **Sales Revenue**, **Vehicle Acquisition Cost**, **Gross Profit**, **Showroom Operating Expenses**, and **Net Profit**.
- Financial waterfall decomposition.
- Unit economics and profit margin yield by vehicle model.

### 8. Explainable AI Business Recommendations (`modules/insights.py`)
- Dynamic, explainable rule-based insights:
  - Critical aging inventory notifications (>60 and >90 days).
  - Imminent stockout warnings on high-velocity SUV/Hybrid models.
  - Predictive staffing and logistics alerts based on next month's forecast.
  - Funnel bottleneck alerts when test drive conversion drops below 50%.

### 9. Academic Research Defense (`modules/research.py`)
- Complete formal research paper presentation for final-year defense.
- Mathematical formulations of autoregressive lag vectors and cyclical features.
- Analysis of empirical results and critical discussion of research limitations.

### 10. Reports & Export Center (`modules/reports.py`)
- Downloadable CSV exports: Full Sales Register, Inventory Audit, ML Forecasts, and Benchmark Metrics.
- **Automated PDF Export:** Generates an executive briefing PDF using ReportLab.

---

## 5. Machine Learning Research Methodology

### Feature Space Vector:
$$\mathbf{X}_t = \left[ t_{\text{year}}, \sin\left(\frac{2\pi m_t}{12}\right), \cos\left(\frac{2\pi m_t}{12}\right), F_{\text{festive}}, y_{t-1}, y_{t-2}, y_{t-3}, \mu_3(y), \sigma_3(y), D_{t-1} \right]$$

### Chronological Train-Test Split (Zero Data Leakage):
Rather than utilizing randomized k-fold splits (which violate temporal ordering and introduce lookahead bias), AutoVision AI utilizes a strict chronological partition where the final 6 months of historical transactions are reserved as an unseen out-of-time test horizon.

### Empirical Validation Results (Actual Computed Scores):

| Model | MAE | RMSE | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence (Baseline)** | 4.33 | 4.65 | 0.3849 | 12.24% |
| **Linear Regression (Champion)** | **2.78** | **2.96** | **0.7509** | **7.76%** |
| **Gradient Boosting Regressor** | 3.42 | 4.88 | 0.3234 | 9.85% |
| **Random Forest Regressor** | 4.00 | 5.12 | 0.2566 | 11.28% |
| **Decision Tree Regressor** | 4.29 | 5.70 | 0.0786 | 12.42% |

*Note: All performance metrics are calculated directly from out-of-time predictions on 36 months of realistic transaction data.*

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- (Optional) MySQL Server 8.0+ and MySQL Workbench. *If MySQL is not installed, the application automatically uses the included SQLite engine.*

### Step 1: Clone or Navigate to Project Directory
```powershell
cd "c:\Users\Naitik\OneDrive\Desktop\AutoVision AI"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
A `.env.example` file is provided in the root directory. Copy it to `.env`:
```powershell
cp .env.example .env
```
If you wish to connect your local MySQL server:
1. Open MySQL Workbench.
2. Edit `.env` with your MySQL credentials:
   ```env
   DB_HOST=127.0.0.1
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_password_here
   DB_NAME=autovision_ai
   ```
*If MySQL credentials are left blank or the server is stopped, the application will automatically connect via the local SQLite database (`database/autovision.db`) without any configuration.*

### Step 4: Seed Database with Realistic Dealership Data
Run the seeding script to initialize the database schema and populate 36 months of realistic transactions, vehicles, customers, inventory, and employees:
```powershell
python -m database.seed_data
```

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 7. Demo Credentials for Testing & Viva

| Account Role | Username | Password | Access Privileges |
| :--- | :--- | :--- | :--- |
| **Executive Director (Admin)** | `admin` | `admin123` | Full administrative control, inventory registration, re-training |
| **General Manager** | `manager` | `manager123` | Operational dealership access, CRM pipeline, sales reports |

*Shortcut buttons ("Login as Admin" / "Login as Manager") are also provided directly on the login screen for rapid presentation demonstration.*

---

## 8. Viva Presentation Talking Points (Q&A)

1. **Why was Linear Regression the best-performing model?**
   - In monthly macroeconomic time series with relatively short sequence lengths (36 months) and clear cyclical and linear trend patterns, regularized linear regression avoids the overfitting and step-function discretization errors common to tree-based regressors.
2. **How is data leakage prevented?**
   - By using chronological train-test splitting (past predicting future) rather than random K-Fold cross-validation, and calculating all rolling statistics and lag variables strictly from prior time periods ($t-1, t-2$).
3. **What is the difference between Gross Profit and Net Profit in the finance module?**
   - Gross profit equals total vehicle selling price minus OEM vehicle acquisition cost plus dealer accessories and insurance margins. Net profit deducts all showroom overhead operating expenses (rent, staff salaries, commercial utilities, marketing).
4. **How does the SQLite fallback work?**
   - `database/connection.py` tests the MySQL socket on startup. If MySQL server is unreachable, it logs a clean fallback message and switches queries to the local `database/autovision.db` with query normalization.

---

## 9. Academic Submission Details

- **Project Title:** AutoVision AI: An Intelligent Car Showroom Analytics and Sales Forecasting System Using Machine Learning
- **Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering
- **Domain:** Artificial Intelligence, Applied Machine Learning, Full-Stack Web Engineering
- **Status:** Complete, Verified, and Production-Ready
