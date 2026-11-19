"""Central configuration for the Customer Retention & Revenue Intelligence Platform.

Every path, seed and modelling constant lives here so that the pipeline,
tests and the notebook all agree on the same values.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Data locations
# --------------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXPORTS = ROOT / "data" / "exports"
POWERBI_EXPORTS = DATA_EXPORTS / "powerbi"
FIGURES = DATA_EXPORTS / "figures"
METRICS_DIR = DATA_EXPORTS / "metrics"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SQL_DIR = ROOT / "sql"
NOTEBOOKS_DIR = ROOT / "notebooks"

RAW_FILE = DATA_RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_FILENAME = RAW_FILE.name

CLEAN_FILE = DATA_PROCESSED / "telco_churn_clean.csv"
SCORED_FILE = DATA_PROCESSED / "customer_risk_scores.csv"
DB_FILE = DATA_PROCESSED / "churn_analytics.sqlite"

PIPELINE_FILE = MODELS_DIR / "churn_pipeline.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_FILE = DATA_EXPORTS / "model_comparison.csv"

SQL_QUERY_FILE = SQL_DIR / "analytics_queries.sql"

REPORT_DATA_QUALITY = REPORTS_DIR / "data_quality_report.md"
REPORT_MODEL_EVALUATION = REPORTS_DIR / "model_evaluation.md"
REPORT_BUSINESS_INSIGHTS = REPORTS_DIR / "business_insights.md"

# --------------------------------------------------------------------------
# Reproducibility and modelling
# --------------------------------------------------------------------------
SEED = 42
TARGET = "churn_label"
TARGET_LABEL = "churn"
ID_COL = "customer_id"

TEST_SIZE = 0.20
VAL_SIZE = 0.20  # taken from the remaining 80% -> 60/20/20 train/val/test

# Primary model selection metric is PR-AUC (average precision) because churn is
# the minority class and the business cares about ranking churners.
SELECTION_METRIC = "pr_auc"

# Recall-oriented threshold rule: F-beta with beta=2 weighs recall twice as
# heavily as precision, which matches a retention team that would rather make
# a few extra calls than miss a churner.
THRESHOLD_BETA = 2.0

CANDIDATE_MODELS = ("logistic_regression", "random_forest", "xgboost")

# Business definition of risk tiers, expressed as the share of the customer
# base (from the validation distribution) above each probability cut-off.
RISK_TIER_SHARES = {
    "Very High": 0.05,
    "High": 0.20,
    "Medium": 0.50,
}
RISK_TIER_ORDER = ("Very High", "High", "Medium", "Low")

PRIMARY_CAPACITY = 0.10  # retention team can contact the top 10% of customers

# Monthly charge bands used for segment reporting (currency units are the
# dataset's own units, i.e. USD-like, undocumented in the source).
REVENUE_BANDS = {
    "Low": (0.0, 35.0),
    "Mid-Low": (35.0, 70.0),
    "Mid-High": (70.0, 90.0),
    "High": (90.0, float("inf")),
}

HIGH_VALUE_PERCENTILE = 0.75

TENURE_BINS = [-1, 6, 12, 24, 48, 1000]
TENURE_LABELS = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]

ADDON_SERVICE_COLUMNS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Columns a model may use. Deliberately excludes the identifier and anything
# derived from the outcome.
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_addon_services",
    "service_count",
    "avg_monthly_charge",
]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "auto_payment",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must never reach the model (identifier / label / leakage).
EXCLUDED_FROM_MODEL = [
    "customer_id",
    "churn",
    "churn_label",
]


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (
        DATA_RAW,
        DATA_PROCESSED,
        DATA_EXPORTS,
        POWERBI_EXPORTS,
        FIGURES,
        METRICS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        SQL_DIR,
        NOTEBOOKS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
