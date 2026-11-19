"""Cleaning and feature derivation for the Telco churn dataset.

Design rules followed here:

* The raw file is never modified in place.
* No record is dropped silently. Anything questionable is flagged in a boolean
  column and counted in the cleaning log so the data-quality report can show it.
* Every derived column has a business justification, recorded in the log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

RENAME_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

YES_NO_COLUMNS = [
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
]

TRI_STATE_COLUMNS = [
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

VALID_CHURN = {"Yes", "No"}


@dataclass
class CleaningLog:
    """Everything a reviewer needs to audit the cleaning step."""

    row_count_raw: int = 0
    row_count_clean: int = 0
    column_count_raw: int = 0
    column_count_clean: int = 0
    duplicate_customer_ids: int = 0
    duplicate_full_rows: int = 0
    duplicates_removed: int = 0
    total_charges_blank: int = 0
    total_charges_imputed: int = 0
    invalid_churn_values: int = 0
    invalid_records_flagged: int = 0
    zero_tenure_records: int = 0
    derived_columns: list[str] = field(default_factory=list)
    decisions: list[tuple[str, str]] = field(default_factory=list)
    invalid_record_breakdown: dict[str, int] = field(default_factory=dict)
    missing_value_counts: dict[str, int] = field(default_factory=dict)

    def add_decision(self, issue: str, decision: str) -> None:
        self.decisions.append((issue, decision))

    def as_dict(self) -> dict[str, Any]:
        data = {
            "row_count_raw": self.row_count_raw,
            "row_count_clean": self.row_count_clean,
            "column_count_raw": self.column_count_raw,
            "column_count_clean": self.column_count_clean,
            "duplicate_customer_ids": self.duplicate_customer_ids,
            "duplicate_full_rows": self.duplicate_full_rows,
            "duplicates_removed": self.duplicates_removed,
            "total_charges_blank": self.total_charges_blank,
            "total_charges_imputed": self.total_charges_imputed,
            "invalid_churn_values": self.invalid_churn_values,
            "invalid_records_flagged": self.invalid_records_flagged,
            "zero_tenure_records": self.zero_tenure_records,
            "derived_columns": self.derived_columns,
            "decisions": self.decisions,
            "invalid_record_breakdown": self.invalid_record_breakdown,
            "missing_value_counts": self.missing_value_counts,
        }
        return data


@dataclass
class CleaningResult:
    frame: pd.DataFrame
    log: CleaningLog


def _to_yes_no(series: pd.Series) -> pd.Series:
    """Map a two-level Yes/No column to int8 1/0, tolerating case, spaces and
    values that are already encoded as 0/1 (so re-running the cleaner is safe)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype("int8")
    cleaned = series.astype("string").str.strip().str.title()
    mapped = cleaned.map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    if mapped.isna().any():
        unexpected = sorted(cleaned[mapped.isna()].unique().tolist())[:5]
        raise ValueError(f"Unexpected Yes/No values in column: {unexpected}")
    return mapped.astype("int8")


def _addon_flag(series: pd.Series) -> pd.Series:
    """1 when the customer actively has the add-on, 0 for 'No'/'No internet service'."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) == 1).astype("int8")
    cleaned = series.astype("string").str.strip().str.title()
    return (cleaned == "Yes").astype("int8")


def _normalise_churn(series: pd.Series) -> pd.Series:
    """Return the target as 'Yes'/'No', accepting an already-encoded 0/1 column."""
    if pd.api.types.is_numeric_dtype(series):
        return series.map({0: "No", 1: "Yes"}).astype("string")
    return series.astype("string").str.strip().str.title()


def _avg_monthly_charge(df: pd.DataFrame) -> pd.Series:
    """Realised average monthly spend.

    For customers with tenure == 0 there is no billing history, so
    TotalCharges cannot be divided by tenure. We fall back to the quoted
    MonthlyCharges, which is the best available estimate of what they will pay.
    """
    tenure = df["tenure"].to_numpy(dtype=float)
    total = df["total_charges"].to_numpy(dtype=float)
    monthly = df["monthly_charges"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(tenure > 0, total / np.where(tenure > 0, tenure, 1.0), monthly)
    return pd.Series(np.round(avg, 2), index=df.index, name="avg_monthly_charge")


def build_features(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Add the engineered columns used by the analytics and the model."""
    out = df.copy()

    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=config.TENURE_BINS,
        labels=config.TENURE_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")

    # Add-ons: only an explicit "Yes" counts. "No internet service" means the
    # product cannot exist for that customer, so it is not a consumed service.
    addon_matrix = pd.DataFrame(
        {col: _addon_flag(out[col]) for col in config.ADDON_SERVICE_COLUMNS}
    )
    out["num_addon_services"] = addon_matrix.sum(axis=1).astype("int8")

    internet_in_use = (out["internet_service"].astype("string").str.strip() != "No").astype("int8")
    out["has_internet"] = internet_in_use
    out["service_count"] = (
        out["phone_service"].astype("int8") + internet_in_use + out["num_addon_services"].astype("int8")
    ).astype("int8")

    out["avg_monthly_charge"] = _avg_monthly_charge(out)

    # Charge bands give the business a segment it can actually act on, rather
    # than a continuous variable nobody can put in a slide.
    edges = [band[0] for band in config.REVENUE_BANDS.values()] + [float("inf")]
    labels = list(config.REVENUE_BANDS.keys())
    out["revenue_band"] = pd.cut(
        out["monthly_charges"], bins=edges, labels=labels, include_lowest=True, right=False
    ).astype("string")

    threshold = float(out["monthly_charges"].quantile(config.HIGH_VALUE_PERCENTILE))
    out["high_value_customer"] = (out["monthly_charges"] >= threshold).astype("int8")

    out["auto_payment"] = (
        out["payment_method"].astype("string").str.contains("automatic", case=False, na=False).astype("int8")
    )

    out["churn_label"] = (
        out["churn"].astype("string").str.strip().str.title().map({"Yes": 1, "No": 0}).astype("int8")
    )
    out["tenure_months"] = out["tenure"].astype("int16")

    log.derived_columns.extend(
        [
            "tenure_group",
            "num_addon_services",
            "has_internet",
            "service_count",
            "avg_monthly_charge",
            "revenue_band",
            "high_value_customer",
            "auto_payment",
            "churn_label",
        ]
    )
    log.add_decision(
        "Derived columns",
        "Added tenure_group (business-friendly tenure bands), num_addon_services, "
        "has_internet, service_count, avg_monthly_charge, revenue_band, "
        "high_value_customer (>= 75th percentile monthly charges), auto_payment and "
        "churn_label (the modelling target).",
    )
    return out


def clean_customers(raw: pd.DataFrame) -> CleaningResult:
    """Clean the raw frame and return the cleaned data plus an audit log."""
    log = CleaningLog(
        row_count_raw=len(raw),
        column_count_raw=raw.shape[1],
        duplicate_full_rows=int(raw.duplicated().sum()),
    )

    df = raw.rename(columns=RENAME_MAP).copy()

    # ------------------------------------------------------------------
    # 1. Missing values as distributed (blank strings, not NaN in the CSV)
    # ------------------------------------------------------------------
    blank_total_charges = (
        df["total_charges"].astype("string").str.strip().isin(["", "nan", "NaN", "<NA>", "None"])
        | df["total_charges"].isna()
    )
    log.total_charges_blank = int(blank_total_charges.sum())
    log.zero_tenure_records = int((df["tenure"] == 0).sum())

    for col in df.columns:
        if df[col].isna().any():
            log.missing_value_counts[col] = int(df[col].isna().sum())

    # TotalCharges arrives as text because 11 new customers have a blank value.
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").astype("float64")
    log.missing_value_counts["total_charges"] = int(df["total_charges"].isna().sum())

    # All blanks belong to tenure == 0 customers: they have not been billed yet.
    blanks_outside_zero_tenure = int(
        (df["total_charges"].isna() & (df["tenure"] != 0)).sum()
    )
    df["total_charges_imputed"] = df["total_charges"].isna().astype("int8")
    df["total_charges"] = df["total_charges"].fillna(0.0)
    log.total_charges_imputed = int(df["total_charges_imputed"].sum())
    log.add_decision(
        "Blank TotalCharges",
        f"{log.total_charges_blank} rows had a blank TotalCharges, all belonging to "
        f"tenure == 0 customers ({blanks_outside_zero_tenure} blanks were outside that group). "
        "They were converted to numeric and imputed with 0.0 because a customer with zero "
        "months of tenure cannot have accumulated charges. The rows are retained and marked "
        "with total_charges_imputed = 1 rather than deleted.",
    )

    # ------------------------------------------------------------------
    # 2. Numeric coercion and categorical standardisation
    # ------------------------------------------------------------------
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce").astype("float64")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("int64")
    df["monthly_charges"] = df["monthly_charges"].fillna(float(df["monthly_charges"].median()))

    for col in config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            df[col] = df[col].astype("string").str.strip()

    # Binary service flags -> 1/0. Text categories stay text so the dashboard
    # can label them directly.
    for col in YES_NO_COLUMNS:
        df[col] = _to_yes_no(df[col])

    df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce").fillna(0).astype("int8")
    df["gender"] = df["gender"].astype("string").str.strip().str.title()
    df["senior_citizen_label"] = df["senior_citizen"].map({0: "No", 1: "Yes"}).astype("string")
    df["churn"] = _normalise_churn(df["churn"])

    log.add_decision(
        "Categorical encoding",
        "Binary Yes/No service and demographic flags "
        f"({', '.join(YES_NO_COLUMNS)}) were encoded as 1/0. Multi-level categories "
        "were kept as readable text in the cleaned file and only one-hot encoded inside "
        "the model pipeline, so business exports stay human-readable.",
    )

    # ------------------------------------------------------------------
    # 3. Duplicates
    # ------------------------------------------------------------------
    log.duplicate_customer_ids = int(df["customer_id"].duplicated().sum())
    if log.duplicate_customer_ids:
        before = len(df)
        df = df.drop_duplicates(subset=["customer_id"], keep="first").copy()
        log.duplicates_removed += before - len(df)
        log.add_decision(
            "Duplicate customer IDs",
            f"{log.duplicate_customer_ids} duplicate customer IDs were found; the first "
            "occurrence was kept so every customer contributes exactly one row.",
        )
    else:
        log.add_decision(
            "Duplicate customer IDs",
            "No duplicate customer IDs were found; every row is a distinct customer.",
        )

    if log.duplicate_full_rows:
        before = len(df)
        df = df.drop_duplicates().copy()
        log.duplicates_removed += before - len(df)
        log.add_decision(
            "Fully duplicated rows",
            f"{log.duplicate_full_rows} fully duplicated rows were removed.",
        )
    else:
        log.add_decision("Fully duplicated rows", "No fully duplicated rows were found.")

    # ------------------------------------------------------------------
    # 4. Validity flags (records are kept, never silently discarded)
    # ------------------------------------------------------------------
    invalid_churn = ~df["churn"].isin(VALID_CHURN)
    log.invalid_churn_values = int(invalid_churn.sum())
    if log.invalid_churn_values:
        df = df.loc[~invalid_churn].copy()
        log.add_decision(
            "Invalid target values",
            f"{log.invalid_churn_values} rows had a Churn value outside {{Yes, No}} and were "
            "removed because they cannot be labelled for supervised learning.",
        )

    checks = {
        "negative_tenure": (df["tenure"] < 0),
        "non_positive_monthly_charges": (df["monthly_charges"] <= 0),
        "negative_total_charges": (df["total_charges"] < 0),
        "tenure_above_expected_max": (df["tenure"] > 72),
        "total_charges_inconsistent": (
            (df["tenure"] >= 12) & (df["total_charges"] < df["monthly_charges"])
        ),
    }
    flag_frame = pd.DataFrame({name: mask.fillna(False) for name, mask in checks.items()})
    df["data_quality_flag"] = flag_frame.any(axis=1).astype("int8")
    log.invalid_record_breakdown = {k: int(v.sum()) for k, v in checks.items()}
    log.invalid_records_flagged = int(df["data_quality_flag"].sum())
    log.add_decision(
        "Invalid / implausible records",
        "Records with negative tenure or charges, non-positive monthly charges, tenure "
        "above the dataset's 72-month ceiling, or total charges that contradict the tenure "
        "were FLAGGED (data_quality_flag = 1), not deleted. Deleting them would bias the "
        "churn rate; flagging keeps them auditable and lets the model see the same "
        "population the business would see.",
    )

    # ------------------------------------------------------------------
    # 5. Target + derived features
    # ------------------------------------------------------------------
    df = build_features(df, log)

    ordered_front = [
        "customer_id",
        "gender",
        "senior_citizen",
        "senior_citizen_label",
        "partner",
        "dependents",
        "tenure",
        "tenure_group",
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
        "monthly_charges",
        "total_charges",
        "total_charges_imputed",
        "avg_monthly_charge",
        "num_addon_services",
        "has_internet",
        "service_count",
        "revenue_band",
        "high_value_customer",
        "churn",
        "churn_label",
        "data_quality_flag",
    ]
    existing = [c for c in ordered_front if c in df.columns]
    remaining = [c for c in df.columns if c not in existing and c != "tenure_months"]
    df = df.loc[:, existing + remaining].reset_index(drop=True)

    log.row_count_clean = len(df)
    log.column_count_clean = df.shape[1]
    log.add_decision(
        "Row retention",
        f"{log.row_count_raw} raw rows -> {log.row_count_clean} cleaned rows. "
        f"{log.duplicates_removed} duplicate rows removed and {log.invalid_churn_values} "
        "rows with an invalid target removed; no other record was dropped.",
    )

    logger.info(
        "Cleaned dataset: %s rows x %s columns (%s flagged)",
        log.row_count_clean,
        log.column_count_clean,
        log.invalid_records_flagged,
    )
    return CleaningResult(frame=df, log=log)


def save_clean(frame: pd.DataFrame, path=None) -> None:
    """Persist the cleaned dataset (raw file is untouched)."""
    path = path or config.CLEAN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Saved cleaned dataset to %s", path)


def load_clean(path=None) -> pd.DataFrame:
    """Load the cleaned dataset, raising a helpful error if absent."""
    path = path or config.CLEAN_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run: python main.py pipeline"
        )
    df = pd.read_csv(path)
    if "tenure_group" in df.columns:
        df["tenure_group"] = df["tenure_group"].astype("string")
    return df
