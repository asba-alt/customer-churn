"""Customer-level analysis: segments, demographics and service adoption."""

from __future__ import annotations

import pandas as pd

TENURE_ORDER = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]


def demographic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Churn by the demographic combinations that exist in the data."""
    grouped = (
        df.groupby(["senior_citizen_label", "partner", "dependents"], observed=True)
        .agg(
            customers=("customer_id", "count"),
            churned=("churn_label", "sum"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["customers"] >= 25].copy()
    grouped["churn_rate"] = (grouped["churned"] / grouped["customers"]).round(4)
    grouped["avg_monthly_charges"] = grouped["avg_monthly_charges"].round(2)
    grouped["avg_tenure"] = grouped["avg_tenure"].round(1)
    grouped["segment_label"] = (
        "Senior: "
        + grouped["senior_citizen_label"].astype(str)
        + " | Partner: "
        + grouped["partner"].map({1: "Yes", 0: "No"}).astype(str)
        + " | Dependents: "
        + grouped["dependents"].map({1: "Yes", 0: "No"}).astype(str)
    )
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def service_adoption_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Adoption rate and churn rate for every service feature.

    'No internet service' / 'No phone service' are reported separately from a
    plain 'No' because they mean the product was never available, not declined.
    """
    service_cols = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "internet_service",
        "paperless_billing",
    ]
    rows = []
    for col in service_cols:
        series = df[col]
        if col in ("phone_service", "paperless_billing"):
            labels = {1: "Yes", 0: "No"}
            series = series.map(labels)
        for value, group in df.groupby(series, observed=True):
            rows.append(
                {
                    "service": col,
                    "value": str(value),
                    "customers": len(group),
                    "adoption_pct": round(len(group) / len(df) * 100, 2),
                    "churn_rate": round(float(group["churn_label"].mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def customer_segments(df: pd.DataFrame, min_customers: int = 40) -> pd.DataFrame:
    """Business-ready segment table: contract x value band.

    Segments below ``min_customers`` are excluded from the ranking so the
    dashboard never shows a "100% churn" cell backed by four people.
    """
    grouped = (
        df.groupby(["contract", "revenue_band"], observed=True)
        .agg(
            customers=("customer_id", "count"),
            churned=("churn_label", "sum"),
            monthly_revenue=("monthly_charges", "sum"),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_addons=("num_addon_services", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["customers"] >= min_customers].copy()
    grouped["churn_rate"] = (grouped["churned"] / grouped["customers"]).round(4)
    grouped["monthly_revenue"] = grouped["monthly_revenue"].round(2)
    grouped["avg_tenure"] = grouped["avg_tenure"].round(1)
    grouped["avg_monthly_charges"] = grouped["avg_monthly_charges"].round(2)
    grouped["avg_addons"] = grouped["avg_addons"].round(2)
    grouped["revenue_at_risk"] = (
        grouped["monthly_revenue"] * grouped["churn_rate"]
    ).round(2)
    grouped["segment"] = grouped["contract"].astype(str) + " / " + grouped["revenue_band"].astype(str)
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def tenure_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive tenure statistics for retained versus churned customers."""
    stats = (
        df.groupby("churn", observed=True)
        .agg(
            customers=("customer_id", "count"),
            tenure_mean=("tenure", "mean"),
            tenure_median=("tenure", "median"),
            tenure_min=("tenure", "min"),
            tenure_max=("tenure", "max"),
            monthly_charges_mean=("monthly_charges", "mean"),
            total_charges_mean=("total_charges", "mean"),
        )
        .reset_index()
    )
    for col in stats.columns[2:]:
        stats[col] = stats[col].round(2)
    return stats
