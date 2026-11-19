"""Churn analysis: which attributes are associated with churn.

All functions are pure (DataFrame in, DataFrame out) so they can be unit
tested independently of the database and of plotting.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TENURE_ORDER = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]


def _rate_table(
    df: pd.DataFrame,
    by: list[str],
    order: list[str] | None = None,
    min_customers: int = 0,
) -> pd.DataFrame:
    """Shared aggregation: customers, churners, churn rate, revenue, revenue at risk."""
    grouped = (
        df.groupby(by, observed=True)
        .agg(
            customers=("customer_id", "count"),
            churned=("churn_label", "sum"),
            monthly_revenue=("monthly_charges", "sum"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .reset_index()
    )
    at_risk = (
        df[df["churn_label"] == 1]
        .groupby(by, observed=True)["monthly_charges"]
        .sum()
        .rename("monthly_revenue_at_risk")
        .reset_index()
    )
    grouped = grouped.merge(at_risk, on=by, how="left").fillna({"monthly_revenue_at_risk": 0.0})

    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    grouped["monthly_revenue"] = grouped["monthly_revenue"].round(2)
    grouped["monthly_revenue_at_risk"] = grouped["monthly_revenue_at_risk"].round(2)
    grouped["avg_monthly_charges"] = grouped["avg_monthly_charges"].round(2)
    grouped["avg_tenure"] = grouped["avg_tenure"].round(1)
    grouped["baseline_churn_rate"] = float(df["churn_label"].mean())
    grouped["churn_rate_vs_baseline_pts"] = (
        (grouped["churn_rate"] - grouped["baseline_churn_rate"]) * 100
    ).round(2)
    grouped["lift_vs_baseline"] = (grouped["churn_rate"] / grouped["baseline_churn_rate"]).round(2)

    if min_customers:
        grouped = grouped[grouped["customers"] >= min_customers].copy()

    if order is not None:
        grouped["__order"] = grouped[by[0]].map({v: i for i, v in enumerate(order)})
        grouped = grouped.sort_values("__order").drop(columns="__order")
    else:
        grouped = grouped.sort_values("churn_rate", ascending=False)

    return grouped.reset_index(drop=True)


def churn_by_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per contract type, ordered by commitment length."""
    order = ["Month-to-month", "One year", "Two year"]
    out = _rate_table(df, ["contract"], order=order)
    out["pct_of_customers"] = (out["customers"] / len(df) * 100).round(2)
    return out


def churn_by_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per tenure band, ordered chronologically."""
    return _rate_table(df, ["tenure_group"], order=TENURE_ORDER)


def churn_by_payment_method(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per payment method, split by automatic vs manual payment."""
    out = _rate_table(df, ["payment_method"])
    out["auto_payment"] = out["payment_method"].str.contains("automatic", case=False).astype(int)
    return out


def churn_by_internet_service(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per internet service tier."""
    return _rate_table(df, ["internet_service"])


def churn_by_addon(df: pd.DataFrame) -> pd.DataFrame:
    """Does having a specific add-on associate with lower churn?

    Reports the with/without churn rates and the difference in percentage points
    for each add-on. No causal claim: customers who buy support may already be
    more engaged, so this is an association, not a treatment effect.
    """
    rows = []
    baseline = float(df["churn_label"].mean())
    for col in ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]:
        has = df[df[col] == "Yes"]
        has_not = df[df[col] == "No"]
        rows.append(
            {
                "addon": col.replace("_", " ").title(),
                "customers_with": len(has),
                "customers_without": len(has_not),
                "churn_rate_with": round(float(has["churn_label"].mean()) if len(has) else float("nan"), 4),
                "churn_rate_without": round(float(has_not["churn_label"].mean()) if len(has_not) else float("nan"), 4),
                "difference_pts": round(
                    (float(has["churn_label"].mean()) - float(has_not["churn_label"].mean())) * 100, 2
                )
                if len(has) and len(has_not)
                else float("nan"),
                "baseline_churn_rate": round(baseline, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("difference_pts")


def churn_driver_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rank categorical values by how far their churn rate sits from the baseline.

    This is a descriptive screen, not a causal model: it shows which single
    attributes carry the strongest raw signal before modelling.
    """
    baseline = float(df["churn_label"].mean())
    categorical_cols = [
        "contract",
        "internet_service",
        "payment_method",
        "tenure_group",
        "online_security",
        "tech_support",
        "multiple_lines",
    ]
    rows = []
    for col in categorical_cols:
        for value, group in df.groupby(col, observed=True):
            if len(group) < 50:
                continue
            rate = float(group["churn_label"].mean())
            rows.append(
                {
                    "feature": col,
                    "value": str(value),
                    "customers": len(group),
                    "churn_rate": round(rate, 4),
                    "vs_baseline_pts": round((rate - baseline) * 100, 2),
                    "lift": round(rate / baseline, 2),
                }
            )
    out = pd.DataFrame(rows)
    return out.reindex(out["vs_baseline_pts"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def churn_by_segment_combination(df: pd.DataFrame, min_customers: int = 50) -> pd.DataFrame:
    """Contract x internet service x payment method combinations, ranked by churn."""
    out = _rate_table(
        df,
        ["contract", "internet_service", "payment_method"],
        min_customers=min_customers,
    )
    out["segment"] = out["contract"] + " | " + out["internet_service"] + " | " + out["payment_method"]
    return out
