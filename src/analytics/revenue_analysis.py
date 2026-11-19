"""Revenue analysis: exposure, contribution and defensible value proxies.

Important framing used throughout: "revenue at risk" means the **monthly
recurring charge attached to customers who churned (or who are predicted to
churn)**. It is not realised loss, not annual loss, and not saved revenue.
"""

from __future__ import annotations

import pandas as pd

import config


def revenue_summary(df: pd.DataFrame) -> dict[str, float]:
    """Headline revenue KPIs computed from the actual dataset."""
    churned = df[df["churn_label"] == 1]
    retained = df[df["churn_label"] == 0]
    total_monthly = float(df["monthly_charges"].sum())

    return {
        "total_customers": int(len(df)),
        "churned_customers": int(len(churned)),
        "retained_customers": int(len(retained)),
        "overall_churn_rate": round(float(df["churn_label"].mean()), 4),
        "avg_tenure": round(float(df["tenure"].mean()), 2),
        "avg_monthly_charges": round(float(df["monthly_charges"].mean()), 2),
        "total_monthly_revenue": round(total_monthly, 2),
        "monthly_revenue_churned": round(float(churned["monthly_charges"].sum()), 2),
        "monthly_revenue_retained": round(float(retained["monthly_charges"].sum()), 2),
        "revenue_share_churned": round(float(churned["monthly_charges"].sum()) / total_monthly, 4),
        "avg_monthly_charge_churned": round(float(churned["monthly_charges"].mean()), 2),
        "avg_monthly_charge_retained": round(float(retained["monthly_charges"].mean()), 2),
        "total_realised_revenue": round(float(df["total_charges"].sum()), 2),
        "realised_revenue_churned": round(float(churned["total_charges"].sum()), 2),
        "avg_lifetime_value_realised_churned": round(float(churned["total_charges"].mean()), 2),
        "avg_lifetime_value_realised_retained": round(float(retained["total_charges"].mean()), 2),
        "high_value_customers": int(df["high_value_customer"].sum()),
        "high_value_churn_rate": round(float(df.loc[df["high_value_customer"] == 1, "churn_label"].mean()), 4),
        "annualised_revenue_proxy": round(float(df["monthly_charges"].sum()) * 12, 2),
    }


def revenue_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue exposure by contract type and by revenue band."""
    by_contract = (
        df.groupby("contract", observed=True)
        .agg(
            customers=("customer_id", "count"),
            monthly_revenue=("monthly_charges", "sum"),
            churned=("churn_label", "sum"),
        )
        .reset_index()
    )
    by_contract["dimension"] = "contract"
    by_contract = by_contract.rename(columns={"contract": "segment"})

    by_band = (
        df.groupby("revenue_band", observed=True)
        .agg(
            customers=("customer_id", "count"),
            monthly_revenue=("monthly_charges", "sum"),
            churned=("churn_label", "sum"),
        )
        .reset_index()
    )
    by_band["dimension"] = "revenue_band"
    by_band = by_band.rename(columns={"revenue_band": "segment"})

    combined = pd.concat([by_contract, by_band], ignore_index=True)
    at_risk = (
        df[df["churn_label"] == 1]
        .groupby("contract", observed=True)["monthly_charges"]
        .sum()
        .rename("at_risk_contract")
        .reset_index()
        .rename(columns={"contract": "segment"})
    )
    at_risk_band = (
        df[df["churn_label"] == 1]
        .groupby("revenue_band", observed=True)["monthly_charges"]
        .sum()
        .rename("at_risk_band")
        .reset_index()
        .rename(columns={"revenue_band": "segment"})
    )
    combined = combined.merge(at_risk, on="segment", how="left").merge(
        at_risk_band, on="segment", how="left"
    )
    combined["monthly_revenue_at_risk"] = combined["at_risk_contract"].fillna(
        combined["at_risk_band"]
    )
    combined = combined.drop(columns=["at_risk_contract", "at_risk_band"])
    combined["churn_rate"] = (combined["churned"] / combined["customers"]).round(4)
    combined["monthly_revenue"] = combined["monthly_revenue"].round(2)
    combined["monthly_revenue_at_risk"] = combined["monthly_revenue_at_risk"].round(2)
    combined["revenue_share_pct"] = (
        combined["monthly_revenue"] / combined["monthly_revenue"].sum() * 100
    ).round(2)
    return combined[["dimension", "segment", "customers", "churned", "churn_rate",
                     "monthly_revenue", "monthly_revenue_at_risk", "revenue_share_pct"]]


def revenue_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue contribution and churn per business segment."""
    grouped = (
        df.groupby(["contract", "revenue_band"], observed=True)
        .agg(
            customers=("customer_id", "count"),
            monthly_revenue=("monthly_charges", "sum"),
            churned=("churn_label", "sum"),
            avg_tenure=("tenure", "mean"),
        )
        .reset_index()
    )
    total_revenue = grouped["monthly_revenue"].sum()
    grouped["revenue_share_pct"] = (grouped["monthly_revenue"] / total_revenue * 100).round(2)
    grouped["churn_rate"] = (grouped["churned"] / grouped["customers"]).round(4)
    grouped["revenue_at_risk"] = (grouped["monthly_revenue"] * grouped["churn_rate"]).round(2)
    grouped["avg_tenure"] = grouped["avg_tenure"].round(1)
    return grouped.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)


def customer_value_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Two transparent value proxies, explicitly labelled as proxies.

    * ``realised_value_to_date`` = total_charges (factual, billed to date)
    * ``annualised_value_proxy`` = monthly_charges * 12 (extrapolation)

    Neither is discounted or survival-adjusted. The dataset has no observation
    dates and no cancellation timing, so a true CLV model is not identifiable.
    """
    out = df[
        ["customer_id", "contract", "tenure", "monthly_charges", "total_charges", "churn", "churn_label"]
    ].copy()
    out["realised_value_to_date"] = out["total_charges"].round(2)
    out["annualised_value_proxy"] = (out["monthly_charges"] * 12).round(2)
    out["value_quintile"] = pd.qcut(
        out["monthly_charges"], 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)  # 1 = highest spenders

    summary = (
        out.groupby("value_quintile", observed=True)
        .agg(
            customers=("customer_id", "count"),
            min_monthly_charges=("monthly_charges", "min"),
            max_monthly_charges=("monthly_charges", "max"),
            avg_annualised_value=("annualised_value_proxy", "mean"),
            avg_realised_value=("realised_value_to_date", "mean"),
            churned=("churn_label", "sum"),
        )
        .reset_index()
    )
    summary["churn_rate"] = (summary["churned"] / summary["customers"]).round(4)
    summary["annualised_value_churned"] = (
        out[out["churn_label"] == 1]
        .groupby("value_quintile", observed=True)["annualised_value_proxy"]
        .sum()
        .reindex(summary["value_quintile"])
        .values
    ).round(2)
    for col in ["min_monthly_charges", "max_monthly_charges", "avg_annualised_value", "avg_realised_value"]:
        summary[col] = summary[col].round(2)
    return summary


def revenue_concentration(df: pd.DataFrame) -> dict[str, float]:
    """How concentrated is monthly revenue in the top-spending customers?"""
    charges = df["monthly_charges"].sort_values(ascending=False).reset_index(drop=True)
    total = charges.sum()
    n = len(charges)

    def share(pct: float) -> float:
        k = max(1, int(round(n * pct)))
        return round(float(charges.head(k).sum() / total), 4)

    return {
        "total_monthly_revenue": round(float(total), 2),
        "top_1pct_revenue_share": share(0.01),
        "top_5pct_revenue_share": share(0.05),
        "top_10pct_revenue_share": share(0.10),
        "top_20pct_revenue_share": share(0.20),
        "revenue_per_customer_monthly": round(float(df["monthly_charges"].mean()), 2),
        "high_value_threshold": round(
            float(df["monthly_charges"].quantile(config.HIGH_VALUE_PERCENTILE)), 2
        ),
    }
