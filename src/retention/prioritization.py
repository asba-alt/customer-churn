"""Retention prioritisation under limited outreach capacity.

The business question: *if we can call only N% of customers, who first?*

The evaluation is done honestly on the **hold-out test set**, where the actual
churn outcome is known. That lets us report realised lift rather than a
simulated one. On the production scoring file the outcome column is simply
unavailable at decision time; nothing here assumes it is known operationally.

Terminology used strictly:

* **predicted churn risk** — the model's probability for a customer.
* **actual churn** — whether the customer in fact left (only known post hoc).
* **revenue exposed** — the monthly recurring charge attached to prioritised
  customers. NOT saved revenue: no campaign was run, no save rate is known.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def build_priority_list(
    scored: pd.DataFrame,
    capacity: float = config.PRIMARY_CAPACITY,
    revenue_column: str = "monthly_charges",
) -> pd.DataFrame:
    """Top-``capacity`` share of customers by predicted risk, with a call list.

    Rankings are made on predicted probability only. If actual outcomes happen to
    be present they are carried along for evaluation, never used for sorting.
    """
    if not 0 < capacity <= 1:
        raise ValueError(f"capacity must be in (0, 1], got {capacity}")

    n = len(scored)
    k = max(1, int(round(n * capacity)))
    ranked = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    priority = ranked.head(k).copy()
    priority["priority_rank"] = np.arange(1, k + 1)
    priority["priority_decile"] = pd.qcut(
        priority["priority_rank"], min(10, k), labels=False, duplicates="drop"
    ) + 1
    priority["capacity_share"] = capacity
    priority["call_action"] = priority["risk_category"].map(
        {
            "Very High": "Immediate retention call + contract offer",
            "High": "Retention call within 7 days",
            "Medium": "Automated offer / monitoring",
            "Low": "No action",
        }
    ).fillna("Review")

    logger.info("Priority list: %s customers (%.1f%% of base)", len(priority), capacity * 100)
    return priority


def lift_at_capacity(
    scored: pd.DataFrame,
    capacities: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
    outcome_column: str = "actual_churn",
    revenue_column: str = "monthly_charges",
) -> pd.DataFrame:
    """Lift table across several capacity levels.

    ``lift`` = churn rate inside the selected group / overall churn rate.
    A value of 3.0 means the prioritised group churns three times as often as the
    base, i.e. the ranking is doing real work.
    """
    if outcome_column not in scored.columns:
        raise ValueError(f"'{outcome_column}' not present; lift needs observed outcomes")

    overall_rate = float(scored[outcome_column].mean())
    overall_revenue = float(scored[revenue_column].sum())
    ranked = scored.sort_values("churn_probability", ascending=False)
    n = len(ranked)

    rows = []
    for capacity in capacities:
        k = max(1, int(round(n * capacity)))
        selected = ranked.head(k)
        selected_rate = float(selected[outcome_column].mean())
        selected_revenue = float(selected[revenue_column].sum())
        churners_caught = int(selected[outcome_column].sum())
        rows.append(
            {
                "capacity": capacity,
                "customers_selected": k,
                "churners_in_group": churners_caught,
                "churn_rate_in_group": round(selected_rate, 4),
                "overall_churn_rate": round(overall_rate, 4),
                "lift": round(selected_rate / overall_rate, 3) if overall_rate else float("nan"),
                "recall_of_churners": round(
                    churners_caught / max(int(scored[outcome_column].sum()), 1), 4
                ),
                "precision_of_selection": round(selected_rate, 4),
                "monthly_revenue_in_group": round(selected_revenue, 2),
                "revenue_share_in_group": round(selected_revenue / overall_revenue, 4)
                if overall_revenue
                else float("nan"),
                "expected_monthly_revenue_at_risk": round(
                    float(selected["expected_monthly_revenue_at_risk"].sum()), 2
                )
                if "expected_monthly_revenue_at_risk" in selected.columns
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def cumulative_gains(
    scored: pd.DataFrame,
    outcome_column: str = "actual_churn",
    points: int = 101,
) -> pd.DataFrame:
    """Cumulative gain and lift curve, used for the gains chart and the dashboard."""
    overall_rate = float(scored[outcome_column].mean())
    ranked = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    n = len(ranked)
    total_churners = int(ranked[outcome_column].sum())
    cumulative_churners = ranked[outcome_column].cumsum()

    fractions = np.linspace(0, 1, points)
    idx = np.clip((fractions * n).astype(int) - 1, 0, n - 1)
    rows = []
    for frac, i in zip(fractions, idx):
        customers = i + 1
        caught = int(cumulative_churners.iloc[i])
        recall = caught / total_churners if total_churners else 0.0
        rows.append(
            {
                "population_share": round(float(frac), 4),
                "customers_contacted": customers,
                "churners_captured": caught,
                "cumulative_recall": round(recall, 4),
                "cumulative_lift": round(recall / frac, 3) if frac > 0 else float("nan"),
                "overall_churn_rate": round(overall_rate, 4),
            }
        )
    return pd.DataFrame(rows)


def prioritisation_summary(
    scored: pd.DataFrame,
    capacity: float = config.PRIMARY_CAPACITY,
    outcome_column: str = "actual_churn",
) -> dict:
    """Everything the retention slide needs, in one dictionary."""
    priority = build_priority_list(scored, capacity=capacity)
    lift_table = lift_at_capacity(scored, capacities=(capacity,), outcome_column=outcome_column)
    row = lift_table.iloc[0]

    has_outcome = outcome_column in scored.columns
    summary = {
        "capacity": capacity,
        "customers_prioritised": int(len(priority)),
        "share_of_base_pct": round(len(priority) / len(scored) * 100, 2),
        "avg_predicted_probability": round(float(priority["churn_probability"].mean()), 4),
        "min_predicted_probability_in_group": round(float(priority["churn_probability"].min()), 4),
        "overall_avg_predicted_probability": round(float(scored["churn_probability"].mean()), 4),
        "monthly_revenue_in_group": round(float(priority["monthly_charges"].sum()), 2),
        "expected_monthly_revenue_at_risk": round(
            float(priority["expected_monthly_revenue_at_risk"].sum()), 2
        ),
        "expected_annual_revenue_at_risk": round(
            float(priority["expected_annual_revenue_at_risk"].sum()), 2
        )
        if "expected_annual_revenue_at_risk" in priority.columns
        else None,
        "share_of_total_monthly_revenue_pct": round(
            float(priority["monthly_charges"].sum()) / float(scored["monthly_charges"].sum()) * 100, 2
        ),
        "contract_mix": priority["contract"].value_counts(normalize=True).round(4).to_dict()
        if "contract" in priority.columns
        else {},
        "risk_tier_mix": priority["risk_category"].value_counts().to_dict()
        if "risk_category" in priority.columns
        else {},
        "has_observed_outcomes": has_outcome,
    }

    if row is not None:
        summary.update(
            {
                "lift_at_capacity": float(row["lift"]),
                "churn_rate_in_group": float(row["churn_rate_in_group"]) if has_outcome else None,
                "overall_churn_rate": float(row["overall_churn_rate"]) if has_outcome else None,
                "churners_captured": int(row["churners_in_group"]) if has_outcome else None,
                "recall_of_churners": float(row["recall_of_churners"]) if has_outcome else None,
            }
        )
    return summary


def segment_priority_profile(priority: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Which segments dominate the priority list, and what are they worth?

    Useful for deciding *how* to contact them: a month-to-month fibre cohort needs
    a different offer than a two-year-contract cohort.
    """
    out = (
        priority.groupby(["contract", "internet_service", "risk_category"], observed=True)
        .agg(
            customers=("customer_id", "count"),
            avg_probability=("churn_probability", "mean"),
            monthly_revenue=("monthly_charges", "sum"),
            avg_tenure=("tenure", "mean"),
            scorable_churners=("actual_churn", "sum")
            if "actual_churn" in priority.columns
            else ("customer_id", "count"),
        )
        .reset_index()
    )
    out["avg_probability"] = out["avg_probability"].round(4)
    out["monthly_revenue"] = out["monthly_revenue"].round(2)
    out["avg_tenure"] = out["avg_tenure"].round(1)
    out["share_of_priority_list_pct"] = (out["customers"] / len(priority) * 100).round(2)
    return out.sort_values("customers", ascending=False).head(top_n).reset_index(drop=True)


def outreach_scenarios(
    scored: pd.DataFrame,
    capacities: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
    outcome_column: str = "actual_churn",
) -> pd.DataFrame:
    """Capacity planning table: how much risk each team size can absorb."""
    rows = []
    for capacity in capacities:
        priority = build_priority_list(scored, capacity=capacity)
        row = {
            "capacity": capacity,
            "customers_contacted": len(priority),
            "mean_predicted_probability": round(float(priority["churn_probability"].mean()), 4),
            "monthly_revenue_covered": round(float(priority["monthly_charges"].sum()), 2),
            "expected_monthly_revenue_at_risk": round(
                float(priority["expected_monthly_revenue_at_risk"].sum()), 2
            ),
            "smallest_probability_contacted": round(float(priority["churn_probability"].min()), 4),
        }
        if outcome_column in scored.columns:
            selected = scored.sort_values("churn_probability", ascending=False).head(len(priority))
            caught = int(selected[outcome_column].sum())
            total = int(scored[outcome_column].sum())
            row["actual_churn_rate_in_group"] = round(float(selected[outcome_column].mean()), 4)
            row["churners_captured"] = caught
            row["recall_of_all_churners"] = round(caught / max(total, 1), 4)
            row["lift"] = round(
                float(selected[outcome_column].mean()) / float(scored[outcome_column].mean()), 3
            )
        rows.append(row)
    return pd.DataFrame(rows)
