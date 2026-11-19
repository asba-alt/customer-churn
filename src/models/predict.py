"""Scoring: generate churn probabilities and risk tiers for every customer."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from .preprocessing import assert_no_leakage, coerce_feature_types, select_features
from .train import load_pipeline

logger = logging.getLogger(__name__)


def assign_risk_tiers(
    probabilities: np.ndarray | pd.Series,
    reference_probabilities: np.ndarray | pd.Series,
    shares: dict[str, float] | None = None,
) -> pd.Series:
    """Assign risk tiers using quantile cut-offs learned from a reference set.

    The cut-offs come from the **validation split**, not from the scored rows, so
    the tier boundaries do not move depending on which batch is scored. "Very
    High" is the top 5% of the reference distribution, "High" the next 15%, and
    so on.
    """
    shares = shares or config.RISK_TIER_SHARES
    reference = np.asarray(reference_probabilities, dtype=float)

    very_high_cut = float(np.quantile(reference, 1 - shares["Very High"]))
    high_cut = float(np.quantile(reference, 1 - shares["High"]))
    medium_cut = float(np.quantile(reference, 1 - shares["Medium"]))

    probs = np.asarray(probabilities, dtype=float)
    tiers = np.full(len(probs), "Low", dtype=object)
    tiers[probs >= medium_cut] = "Medium"
    tiers[probs >= high_cut] = "High"
    tiers[probs >= very_high_cut] = "Very High"

    # Carry the input index through. Without this, assigning the result onto a
    # frame with a non-contiguous index (e.g. a filtered subset) silently
    # produces NaN through label alignment.
    index = probabilities.index if isinstance(probabilities, pd.Series) else None
    return pd.Series(
        pd.Categorical(tiers, categories=list(config.RISK_TIER_ORDER), ordered=True),
        index=index,
        name="risk_category",
    )


def tier_thresholds(reference_probabilities: np.ndarray | pd.Series) -> dict[str, float]:
    """Expose the tier cut-offs so they can be reported and tested."""
    reference = np.asarray(reference_probabilities, dtype=float)
    return {
        "very_high_min_probability": round(float(np.quantile(reference, 1 - config.RISK_TIER_SHARES["Very High"])), 4),
        "high_min_probability": round(float(np.quantile(reference, 1 - config.RISK_TIER_SHARES["High"])), 4),
        "medium_min_probability": round(float(np.quantile(reference, 1 - config.RISK_TIER_SHARES["Medium"])), 4),
    }


def score_customers(
    clean: pd.DataFrame,
    pipeline: Pipeline | None = None,
    threshold: float | None = None,
    reference_probabilities: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Score every customer and attach risk category, revenue impact and detail.

    ``reference_probabilities`` should be the validation-split probabilities. If
    omitted, the scored population's own distribution is used (fine for a
    one-off report, less appropriate across batches).
    """
    if pipeline is None or threshold is None:
        pipeline, threshold, _ = load_pipeline()

    features = coerce_feature_types(select_features(clean))
    assert_no_leakage(features)

    probabilities = pd.Series(
        pipeline.predict_proba(features)[:, 1],
        index=clean.index,
        name="churn_probability",
    )
    reference = probabilities if reference_probabilities is None else reference_probabilities

    out = clean[
        [
            "customer_id",
            "contract",
            "tenure",
            "tenure_group",
            "monthly_charges",
            "total_charges",
            "internet_service",
            "payment_method",
            "online_security",
            "tech_support",
            "num_addon_services",
            "service_count",
            "revenue_band",
            "high_value_customer",
            "churn",
            "churn_label",
        ]
    ].copy()

    out["churn_probability"] = probabilities.round(6)
    out["risk_category"] = assign_risk_tiers(probabilities, reference).astype(str)
    out["predicted_churn"] = (probabilities >= threshold).astype(int)
    out["decision_threshold"] = round(float(threshold), 4)
    out["monthly_revenue_exposed"] = np.round(out["monthly_charges"], 2)
    out["expected_monthly_revenue_at_risk"] = np.round(
        out["monthly_charges"] * out["churn_probability"], 2
    )
    out["expected_annual_revenue_at_risk"] = np.round(
        out["monthly_charges"] * out["churn_probability"] * 12, 2
    )
    out["actual_churn"] = out["churn_label"]

    # Guard: an index-alignment slip would leave tiers as NaN, which would
    # silently corrupt every downstream risk summary and dashboard slice.
    if out["risk_category"].isna().any():
        raise RuntimeError(
            f"risk_category has {int(out['risk_category'].isna().sum())} missing values "
            "after scoring — check index alignment between probabilities and the frame."
        )

    out = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    out["risk_rank"] = np.arange(1, len(out) + 1)
    out["risk_percentile"] = (out["risk_rank"] / len(out) * 100).round(2)

    logger.info(
        "Scored %s customers | mean probability %.4f | predicted churners %s",
        len(out), float(np.mean(probabilities)), int(out["predicted_churn"].sum()),
    )
    return out


def save_scores(scored: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else config.SCORED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    logger.info("Saved customer risk scores to %s", path)
    return path


def load_scores(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else config.SCORED_FILE
    if not path.exists():
        raise FileNotFoundError(f"No scored file at {path}. Run: python main.py score")
    return pd.read_csv(path)


def risk_distribution(scored: pd.DataFrame) -> pd.DataFrame:
    """Counts and revenue by risk tier."""
    out = (
        scored.groupby("risk_category", observed=True)
        .agg(
            customers=("customer_id", "count"),
            mean_probability=("churn_probability", "mean"),
            monthly_revenue=("monthly_charges", "sum"),
            expected_monthly_at_risk=("expected_monthly_revenue_at_risk", "sum"),
            actual_churn_rate=("actual_churn", "mean"),
        )
        .reset_index()
    )
    out["pct_of_customers"] = (out["customers"] / len(scored) * 100).round(2)
    out["mean_probability"] = out["mean_probability"].round(4)
    out["monthly_revenue"] = out["monthly_revenue"].round(2)
    out["expected_monthly_at_risk"] = out["expected_monthly_at_risk"].round(2)
    out["actual_churn_rate"] = out["actual_churn_rate"].round(4)
    order = {tier: i for i, tier in enumerate(config.RISK_TIER_ORDER)}
    out["__o"] = out["risk_category"].map(order)
    return out.sort_values("__o").drop(columns="__o").reset_index(drop=True)
