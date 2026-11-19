"""Tests for scoring, risk tiers and retention prioritisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.models.predict import (
    assign_risk_tiers,
    risk_distribution,
    score_customers,
    tier_thresholds,
)
from src.retention.prioritization import (
    build_priority_list,
    cumulative_gains,
    lift_at_capacity,
    outreach_scenarios,
    prioritisation_summary,
    segment_priority_profile,
)


# ----------------------------------------------------------------------
# Scoring output
# ----------------------------------------------------------------------
def test_scores_cover_every_customer(scored_frame, clean_frame):
    assert len(scored_frame) == len(clean_frame)
    assert scored_frame["customer_id"].is_unique


def test_probabilities_are_valid(scored_frame):
    assert scored_frame["churn_probability"].between(0, 1).all()
    assert scored_frame["churn_probability"].notna().all()


def test_scored_frame_is_sorted_by_risk(scored_frame):
    probabilities = scored_frame["churn_probability"].tolist()
    assert probabilities == sorted(probabilities, reverse=True)


def test_risk_rank_starts_at_one(scored_frame):
    assert scored_frame["risk_rank"].min() == 1
    assert scored_frame["risk_rank"].max() == len(scored_frame)


def test_risk_categories_use_known_labels(scored_frame):
    assert set(scored_frame["risk_category"].unique()) <= set(config.RISK_TIER_ORDER)


def test_risk_tiers_are_ordered_by_probability(scored_frame):
    order = list(config.RISK_TIER_ORDER)
    means = scored_frame.groupby("risk_category", observed=True)["churn_probability"].mean()
    present = [t for t in order if t in means.index]
    assert means[present].is_monotonic_decreasing


def test_predicted_churn_matches_threshold(scored_frame):
    threshold = scored_frame["decision_threshold"].iloc[0]
    expected = (scored_frame["churn_probability"] >= threshold).astype(int)
    assert (scored_frame["predicted_churn"] == expected).all()


def test_expected_revenue_is_probability_weighted(scored_frame):
    expected = scored_frame["monthly_charges"] * scored_frame["churn_probability"]
    assert np.allclose(scored_frame["expected_monthly_revenue_at_risk"], expected, atol=0.02)


def test_expected_revenue_never_exceeds_actual_charge(scored_frame):
    assert (scored_frame["expected_monthly_revenue_at_risk"] <= scored_frame["monthly_charges"] + 1e-6).all()


# ----------------------------------------------------------------------
# Risk tiers
# ----------------------------------------------------------------------
def test_assign_risk_tiers_uses_reference_quantiles():
    reference = np.linspace(0.01, 0.99, 1000)
    probabilities = np.array([0.999, 0.9, 0.7, 0.2])
    tiers = assign_risk_tiers(probabilities, reference)
    assert tiers.iloc[0] == "Very High"
    assert tiers.iloc[3] == "Low"


def test_assign_risk_tiers_preserves_series_index():
    """Regression: a non-contiguous index must survive tier assignment.

    Assigning a RangeIndex result onto a filtered frame silently produced NaN
    through label alignment, which corrupted the test-set export.
    """
    reference = np.linspace(0.01, 0.99, 1000)
    probabilities = pd.Series([0.99, 0.05], index=[900, 901])
    tiers = assign_risk_tiers(probabilities, reference)
    assert list(tiers.index) == [900, 901]
    assert not tiers.isna().any()


def test_scoring_a_filtered_subset_produces_no_nulls(clean_frame):
    """Scoring a non-contiguous slice must fully populate every derived column."""
    from src.models.train import load_pipeline

    subset = clean_frame.iloc[[3, 17, 42, 101, 500]].copy()
    subset.index = [40, 91, 305, 777, 1200]  # deliberately non-contiguous

    pipeline, threshold, _ = load_pipeline()
    rescored = score_customers(subset, pipeline, threshold)
    assert rescored["risk_category"].isna().sum() == 0
    assert rescored["churn_probability"].notna().all()
    assert set(rescored["risk_category"].unique()) <= set(config.RISK_TIER_ORDER)


def test_tier_thresholds_are_descending():
    reference = np.linspace(0, 1, 1000)
    thresholds = tier_thresholds(reference)
    assert thresholds["very_high_min_probability"] > thresholds["high_min_probability"]
    assert thresholds["high_min_probability"] > thresholds["medium_min_probability"]


def test_risk_distribution_covers_all_customers(scored_frame):
    distribution = risk_distribution(scored_frame)
    assert distribution["customers"].sum() == len(scored_frame)
    assert distribution["pct_of_customers"].sum() == pytest.approx(100, abs=0.1)


# ----------------------------------------------------------------------
# Priority list
# ----------------------------------------------------------------------
def test_priority_list_size_matches_capacity(scored_frame):
    for capacity in (0.05, 0.10, 0.25):
        priority = build_priority_list(scored_frame, capacity=capacity)
        expected = max(1, int(round(len(scored_frame) * capacity)))
        assert len(priority) == expected


def test_priority_list_takes_highest_risk(scored_frame):
    priority = build_priority_list(scored_frame, capacity=0.10)
    cutoff = priority["churn_probability"].min()
    non_priority = scored_frame[~scored_frame["customer_id"].isin(priority["customer_id"])]
    assert non_priority["churn_probability"].max() <= cutoff + 1e-9


def test_priority_list_is_ranked(scored_frame):
    priority = build_priority_list(scored_frame, capacity=0.10)
    assert priority["priority_rank"].tolist() == list(range(1, len(priority) + 1))


def test_invalid_capacity_is_rejected(scored_frame):
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            build_priority_list(scored_frame, capacity=bad)


def test_priority_list_has_call_actions(scored_frame):
    priority = build_priority_list(scored_frame, capacity=0.10)
    assert priority["call_action"].notna().all()
    assert (priority["call_action"] != "").all()


# ----------------------------------------------------------------------
# Lift (the core business metric)
# ----------------------------------------------------------------------
def test_lift_at_capacity_has_expected_shape(scored_frame):
    table = lift_at_capacity(scored_frame)
    assert len(table) == 5
    for column in ["lift", "churn_rate_in_group", "overall_churn_rate", "recall_of_churners"]:
        assert column in table.columns


def test_overall_rate_is_constant_across_rows(scored_frame):
    table = lift_at_capacity(scored_frame)
    assert table["overall_churn_rate"].nunique() == 1


def test_lift_exceeds_one_for_a_useful_model(scored_frame):
    table = lift_at_capacity(scored_frame, capacities=(0.10,))
    assert table.iloc[0]["lift"] > 1.0


def test_lift_decreases_as_capacity_grows(scored_frame):
    table = lift_at_capacity(scored_frame, capacities=(0.05, 0.10, 0.20, 0.50))
    lifts = table["lift"].tolist()
    assert lifts == sorted(lifts, reverse=True)


def test_recall_increases_with_capacity(scored_frame):
    table = lift_at_capacity(scored_frame, capacities=(0.05, 0.20, 0.50))
    recalls = table["recall_of_churners"].tolist()
    assert recalls == sorted(recalls)


def test_lift_requires_outcomes(scored_frame):
    without = scored_frame.drop(columns=["actual_churn"])
    with pytest.raises(ValueError, match="observed outcomes"):
        lift_at_capacity(without)


def test_churners_in_group_never_exceed_group_size(scored_frame):
    table = lift_at_capacity(scored_frame)
    assert (table["churners_in_group"] <= table["customers_selected"]).all()


# ----------------------------------------------------------------------
# Gains and scenarios
# ----------------------------------------------------------------------
def test_cumulative_gains_curve_is_monotonic(scored_frame):
    gains = cumulative_gains(scored_frame)
    assert gains["churners_captured"].is_monotonic_increasing
    assert gains["cumulative_recall"].is_monotonic_increasing


def test_cumulative_gains_reaches_100_percent(scored_frame):
    gains = cumulative_gains(scored_frame)
    assert gains["cumulative_recall"].iloc[-1] == pytest.approx(1.0, abs=0.01)
    assert gains["churners_captured"].iloc[-1] == int(scored_frame["actual_churn"].sum())


def test_scenarios_are_consistent(scored_frame):
    scenarios = outreach_scenarios(scored_frame)
    assert len(scenarios) == 5
    assert scenarios["customers_contacted"].is_monotonic_increasing
    assert scenarios["lift"].is_monotonic_decreasing


def test_prioritisation_summary_keys(scored_frame):
    summary = prioritisation_summary(scored_frame, capacity=0.10)
    for key in [
        "customers_prioritised", "lift_at_capacity", "churn_rate_in_group",
        "overall_churn_rate", "monthly_revenue_in_group", "expected_monthly_revenue_at_risk",
    ]:
        assert key in summary, key
    assert summary["lift_at_capacity"] > 1.0
    assert summary["churn_rate_in_group"] > summary["overall_churn_rate"]


def test_priority_summary_revenue_is_a_share_of_total(scored_frame):
    summary = prioritisation_summary(scored_frame, capacity=0.10)
    assert 0 < summary["share_of_total_monthly_revenue_pct"] < 100


def test_segment_priority_profile_sums_to_list(scored_frame):
    priority = build_priority_list(scored_frame, capacity=0.10)
    profile = segment_priority_profile(priority)
    assert profile["share_of_priority_list_pct"].sum() <= 100.5


def test_priority_list_does_not_use_actual_outcome_for_ranking(scored_frame):
    """Ranking must be driven by predictions only; shuffling the observed
    outcomes must not change who is selected."""
    shuffled = scored_frame.copy()
    shuffled["actual_churn"] = np.random.RandomState(0).permutation(shuffled["actual_churn"].values)
    a = build_priority_list(scored_frame, capacity=0.10)["customer_id"].tolist()
    b = build_priority_list(shuffled, capacity=0.10)["customer_id"].tolist()
    assert a == b
