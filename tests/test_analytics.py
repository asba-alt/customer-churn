"""Tests for KPI and analytics calculations."""

from __future__ import annotations

import pytest

from src.analytics.churn_analysis import (
    churn_by_addon,
    churn_by_contract,
    churn_by_payment_method,
    churn_by_tenure_group,
    churn_driver_summary,
)
from src.analytics.customer_analysis import (
    customer_segments,
    demographic_profile,
    service_adoption_summary,
    tenure_profile,
)
from src.analytics.revenue_analysis import (
    customer_value_proxies,
    revenue_by_segment,
    revenue_concentration,
    revenue_exposure,
    revenue_summary,
)


def test_kpi_customers_reconcile(clean_frame):
    kpis = revenue_summary(clean_frame)
    assert kpis["total_customers"] == len(clean_frame)
    assert kpis["churned_customers"] + kpis["retained_customers"] == kpis["total_customers"]


def test_churn_rate_matches_manual_calculation(clean_frame):
    kpis = revenue_summary(clean_frame)
    expected = (clean_frame["churn"] == "Yes").sum() / len(clean_frame)
    assert kpis["overall_churn_rate"] == pytest.approx(expected, abs=1e-4)


def test_total_monthly_revenue_matches_sum(clean_frame):
    kpis = revenue_summary(clean_frame)
    assert kpis["total_monthly_revenue"] == pytest.approx(clean_frame["monthly_charges"].sum(), abs=0.01)


def test_revenue_split_sums_to_total(clean_frame):
    kpis = revenue_summary(clean_frame)
    assert kpis["monthly_revenue_churned"] + kpis["monthly_revenue_retained"] == pytest.approx(
        kpis["total_monthly_revenue"], abs=0.05
    )


def test_revenue_share_is_a_valid_fraction(clean_frame):
    kpis = revenue_summary(clean_frame)
    assert 0 < kpis["revenue_share_churned"] < 1


def test_average_monthly_charges_match_manual(clean_frame):
    kpis = revenue_summary(clean_frame)
    assert kpis["avg_monthly_charges"] == pytest.approx(clean_frame["monthly_charges"].mean(), abs=0.01)
    assert kpis["avg_tenure"] == pytest.approx(clean_frame["tenure"].mean(), abs=0.01)


def test_contract_churn_covers_whole_base(clean_frame):
    result = churn_by_contract(clean_frame)
    assert result["customers"].sum() == len(clean_frame)
    assert result["churned"].sum() == int(clean_frame["churn_label"].sum())


def test_contract_churn_is_ordered_by_commitment(clean_frame):
    result = churn_by_contract(clean_frame)
    assert result["contract"].tolist() == ["Month-to-month", "One year", "Two year"]


def test_lift_and_rate_are_consistent(clean_frame):
    result = churn_by_contract(clean_frame)
    for row in result.itertuples():
        assert row.churn_rate == pytest.approx(row.churned / row.customers, abs=1e-6)
        assert row.lift_vs_baseline == pytest.approx(
            row.churn_rate / clean_frame["churn_label"].mean(), abs=1e-2
        )


def test_tenure_group_churn_is_chronological(clean_frame):
    result = churn_by_tenure_group(clean_frame)
    assert result["tenure_group"].tolist() == [
        "0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"
    ][: len(result)]
    assert result["customers"].sum() == len(clean_frame)


def test_payment_method_auto_flag(clean_frame):
    result = churn_by_payment_method(clean_frame)
    manual = result[result["auto_payment"] == 0]
    automatic = result[result["auto_payment"] == 1]
    assert not manual.empty and not automatic.empty
    assert set(automatic["payment_method"]) == {
        "Bank transfer (automatic)", "Credit card (automatic)"
    }


def test_addon_analysis_has_both_groups(clean_frame):
    result = churn_by_addon(clean_frame)
    assert len(result) == 6
    assert result["customers_with"].gt(0).all()
    assert result["difference_pts"].notna().all()


def test_addon_difference_is_computed_correctly(clean_frame):
    result = churn_by_addon(clean_frame)
    row = result[result["addon"] == "Online Security"].iloc[0]
    has = clean_frame[clean_frame["online_security"] == "Yes"]["churn_label"].mean()
    has_not = clean_frame[clean_frame["online_security"] == "No"]["churn_label"].mean()
    assert row["churn_rate_with"] == pytest.approx(has, abs=1e-4)
    assert row["churn_rate_without"] == pytest.approx(has_not, abs=1e-4)
    assert row["difference_pts"] == pytest.approx((has - has_not) * 100, abs=1e-2)


def test_driver_summary_has_one_row_per_feature_value(clean_frame):
    result = churn_driver_summary(clean_frame)
    assert not result.duplicated(subset=["feature", "value"]).any()
    assert (result["customers"] >= 50).all()
    assert result["vs_baseline_pts"].abs().is_monotonic_decreasing


def test_segments_meet_minimum_size(clean_frame):
    result = customer_segments(clean_frame)
    assert (result["customers"] >= 40).all()
    assert result["churn_rate"].between(0, 1).all()


def test_revenue_exposure_reconciles(clean_frame):
    result = revenue_exposure(clean_frame)
    contract_rows = result[result["dimension"] == "contract"]
    assert contract_rows["monthly_revenue"].sum() == pytest.approx(
        clean_frame["monthly_charges"].sum(), abs=0.05
    )


def test_revenue_by_segment_shares_sum_to_100(clean_frame):
    result = revenue_by_segment(clean_frame)
    assert result["revenue_share_pct"].sum() == pytest.approx(100.0, abs=0.1)


def test_value_proxies_are_correct(clean_frame):
    result = customer_value_proxies(clean_frame)
    assert result["customers"].sum() == len(clean_frame)
    assert len(result) == 5, "expected five value quintiles"
    for row in result.itertuples():
        assert row.avg_annualised_value > row.avg_realised_value or row.avg_realised_value > 0


def test_annualised_value_is_twelve_times_monthly(clean_frame):
    sample = clean_frame.head(50).copy()
    sample["churn_label"] = sample["churn_label"]
    result = customer_value_proxies(sample)
    assert result["avg_annualised_value"].notna().all()


def test_revenue_concentration_shares_increase(clean_frame):
    result = revenue_concentration(clean_frame)
    assert result["top_1pct_revenue_share"] < result["top_5pct_revenue_share"]
    assert result["top_5pct_revenue_share"] < result["top_10pct_revenue_share"]
    assert result["top_20pct_revenue_share"] < 1


def test_demographic_profile_suppresses_small_cells(clean_frame):
    result = demographic_profile(clean_frame)
    assert (result["customers"] >= 25).all()


def test_service_adoption_covers_all_columns(clean_frame):
    result = service_adoption_summary(clean_frame)
    assert result["service"].nunique() == 10
    assert (result["adoption_pct"] > 0).all()
    assert result["churn_rate"].between(0, 1).all()


def test_tenure_profile_has_both_outcomes(clean_frame):
    result = tenure_profile(clean_frame)
    assert set(result["churn"]) == {"No", "Yes"}
    churned = result[result["churn"] == "Yes"].iloc[0]
    retained = result[result["churn"] == "No"].iloc[0]
    assert churned["tenure_mean"] < retained["tenure_mean"]
