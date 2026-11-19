"""Tests for the cleaning and validation stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data.cleaning import clean_customers, save_clean
from src.data.validation import run_validation_checks


def test_cleaning_preserves_row_count(cleaning_result, raw_frame):
    """This dataset has no duplicates and no invalid targets, so nothing should drop."""
    assert len(cleaning_result.frame) == len(raw_frame)


def test_cleaning_is_idempotent(clean_frame):
    """Cleaning an already-clean frame must not change its shape."""
    second = clean_customers(clean_frame).frame
    assert second.shape[0] == clean_frame.shape[0]


def test_no_duplicate_customer_ids(clean_frame):
    assert clean_frame["customer_id"].is_unique


def test_total_charges_is_numeric_and_complete(clean_frame):
    assert pd.api.types.is_numeric_dtype(clean_frame["total_charges"])
    assert clean_frame["total_charges"].isna().sum() == 0


def test_total_charges_blanks_are_imputed_to_zero_and_flagged(clean_frame):
    imputed = clean_frame[clean_frame["total_charges_imputed"] == 1]
    assert len(imputed) == 11
    assert (imputed["total_charges"] == 0).all()
    assert (imputed["tenure"] == 0).all()


def test_target_survives_cleaning(clean_frame):
    assert set(clean_frame["churn"].unique()) == {"No", "Yes"}
    assert set(clean_frame["churn_label"].unique()) == {0, 1}
    assert (clean_frame["churn"] == "Yes").sum() == int(clean_frame["churn_label"].sum())


def test_derived_columns_exist_and_are_sane(clean_frame):
    for column in [
        "tenure_group", "num_addon_services", "service_count",
        "avg_monthly_charge", "revenue_band", "high_value_customer", "auto_payment",
    ]:
        assert column in clean_frame.columns

    assert clean_frame["num_addon_services"].between(0, 6).all()
    assert clean_frame["service_count"].between(0, 8).all()
    assert clean_frame["high_value_customer"].isin([0, 1]).all()


def test_tenure_group_has_no_missing_values(clean_frame):
    assert clean_frame["tenure_group"].isna().sum() == 0
    assert set(clean_frame["tenure_group"].unique()) <= set(config.TENURE_LABELS)


def test_avg_monthly_charge_matches_total_over_tenure(clean_frame):
    mask = (clean_frame["tenure"] > 0) & (clean_frame["total_charges"] > 0)
    subset = clean_frame[mask]
    expected = subset["total_charges"] / subset["tenure"]
    assert np.allclose(subset["avg_monthly_charge"], expected, atol=0.02)


def test_zero_tenure_uses_monthly_charges_as_fallback(clean_frame):
    zero = clean_frame[clean_frame["tenure"] == 0]
    assert len(zero) > 0
    assert np.allclose(zero["avg_monthly_charge"], zero["monthly_charges"], atol=0.02)


def test_validation_checks_all_pass(clean_frame):
    results = run_validation_checks(clean_frame)
    failures = [(r.name, r.detail) for r in results if not r.passed]
    assert not failures, f"failing checks: {failures}"


def test_binary_flags_are_zero_or_one(clean_frame):
    for column in ["partner", "dependents", "phone_service", "paperless_billing", "auto_payment"]:
        assert set(clean_frame[column].unique()) <= {0, 1}, column


def test_duplicate_injection_is_detected(clean_frame):
    """A deliberately duplicated customer must be collapsed by the cleaner."""
    duplicated = pd.concat([clean_frame, clean_frame.iloc[[0]]], ignore_index=True)
    result = clean_customers(duplicated)
    assert len(result.frame) == len(clean_frame)
    assert result.log.duplicate_customer_ids == 1


def test_invalid_churn_value_is_dropped_and_logged(clean_frame):
    corrupted = clean_frame.copy()
    corrupted.loc[0, "churn"] = "Maybe"
    result = clean_customers(corrupted)
    assert len(result.frame) == len(clean_frame) - 1
    assert result.log.invalid_churn_values == 1


def test_cleaning_decisions_are_documented(cleaning_result):
    log = cleaning_result.log
    issues = [issue.lower() for issue, _ in log.decisions]
    assert any("totalcharges" in i for i in issues)
    assert any("duplicate" in i for i in issues)
    assert len(log.decisions) >= 5
    assert all(decision.strip() for _, decision in log.decisions)


def test_save_clean_roundtrip(clean_frame, tmp_path):
    path = tmp_path / "clean.csv"
    save_clean(clean_frame, path)
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(clean_frame)
    assert "tenure_group" in reloaded.columns
