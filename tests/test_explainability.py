"""Tests for explainability, saved artefacts and the end-to-end pipeline smoke path."""

from __future__ import annotations

import json

import numpy as np
import pytest

import config
from src.models.explainability import ChurnExplainer, explain_individual, global_importance


@pytest.fixture(scope="module")
def explainer(trained_model):
    return ChurnExplainer(trained_model)


def test_explainer_feature_names_are_readable(explainer):
    assert len(explainer.feature_names) > 0
    assert all("__" not in name for name in explainer.feature_names)
    assert any("=" in name for name in explainer.feature_names), "categorical names should be readable"


def test_transform_matches_preprocessor(explainer, splits):
    matrix = explainer.transform(splits.X_train.head(20))
    assert matrix.shape[0] == 20
    assert np.isfinite(matrix.values).all()


def test_shap_values_have_one_column_per_feature(explainer, splits):
    values = explainer.shap_values(splits.X_train.head(50))
    assert values.shape == (50, len(explainer.feature_names))


def test_shap_values_reconstruct_prediction(explainer, splits):
    """Additivity check: base + sum(shap) must reproduce the model's own output.

    Gradient-boosted models add up in log-odds; scikit-learn forests add up in
    probability space. Both are checked against the matching transform.
    """
    sample = splits.X_train.head(15)
    values = explainer.shap_values(sample)
    base = explainer.base_value()
    reconstructed = base + values.sum(axis=1)
    proba = explainer.pipeline.predict_proba(sample)[:, 1]

    if explainer.output_space == "log_odds":
        expected = np.log(np.clip(proba, 1e-9, 1 - 1e-9) / np.clip(1 - proba, 1e-9, 1))
    else:
        expected = proba

    assert np.allclose(reconstructed, expected, atol=1e-3)


def test_output_space_matches_estimator(explainer):
    assert explainer.output_space in {"log_odds", "probability"}


def test_global_importance_is_ranked(explainer, splits):
    importance = global_importance(explainer, splits.X_train.head(200), top_n=10)
    assert len(importance) <= 10
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert importance["share_of_total_importance"].sum() <= 100.5


def test_individual_explanation_structure(explainer, splits, clean_frame):
    customer_id = clean_frame["customer_id"].iloc[0]
    explanation = explain_individual(explainer, clean_frame, customer_id)
    assert explanation["customer_id"] == customer_id
    assert explanation["output_space"] in {"log_odds", "probability"}
    assert 0 <= explanation["predicted_probability"] <= 1
    assert not explanation["all_contributions"].empty
    assert set(explanation["customer_context"]) >= {
        "contract", "tenure", "monthly_charges", "internet_service"
    }


def test_individual_explanation_directions(explainer, clean_frame):
    customer_id = clean_frame["customer_id"].iloc[0]
    explanation = explain_individual(explainer, clean_frame, customer_id)
    increasing = explanation["top_risk_increasing"]
    decreasing = explanation["top_risk_decreasing"]
    assert (increasing["shap_value"] > 0).all()
    assert (decreasing["shap_value"] < 0).all()


def test_unknown_customer_raises(explainer, clean_frame):
    with pytest.raises(KeyError):
        explain_individual(explainer, clean_frame, "NOT-A-CUSTOMER")


def test_pipeline_already_run_produced_reports():
    """If the pipeline has run, the headline artefacts must exist and be non-empty."""
    if not config.CLEAN_FILE.exists():
        pytest.skip("pipeline has not been run yet")

    for path in [
        config.CLEAN_FILE,
        config.DB_FILE,
        config.REPORT_DATA_QUALITY,
        config.MODELS_DIR / "model_metadata.json",
    ]:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_saved_metadata_is_valid_json():
    if not config.MODEL_METADATA_FILE.exists():
        pytest.skip("no trained model metadata yet")
    metadata = json.loads(config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    assert "model_name" in metadata
    assert "decision_threshold" in metadata
    assert 0 < metadata["decision_threshold"] < 1
    assert metadata["split"]["train"] > metadata["split"]["test"]


def test_exported_powerbi_files_exist():
    if not config.POWERBI_EXPORTS.exists():
        pytest.skip("pipeline has not been run yet")
    expected = [
        "customers.csv",
        "customer_risk_scores.csv",
        "retention_priority_list.csv",
        "retention_lift.csv",
        "risk_distribution.csv",
        "data_dictionary.csv",
    ]
    for name in expected:
        path = config.POWERBI_EXPORTS / name
        assert path.exists(), f"missing Power BI export: {name}"
        assert path.stat().st_size > 0
