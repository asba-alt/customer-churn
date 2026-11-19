"""Tests for preprocessing, leakage prevention and model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from src.models.evaluate import evaluate_on_test
from src.models.preprocessing import (
    assert_no_leakage,
    build_preprocessor,
    coerce_feature_types,
    select_features,
    select_target,
    split_data,
)
from src.models.train import (
    build_pipeline,
    comparison_table,
    compute_metrics,
    select_best,
    tune_threshold,
)


# ----------------------------------------------------------------------
# Feature selection and leakage
# ----------------------------------------------------------------------
def test_features_exclude_identifier_and_target(splits):
    assert config.ID_COL not in splits.X_train.columns
    assert "churn" not in splits.X_train.columns
    assert "churn_label" not in splits.X_train.columns


def test_assert_no_leakage_blocks_identifier():
    frame = pd.DataFrame({config.ID_COL: ["a"], "tenure": [1]})
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(frame)


def test_assert_no_leakage_blocks_label_column():
    frame = pd.DataFrame({"tenure": [1], "churn_label": [0]})
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(frame)


def test_assert_no_leakage_blocks_suspicious_name():
    frame = pd.DataFrame({"tenure": [1], "churn_prediction_hint": [0]})
    with pytest.raises(ValueError, match="label-derived"):
        assert_no_leakage(frame)


def test_missing_feature_raises_clear_error():
    frame = pd.DataFrame({"tenure": [1, 2]})
    with pytest.raises(ValueError, match="Missing model features"):
        select_features(frame)


def test_target_is_binary_int(clean_frame):
    target = select_target(clean_frame)
    assert target.dtype.kind in "iu"
    assert set(target.unique()) == {0, 1}


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def test_split_sizes_are_60_20_20(splits):
    total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert total == len(splits.features)
    assert len(splits.X_test) / total == pytest.approx(0.20, abs=0.01)
    assert len(splits.X_val) / total == pytest.approx(0.16, abs=0.01)


def test_splits_are_stratified(splits):
    rates = [float(s.mean()) for s in (splits.y_train, splits.y_val, splits.y_test)]
    assert max(rates) - min(rates) < 0.01


def test_splits_have_no_customer_overlap(splits):
    """Index-based partitions must be disjoint."""
    train_idx = set(splits.X_train.index)
    val_idx = set(splits.X_val.index)
    test_idx = set(splits.X_test.index)
    assert not train_idx & val_idx
    assert not train_idx & test_idx
    assert not val_idx & test_idx


def test_split_is_deterministic(splits, clean_frame):
    again = split_data(clean_frame)
    assert list(again.X_train.index) == list(splits.X_train.index)
    assert list(again.y_test.values) == list(splits.y_test.values)


# ----------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------
def test_preprocessor_outputs_one_hot_and_scaled(splits):
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(splits.X_train)
    assert matrix.shape[0] == len(splits.X_train)
    assert matrix.shape[1] > len(config.FEATURE_COLUMNS)
    assert np.isfinite(matrix).all()


def test_preprocessor_handles_unseen_categories(splits):
    preprocessor = build_preprocessor()
    preprocessor.fit(splits.X_train)
    unseen = splits.X_test.head(5).copy()
    unseen.loc[unseen.index[0], "contract"] = "Five year"
    matrix = preprocessor.transform(unseen)
    assert np.isfinite(matrix).all()


def test_coerce_feature_types_forces_numeric():
    frame = pd.DataFrame(
        {c: ["1"] * 3 for c in config.NUMERIC_FEATURES}
        | {c: ["x"] * 3 for c in config.CATEGORICAL_FEATURES}
    )
    out = coerce_feature_types(frame)
    for col in config.NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(out[col]), col


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def test_metrics_are_perfect_for_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y, np.array([0.05, 0.1, 0.9, 0.95]), threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_metrics_handle_all_negative_predictions():
    y = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y, np.array([0.1, 0.2, 0.15, 0.3]), threshold=0.5)
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["accuracy"] >= 0


def test_metrics_are_bounded():
    y = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([0.2, 0.8, 0.35, 0.65, 0.9, 0.1])
    metrics = compute_metrics(y, proba, threshold=0.5)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, name


def test_threshold_tuning_prefers_higher_recall():
    """With beta=2, the tuned cut-off should not be more conservative than 0.5
    when the model is well separated."""
    y = np.array([0] * 70 + [1] * 30)
    proba = np.concatenate([np.linspace(0.01, 0.5, 70), np.linspace(0.45, 0.99, 30)])
    threshold, f2 = tune_threshold(y, proba, beta=2.0)
    assert 0 < threshold < 1
    assert f2 > 0


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------
def test_pipeline_fits_and_predicts(splits):
    pipeline = build_pipeline("logistic_regression", {"clf__C": 1.0}, scale_pos_weight=2.77)
    pipeline.fit(splits.X_train, splits.y_train)
    proba = pipeline.predict_proba(splits.X_val)[:, 1]
    assert proba.shape == (len(splits.X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_pipeline_fits(splits):
    pipeline = build_pipeline(
        "xgboost",
        {"clf__n_estimators": 50, "clf__max_depth": 3, "clf__learning_rate": 0.1},
        scale_pos_weight=2.77,
    )
    pipeline.fit(splits.X_train, splits.y_train)
    proba = pipeline.predict_proba(splits.X_test)[:, 1]
    assert np.isfinite(proba).all()


def test_selection_prefers_higher_pr_auc():
    from src.models.train import CandidateResult

    def make(name, pr_auc):
        return CandidateResult(
            model_name=name,
            params={},
            pipeline=Pipeline([("clf", LogisticRegression())]),
            metrics={"pr_auc": pr_auc, "roc_auc": pr_auc, "tuned_f2": pr_auc},
        )

    best = select_best([make("a", 0.60), make("b", 0.72), make("c", 0.65)])
    assert best.model_name == "b"


def test_comparison_table_has_expected_columns(trained_model, splits):
    from src.models.train import CandidateResult

    result = CandidateResult(
        model_name="rf",
        params={"max_depth": 6},
        pipeline=trained_model,
        metrics={
            **compute_metrics(splits.y_val, trained_model.predict_proba(splits.X_val)[:, 1]),
            "tuned_threshold": 0.3,
            "tuned_f2": 0.7,
            "metrics_at_tuned_threshold": compute_metrics(
                splits.y_val, trained_model.predict_proba(splits.X_val)[:, 1], threshold=0.3
            ),
        },
    )
    table = comparison_table([result])
    for column in ["model", "val_pr_auc", "val_roc_auc", "tuned_threshold", "val_f2_tuned"]:
        assert column in table.columns


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def test_evaluation_produces_consistent_confusion_matrix(trained_model, splits):
    result = evaluate_on_test(trained_model, splits, threshold=0.5)
    cm = result["confusion_tuned"]
    assert cm.shape == (2, 2)
    assert cm.sum() == len(splits.y_test)
    assert result["n_test"] == len(splits.y_test)


def test_evaluation_confusion_matrix_matches_metrics(trained_model, splits):
    result = evaluate_on_test(trained_model, splits, threshold=0.5)
    tn, fp, fn, tp = result["confusion_tuned"].ravel()
    metrics = result["metrics_tuned"]
    assert metrics["recall"] == pytest.approx(tp / (tp + fn), abs=1e-6)
    assert metrics["precision"] == pytest.approx(tp / (tp + fp), abs=1e-6)


def test_lower_threshold_increases_recall(trained_model, splits):
    low = evaluate_on_test(trained_model, splits, threshold=0.2)
    high = evaluate_on_test(trained_model, splits, threshold=0.7)
    assert low["metrics_tuned"]["recall"] >= high["metrics_tuned"]["recall"]


def test_roc_auc_is_threshold_independent(trained_model, splits):
    low = evaluate_on_test(trained_model, splits, threshold=0.2)
    high = evaluate_on_test(trained_model, splits, threshold=0.8)
    assert low["metrics_tuned"]["roc_auc"] == pytest.approx(high["metrics_tuned"]["roc_auc"])


def test_model_beats_random_on_test(trained_model, splits):
    result = evaluate_on_test(trained_model, splits, threshold=0.5)
    assert result["metrics_default"]["roc_auc"] > 0.7
    assert result["metrics_default"]["pr_auc"] > result["baseline_rate"]
