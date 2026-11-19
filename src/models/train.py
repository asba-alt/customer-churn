"""Model training: candidate comparison, selection and threshold tuning.

Methodology (reproducible end to end):

1. Split into train / validation / test (stratified, seed 42).
2. Fit a handful of hyper-parameter settings per candidate model on **train**
   only, scored on **validation** with PR-AUC (average precision).
3. Select the best (model, parameters) pair on validation PR-AUC. PR-AUC is
   used because churn is a 26.5% minority class and the business decision is a
   ranked outreach list, not a balanced classification.
4. Tune the decision threshold on **validation** by maximising F-beta with
   beta = 2 (recall weighted twice as heavily as precision).
5. Refit the selected configuration on train + validation and evaluate once on
   the untouched test set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config
from .preprocessing import DataSplits, build_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """One fitted candidate with its validation scores."""

    model_name: str
    params: dict[str, Any]
    pipeline: Pipeline
    metrics: dict[str, float]


def candidate_grids() -> dict[str, list[dict[str, Any]]]:
    """Small, defensible hyper-parameter grids.

    Deliberately compact: this dataset has ~4,200 training rows, so wide random
    search would mostly fit noise. The grids probe the one or two settings that
    actually matter for each family.
    """
    return {
        "logistic_regression": [
            {"clf__C": 0.1, "clf__solver": "liblinear"},
            {"clf__C": 1.0, "clf__solver": "liblinear"},
            {"clf__C": 1.0, "clf__solver": "lbfgs"},
            {"clf__C": 3.0, "clf__solver": "lbfgs"},
        ],
        "random_forest": [
            {"clf__n_estimators": 300, "clf__max_depth": 6, "clf__min_samples_leaf": 5},
            {"clf__n_estimators": 300, "clf__max_depth": 10, "clf__min_samples_leaf": 3},
            {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_leaf": 1},
        ],
        "xgboost": [
            {"clf__n_estimators": 200, "clf__max_depth": 3, "clf__learning_rate": 0.05, "clf__subsample": 0.9},
            {"clf__n_estimators": 400, "clf__max_depth": 3, "clf__learning_rate": 0.05, "clf__subsample": 0.9},
            {"clf__n_estimators": 300, "clf__max_depth": 5, "clf__learning_rate": 0.05, "clf__subsample": 0.8},
            {"clf__n_estimators": 400, "clf__max_depth": 4, "clf__learning_rate": 0.03, "clf__subsample": 0.9},
        ],
    }


def build_estimator(model_name: str, params: dict[str, Any], scale_pos_weight: float) -> Any:
    """Instantiate an estimator with imbalance handling appropriate to its family."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.SEED,
            **{k.replace("clf__", ""): v for k, v in params.items()},
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.SEED,
            **{k.replace("clf__", ""): v for k, v in params.items()},
        )
    if model_name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=config.SEED,
            tree_method="hist",
            **{k.replace("clf__", ""): v for k, v in params.items()},
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, params: dict[str, Any], scale_pos_weight: float) -> Pipeline:
    """Preprocessor + estimator as one object, saved and reused at inference."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", build_estimator(model_name, params, scale_pos_weight)),
        ]
    )


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Full metric set for a probability vector at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=config.THRESHOLD_BETA, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def tune_threshold(y_true, y_proba, beta: float = config.THRESHOLD_BETA) -> tuple[float, float]:
    """Pick the threshold that maximises F-beta on the validation split.

    A probability of 0.5 is the wrong cut-off for an imbalanced problem where
    the cost of a missed churner exceeds the cost of a spare call. F-beta with
    beta=2 encodes that preference explicitly.
    """
    grid = np.unique(np.clip(np.round(np.asarray(y_proba), 3), 0.01, 0.99))
    best_threshold, best_score = 0.5, -1.0
    for threshold in grid:
        score = fbeta_score(y_true, (np.asarray(y_proba) >= threshold).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def train_candidates(splits: DataSplits) -> list[CandidateResult]:
    """Fit every candidate on train, score on validation."""
    positives = int(splits.y_train.sum())
    negatives = int(len(splits.y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    logger.info("Train imbalance: %s negatives / %s positives -> scale_pos_weight %.2f",
                negatives, positives, scale_pos_weight)

    results: list[CandidateResult] = []
    for model_name, grid in candidate_grids().items():
        for params in grid:
            pipeline = build_pipeline(model_name, params, scale_pos_weight)
            pipeline.fit(splits.X_train, splits.y_train)
            proba = pipeline.predict_proba(splits.X_val)[:, 1]
            metrics = compute_metrics(splits.y_val, proba, threshold=0.5)
            threshold, f2 = tune_threshold(splits.y_val, proba)
            metrics["tuned_threshold"] = threshold
            metrics["tuned_f2"] = f2
            metrics["metrics_at_tuned_threshold"] = compute_metrics(
                splits.y_val, proba, threshold=threshold
            )
            results.append(
                CandidateResult(
                    model_name=model_name,
                    params={k.replace("clf__", ""): v for k, v in params.items()},
                    pipeline=pipeline,
                    metrics=metrics,
                )
            )
            logger.info(
                "%-20s %-70s val PR-AUC %.4f | ROC-AUC %.4f | F2 %.4f @ thr %.2f",
                model_name,
                str({k.replace('clf__', ''): v for k, v in params.items()}),
                metrics["pr_auc"],
                metrics["roc_auc"],
                f2,
                threshold,
            )
    return results


def select_best(results: list[CandidateResult]) -> CandidateResult:
    """Highest validation PR-AUC; ties broken by ROC-AUC then F2."""
    return sorted(
        results,
        key=lambda r: (r.metrics["pr_auc"], r.metrics["roc_auc"], r.metrics["tuned_f2"]),
        reverse=True,
    )[0]


def comparison_table(results: list[CandidateResult]) -> pd.DataFrame:
    """Model comparison table sorted by the selection metric."""
    rows = []
    for r in results:
        at_thr = r.metrics["metrics_at_tuned_threshold"]
        rows.append(
            {
                "model": r.model_name,
                "params": json.dumps(r.params, sort_keys=True),
                "val_accuracy_at_0.5": round(r.metrics["accuracy"], 4),
                "val_precision_at_0.5": round(r.metrics["precision"], 4),
                "val_recall_at_0.5": round(r.metrics["recall"], 4),
                "val_f1_at_0.5": round(r.metrics["f1"], 4),
                "val_roc_auc": round(r.metrics["roc_auc"], 4),
                "val_pr_auc": round(r.metrics["pr_auc"], 4),
                "tuned_threshold": round(r.metrics["tuned_threshold"], 3),
                "val_precision_tuned": round(at_thr["precision"], 4),
                "val_recall_tuned": round(at_thr["recall"], 4),
                "val_f1_tuned": round(at_thr["f1"], 4),
                "val_f2_tuned": round(at_thr["f2"], 4),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)


def refit_final(
    best: CandidateResult, splits: DataSplits
) -> tuple[Pipeline, float]:
    """Refit the winning configuration on train + validation.

    The test set stays untouched. Refitting on train+val uses more data for the
    final model, and the validation-derived threshold carries over because the
    decision rule was already selected.
    """
    positives = int(splits.y_train.sum() + splits.y_val.sum())
    total = len(splits.y_train) + len(splits.y_val)
    scale_pos_weight = (total - positives) / max(positives, 1)

    params = {f"clf__{k}": v for k, v in best.params.items()}
    pipeline = build_pipeline(best.model_name, params, scale_pos_weight)

    X_combined = pd.concat([splits.X_train, splits.X_val])
    y_combined = pd.concat([splits.y_train, splits.y_val])
    pipeline.fit(X_combined, y_combined)
    logger.info("Refit %s on %s train+val rows", best.model_name, len(X_combined))
    return pipeline, best.metrics["tuned_threshold"]


def save_pipeline(pipeline: Pipeline, threshold: float, best: CandidateResult, splits: DataSplits) -> dict:
    """Persist the pipeline and the metadata needed to reuse it."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "model_name": best.model_name,
            "params": best.params,
            "feature_columns": config.FEATURE_COLUMNS,
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "trained_on": {"rows": int(len(splits.X_train) + len(splits.X_val))},
            "seed": config.SEED,
        },
        config.PIPELINE_FILE,
    )
    metadata = {
        "model_name": best.model_name,
        "params": best.params,
        "decision_threshold": round(threshold, 4),
        "selection_metric": config.SELECTION_METRIC,
        "threshold_rule": f"max F{config.THRESHOLD_BETA:g} on validation",
        "validation_metrics": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in best.metrics.items()
            if not isinstance(v, dict)
        },
        "validation_metrics_at_tuned_threshold": {
            k: round(v, 4) for k, v in best.metrics["metrics_at_tuned_threshold"].items()
        },
        "split": {
            "train": len(splits.X_train),
            "validation": len(splits.X_val),
            "test": len(splits.X_test),
            "train_churn_rate": round(float(splits.y_train.mean()), 4),
            "val_churn_rate": round(float(splits.y_val.mean()), 4),
            "test_churn_rate": round(float(splits.y_test.mean()), 4),
        },
        "feature_count_raw": len(config.FEATURE_COLUMNS),
        "seed": config.SEED,
    }
    config.MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved pipeline to %s", config.PIPELINE_FILE)
    return metadata


def load_pipeline() -> tuple[Pipeline, float, dict]:
    """Load the saved pipeline, its threshold and its metadata."""
    if not config.PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"No trained pipeline at {config.PIPELINE_FILE}. Run: python main.py train"
        )
    bundle = joblib.load(config.PIPELINE_FILE)
    metadata = json.loads(config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    return bundle["pipeline"], bundle["threshold"], metadata
