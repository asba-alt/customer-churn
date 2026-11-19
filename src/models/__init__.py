"""Machine learning package."""

from .evaluate import evaluate_on_test
from .explainability import ChurnExplainer, save_explanations
from .predict import (
    assign_risk_tiers,
    load_scores,
    risk_distribution,
    save_scores,
    score_customers,
    tier_thresholds,
)
from .preprocessing import DataSplits, build_preprocessor, split_data
from .train import (
    comparison_table,
    load_pipeline,
    refit_final,
    save_pipeline,
    select_best,
    train_candidates,
)

__all__ = [
    "evaluate_on_test",
    "ChurnExplainer",
    "save_explanations",
    "assign_risk_tiers",
    "load_scores",
    "risk_distribution",
    "save_scores",
    "score_customers",
    "tier_thresholds",
    "DataSplits",
    "build_preprocessor",
    "split_data",
    "comparison_table",
    "load_pipeline",
    "refit_final",
    "save_pipeline",
    "select_best",
    "train_candidates",
]
