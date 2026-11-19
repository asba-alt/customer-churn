"""Shared pytest fixtures: one pipeline run reused across the suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


@pytest.fixture(scope="session")
def raw_frame():
    """Raw dataset, or skip the whole suite with clear instructions if absent."""
    from src.data.ingestion import dataset_is_available, load_raw, missing_dataset_message

    if not dataset_is_available():
        pytest.skip(missing_dataset_message(), allow_module_level=True)
    return load_raw()


@pytest.fixture(scope="session")
def cleaning_result(raw_frame):
    from src.data.cleaning import clean_customers

    return clean_customers(raw_frame)


@pytest.fixture(scope="session")
def clean_frame(cleaning_result):
    return cleaning_result.frame


@pytest.fixture(scope="session")
def db_path(clean_frame, tmp_path_factory):
    """A dedicated SQLite database built from the cleaned frame."""
    from src.database.db_setup import build_database

    path = tmp_path_factory.mktemp("db") / "test_churn.sqlite"
    build_database(clean_frame, path)
    return path


@pytest.fixture(scope="session")
def splits(clean_frame):
    from src.models.preprocessing import split_data

    return split_data(clean_frame)


@pytest.fixture(scope="session")
def trained_model(splits, tmp_path_factory):
    """A small real model, trained once for the whole session."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from src.models.preprocessing import build_preprocessor

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=120, max_depth=6, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=config.SEED, n_jobs=-1,
            )),
        ]
    )
    pipeline.fit(splits.X_train, splits.y_train)
    return pipeline


@pytest.fixture(scope="session")
def scored_frame(clean_frame, trained_model):
    from src.models.predict import score_customers

    return score_customers(clean_frame, trained_model, threshold=0.5)
