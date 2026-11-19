"""Preprocessing: feature selection, column transforms and data splitting.

Leakage rules enforced here:

* ``customer_id`` never reaches the model.
* The target (``churn`` / ``churn_label``) and any column derived from it is
  dropped before the transformer is built.
* All fitting (scaling, one-hot vocabulary, imputation) happens inside a
  scikit-learn ``Pipeline`` that is fitted on the training split only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logger = logging.getLogger(__name__)


@dataclass
class DataSplits:
    """Stratified train / validation / test partitions plus the full frame."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    features: pd.DataFrame
    target: pd.Series
    ids: pd.Series


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the modelling columns, raising if one is missing."""
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}. Run the data pipeline first.")
    return df.loc[:, config.FEATURE_COLUMNS].copy()


def select_target(df: pd.DataFrame) -> pd.Series:
    """Binary target as int (1 = churned)."""
    return df[config.TARGET].astype(int).rename("churn_label")


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Hard guard: no identifier, label or label-derived column in the matrix."""
    leaked = [c for c in features.columns if c in config.EXCLUDED_FROM_MODEL]
    if leaked:
        raise ValueError(f"Leakage detected, these columns must not be features: {leaked}")

    suspicious = [
        c for c in features.columns
        if any(token in c.lower() for token in ("churn", "target", "outcome", "label"))
    ]
    if suspicious:
        raise ValueError(f"Columns look label-derived and cannot be features: {suspicious}")


def coerce_feature_types(features: pd.DataFrame) -> pd.DataFrame:
    """Force the dtypes the transformer expects.

    Numeric columns must be numeric (a CSV round-trip can turn them into text)
    and categorical columns must be plain object/text so OneHotEncoder sees the
    categories it expects.
    """
    out = features.copy()
    for col in config.NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in config.CATEGORICAL_FEATURES:
        out[col] = out[col].astype(object)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Numeric scaling + categorical one-hot encoding, fitted inside the pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def split_data(df: pd.DataFrame) -> DataSplits:
    """Stratified 60/20/20 split with a fixed seed."""
    features = coerce_feature_types(select_features(df))
    target = select_target(df)
    assert_no_leakage(features)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        stratify=target,
        random_state=config.SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=config.VAL_SIZE,
        stratify=y_train_val,
        random_state=config.SEED,
    )

    splits = DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        features=features,
        target=target,
        ids=df[config.ID_COL].astype(str).reset_index(drop=True),
    )

    logger.info(
        "Split sizes -> train %s (churn %.2f%%), val %s (churn %.2f%%), test %s (churn %.2f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_val), y_val.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )

    # Guard against a split that accidentally leaks rows across partitions.
    train_ids = set(df.loc[X_train.index, config.ID_COL])
    test_ids = set(df.loc[X_test.index, config.ID_COL])
    if train_ids & test_ids:
        raise RuntimeError("Train/test customer overlap detected — split is not clean.")

    return splits


def feature_names_after_transform(preprocessor: ColumnTransformer) -> list[str]:
    """Readable one-hot feature names, used by SHAP and importance tables."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - only if unfitted
        return list(config.FEATURE_COLUMNS)
