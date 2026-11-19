"""Raw dataset ingestion.

The IBM Telco Customer Churn file is the single source of truth for this
project. Nothing here rewrites or samples it: the raw file is read-only and
the cleaned copy is written elsewhere by :mod:`src.data.cleaning`.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

EXPECTED_ROW_COUNT = 7043


def dataset_is_available(path: Path | None = None) -> bool:
    """Return True when the raw CSV exists and is non-empty."""
    path = Path(path) if path is not None else config.RAW_FILE
    return path.exists() and path.stat().st_size > 0


def missing_dataset_message(path: Path | None = None) -> str:
    """Actionable instructions shown when the raw file cannot be found."""
    path = Path(path) if path is not None else config.RAW_FILE
    return (
        "Raw dataset not found.\n"
        f"  Expected file : {path}\n"
        f"  Expected dir  : {path.parent}\n\n"
        "How to fix:\n"
        "  1. Download WA_Fn-UseC_-Telco-Customer-Churn.csv from\n"
        "     https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        "  2. Place it in the data/raw/ directory without renaming it.\n"
        "  3. Re-run: python main.py all\n\n"
        "Everything that does not need the data (SQL files, DAX measures,\n"
        "dashboard specification, test-suite scaffolding) is still generated."
    )


def file_fingerprint(path: Path | None = None) -> dict[str, object]:
    """Cheap provenance record: name, size and SHA-256 of the raw file."""
    path = Path(path) if path is not None else config.RAW_FILE
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Telco churn CSV exactly as distributed.

    Raises
    ------
    FileNotFoundError
        With placement instructions when the CSV is absent.
    ValueError
        When the file does not look like the expected dataset (wrong schema).
    """
    path = Path(path) if path is not None else config.RAW_FILE
    if not dataset_is_available(path):
        raise FileNotFoundError(missing_dataset_message(path))

    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %s rows x %s columns", *df.shape)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw file is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep the official column order, drop nothing silently.
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        logger.warning("Unexpected extra columns in raw file: %s", extra)

    df = df.loc[:, EXPECTED_COLUMNS + extra].copy()
    if len(df) != EXPECTED_ROW_COUNT:
        logger.warning(
            "Row count %s differs from the documented %s rows for this dataset",
            len(df),
            EXPECTED_ROW_COUNT,
        )
    return df


def read_raw_as_strings(path: Path | None = None) -> pd.DataFrame:
    """Utility for schema inspection: read every column as text."""
    path = Path(path) if path is not None else config.RAW_FILE
    if not dataset_is_available(path):
        raise FileNotFoundError(missing_dataset_message(path))
    return pd.read_csv(path, dtype=str)
