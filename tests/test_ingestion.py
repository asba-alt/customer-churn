"""Tests for dataset ingestion."""

from __future__ import annotations

import pytest

import config
from src.data.ingestion import (
    EXPECTED_COLUMNS,
    dataset_is_available,
    file_fingerprint,
    load_raw,
    missing_dataset_message,
)


def test_dataset_exists_on_disk():
    assert dataset_is_available(), missing_dataset_message()


def test_load_raw_shape_and_columns(raw_frame):
    assert raw_frame.shape == (7043, 21)
    assert list(raw_frame.columns) == EXPECTED_COLUMNS


def test_raw_has_expected_customer_ids(raw_frame):
    assert raw_frame["customerID"].nunique() == 7043
    assert raw_frame["customerID"].is_unique


def test_loading_missing_file_raises_with_instructions(tmp_path):
    absent = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_raw(absent)
    message = str(excinfo.value)
    assert "Kaggle" in message or "kaggle" in message
    assert "data/raw" in message
    assert str(absent) in message


def test_dataset_is_available_false_for_missing_path(tmp_path):
    assert not dataset_is_available(tmp_path / "missing.csv")


def test_empty_file_is_not_treated_as_available(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert not dataset_is_available(empty)


def test_file_fingerprint_is_stable():
    first = file_fingerprint()
    second = file_fingerprint()
    assert first == second
    assert len(first["sha256"]) == 64
    assert first["size_bytes"] > 0


def test_raw_loader_does_not_mutate_source_file():
    path = config.RAW_FILE
    before = path.stat().st_mtime_ns
    load_raw()
    assert path.stat().st_mtime_ns == before
