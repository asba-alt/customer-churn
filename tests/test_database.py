"""Tests for the SQLite database layer and the SQL analytics library."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.database.db_setup import get_connection, table_names
from src.database.queries import parse_query_file, run_all_queries, run_query


def test_database_tables_created(db_path):
    names = table_names(db_path)
    for table in ["dim_customer", "dim_service", "dim_contract", "fact_billing", "v_customer_360"]:
        assert table in names


def test_row_counts_match_dataset(db_path, clean_frame):
    conn = get_connection(db_path)
    try:
        for table in ["dim_customer", "dim_service", "dim_contract", "fact_billing"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == len(clean_frame), table
    finally:
        conn.close()


def test_view_joins_to_full_row_count(db_path, clean_frame):
    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM v_customer_360").fetchone()[0]
    finally:
        conn.close()
    assert count == len(clean_frame)


def test_primary_keys_reject_duplicates(db_path):
    conn = get_connection(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO dim_customer (customer_id, gender, senior_citizen, partner, dependents) "
                "VALUES ('7590-VHVEG', 'Female', 0, 1, 0)"
            )
    finally:
        conn.close()


def test_no_orphan_rows_in_child_tables(db_path):
    conn = get_connection(db_path)
    try:
        for table in ["dim_service", "dim_contract", "fact_billing"]:
            orphans = conn.execute(
                f"SELECT COUNT(*) FROM {table} t "
                f"LEFT JOIN dim_customer c ON c.customer_id = t.customer_id "
                f"WHERE c.customer_id IS NULL"
            ).fetchone()[0]
            assert orphans == 0, table
    finally:
        conn.close()


def test_query_file_parses_all_blocks():
    blocks = parse_query_file()
    assert len(blocks) >= 15
    names = [b.name for b in blocks]
    assert len(names) == len(set(names)), "duplicate query names"
    for block in blocks:
        assert block.sql.strip()
        assert block.title


def test_every_query_executes(db_path):
    results = run_all_queries(db_path, save=False)
    assert len(results) == len(parse_query_file())
    for name, frame in results.items():
        assert isinstance(frame, pd.DataFrame), name
        assert not frame.empty, f"{name} returned no rows"


def test_overall_churn_rate_matches_pandas(db_path, clean_frame):
    result = run_query("q01_overall_churn_rate", db_path)
    row = result.iloc[0]
    assert row["total_customers"] == len(clean_frame)
    assert row["churned_customers"] == int(clean_frame["churn_label"].sum())
    assert row["churn_rate_pct"] == pytest.approx(clean_frame["churn_label"].mean() * 100, abs=0.01)


def test_revenue_aggregates_match_pandas(db_path, clean_frame):
    result = run_query("q06_monthly_revenue_lost_to_churn", db_path)
    row = result.iloc[0]
    churned = clean_frame[clean_frame["churn_label"] == 1]
    assert row["monthly_revenue_churned"] == pytest.approx(churned["monthly_charges"].sum(), abs=0.05)
    assert row["monthly_revenue_total"] == pytest.approx(clean_frame["monthly_charges"].sum(), abs=0.05)


def test_contract_churn_sums_to_total_churners(db_path, clean_frame):
    result = run_query("q02_churn_by_contract", db_path)
    assert result["churned"].sum() == int(clean_frame["churn_label"].sum())
    assert result["customers"].sum() == len(clean_frame)


def test_tenure_group_covers_every_customer(db_path, clean_frame):
    result = run_query("q03_churn_by_tenure_group", db_path)
    assert result["customers"].sum() == len(clean_frame)


def test_window_function_ranks_are_within_partitions(db_path):
    result = run_query("q10_churn_ranking_by_segment", db_path)
    for contract, group in result.groupby("contract"):
        assert group["churn_rank_in_contract"].min() == 1, contract


def test_unknown_query_name_raises():
    with pytest.raises(KeyError):
        run_query("q99_does_not_exist")


def test_cumulative_revenue_is_monotonic(db_path):
    result = run_query("q13_tenure_revenue_cohorts", db_path)
    cumulative = result["cumulative_monthly_revenue"].tolist()
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == pytest.approx(result["monthly_revenue"].sum(), abs=0.05)
