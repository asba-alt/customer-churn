"""SQLite analytical database: schema creation and loading.

The cleaned CSV is loaded into a small star-ish schema rather than one flat
table. That is deliberate: churn questions cut across customer attributes,
subscribed services and billing, so separate tables make the JOINs in
``sql/analytics_queries.sql`` meaningful instead of decorative.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

SCHEMA = """
DROP VIEW  IF EXISTS v_customer_360;
DROP TABLE IF EXISTS fact_billing;
DROP TABLE IF EXISTS dim_contract;
DROP TABLE IF EXISTS dim_service;
DROP TABLE IF EXISTS dim_customer;

CREATE TABLE dim_customer (
    customer_id        TEXT PRIMARY KEY,
    gender             TEXT    NOT NULL,
    senior_citizen     INTEGER NOT NULL,
    partner            INTEGER NOT NULL,
    dependents         INTEGER NOT NULL
);

CREATE TABLE dim_service (
    customer_id        TEXT PRIMARY KEY REFERENCES dim_customer(customer_id),
    phone_service      INTEGER NOT NULL,
    multiple_lines     TEXT    NOT NULL,
    internet_service   TEXT    NOT NULL,
    online_security    TEXT    NOT NULL,
    online_backup      TEXT    NOT NULL,
    device_protection  TEXT    NOT NULL,
    tech_support       TEXT    NOT NULL,
    streaming_tv       TEXT    NOT NULL,
    streaming_movies   TEXT    NOT NULL,
    has_internet       INTEGER NOT NULL,
    num_addon_services INTEGER NOT NULL,
    service_count      INTEGER NOT NULL
);

CREATE TABLE dim_contract (
    customer_id        TEXT PRIMARY KEY REFERENCES dim_customer(customer_id),
    contract           TEXT    NOT NULL,
    paperless_billing  INTEGER NOT NULL,
    payment_method     TEXT    NOT NULL,
    auto_payment       INTEGER NOT NULL
);

CREATE TABLE fact_billing (
    customer_id        TEXT PRIMARY KEY REFERENCES dim_customer(customer_id),
    tenure             INTEGER NOT NULL,
    tenure_group       TEXT    NOT NULL,
    monthly_charges    REAL    NOT NULL,
    total_charges      REAL    NOT NULL,
    avg_monthly_charge REAL    NOT NULL,
    revenue_band       TEXT    NOT NULL,
    high_value_customer INTEGER NOT NULL,
    churn              TEXT    NOT NULL,
    churn_label        INTEGER NOT NULL,
    data_quality_flag  INTEGER NOT NULL
);

CREATE INDEX idx_dim_contract_contract ON dim_contract(contract);
CREATE INDEX idx_dim_service_internet   ON dim_service(internet_service);
CREATE INDEX idx_fact_churn             ON fact_billing(churn);
CREATE INDEX idx_fact_tenure_group      ON fact_billing(tenure_group);

CREATE VIEW v_customer_360 AS
SELECT
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    s.has_internet,
    s.num_addon_services,
    s.service_count,
    k.contract,
    k.paperless_billing,
    k.payment_method,
    k.auto_payment,
    b.tenure,
    b.tenure_group,
    b.monthly_charges,
    b.total_charges,
    b.avg_monthly_charge,
    b.revenue_band,
    b.high_value_customer,
    b.churn,
    b.churn_label
FROM dim_customer   c
JOIN dim_service    s ON s.customer_id = c.customer_id
JOIN dim_contract   k ON k.customer_id = c.customer_id
JOIN fact_billing   b ON b.customer_id = c.customer_id;
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to the analytics database (created on demand)."""
    db_path = Path(db_path) if db_path is not None else config.DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _frames(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "dim_customer": clean[
            ["customer_id", "gender", "senior_citizen", "partner", "dependents"]
        ].copy(),
        "dim_service": clean[
            [
                "customer_id",
                "phone_service",
                "multiple_lines",
                "internet_service",
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
                "has_internet",
                "num_addon_services",
                "service_count",
            ]
        ].copy(),
        "dim_contract": clean[
            ["customer_id", "contract", "paperless_billing", "payment_method", "auto_payment"]
        ].copy(),
        "fact_billing": clean[
            [
                "customer_id",
                "tenure",
                "tenure_group",
                "monthly_charges",
                "total_charges",
                "avg_monthly_charge",
                "revenue_band",
                "high_value_customer",
                "churn",
                "churn_label",
                "data_quality_flag",
            ]
        ].copy(),
    }


def build_database(clean: pd.DataFrame, db_path: Path | None = None) -> Path:
    """(Re)build the SQLite database from the cleaned customer table."""
    db_path = Path(db_path) if db_path is not None else config.DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Build into a fresh file so a stale schema can never linger.
    if db_path.exists():
        db_path.unlink()

    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        for name, frame in _frames(clean).items():
            frame.to_sql(name, conn, if_exists="append", index=False)
        conn.commit()

        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("dim_customer", "dim_service", "dim_contract", "fact_billing")
        }
        view_count = conn.execute("SELECT COUNT(*) FROM v_customer_360").fetchone()[0]
        if not all(v == len(clean) for v in counts.values()) or view_count != len(clean):
            raise RuntimeError(f"Row counts do not match the cleaned dataset: {counts}")
        logger.info("Built SQLite database at %s %s", db_path, counts)
    finally:
        conn.close()
    return db_path


def table_names(db_path: Path | None = None) -> list[str]:
    """List user tables and views (used by the tests)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()
