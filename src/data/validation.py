"""Dataset validation and the generated data-quality report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from .cleaning import CleaningLog
from .ingestion import file_fingerprint

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "customer_id",
    "tenure",
    "contract",
    "monthly_charges",
    "total_charges",
    "internet_service",
    "payment_method",
    "churn",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_validation_checks(clean: pd.DataFrame) -> list[CheckResult]:
    """Post-cleaning assertions that must hold before analytics run."""
    results: list[CheckResult] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in clean.columns]
    results.append(
        CheckResult(
            "required_columns_present",
            not missing,
            "all required columns present" if not missing else f"missing: {missing}",
        )
    )

    results.append(
        CheckResult(
            "unique_customer_ids",
            clean["customer_id"].is_unique,
            f"{clean['customer_id'].nunique()} unique ids across {len(clean)} rows",
        )
    )

    null_counts = clean[REQUIRED_COLUMNS].isna().sum()
    results.append(
        CheckResult(
            "no_nulls_in_required_columns",
            bool((null_counts == 0).all()),
            "no nulls" if (null_counts == 0).all() else f"nulls: {null_counts[null_counts > 0].to_dict()}",
        )
    )

    results.append(
        CheckResult(
            "target_is_binary",
            set(clean["churn"].unique()) <= {"Yes", "No"},
            f"values: {sorted(clean['churn'].unique().tolist())}",
        )
    )

    results.append(
        CheckResult(
            "total_charges_numeric",
            pd.api.types.is_numeric_dtype(clean["total_charges"]),
            f"dtype: {clean['total_charges'].dtype}",
        )
    )

    results.append(
        CheckResult(
            "monthly_charges_positive",
            bool((clean["monthly_charges"] > 0).all()),
            f"min: {clean['monthly_charges'].min():.2f}",
        )
    )

    results.append(
        CheckResult(
            "tenure_non_negative",
            bool((clean["tenure"] >= 0).all()),
            f"min: {clean['tenure'].min()}, max: {clean['tenure'].max()}",
        )
    )

    churn_rate = (clean["churn"] == "Yes").mean()
    results.append(
        CheckResult(
            "churn_rate_plausible",
            0.0 < churn_rate < 1.0,
            f"{churn_rate:.2%} of customers churned",
        )
    )

    return results


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavoured markdown table."""
    if frame.empty:
        return "_No rows._"
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _raw_missing_summary(raw: pd.DataFrame) -> pd.DataFrame:
    blank_like = ["", " ", "nan", "NaN", "None", "null", "<NA>"]
    counts = {}
    for col in raw.columns:
        series = raw[col].astype("string")
        n_blank = int(series.str.strip().isin(blank_like).sum() + series.isna().sum())
        counts[col] = n_blank
    out = pd.DataFrame(
        {"column": list(counts), "blank_or_null": list(counts.values())}
    )
    out["pct_of_rows"] = (out["blank_or_null"] / len(raw) * 100).round(2).astype(str) + "%"
    return out.sort_values("blank_or_null", ascending=False).reset_index(drop=True)


def build_data_quality_report(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    log: CleaningLog,
    output_path: Path | None = None,
) -> str:
    """Write reports/data_quality_report.md and return the markdown text."""
    output_path = output_path or config.REPORT_DATA_QUALITY
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checks = run_validation_checks(clean)
    fingerprint = file_fingerprint()
    blank_table = _raw_missing_summary(raw)

    schema_table = pd.DataFrame(
        {
            "column": list(clean.columns),
            "dtype": [str(clean[c].dtype) for c in clean.columns],
            "non_null": [int(clean[c].notna().sum()) for c in clean.columns],
            "distinct": [int(clean[c].nunique(dropna=True)) for c in clean.columns],
        }
    )

    churn_counts = clean["churn"].value_counts()
    churn_rate = (clean["churn"] == "Yes").mean()

    tenure_group_table = (
        clean.groupby("tenure_group", observed=True)
        .agg(customers=("customer_id", "count"), churn_rate=("churn_label", "mean"))
        .reset_index()
    )
    tenure_group_table["churn_rate"] = (tenure_group_table["churn_rate"] * 100).round(2).astype(str) + "%"

    contract_table = (
        clean.groupby("contract", observed=True)
        .agg(customers=("customer_id", "count"), churn_rate=("churn_label", "mean"))
        .reset_index()
    )
    contract_table["churn_rate"] = (contract_table["churn_rate"] * 100).round(2).astype(str) + "%"

    check_table = pd.DataFrame(
        {
            "check": [c.name for c in checks],
            "status": ["PASS" if c.passed else "FAIL" for c in checks],
            "detail": [c.detail for c in checks],
        }
    )

    decision_lines = "\n".join(
        f"{i}. **{issue}.** {decision}" for i, (issue, decision) in enumerate(log.decisions, start=1)
    )

    flagged_table = pd.DataFrame(
        {
            "quality_issue": list(log.invalid_record_breakdown.keys()),
            "records": list(log.invalid_record_breakdown.values()),
        }
    )

    if log.invalid_records_flagged:
        dominant = max(log.invalid_record_breakdown, key=log.invalid_record_breakdown.get)
        flagged_note = (
            f"`data_quality_flag = 1` marks **{log.invalid_records_flagged}** records. "
            f"They are kept so the modelled population matches the population the business "
            f"actually has. The largest contributor is `{dominant}` "
            f"({log.invalid_record_breakdown[dominant]} records), which fires when a "
            "customer's total charges contradict their tenure."
        )
    else:
        flagged_note = (
            "**No records failed these checks.** `data_quality_flag` is present on every row "
            "and set to 0 throughout, so the guard is in place for future data without "
            "deleting anything from this sample."
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    all_checks_pass = all(c.passed for c in checks)

    markdown = f"""# Data Quality Report — Telco Customer Churn

**Generated:** {generated_at}
**Source file:** `{config.RAW_FILENAME}`
**SHA-256:** `{fingerprint['sha256']}`
**Raw shape:** {log.row_count_raw:,} rows x {log.column_count_raw} columns
**Cleaned shape:** {log.row_count_clean:,} rows x {log.column_count_clean} columns
**Validation status:** {'ALL CHECKS PASSED' if all_checks_pass else 'ATTENTION REQUIRED — see checks table'}

---

## 1. Dataset and scope

This is the public **IBM Telco Customer Churn** sample dataset — a single
cross-sectional snapshot of ~7,043 fictional customers of a fictional
telecommunications company. It is used as a customer-churn classification and
retention-prioritisation problem.

Limitations that matter when reading anything downstream:

- **Fictional sample data.** Numbers describe the sample, not a real operator.
- **No time dimension.** There is one row per customer, not a monthly panel, so
  nothing in this project is a longitudinal forecast.
- **No monetary unit is documented.** Revenue figures are in the dataset's own
  charge units and are described as charge units, not a currency.
- **No intervention or cost data.** We can rank risk, but we cannot claim saved
  revenue because no retention campaign was run and no campaign cost exists.
- **No dates.** Tenure is months since acquisition; there is no calendar date to
  join against macro or seasonal effects.

---

## 2. Raw-file missing values

The CSV ships blank strings rather than nulls. Counting blank-like values per
column before any treatment:

{_markdown_table(blank_table)}

Only `TotalCharges` has genuinely missing content: **{log.total_charges_blank} blank
values**, and every one of them belongs to a customer with `tenure == 0` (a
customer who has not yet been billed). `SeniorCitizen` is stored as 0/1 despite
looking categorical; it is treated as a binary flag.

---

## 3. Duplicates

| Check | Count |
|---|---|
| Duplicate `customerID` values | {log.duplicate_customer_ids} |
| Fully duplicated rows | {log.duplicate_full_rows} |
| Rows removed as duplicates | {log.duplicates_removed} |

---

## 4. Cleaning decisions (every one is deliberate)

{decision_lines}

### Records flagged as implausible (retained, not deleted)

{_markdown_table(flagged_table)}

{flagged_note}

---

## 5. Cleaned schema

{_markdown_table(schema_table)}

---

## 6. Validation checks

{_markdown_table(check_table)}

---

## 7. Distributions after cleaning

| Metric | Value |
|---|---|
| Customers | {len(clean):,} |
| Churned (Yes) | {int(churn_counts.get('Yes', 0)):,} |
| Retained (No) | {int(churn_counts.get('No', 0)):,} |
| Overall churn rate | {churn_rate:.2%} |
| Class imbalance (retained:churned) | {churn_counts.get('No', 0) / max(churn_counts.get('Yes', 1), 1):.2f} : 1 |
| Mean tenure (months) | {clean['tenure'].mean():.2f} |
| Median tenure (months) | {clean['tenure'].median():.1f} |
| Mean monthly charges | {clean['monthly_charges'].mean():.2f} |
| Median monthly charges | {clean['monthly_charges'].median():.2f} |
| Mean total charges | {clean['total_charges'].mean():.2f} |
| Total monthly revenue (sum of monthly charges) | {clean['monthly_charges'].sum():,.2f} |
| Duplicate rows remaining | {int(clean['customer_id'].duplicated().sum())} |
| Null cells in cleaned file | {int(clean.isna().sum().sum())} |

### Churn by tenure group

{_markdown_table(tenure_group_table)}

### Churn by contract

{_markdown_table(contract_table)}

---

## 8. Reproducibility

```
python main.py pipeline    # ingest -> clean -> validate -> this report
python main.py all         # full run: data, SQL, EDA, models, scoring, reports
```

Deterministic seed: `{config.SEED}`. The raw file is opened read-only and is
never rewritten by any stage; `data/processed/telco_churn_clean.csv` is always
regenerated from it.
"""

    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote data quality report to %s", output_path)
    return markdown
