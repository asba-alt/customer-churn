# Data Quality Report — Telco Customer Churn

**Generated:** 2026-11-19 15:25 UTC
**Source file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
**SHA-256:** `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
**Raw shape:** 7,043 rows x 21 columns
**Cleaned shape:** 7,043 rows x 33 columns
**Validation status:** ALL CHECKS PASSED

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

| column | blank_or_null | pct_of_rows |
|---|---|---|
| TotalCharges | 11 | 0.16% |
| gender | 0 | 0.0% |
| SeniorCitizen | 0 | 0.0% |
| Partner | 0 | 0.0% |
| customerID | 0 | 0.0% |
| Dependents | 0 | 0.0% |
| tenure | 0 | 0.0% |
| MultipleLines | 0 | 0.0% |
| PhoneService | 0 | 0.0% |
| OnlineSecurity | 0 | 0.0% |
| OnlineBackup | 0 | 0.0% |
| DeviceProtection | 0 | 0.0% |
| InternetService | 0 | 0.0% |
| TechSupport | 0 | 0.0% |
| StreamingTV | 0 | 0.0% |
| Contract | 0 | 0.0% |
| StreamingMovies | 0 | 0.0% |
| PaperlessBilling | 0 | 0.0% |
| PaymentMethod | 0 | 0.0% |
| MonthlyCharges | 0 | 0.0% |
| Churn | 0 | 0.0% |

Only `TotalCharges` has genuinely missing content: **11 blank
values**, and every one of them belongs to a customer with `tenure == 0` (a
customer who has not yet been billed). `SeniorCitizen` is stored as 0/1 despite
looking categorical; it is treated as a binary flag.

---

## 3. Duplicates

| Check | Count |
|---|---|
| Duplicate `customerID` values | 0 |
| Fully duplicated rows | 0 |
| Rows removed as duplicates | 0 |

---

## 4. Cleaning decisions (every one is deliberate)

1. **Blank TotalCharges.** 11 rows had a blank TotalCharges, all belonging to tenure == 0 customers (0 blanks were outside that group). They were converted to numeric and imputed with 0.0 because a customer with zero months of tenure cannot have accumulated charges. The rows are retained and marked with total_charges_imputed = 1 rather than deleted.
2. **Categorical encoding.** Binary Yes/No service and demographic flags (partner, dependents, phone_service, paperless_billing) were encoded as 1/0. Multi-level categories were kept as readable text in the cleaned file and only one-hot encoded inside the model pipeline, so business exports stay human-readable.
3. **Duplicate customer IDs.** No duplicate customer IDs were found; every row is a distinct customer.
4. **Fully duplicated rows.** No fully duplicated rows were found.
5. **Invalid / implausible records.** Records with negative tenure or charges, non-positive monthly charges, tenure above the dataset's 72-month ceiling, or total charges that contradict the tenure were FLAGGED (data_quality_flag = 1), not deleted. Deleting them would bias the churn rate; flagging keeps them auditable and lets the model see the same population the business would see.
6. **Derived columns.** Added tenure_group (business-friendly tenure bands), num_addon_services, has_internet, service_count, avg_monthly_charge, revenue_band, high_value_customer (>= 75th percentile monthly charges), auto_payment and churn_label (the modelling target).
7. **Row retention.** 7043 raw rows -> 7043 cleaned rows. 0 duplicate rows removed and 0 rows with an invalid target removed; no other record was dropped.

### Records flagged as implausible (retained, not deleted)

| quality_issue | records |
|---|---|
| negative_tenure | 0 |
| non_positive_monthly_charges | 0 |
| negative_total_charges | 0 |
| tenure_above_expected_max | 0 |
| total_charges_inconsistent | 0 |

**No records failed these checks.** `data_quality_flag` is present on every row and set to 0 throughout, so the guard is in place for future data without deleting anything from this sample.

---

## 5. Cleaned schema

| column | dtype | non_null | distinct |
|---|---|---|---|
| customer_id | str | 7043 | 7043 |
| gender | string | 7043 | 2 |
| senior_citizen | int8 | 7043 | 2 |
| senior_citizen_label | string | 7043 | 2 |
| partner | int8 | 7043 | 2 |
| dependents | int8 | 7043 | 2 |
| tenure | int64 | 7043 | 73 |
| tenure_group | string | 7043 | 5 |
| phone_service | int8 | 7043 | 2 |
| multiple_lines | string | 7043 | 3 |
| internet_service | string | 7043 | 3 |
| online_security | string | 7043 | 3 |
| online_backup | string | 7043 | 3 |
| device_protection | string | 7043 | 3 |
| tech_support | string | 7043 | 3 |
| streaming_tv | string | 7043 | 3 |
| streaming_movies | string | 7043 | 3 |
| contract | string | 7043 | 3 |
| paperless_billing | int8 | 7043 | 2 |
| payment_method | string | 7043 | 4 |
| auto_payment | int8 | 7043 | 2 |
| monthly_charges | float64 | 7043 | 1585 |
| total_charges | float64 | 7043 | 6531 |
| total_charges_imputed | int8 | 7043 | 2 |
| avg_monthly_charge | float64 | 7043 | 4579 |
| num_addon_services | int8 | 7043 | 7 |
| has_internet | int8 | 7043 | 2 |
| service_count | int8 | 7043 | 8 |
| revenue_band | string | 7043 | 4 |
| high_value_customer | int8 | 7043 | 2 |
| churn | string | 7043 | 2 |
| churn_label | int8 | 7043 | 2 |
| data_quality_flag | int8 | 7043 | 1 |

---

## 6. Validation checks

| check | status | detail |
|---|---|---|
| required_columns_present | PASS | all required columns present |
| unique_customer_ids | PASS | 7043 unique ids across 7043 rows |
| no_nulls_in_required_columns | PASS | no nulls |
| target_is_binary | PASS | values: ['No', 'Yes'] |
| total_charges_numeric | PASS | dtype: float64 |
| monthly_charges_positive | PASS | min: 18.25 |
| tenure_non_negative | PASS | min: 0, max: 72 |
| churn_rate_plausible | PASS | 26.54% of customers churned |

---

## 7. Distributions after cleaning

| Metric | Value |
|---|---|
| Customers | 7,043 |
| Churned (Yes) | 1,869 |
| Retained (No) | 5,174 |
| Overall churn rate | 26.54% |
| Class imbalance (retained:churned) | 2.77 : 1 |
| Mean tenure (months) | 32.37 |
| Median tenure (months) | 29.0 |
| Mean monthly charges | 64.76 |
| Median monthly charges | 70.35 |
| Mean total charges | 2279.73 |
| Total monthly revenue (sum of monthly charges) | 456,116.60 |
| Duplicate rows remaining | 0 |
| Null cells in cleaned file | 0 |

### Churn by tenure group

| tenure_group | customers | churn_rate |
|---|---|---|
| 0-6 months | 1481 | 52.94% |
| 13-24 months | 1024 | 28.71% |
| 25-48 months | 1594 | 20.39% |
| 49+ months | 2239 | 9.51% |
| 7-12 months | 705 | 35.89% |

### Churn by contract

| contract | customers | churn_rate |
|---|---|---|
| Month-to-month | 3875 | 42.71% |
| One year | 1473 | 11.27% |
| Two year | 1695 | 2.83% |

---

## 8. Reproducibility

```
python main.py pipeline    # ingest -> clean -> validate -> this report
python main.py all         # full run: data, SQL, EDA, models, scoring, reports
```

Deterministic seed: `42`. The raw file is opened read-only and is
never rewritten by any stage; `data/processed/telco_churn_clean.csv` is always
regenerated from it.
