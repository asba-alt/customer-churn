# Data Dictionary — Cleaned Customer Table

**File:** `data/processed/telco_churn_clean.csv` (~7,043 rows × 33 columns)
**Also exported as:** `data/exports/powerbi/customers.csv`

All columns are non-null. The original 21 raw columns are renamed to snake_case;
the remaining columns are documented derived features.

> Monetary columns carry **no documented currency** in the source dataset. Treat
> the units as generic "charge units" throughout.

---

## Identifier

| Column | Type | Description |
|---|---|---|
| `customer_id` | text | Unique customer identifier (from `customerID`). Excluded from all models. |

---

## Demographics

| Column | Type | Description |
|---|---|---|
| `gender` | text | `Female` / `Male` |
| `senior_citizen` | 0/1 | From `SeniorCitizen`, stored 0/1 in the source |
| `senior_citizen_label` | text | `Yes` / `No` — readable mirror of the flag, for visuals |
| `partner` | 0/1 | Has a partner |
| `dependents` | 0/1 | Has dependents |

---

## Tenure

| Column | Type | Description |
|---|---|---|
| `tenure` | int | Months as a customer, 0–72 |
| `tenure_group` | text | `0-6 months`, `7-12`, `13-24`, `25-48`, `49+ months` |

`tenure_group` exists so the business can talk in cohorts rather than a raw
month count. The bands reflect the observed churn cliff: risk is concentrated in
the first 12 months.

---

## Services

| Column | Type | Description |
|---|---|---|
| `phone_service` | 0/1 | Subscribes to phone service |
| `multiple_lines` | text | `Yes`, `No`, `No phone service` |
| `internet_service` | text | `DSL`, `Fiber optic`, `No` |
| `online_security` | text | `Yes`, `No`, `No internet service` |
| `online_backup` | text | as above |
| `device_protection` | text | as above |
| `tech_support` | text | as above |
| `streaming_tv` | text | as above |
| `streaming_movies` | text | as above |
| `has_internet` | 0/1 | Derived: `internet_service != 'No'` |
| `num_addon_services` | int 0–6 | Derived: count of the six add-ons equal to `Yes` |
| `service_count` | int 0–8 | Derived: phone + internet + add-ons in use |

**Why `No internet service` is not treated as a "No".** It means the product is
unavailable to that customer, not that they declined it. Conflating the two would
misattribute churn to the absence of a service the customer could never buy.
`num_addon_services` therefore counts only explicit `Yes`.

---

## Contract and billing

| Column | Type | Description |
|---|---|---|
| `contract` | text | `Month-to-month`, `One year`, `Two year` |
| `paperless_billing` | 0/1 | Enrolled in paperless billing |
| `payment_method` | text | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `auto_payment` | 0/1 | Derived: `payment_method` contains "automatic" |

---

## Charges

| Column | Type | Description |
|---|---|---|
| `monthly_charges` | float | Current monthly recurring charge |
| `total_charges` | float | Cumulative charges billed to date |
| `total_charges_imputed` | 0/1 | 1 where the source value was blank and was imputed as 0.0 |
| `avg_monthly_charge` | float | Derived: `total_charges / tenure`, falling back to `monthly_charges` when `tenure = 0` |
| `revenue_band` | text | Derived monthly charge band: `Low` (<35), `Mid-Low` (35–70), `Mid-High` (70–90), `High` (≥90) |
| `high_value_customer` | 0/1 | Derived: `monthly_charges` ≥ 75th percentile (89.85) |

### On the 11 imputed `total_charges` values

Every blank `TotalCharges` belongs to a customer with `tenure = 0` — they have not
been billed yet, so 0.0 is the correct value, not a guess. The
`total_charges_imputed` flag preserves that fact so any downstream analysis can
include or exclude them. No rows were dropped.

### On `avg_monthly_charge`

For `tenure > 0` this is the realised average spend per month. For `tenure = 0`
it falls back to `monthly_charges`, which is the best available forward estimate.
Dividing by zero is never allowed.

---

## Target

| Column | Type | Description |
|---|---|---|
| `churn` | text | `Yes` / `No` — the source target, kept readable |
| `churn_label` | 0/1 | Modelling target, 1 = churned |

Both are kept: `churn` for human-facing exports and visuals, `churn_label` for
arithmetic and modelling. They always agree.

---

## Quality

| Column | Type | Description |
|---|---|---|
| `data_quality_flag` | 0/1 | 1 where the record failed an implausibility check (negative charges, tenure > 72, total charges contradicting tenure). **No rows are deleted**; this sample has 0 flagged records. |

---

## Deliberately absent columns

| Not present | Reason |
|---|---|
| `clv` / `lifetime_value` | Not identifiable: no dates, no cancellation timing, no discount rate. `total_charges` is the factual realised value; `monthly_charges × 12` is a labelled proxy. |
| `churn_prediction_from_leak` | Any feature derived from the target is blocked by an assertion in `src/models/preprocessing.py`. |
| `mrr_growth`, YoY, MoM | No time dimension exists in the dataset. |
| Currency | The source documents none. Do not label these numbers as USD/EUR. |

---

## Power BI export files

| File | Grain | Rows |
|---|---|---|
| `customers.csv` | customer | 7,043 |
| `customer_risk_scores.csv` | customer | 7,043 |
| `retention_priority_list.csv` | prioritised customer | 704 |
| `retention_test_evaluation.csv` | held-out customer | 1,409 |
| `retention_lift.csv` | capacity level | 5 |
| `retention_scenarios.csv` | capacity level | 5 |
| `retention_gains_curve.csv` | population share point | 101 |
| `retention_segment_profile.csv` | priority segment | 4 |
| `risk_distribution.csv` | risk tier | 4 |
| `customer_segments.csv` | contract × revenue band | 12 |
| `churn_by_contract.csv` | contract | 3 |
| `revenue_exposure.csv` | segment | 7 |
| `revenue_by_segment.csv` | contract × revenue band | 12 |
| `customer_value_proxies.csv` | value quintile | 5 |
| `tenure_profile.csv` | outcome | 2 |
| `data_dictionary.csv` | file | 16 |

Generated automatically — see `src/pipeline.py`.
