# DAX Measures — Customer Retention & Revenue Intelligence

Import `data/exports/powerbi/customers.csv` as table **Customers** and
`customer_risk_scores.csv` as **RiskScores**. Measures below assume that naming.

Metric values quoted in comments are the verified results of the current run; they
are documentation, not calculations. The DAX itself always recomputes.

---

## 1. Base customer measures

```dax
Total Customers =
COUNTROWS ( Customers )
-- Verified: 7,043

Churned Customers =
CALCULATE ( COUNTROWS ( Customers ), Customers[churn] = "Yes" )
-- Verified: 1,869

Retained Customers =
CALCULATE ( COUNTROWS ( Customers ), Customers[churn] = "No" )
-- Verified: 5,174

Churn Rate % =
DIVIDE ( [Churned Customers], [Total Customers] )
-- Verified: 0.2654  (format as percentage, 1 dp)

Retention Rate % =
DIVIDE ( [Retained Customers], [Total Customers] )
```

---

## 2. Revenue measures

```dax
Total Monthly Revenue =
SUM ( Customers[monthly_charges] )
-- Verified: 456,116.60
-- Recurring revenue for ONE month. Never multiply this into a year unless the
-- assumption is stated explicitly.

Monthly Revenue Churned =
CALCULATE (
    SUM ( Customers[monthly_charges] ),
    Customers[churn] = "Yes"
)
-- Verified: 139,130.85

Monthly Revenue Retained =
CALCULATE (
    SUM ( Customers[monthly_charges] ),
    Customers[churn] = "No"
)

Revenue Share Lost % =
DIVIDE ( [Monthly Revenue Churned], [Total Monthly Revenue] )
-- Verified: 0.3050

Avg Monthly Charges =
AVERAGE ( Customers[monthly_charges] )
-- Verified: 64.76

Avg Monthly Charges Churned =
CALCULATE (
    AVERAGE ( Customers[monthly_charges] ),
    Customers[churn] = "Yes"
)
-- Verified: 74.44

Avg Monthly Charges Retained =
CALCULATE (
    AVERAGE ( Customers[monthly_charges] ),
    Customers[churn] = "No"
)
-- Verified: 61.27

Revenue Realised To Date =
SUM ( Customers[total_charges] )
-- Verified: 16,056,168.70
-- This is what has actually been billed. It is the only factual revenue total.

Annualised Revenue Proxy =
SUMX ( Customers, Customers[monthly_charges] * 12 )
-- A proxy that ignores churn, discounting and price change. Label it as such on
-- the report; it is NOT annual revenue.
```

---

## 3. Tenure measures

```dax
Avg Tenure Months =
AVERAGE ( Customers[tenure] )
-- Verified: 32.37

Median Tenure Months =
MEDIAN ( Customers[tenure] )

Avg Tenure Churned =
CALCULATE ( AVERAGE ( Customers[tenure] ), Customers[churn] = "Yes" )
-- Verified: 17.98

Avg Tenure Retained =
CALCULATE ( AVERAGE ( Customers[tenure] ), Customers[churn] = "No" )
-- Verified: 37.57
```

---

## 4. Segment measures

```dax
Contract Customers =
COUNTROWS ( Customers )

Contract Churn Rate % =
DIVIDE ( SUM ( Customers[churn_label] ), COUNTROWS ( Customers ) )

Churn Lift vs Base =
VAR SegmentRate = [Contract Churn Rate %]
VAR BaseRate = DIVIDE (
    CALCULATE ( SUM ( Customers[churn_label] ), ALL ( Customers ) ),
    CALCULATE ( COUNTROWS ( Customers ), ALL ( Customers ) )
)
RETURN DIVIDE ( SegmentRate, BaseRate )
-- Verified for Month-to-month: 1.61x ; Two year: 0.11x

High Value Customers =
CALCULATE ( COUNTROWS ( Customers ), Customers[high_value_customer] = 1 )
-- Verified: 1,771

High Value Churn Rate % =
CALCULATE (
    AVERAGE ( Customers[churn_label] ),
    Customers[high_value_customer] = 1
)
-- Verified: 0.3275

Auto Payment Share % =
DIVIDE (
    CALCULATE ( COUNTROWS ( Customers ), Customers[auto_payment] = 1 ),
    COUNTROWS ( Customers )
)
```

---

## 5. Predictive / risk measures

```dax
Avg Churn Probability =
AVERAGE ( RiskScores[churn_probability] )
-- Verified: 0.4039 (all customers)

Avg Churn Probability High Risk =
CALCULATE (
    AVERAGE ( RiskScores[churn_probability] ),
    RiskScores[risk_category] IN { "Very High", "High" }
)

Predicted Churners =
CALCULATE (
    COUNTROWS ( RiskScores ),
    RiskScores[predicted_churn] = 1
)

Expected Monthly Revenue At Risk =
SUM ( RiskScores[expected_monthly_revenue_at_risk] )
-- Σ(monthly charge × predicted probability). An EXPOSURE estimate, never savings.

Expected Annual Revenue At Risk =
SUM ( RiskScores[expected_annual_revenue_at_risk] )
-- Verified (top decile): 624,301.49

Revenue At Risk Share % =
DIVIDE (
    [Expected Monthly Revenue At Risk],
    SUM ( RiskScores[monthly_charges] )
)
```

---

## 6. Risk tier measures

```dax
Customers Very High Risk =
CALCULATE (
    COUNTROWS ( RiskScores ),
    RiskScores[risk_category] = "Very High"
)
-- Verified: 382  (5.42% of base; tiers use validation-derived cut-offs)

Customers High Risk =
CALCULATE (
    COUNTROWS ( RiskScores ),
    RiskScores[risk_category] = "High"
)
-- Verified: 1,040  (14.77% of base)

High Risk Share % =
DIVIDE (
    [Customers Very High Risk] + [Customers High Risk],
    COUNTROWS ( RiskScores )
)
```

---

## 7. Retention prioritisation measures

These read the **priority list**, which is the top decile of the full customer
base. Use these for the operational call list, not for lift evaluation.

```dax
Prioritised Customers =
COUNTROWS ( PriorityList )
-- Verified: 704 (top 10% of 7,043)

Priority Monthly Revenue =
SUM ( PriorityList[monthly_charges] )
-- Verified: 57,906.95

Priority Expected Revenue At Risk =
SUM ( PriorityList[expected_monthly_revenue_at_risk] )
-- Verified: 52,025.23

Priority Avg Probability =
AVERAGE ( PriorityList[churn_probability] )
-- Verified: 0.8973
```

---

## 8. Lift and model performance measures

These use the **out-of-sample** test table (`retention_test_evaluation.csv`) and
the pre-computed summary tables. That distinction is the whole point: lift must be
measured where the model had no sight of the outcome.

```dax
Lift at 10 Percent =
LOOKUPVALUE ( Lift[lift], Lift[capacity], 0.1 )
-- Verified: 2.91

OOS Churn Rate Top Decile =
LOOKUPVALUE ( Lift[churn_rate_in_group], Lift[capacity], 0.1 )
-- Verified: 0.773  (77.3%)

OOS Overall Churn Rate =
LOOKUPVALUE ( Lift[overall_churn_rate], Lift[capacity], 0.1 )
-- Verified: 0.2654

OOS Recall Top Decile =
LOOKUPVALUE ( Lift[recall_of_churners], Lift[capacity], 0.1 )
-- Verified: 0.2914  (29.1% of churners captured)

Lift vs Random =
[Lift at 10 Percent] - 1
-- The multiple of extra churners reached compared with random targeting.

-- Model metrics are constants from the latest training run.
-- Source of truth: models/model_metadata.json
ROC AUC = 0.8438
PR AUC = 0.6583
Recall at Threshold = 0.9225
Precision at Threshold = 0.4390
Decision Threshold = 0.293

Model Coverage % =
DIVIDE (
    CALCULATE ( COUNTROWS ( RiskScores ), NOT ISBLANK ( RiskScores[churn_probability] ) ),
    COUNTROWS ( Customers )
)
```

**Important:** `ROC AUC`, `PR AUC` and the threshold measures are **static
constants**. They are not recalculated by Power BI. If you retrain the model,
update them from `models/model_metadata.json`, or the dashboard will silently
misreport performance.

---

## 9. Formatting guidance

| Measure pattern | Format string |
|---|---|
| Anything ending `%` | Percentage, 1–2 dp |
| `Total Monthly Revenue`, `* At Risk` | `#,##0` (no decimals — the units are fictional) |
| `* Lift *` | `0.00"x"` |
| `churn_probability` | Percentage, 1 dp |
| `Avg Tenure Months` | `0.0` |

---

## 10. Measures deliberately NOT included

These would be easy to write and would be wrong:

- **`Monthly Recurring Revenue Growth`** — no time dimension exists in the data.
- **`CLV`** — no dates, no cancellation timing, no discount rate. Any "CLV" here
  would be an invented number. Use `Realised Revenue To Date` instead.
- **`Revenue Saved`** — no campaign ran, no save rate is known. Reporting this
  would be fabrication. Use `Expected Monthly Revenue At Risk`, clearly labelled
  as exposure.
- **`Churn Prediction Accuracy`** — accuracy is a misleading headline on a 26.5%
  positive class. Report PR-AUC and lift instead.
