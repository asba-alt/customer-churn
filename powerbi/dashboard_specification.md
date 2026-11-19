# Power BI Dashboard Specification

**Report name:** Customer Retention & Revenue Intelligence
**Pages:** 3
**Data source:** CSVs in `data/exports/powerbi/` (import mode)
**Status:** *Specification and import-ready files complete. The `.pbix` must be
opened and validated in Power BI Desktop — this document does not claim a
validated report exists.*

> No `.pbix` file is shipped. Creating one requires Power BI Desktop, which
> cannot be driven headlessly here. Everything needed to build and validate the
> report is in this folder: import-ready CSVs, a documented data model, a full
> DAX measure library, and page-by-page layout instructions.

---

## 1. Import the data

`Get Data → Text/CSV` for each file below. Use **Import** mode, not DirectQuery.

| File | Grain | Rows | Purpose |
|---|---|---|---|
| `customers.csv` | one row per customer | 7,043 | Main analytics table |
| `customer_risk_scores.csv` | one row per customer | 7,043 | Model scores and risk tiers |
| `retention_priority_list.csv` | one row per prioritised customer | 704 | The call list for the top decile |
| `retention_lift.csv` | one row per capacity level | 5 | Lift and capture table |
| `retention_scenarios.csv` | one row per capacity level | 5 | Capacity planning |
| `retention_segment_profile.csv` | priority-list segment | 4 | Who dominates the call list |
| `retention_gains_curve.csv` | 101 points | 101 | Cumulative gains chart |
| `retention_test_evaluation.csv` | one row per held-out customer | 1,409 | Out-of-sample evaluation |
| `risk_distribution.csv` | one row per risk tier | 4 | Risk tier summary |
| `customer_segments.csv` | segment level | 12 | Contract × value band segments |
| `churn_by_contract.csv` | contract level | 3 | Contract analysis |
| `revenue_exposure.csv` | segment level | 7 | Revenue concentration |
| `revenue_by_segment.csv` | segment level | 12 | Revenue contribution |
| `customer_value_proxies.csv` | value quintile | 5 | Value proxies |
| `tenure_profile.csv` | outcome level | 2 | Tenure comparison |
| `data_dictionary.csv` | file level | — | Self-documentation |

### Import cleanup steps (do these in Power Query)

1. `customers.csv` — set `customer_id` to **Text**. Set `senior_citizen`,
   `partner`, `dependents`, `phone_service`, `paperless_billing`,
   `high_value_customer`, `churn_label`, `auto_payment` to **Whole Number**.
2. `customer_risk_scores.csv` — set `customer_id`, `risk_category`, `contract`,
   `internet_service`, `payment_method` to **Text**; `churn_probability` to
   **Decimal Number**.
3. Remove the `data_quality_flag` column from any visual that counts customers —
   it is an audit column, not a business dimension.
4. **Do not** sum `total_charges`, `monthly_charges`, or `churn_label` across
   rows without an explicit filter. `churn_label` is a 0/1 flag, so its SUM gives
   the churner count deliberately; see the DAX notes.

---

## 2. Data model

Use a **star schema**, not one flat table. The flat `customers.csv` is convenient
but it duplicates every contract attribute 7,043 times and will produce wrong
non-additive totals if someone sums a text-derived column.

```
                    ┌─────────────────────┐
                    │      dim_date       │
                    │  (optional: none)   │
                    └─────────────────────┘
                              │
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ dim_customer │───│  fact_churn   │───│ dim_contract │
   │  customer_id │  1:N│ customer_id  │N:1│  customer_id │
   └──────────────┘    └──────────────┘    └──────────────┘
                              │
                     ┌────────────────┐
                     │ dim_service    │
                     │  customer_id   │
                     └────────────────┘
```

**Practical alternative for this delivery (recommended):** keep
`customers.csv` as a single denormalised table named `Customers` and add the
score columns via a merge. This dataset has one row per customer, so a flat table
is not actually a fan-out risk — the grain is the same. The star schema matters
when you add a monthly snapshot table later.

**Minimum viable model for these 3 pages:**

| Table | Role | Key |
|---|---|---|
| `Customers` | fact + all dimensions | `customer_id` |
| `RiskScores` | model output, related 1:1 | `customer_id` |
| `PriorityList` | filtered subset | `customer_id` |
| `Lift` | disconnected metrics table | — |
| `RiskDistribution` | disconnected summary | — |

Relationships:
- `Customers[customer_id]` 1 ──→ * `RiskScores[customer_id]` (single direction)
- `Customers[customer_id]` 1 ──→ * `PriorityList[customer_id]` (single direction)
- `Lift` and `RiskDistribution` stay **disconnected** — they are pre-aggregated
  measures and must not be filtered by page slicers.

---

## 3. PAGE 1 — Executive Overview

**Purpose:** one screen that a VP can read in fifteen seconds.

### Layout (16:9, 12-column grid)

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Customer Retention & Revenue Intelligence        [Dataset: 7,043 sample] │
├───────────┬───────────┬───────────┬───────────┬───────────┬───────────────┤
│ Total     │ Churn     │ Retained  │ Monthly   │ Revenue   │ Revenue       │
│ Customers │ Rate      │ Customers │ Revenue   │ Lost      │ Share Lost    │
│  7,043    │  26.5%    │  5,174    │  456,117  │  139,131  │  30.5%        │
├───────────┴───────────┴───────────┴───────────┴───────────┴───────────────┤
│                                                                           │
│  Retained vs Churned (donut)        │  Churn Rate by Contract (bar)       │
│                                     │                                     │
├─────────────────────────────────────┴─────────────────────────────────────┤
│                                                                           │
│  Monthly Revenue by Contract (stacked: retained vs churned)               │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Visuals

| # | Type | Fields | Notes |
|---|---|---|---|
| 1 | Card | `[Total Customers]` | Large number, no decimals |
| 2 | Card | `[Churn Rate %]` | 1 decimal |
| 3 | Card | `[Retained Customers]` | |
| 4 | Card | `[Total Monthly Revenue]` | 0 decimals |
| 5 | Card | `[Monthly Revenue Churned]` | 0 decimals |
| 6 | Card | `[Revenue Share Lost %]` | 1 decimal, red accent |
| 7 | Donut | Legend `Customers[churn]`, Values `[Total Customers]` | 2 slices |
| 8 | Clustered bar | Axis `Customers[contract]`, Values `[Churn Rate %]` | Sort desc |
| 9 | Stacked bar | Axis `Customers[contract]`, Legend `Customers[churn]`, Values `[Total Monthly Revenue]` | |

### Conditional formatting
- Card 2 (Churn Rate): font colour `#C1121F` when > 25%, else `#2A9D8F`.
- Card 5 and 6: red accent `#E4572E`.
- Donut: retained `#2A9D8F`, churned `#E4572E`.

### Text box (bottom footnote — required)
> Fictional IBM Telco sample dataset. Revenue figures are monthly recurring
> charges in dataset units, not a currency. "Revenue lost" is monthly recurring
> charge attached to churned customers, not a cumulative realised loss.

---

## 4. PAGE 2 — Customer Analytics

**Purpose:** explain *where* churn concentrates, so someone can act on it.

### Layout

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Slicers: [Contract ▾]  [Internet Service ▾]  [Payment Method ▾]           │
├───────────────────────────────┬───────────────────────────────────────────┤
│ Churn Rate by Tenure Group    │ Churn Rate by Payment Method              │
│ (column)                      │ (bar, sorted)                             │
├───────────────────────────────┼───────────────────────────────────────────┤
│ Churn Rate by Internet Service│ Churn by Add-on Depth (line)              │
├───────────────────────────────┴───────────────────────────────────────────┤
│ Churn Rate Heat Map: Contract (rows) × Monthly Charge Band (columns)       │
├───────────────────────────────┬───────────────────────────────────────────┤
│ Revenue by Customer Segment   │ Churn vs Revenue Exposure (scatter)       │
│ (bar)                         │ x = avg charge, y = churn rate, size = rev│
└───────────────────────────────┴───────────────────────────────────────────┘
```

### Visuals

| # | Type | Fields |
|---|---|---|
| 1 | Slicer | `Customers[contract]`, `Customers[internet_service]`, `Customers[payment_method]` |
| 2 | Clustered column | Axis `Customers[tenure_group]`, Values `[Churn Rate %]` |
| 3 | Clustered bar | Axis `Customers[payment_method]`, Values `[Churn Rate %]` |
| 4 | Clustered column | Axis `Customers[internet_service]`, Values `[Churn Rate %]` |
| 5 | Line + column | Axis `Customers[num_addon_services]`, Columns `[Total Customers]`, Line `[Churn Rate %]` |
| 6 | Matrix | Rows `Customers[contract]`, Columns `Customers[revenue_band]`, Values `[Churn Rate %]` |
| 7 | Clustered bar | Axis `Customers[revenue_band]`, Values `[Total Monthly Revenue]`, Legend `Customers[churn]` |
| 8 | Scatter | X `[Avg Monthly Charges]`, Y `[Churn Rate %]`, Size `[Total Monthly Revenue]`, Details `Customers[contract]` |

**Matrix conditional formatting (visual 6):** colour scale on the value,
min `#F7FBFF`, max `#C1121F`. This single visual carries the core finding of the
whole page.

### Required annotation
Add a text box: *"Churn falls as tenure and add-on depth rise. These are
associations in a snapshot dataset — not proven causal effects."*

---

## 5. PAGE 3 — Predictive Retention

**Purpose:** show the model earns its place, and hand over an actionable list.

### Layout

```
┌───────────────────────────────────────────────────────────────────────────┐
│  MODEL PERFORMANCE                    │  RISK DISTRIBUTION                 │
│  ROC-AUC 0.844  PR-AUC 0.658          │  (funnel / bar by tier)            │
│  Recall 92.2%   Precision 43.9%       │                                    │
├───────────────────────────────────────┴───────────────────────────────────┤
│  LIFT: contact top 10% → 77.3% churn vs 26.5% base = 2.91x                 │
│  (big callout card + gains curve)                                          │
├───────────────────────────────────────────────────────────────────────────┤
│  HIGH-RISK CUSTOMER TABLE (top 100, sortable)                             │
│  customer_id | risk | prob | contract | tenure | monthly | revenue at risk│
├───────────────────────────────────────────────────────────────────────────┤
│  Lift by capacity (line)  │  Expected revenue at risk by tier (bar)       │
└───────────────────────────────────────────────────────────────────────────┘
```

### Visuals

| # | Type | Fields |
|---|---|---|
| 1 | Multi-row card | `[ROC AUC]`, `[PR AUC]`, `[Recall at Threshold]`, `[Precision at Threshold]` |
| 2 | Funnel or bar | `RiskDistribution[risk_category]`, `RiskDistribution[customers]` |
| 3 | Card | `[Lift at 10 Percent]` — formatted as `2.91x` |
| 4 | Card | `[OOS Churn Rate Top Decile]` |
| 5 | Line | `retention_gains_curve`: X `population_share`, Y `cumulative_recall` + random diagonal |
| 6 | Table | `PriorityList`: `customer_id`, `risk_category`, `churn_probability` (%), `contract`, `tenure`, `monthly_charges`, `expected_monthly_revenue_at_risk`, `call_action` |
| 7 | Line | `retention_lift`: X `capacity`, Y `lift` |
| 8 | Clustered bar | `risk_distribution`: `risk_category`, `expected_monthly_at_risk` |

**Table formatting (visual 6):** Top N filter = 100 by `churn_probability`
descending. Add data bars on `churn_probability`. Colour `risk_category` by
`#C1121F` (Very High) → `#2A9D8F` (Low).

### Required annotation (critical — do not omit)
> Predictions are from a model validated on a held-out test set. Lift of 2.91×
> measures **ranking quality**, not campaign success. No retention intervention
> was executed; "expected revenue at risk" is Σ(monthly charge × predicted
> probability), not saved revenue.

---

## 6. Interactivity

| Control | Location | Action |
|---|---|---|
| Contract / Internet / Payment slicers | Page 2 | Cross-filter page 2 visuals |
| Risk tier slicer | Page 3 | Filter the high-risk table only |
| Drill-through: customer_id | Page 3 table → detail page | Optional 4th page with SHAP drivers |
| Sync slicers | Pages 1 ↔ 2 | Contract slicer synced |
| Bookmark: "Top 10% only" | Page 3 | Toggles priority list between full and top decile |

---

## 7. Theme

Save as `theme.json` and apply with `View → Themes → Browse for themes`.

```json
{
  "name": "Retention Intelligence",
  "dataColors": ["#1F3B73", "#E4572E", "#2A9D8F", "#E9C46A", "#8D99AE", "#C1121F"],
  "background": "#FFFFFF",
  "foreground": "#333333",
  "tableAccent": "#1F3B73",
  "textClasses": {
    "title":     { "fontFace": "Segoe UI Semibold", "fontSize": 16, "color": "#1F3B73" },
    "header":    { "fontFace": "Segoe UI Semibold", "fontSize": 12, "color": "#1F3B73" },
    "label":     { "fontFace": "Segoe UI", "fontSize": 10, "color": "#333333" },
    "callout":   { "fontFace": "Segoe UI Light", "fontSize": 40, "color": "#1F3B73" }
  }
}
```

---

## 8. Validation checklist (run this in Power BI Desktop)

Do not mark this report as complete until each line is confirmed.

- [ ] All 14 CSVs imported without type errors.
- [ ] `Customers` row count = **7,043**.
- [ ] `[Total Customers]` card = 7,043.
- [ ] `[Churn Rate %]` card = **26.5%** (matches `reports/business_insights.md`).
- [ ] `[Total Monthly Revenue]` = **456,116.60** (matches KPI report).
- [ ] Contract bar sums: 3,875 + 1,473 + 1,695 = 7,043.
- [ ] Donut slices = 5,174 retained + 1,869 churned.
- [ ] `[Lift at 10 Percent]` = **2.91** (matches `retention_summary.json`).
- [ ] Risk tier counts sum to 7,043.
- [ ] No visual shows a blank or "(Blank)" category.
- [ ] All three pages render at 16:9 without scroll bars.
- [ ] Disclaimers present on pages 1, 2 and 3.
- [ ] Slicers filter as expected and do not change disconnected-table cards.

---

## 9. Known limitations to state when presenting

1. **Not a `.pbix` delivery.** The report structure, measures and data are
   complete and validated as CSVs; the visual layer must be assembled in Desktop.
2. **No time intelligence.** No date table exists because the source has no dates.
   There is nothing to trend and no YTD/MTD measure that would be meaningful.
3. **Model metrics are static.** `[ROC AUC]` and friends are hard values from the
   latest run (`models/model_metadata.json`), not live calculations. If you retrain,
   update them or import `retention_test_evaluation.json`.
4. **Sample data.** 7,043 fictional customers; treat every number as illustrative.
