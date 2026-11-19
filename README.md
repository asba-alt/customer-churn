# Customer Churn & Retention Prioritisation

An end-to-end churn analytics project on the IBM Telco Customer Churn dataset
(7,043 customers, 21 columns): clean the data, explore it, model it, explain the
model, and rank customers for a retention call list.

Everything in this README is computed from the actual run. Nothing is hardcoded.

---

## The result in one table

| | |
|---|---|
| Churn rate | **26.54%** (1,869 of 7,043) |
| Monthly revenue attached to churners | **139,131** (30.5% of the base) |
| Model (held-out test) | **XGBoost** — ROC-AUC **0.844**, PR-AUC **0.658** |
| Threshold | 0.293 (F2-optimal on validation) → recall **92.2%**, precision **43.9%** |
| Lift at top 10% | **2.91×** the base churn rate, catching **29.1%** of all churners |

If the retention team can only call 141 customers (the top decile), those
customers churn at 77.3% against a 26.5% base. Same call capacity, roughly three
times the churners reached.

---

## Three findings

**1. Churn is front-loaded.** The 0–6 month cohort churns at 52.9%, falling to
9.5% past 48 months.

**2. Contract type is the sharpest split.** Month-to-month churns at 42.7%;
two-year contracts at 2.8%.

**3. The customers leaving are the expensive ones.** Churners average 74.44 per
month versus 61.27 for retained customers, so exposure sits in high-value,
uncommitted accounts.

---

## Quick start

```bash
pip install -r requirements.txt

python main.py check    # verify environment and dataset
python main.py all      # full pipeline (~25s)
python -m pytest tests  # 132 tests
```

The dataset must be at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
`python main.py check` prints exact placement instructions if it is missing.

### Individual stages

```bash
python main.py pipeline   # ingest → clean → validate → SQLite
python main.py sql        # execute the SQL analytics library
python main.py train      # compare 3 model families, select, refit
python main.py evaluate   # test-set metrics, curves, evaluation report
python main.py score      # score customers, build the priority list
python main.py explain    # SHAP global + individual explanations
python main.py report     # EDA figures and business reports
```

---

## Where to look

The README is the summary. Everything detailed lives in one of these files:

| File | What it covers |
|---|---|
| [`reports/business_insights.md`](reports/business_insights.md) | Full business narrative, findings, recommendations, limitations |
| [`reports/model_evaluation.md`](reports/model_evaluation.md) | Model comparison, test metrics, curves, methodology, leakage controls |
| [`reports/data_quality_report.md`](reports/data_quality_report.md) | Schema, missing values, duplicates, every cleaning decision |
| [`reports/sql_results.md`](reports/sql_results.md) | Output of all 18 executed SQL queries |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every column in the cleaned table |
| [`notebooks/exploratory_analysis.ipynb`](notebooks/exploratory_analysis.ipynb) | Executed EDA — 45 cells, 13 inline charts |
| [`powerbi/`](powerbi) | Dashboard spec and DAX measures (no `.pbix`; see note below) |

Charts are in `data/exports/figures/` (22 PNGs). The reports embed them
directly, so open a report rather than browsing the folder.

---

## Project structure

```
customer_churn/
├── main.py                 CLI entry point
├── config.py               every path, seed and constant
├── data/
│   ├── raw/                source CSV (never modified)
│   ├── processed/          cleaned data, SQLite DB, scored customers
│   └── exports/
│       ├── figures/        all PNG charts
│       ├── metrics/        JSON/KPI artefacts
│       └── powerbi/        import-ready CSVs
├── sql/                    analytics_queries.sql — 18 query blocks
├── src/
│   ├── data/               ingestion, cleaning, validation
│   ├── analytics/          churn, customer, revenue analysis + EDA charts
│   ├── database/           SQLite schema, query runner
│   ├── models/             preprocessing, train, evaluate, explain, predict
│   ├── retention/          prioritisation and lift
│   ├── reporting/          business report generation
│   └── pipeline.py         stage orchestration
├── tests/                  132 tests
├── reports/                the written deliverables
├── notebooks/              executed exploratory analysis
├── powerbi/                dashboard spec + DAX measures
└── docs/                   data dictionary
```

---

## Method, briefly

**Data.** The raw file is read-only. Eleven blank `TotalCharges` values all
belong to `tenure = 0` customers; they are imputed as `0.0` and flagged with
`total_charges_imputed = 1`. Implausible records are flagged, not deleted, so the
modelled population matches the business population.

**Model.** Stratified 60/20/20 split, seed 42. Three candidates — logistic
regression, random forest, XGBoost — selected on **validation PR-AUC**, never
accuracy: on a 26.5% positive class, a "nobody churns" model scores 73.5%
accuracy and is worthless. Winner is XGBoost (200 trees, depth 3, lr 0.05).
The decision threshold is F2-optimal on validation (0.293) and applied unchanged
to test.

**Lift is out-of-sample.** Every lift figure comes from the held-out test split,
never from rows the model was fitted on.

**Explainability.** SHAP applied to transformed features, so SHAP sees exactly
what the model sees. Top drivers: contract type (21.7%), tenure (14.0%), no
online security (7.6%), fibre optic (6.7%), no tech support (6.1%).

---

## Limitations

1. **Fictional sample data.** All figures describe this 7,043-row sample.
2. **Cross-sectional.** One row per customer, no dates. Nothing here is a time
   forecast.
3. **No causality.** Every relationship is an association. The model ranks risk;
   it does not estimate treatment effects.
4. **No intervention.** No campaign was run and no save rate exists. Revenue
   figures are exposure measures, not savings.
5. **No temporal validation.** Performance is measured on a random hold-out. A
   time-based split would be stronger and is impossible without dates.
6. **Power BI layer unvalidated.** Import-ready and specified; not opened in
   Desktop. No `.pbix` is claimed, because creating one requires Power BI Desktop,
   which cannot be driven headlessly.
7. **Threshold is a judgement.** Without campaign cost data, the F2 cut-off is
   defensible, not optimal.

---

## Stack

Python 3.14 · pandas 3.0 · scikit-learn 1.8 · XGBoost 3.4 · SHAP 0.54 ·
SQLite · matplotlib · seaborn · pytest

Deterministic seed `42`. Full run completes in ~25s. Deleting `data/processed/`,
`data/exports/` and `models/*.joblib` and re-running produces identical results.
