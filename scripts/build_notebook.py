"""Build and execute notebooks/exploratory_analysis.ipynb.

The notebook is generated from this script so its source stays reviewable in
version control, then executed with nbclient so the committed notebook contains
real outputs and real charts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

OUTPUT = config.NOTEBOOKS_DIR / "exploratory_analysis.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    cells: list = []

    def md(text: str) -> None:
        cells.append(new_markdown_cell(text.strip()))

    def code(text: str) -> None:
        cells.append(new_code_cell(text.strip()))

    md(
        """
# Customer Retention & Revenue Intelligence — Exploratory Analysis

**Dataset:** IBM Telco Customer Churn (7,043 fictional customers, 21 columns)
**Purpose:** understand why customers leave, where revenue is exposed, and who the
retention team should call first.

This notebook reads the artefacts produced by the pipeline (`python main.py all`),
so every number here is the same number the reports and dashboard use.

> **Scope.** Cross-sectional customer churn. Not a time-series forecast — the
> dataset has no dates. All findings are associations, not causal effects.
"""
    )

    code(
        """
import sys
from pathlib import Path

# Make the project importable when the notebook runs from notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

def show(path):
    \"\"\"Render a saved figure inline at a readable width.\"\"\"
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return display(Image(filename=str(path), width=880))

import config
from src.data.cleaning import load_clean
from src.analytics import style

style.apply_style()
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

print("Project root:", ROOT)
print("pandas", pd.__version__)
"""
    )

    md(
        """
## 1. Load the cleaned dataset

The raw CSV is never modified. `src/data/cleaning.py` produces
`data/processed/telco_churn_clean.csv` with documented imputation and derived
features. See `reports/data_quality_report.md` for the audit trail.
"""
    )

    code(
        """
df = load_clean()
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()
"""
    )

    code(
        """
# Business KPI baseline — computed, not asserted
kpis = {
    "total_customers": len(df),
    "churned": int((df["churn"] == "Yes").sum()),
    "retained": int((df["churn"] == "No").sum()),
    "churn_rate": df["churn_label"].mean(),
    "avg_tenure": df["tenure"].mean(),
    "avg_monthly_charges": df["monthly_charges"].mean(),
    "monthly_revenue": df["monthly_charges"].sum(),
    "monthly_revenue_churned": df.loc[df["churn_label"] == 1, "monthly_charges"].sum(),
}
for key, value in kpis.items():
    print(f"{key:28s} {value:,.4f}" if isinstance(value, float) else f"{key:28s} {value:,}")
"""
    )

    md(
        """
### Reading the numbers

Churn sits at **26.5%** — a genuine minority class. That single fact drives most
methodological choices downstream: accuracy is a useless headline metric, PR-AUC
and lift are the honest ones, and the model needs explicit imbalance handling.

Note also that churners carry a **higher average monthly charge** than retained
customers. The customers leaving are the expensive ones.
"""
    )

    md("## 2. Churn distribution")

    code(
        """
from src.analytics.eda import plot_churn_distribution
path = plot_churn_distribution(df)
show(path)
print("Saved:", path)
"""
    )

    md(
        """
**Observed:** 5,174 retained vs 1,869 churned (26.5% churn).
**Why it matters:** a naive "nobody churns" model would score 73.5% accuracy and be
worthless. Every metric reported later is read against this base rate.
"""
    )

    md("## 3. Tenure — the strongest continuous signal")

    code(
        """
from src.analytics.churn_analysis import churn_by_tenure_group

tenure_table = churn_by_tenure_group(df)
tenure_table[["tenure_group", "customers", "churned", "churn_rate", "avg_tenure", "monthly_revenue_at_risk"]]
"""
    )

    code(
        """
from src.analytics.eda import plot_tenure_analysis
show(plot_tenure_analysis(df))
"""
    )

    md(
        """
**Observed:** the 0–6 month cohort churns at **52.9%**, falling monotonically to
**9.5%** for customers past 48 months. Median tenure is 10 months for churners
versus 38 for retained customers.

**Why it matters:** churn is front-loaded. Retention spend in the first 90 days
has far more leverage than the same spend on a three-year customer.

**What to investigate next:** onboarding quality, first-bill shock, early support
ticket volume, and whether the first contract renewal is where customers decide.

**Caveat:** tenure and churn are observed at the same instant. A customer with
long tenure has already survived many churn opportunities, so part of this pattern
is survivorship rather than a protective effect of time itself.
"""
    )

    md("## 4. Contract type — the sharpest structural split")

    code(
        """
from src.analytics.churn_analysis import churn_by_contract
churn_by_contract(df)[["contract", "customers", "churned", "churn_rate", "lift_vs_baseline",
                       "monthly_revenue", "monthly_revenue_at_risk"]]
"""
    )

    code(
        """
from src.analytics.eda import plot_contract_and_payment
show(plot_contract_and_payment(df))
"""
    )

    md(
        """
**Observed:** Month-to-month churns at **42.7%** (1.61× base); two-year contracts at
**2.8%** (0.11× base). A 40-percentage-point spread.

**Why it matters:** this is the single largest controllable lever in the dataset.
Month-to-month customers also hold the majority of exposed revenue.

**What to investigate next:** the economics of converting month-to-month customers
to fixed terms — what discount is needed, and what the resulting margin looks like
after accounting for the churn avoided.

**Caveat:** contract choice is not random. Customers who accept a two-year term may
already be more settled, more price-insensitive, or planning to stay. The observed
gap is an association, and part of it is selection.
"""
    )

    md("## 5. Payment method")

    code(
        """
from src.analytics.churn_analysis import churn_by_payment_method
churn_by_payment_method(df)[["payment_method", "auto_payment", "customers", "churn_rate",
                             "lift_vs_baseline", "monthly_revenue_at_risk"]]
"""
    )

    md(
        """
**Observed:** electronic check churns at **45.3%**; automatic bank transfer and
credit card both sit near **15–17%**.

**Why it matters:** enrolling customers in automatic payment is operationally cheap
relative to any pricing change, and it removes a monthly moment where the customer
actively re-decides to stay.

**What to investigate next:** whether the payment method itself matters, or whether
it proxies for something else (income stability, tech comfort, engagement). A
targeted enrolment test with a control group would answer this properly.
"""
    )

    md("## 6. Services and add-on depth")

    code(
        """
from src.analytics.churn_analysis import churn_by_addon, churn_by_internet_service
display(churn_by_internet_service(df)[["internet_service", "customers", "churn_rate", "monthly_revenue_at_risk"]])
churn_by_addon(df)[["addon", "churn_rate_with", "churn_rate_without", "difference_pts"]]
"""
    )

    code(
        """
from src.analytics.eda import plot_service_analysis, plot_addon_effect
show(plot_service_analysis(df))
show(plot_addon_effect(df))
"""
    )

    md(
        """
**Observed:** fibre-optic churns at **41.9%** — higher than DSL (19.0%) despite
costing more. Add-on depth is protective: online security shows a
**27.2-point** churn gap, tech support **26.5 points**.

**Why it matters:** the fibre paradox is the most surprising finding and the one
worth the most investigation. Customers paying a premium for the best product are
leaving fastest.

**What to investigate next:** fibre price sensitivity, competitor fibre offers in
the same service areas, installation and reliability issues, and whether support
add-ons are being sold to the customers who need them most.

**Caveat:** add-on adoption correlates with engagement. Customers who buy security
and support may already be committed; buying the add-on may not be what keeps them.
"""
    )

    md("## 7. Charges and revenue exposure")

    code(
        """
from src.analytics.revenue_analysis import revenue_summary, revenue_concentration
summary = revenue_summary(df)
for key in ["total_monthly_revenue", "monthly_revenue_churned", "revenue_share_churned",
            "avg_monthly_charge_churned", "avg_monthly_charge_retained"]:
    print(f"{key:32s} {summary[key]:>14,.2f}")
print()
for key, value in revenue_concentration(df).items():
    print(f"{key:32s} {value}")
"""
    )

    code(
        """
from src.analytics.eda import plot_charges_analysis, plot_revenue_exposure, plot_segment_heatmap
show(plot_charges_analysis(df))
show(plot_revenue_exposure(df))
show(plot_segment_heatmap(df))
"""
    )

    md(
        """
**Observed:** **139,131** in monthly recurring charges sits with customers who
already churned — **30.5%** of total monthly revenue. Churners average
**74.44** per month versus **61.27** for retained customers.

The contract × charge-band heat map shows the danger zone clearly: month-to-month
customers in the Mid-High and High bands churn at **53.2%** and **52.2%**.

**Why it matters:** the exposure is not concentrated in low-value accounts. It is
concentrated in the *higher-value, uncommitted* segment — exactly the customers a
retention team should care most about.

**Careful reading:** this is one month of recurring charge, not cumulative loss.
No campaign cost or save rate is modelled, so nothing here is "revenue saved".
"""
    )

    md("## 8. Correlation — and its limits")

    code(
        """
from src.analytics.eda import plot_correlation
show(plot_correlation(df))

numeric = ["tenure", "monthly_charges", "total_charges", "num_addon_services", "service_count", "churn_label"]
df[numeric].corr()["churn_label"].sort_values()
"""
    )

    md(
        """
**Observed:** tenure has the strongest linear correlation with churn
(**negative** — longer tenure, less churn). `total_charges` and `monthly_charges`
correlate positively but weakly.

**Why this matters as a warning:** Pearson correlation only captures linear
co-movement, and several strong effects in this data are non-linear (contract type
is categorical; tenure's effect flattens). Correlation here is a screening tool, not
the analysis. It also says nothing about causation — and the correlation between
`tenure` and `total_charges` is largely mechanical, since total charges accumulate
over tenure.
"""
    )

    md(
        """
## 9. From description to prediction

The analysis so far says *which attributes travel with churn*. It cannot say which
matter once the others are controlled for, and it cannot rank an individual
customer.

That is the job of the model. Full methodology and metrics are in
`reports/model_evaluation.md`; here we load the trained model's actual output.
"""
    )

    code(
        """
from src.models.predict import load_scores, risk_distribution

scored = load_scores()
print(f"Scored customers: {len(scored):,}")
risk_distribution(scored)
"""
    )

    code(
        """
from src.analytics.eda import plot_risk_distribution
show(plot_risk_distribution(scored))
"""
    )

    md(
        """
**Observed:** the model separates the base into a **5.4% "Very High"** tier averaging
**0.92** predicted probability and a **50.3% "Low"** tier averaging **0.13**. The
actual churn rate in the Very High tier is **88.2%** — the tiers are real, not
cosmetic.

**Why it matters:** a wide probability spread is what makes prioritisation
possible. If every customer scored near the base rate, no call list could be built.
"""
    )

    md("## 10. Retention prioritisation — the business payoff")

    code(
        """
from src.retention.prioritization import lift_at_capacity, prioritisation_summary

lift = lift_at_capacity(scored)
lift
"""
    )

    code(
        """
from src.retention.prioritization import cumulative_gains
from src.analytics.eda import plot_lift_curve

primary = lift[lift["capacity"] == 0.10].iloc[0].to_dict()
show(plot_lift_curve(cumulative_gains(scored), primary))
"""
    )

    code(
        """
import json
summary = json.loads((config.METRICS_DIR / "retention_summary.json").read_text())["out_of_sample"]
for key, value in summary.items():
    print(f"{key:46s} {value}")
"""
    )

    md(
        """
**Observed:** contacting the top **10%** reaches a group where **81.1%** actually
churned, against a **26.5%** base rate — a lift of **3.06×** in-sample. The
out-of-sample figure in the production report (2.91× on the held-out test split) is
the one to quote publicly, because it reflects data the model never saw.

**What the business gets:** the same number of retention calls reaches roughly three
times as many genuine churners as an untargeted list. That is a real, measurable
efficiency gain in *targeting*.

**What it does not get:** proof that contacting them works. No intervention was
run. The revenue figure is exposure, not savings.
"""
    )

    md("## 11. Explaining individual predictions")

    code(
        """
import json
from pathlib import Path

manifest_path = config.DATA_EXPORTS / "explainability" / "explainability_manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    show(manifest["figures"]["global_importance"])
    show(manifest["figures"]["waterfall"])
else:
    print("Run `python main.py explain` to generate SHAP artefacts.")
"""
    )

    code(
        """
if manifest_path.exists():
    individual = json.loads((config.DATA_EXPORTS / "explainability" / "individual_explanation.json").read_text())
    print(f"Customer {individual['customer_id']} — predicted churn probability "
          f"{individual['predicted_probability']:.1%} (base {individual['base_probability']:.1%})")
    print("\\nContext:", individual["customer_context"])
    print("\\nFactors increasing risk:")
    for factor in individual["risk_increasing_factors"][:5]:
        print(f"  {factor['feature']:45s} {factor['probability_delta_pts']:+7.1f} pts")
    print("\\nFactors decreasing risk:")
    for factor in individual["risk_decreasing_factors"][:5]:
        print(f"  {factor['feature']:45s} {factor['probability_delta_pts']:+7.1f} pts")
"""
    )

    md(
        """
**How to read this:** SHAP attributes the gap between the base rate and this
customer's predicted risk to specific feature values. A positive contribution means
that value pushed the prediction up.

**Feature importance ≠ correlation ≠ causation.** Importance is about what the
model uses. Correlation is about co-movement in the data. Neither establishes that
changing a value would change the outcome. The model is a ranking tool, not a
causal engine.
"""
    )

    md(
        """
## 12. Conclusions and recommendations

1. **Churn is front-loaded and structural.** 52.9% of 0–6 month customers leave;
   month-to-month contracts churn at 42.7%. Both point at the same intervention
   window: the first 90 days, before commitment is established.
2. **Exposed revenue is concentrated in high-value, uncommitted customers.**
   Month-to-month Mid-High/High band customers churn above 52% while carrying
   disproportionate monthly revenue.
3. **The model is usable, not perfect.** Test PR-AUC 0.658 against a 0.265 base
   rate; at the tuned threshold it catches ~92% of churners but roughly half the
   flagged customers stay. That trade-off is deliberate and should be revisited
   with real campaign economics.
4. **Prioritisation delivers ~3× lift.** The top decile churns at roughly three
   times the base rate on held-out data. Same call capacity, three times the
   churners reached.
5. **Nothing here proves retention works.** No intervention was run and no save
   rate exists. The honest next step is a controlled pilot with a holdout group,
   tracked against cost per save.

Business recommendations, the full limitations section and the additional data
that would materially improve this are in `reports/business_insights.md`.
"""
    )

    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )


def execute_notebook(notebook, timeout: int = 600):
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(config.NOTEBOOKS_DIR)}},
        allow_errors=False,
    )
    return client.execute()


def main() -> int:
    config.NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote notebook source to {OUTPUT}")

    executed = execute_notebook(notebook)
    nbformat.write(executed, OUTPUT)

    errors = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    print(f"Executed {len(executed.cells)} cells, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
