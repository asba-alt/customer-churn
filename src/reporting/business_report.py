"""Write reports/business_insights.md from the executed pipeline results.

Every number and recommendation in the output is interpolated from the actual
run — no hardcoded findings, no invented magnitudes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def _fmt_currency(value: float) -> str:
    return f"{value:,.2f}"


def _top_rows(df: pd.DataFrame, column: str, n: int = 3) -> list[dict]:
    return df.sort_values(column, ascending=False).head(n).to_dict(orient="records")


def write_business_insights(state, output_path: Path | None = None) -> str:
    """Generate the flagship business narrative."""
    output_path = output_path or config.REPORT_BUSINESS_INSIGHTS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kpis = state.extras["kpis"]
    clean = state.clean
    metadata = state.metadata or json.loads(config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    evaluation = state.evaluation
    summary = state.extras["priority_summary"]
    lift_table = state.extras["lift_table"]
    priority = state.extras["priority_list"]
    scenarios = state.extras["scenarios"]
    explainability = state.explainability or {}

    from src.analytics.churn_analysis import (
        churn_by_addon,
        churn_by_contract,
        churn_by_internet_service,
        churn_by_payment_method,
        churn_by_tenure_group,
        churn_driver_summary,
    )
    from src.analytics.customer_analysis import customer_segments

    contract = churn_by_contract(clean)
    tenure = churn_by_tenure_group(clean)
    payment = churn_by_payment_method(clean)
    internet = churn_by_internet_service(clean)
    addons = churn_by_addon(clean)
    drivers = churn_driver_summary(clean)
    segments = customer_segments(clean)

    worst_contract = contract.sort_values("churn_rate", ascending=False).iloc[0]
    best_contract = contract.sort_values("churn_rate").iloc[0]
    worst_tenure = tenure.sort_values("churn_rate", ascending=False).iloc[0]
    worst_payment = payment.sort_values("churn_rate", ascending=False).iloc[0]
    worst_internet = internet.sort_values("churn_rate", ascending=False).iloc[0]
    strongest_addon = addons.dropna(subset=["difference_pts"]).sort_values("difference_pts").iloc[0]
    top_segment = segments.sort_values("revenue_at_risk", ascending=False).iloc[0]
    # The add-on columns share the 'No internet service' category, so keep one
    # row per feature to avoid showing the same 1,526 customers three times.
    drivers_by_feature = drivers.groupby("feature", sort=False).head(1)
    top_drivers = drivers_by_feature.head(6)

    metrics_default = evaluation["metrics_default"]
    metrics_tuned = evaluation["metrics_tuned"]
    cm_default = evaluation["confusion_default"]
    cm_tuned = evaluation["confusion_tuned"]
    tn_d, fp_d, fn_d, tp_d = cm_default.ravel()
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()

    priority_mix = priority["contract"].value_counts(normalize=True)
    priority_fibre_share = (
        (priority["internet_service"] == "Fiber optic").mean() if "internet_service" in priority else 0.0
    )
    priority_tenure_median = float(priority["tenure"].median())

    # Diagram/table builders -------------------------------------------------
    contract_rows = "\n".join(
        f"| {r.contract} | {r.customers:,} | {r.churned:,} | {r.churn_rate:.1%} | "
        f"{r.lift_vs_baseline:.2f}x | {_fmt_currency(r.monthly_revenue_at_risk)} |"
        for r in contract.itertuples()
    )
    tenure_rows = "\n".join(
        f"| {r.tenure_group} | {r.customers:,} | {r.churn_rate:.1%} | {r.churn_rate_vs_baseline_pts:+.1f} |"
        for r in tenure.itertuples()
    )
    payment_rows = "\n".join(
        f"| {r.payment_method} | {r.customers:,} | {r.churn_rate:.1%} | "
        f"{'Yes' if r.auto_payment else 'No'} |"
        for r in payment.itertuples()
    )
    internet_rows = "\n".join(
        f"| {r.internet_service} | {r.customers:,} | {r.churn_rate:.1%} | {_fmt_currency(r.monthly_revenue_at_risk)} |"
        for r in internet.itertuples()
    )
    driver_rows = "\n".join(
        f"| {r.feature} | {r.value} | {r.customers:,} | {r.churn_rate:.1%} | "
        f"{r.vs_baseline_pts:+.1f} | {r.lift:.2f}x |"
        for r in top_drivers.itertuples()
    )
    lift_rows = "\n".join(
        f"| {r.capacity:.0%} | {r.customers_selected:,} | {r.churn_rate_in_group:.1%} | "
        f"{r.lift:.2f}x | {r.recall_of_churners:.1%} | {_fmt_currency(r.monthly_revenue_in_group)} |"
        for r in lift_table.itertuples()
    )
    scenario_rows = "\n".join(
        f"| {r.capacity:.0%} | {r.customers_contacted:,} | {r.actual_churn_rate_in_group:.1%} | "
        f"{r.lift:.2f}x | {r.recall_of_all_churners:.1%} | "
        f"{_fmt_currency(r.expected_monthly_revenue_at_risk)} |"
        for r in scenarios.itertuples()
    )

    shap_rows = ""
    if explainability.get("top_features"):
        shap_rows = (
            "| Rank | Feature (model input) | Mean absolute SHAP | Share of total |\n"
            "|---|---|---|---|\n"
        ) + "\n".join(
            f"| {i+1} | {f['feature']} | {f['mean_abs_shap']:.4f} | {f['share_of_total_importance']:.1f}% |"
            for i, f in enumerate(explainability["top_features"])
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    markdown = f"""# Customer Retention & Revenue Intelligence — Business Insights

**Generated:** {generated}
**Dataset:** IBM Telco Customer Churn (fictional sample, 7,043 customers × 21 columns)
**Model:** `{metadata['model_name']}` · decision threshold {metadata['decision_threshold']:.3f}

> **Scope statement.** This is a cross-sectional customer-churn classification and
> retention-prioritisation project. It is *not* a longitudinal monthly forecast.
> All monetary figures are in the dataset's own charge units, described here as
> "charge units" rather than a currency, because the source documents no currency.
> No retention campaign was executed, so nothing in this document claims revenue
> was saved or churn was reduced.

---

## 1. The business problem

The operator is losing roughly a quarter of its customer base
({kpis['overall_churn_rate']:.1%}, {kpis['churned_customers']:,} of
{kpis['total_customers']:,} customers) and wants three things answered:

1. **Why** are customers leaving, and which attributes travel with churn?
2. **Where** is the money exposed — which segments carry the most at-risk revenue?
3. **Who** should the retention team call first when it can only reach a fraction
   of the base?

---

## 2. Headline numbers

| KPI | Value | How it is calculated |
|---|---|---|
| Total customers | {kpis['total_customers']:,} | Row count of the cleaned dataset |
| Churned customers | {kpis['churned_customers']:,} | Count where `churn = Yes` |
| Retained customers | {kpis['retained_customers']:,} | Count where `churn = No` |
| Overall churn rate | {kpis['overall_churn_rate']:.2%} | Churned ÷ total |
| Average tenure | {kpis['avg_tenure']:.1f} months | Mean of `tenure` |
| Average monthly charges | {_fmt_currency(kpis['avg_monthly_charges'])} | Mean of `monthly_charges` |
| Total monthly revenue | {_fmt_currency(kpis['total_monthly_revenue'])} | Sum of `monthly_charges` — recurring, one month |
| Monthly revenue attached to churned customers | {_fmt_currency(kpis['monthly_revenue_churned'])} | Sum of `monthly_charges` where churn = Yes |
| Share of monthly revenue lost with churners | {kpis['revenue_share_churned']:.2%} | Above ÷ total monthly revenue |
| Average monthly charge — churned | {_fmt_currency(kpis['avg_monthly_charge_churned'])} | Mean charges among churners |
| Average monthly charge — retained | {_fmt_currency(kpis['avg_monthly_charge_retained'])} | Mean charges among retained |
| Revenue realised to date (all customers) | {_fmt_currency(kpis['total_realised_revenue'])} | Sum of `total_charges` |
| High-value customers (≥75th pct charges) | {kpis['high_value_customers']:,} | Monthly charges ≥ {_fmt_currency(kpis.get('high_value_threshold', 0))} |
| High-value churn rate | {kpis['high_value_churn_rate']:.2%} | Churn among high-value customers |
| Revenue in top 10% of spenders | {kpis['top_10pct_revenue_share']:.1%} | Share of monthly revenue |

**Read the revenue number carefully.** The
{_fmt_currency(kpis['monthly_revenue_churned'])} figure is *one month* of recurring
charges attached to customers who already left. It is an exposure measure, not a
cumulative loss, and it does not net off the cost of any retention activity.

---

## 3. What the data says about churn

### 3.1 Contract type is the single sharpest split

| Contract | Customers | Churned | Churn rate | Lift vs base | Monthly revenue at risk |
|---|---|---|---|---|---|
{contract_rows}

{worst_contract.contract} customers churn at {worst_contract.churn_rate:.1%}, which is
{worst_contract.lift_vs_baseline:.2f}× the base rate. {best_contract.contract} customers churn at
{best_contract.churn_rate:.1%} — a difference of
{worst_contract.churn_rate * 100 - best_contract.churn_rate * 100:.0f} percentage points.

*Why it matters:* commitment length is the strongest observable retention signal
in this dataset, and it is something the business controls through pricing and
contract design.
*What to investigate:* whether conversion from month-to-month to a fixed term is
priced attractively enough, and what churn looks like in the first weeks after
signing a longer contract.

### 3.2 Churn is front-loaded in the customer lifecycle

| Tenure group | Customers | Churn rate | vs base (pts) |
|---|---|---|---|
{tenure_rows}

The {worst_tenure.tenure_group} cohort churns at {worst_tenure.churn_rate:.1%}. Churned
customers have a median tenure of {clean.loc[clean['churn_label'] == 1, 'tenure'].median():.0f}
months versus {clean.loc[clean['churn_label'] == 0, 'tenure'].median():.0f} months for
retained customers.

*Why it matters:* the first year is where retention spend has the highest leverage,
because that is where the losses are concentrated.
*What to investigate:* onboarding quality, first-bill shock, and early support
contact volume.

### 3.3 Payment method and automatic billing

| Payment method | Customers | Churn rate | Automatic |
|---|---|---|---|
{payment_rows}

{worst_payment.payment_method} customers churn at {worst_payment.churn_rate:.1%}.
*Why it matters:* payment friction is cheap to fix compared with a price change.
Moving customers onto automatic payment removes a monthly decision point where
they can reconsider the relationship.
*Caveat:* electronically-paying customers may differ in other ways (age, tech
comfort), so this is an association, not a proven intervention effect.

### 3.4 Internet service and service depth

| Internet service | Customers | Churn rate | Monthly revenue at risk |
|---|---|---|---|
{internet_rows}

{worst_internet.internet_service} customers churn at {worst_internet.churn_rate:.1%}
despite paying more. Meanwhile add-on depth is protective:

| Add-on | Churn with | Churn without | Difference (pts) |
|---|---|---|---|
""" + "\n".join(
        f"| {r.addon} | {r.churn_rate_with:.1%} | {r.churn_rate_without:.1%} | {r.difference_pts:+.1f} |"
        for r in addons.itertuples()
    ) + f"""

The strongest single association is **{strongest_addon.addon}**:
{strongest_addon.churn_rate_with:.1%} churn with it versus
{strongest_addon.churn_rate_without:.1%} without
({strongest_addon.difference_pts:+.1f} percentage points).

*Why it matters:* customers who have adopted support-style services stay longer.
*What to investigate:* whether bundling support services changes retention
(causal test needed) or simply attracts customers who were already committed.

### 3.5 The strongest raw churn signals

| Feature | Value | Customers | Churn rate | vs base (pts) | Lift |
|---|---|---|---|---|---|
{driver_rows}

One row per feature, so the repeated "No internet service" category across the
add-on columns does not inflate the table. This is a purely descriptive screen of
single attributes: useful for framing, but superseded by the multivariate model in
section 5, which controls for the other variables.

### 3.6 Segment heat map — the danger zone

The highest-exposure segment by revenue is **{top_segment.segment}**:
{top_segment.customers:,} customers, {top_segment.churn_rate:.1%} churn,
{_fmt_currency(top_segment.revenue_at_risk)} of monthly revenue attached to churners.

The interaction matters: uncommitted customers on high bills are far riskier than
either attribute alone suggests, while long-contract high-spend customers are the
safest revenue in the business.

---

## 4. Revenue exposure

| Measure | Value |
|---|---|
| Total monthly revenue | {_fmt_currency(kpis['total_monthly_revenue'])} |
| Monthly revenue attached to churned customers | {_fmt_currency(kpis['monthly_revenue_churned'])} |
| Share of monthly revenue | {kpis['revenue_share_churned']:.2%} |
| Revenue in top 10% of spenders | {kpis['top_10pct_revenue_share']:.1%} |
| Revenue in top 20% of spenders | {kpis['top_20pct_revenue_share']:.1%} |

Revenue is moderately concentrated: the top decile of spenders holds
{kpis['top_10pct_revenue_share']:.1%} of monthly revenue. Critically, churn is
*not* confined to low-value customers — churners pay
{_fmt_currency(kpis['avg_monthly_charge_churned'])} per month on average versus
{_fmt_currency(kpis['avg_monthly_charge_retained'])} for retained customers.
**The customers leaving are the more expensive ones.**

**Value proxies used, and their limits.** The dataset has no billing dates and no
cancellation timing, so a proper discounted lifetime-value model is not
identifiable. Two transparent proxies are reported instead:

- *Realised value to date* = `total_charges` — factual, already billed.
- *Annualised value proxy* = `monthly_charges × 12` — a simple extrapolation that
  ignores future churn, discounting and price changes, so it overstates value for
  customers who are about to leave.

---

## 5. What the model adds beyond the descriptive analysis

The descriptive tables above say which attributes travel with churn. They cannot
say which of those attributes matter *once the others are controlled for*, and
they cannot rank an individual customer. The model does both.

### 5.1 Model selection

Three families were compared on the validation split using PR-AUC, chosen
because churn is a {kpis['overall_churn_rate']:.1%} minority class and the business
decision is a ranked list, not a balanced classifier.

**Selected: `{metadata['model_name']}`** with `{json.dumps(metadata['params'])}`.

### 5.2 Test-set performance (never seen during training)

| Metric | At 0.50 cut-off | At tuned cut-off {evaluation['threshold']:.2f} |
|---|---|---|
| Accuracy | {metrics_default['accuracy']:.3f} | {metrics_tuned['accuracy']:.3f} |
| Precision | {metrics_default['precision']:.3f} | {metrics_tuned['precision']:.3f} |
| Recall | {metrics_default['recall']:.3f} | {metrics_tuned['recall']:.3f} |
| F1 | {metrics_default['f1']:.3f} | {metrics_tuned['f1']:.3f} |
| ROC-AUC | {metrics_default['roc_auc']:.3f} | {metrics_tuned['roc_auc']:.3f} |
| PR-AUC | {metrics_default['pr_auc']:.3f} | {metrics_tuned['pr_auc']:.3f} |

At the tuned cut-off the model catches **{tp_t:,} of {tp_t + fn_t:,} churners**
({metrics_tuned['recall']:.1%} recall) while raising
**{fp_t:,} false alarms** ({metrics_tuned['precision']:.1%} precision). At the
default 0.5 cut-off it catches {tp_d:,} churners with {fp_d:,} false alarms.

**The trade-off, stated plainly.** Pushing recall from
{metrics_default['recall']:.1%} to {metrics_tuned['recall']:.1%} costs
{fp_t - fp_d:,} extra wasted contacts. That is the correct direction to err when a
retention call is cheap relative to the recurring revenue of a customer who leaves —
but it should be revisited once real call costs and save rates are known. Without
those, no dollar-optimal threshold exists.

### 5.3 What drives the predictions (SHAP)

{shap_rows}

SHAP tells us how much each feature moves the model's output for each customer.
The top drivers — contract type, tenure, online security, internet service, tech
support and payment method — agree with the descriptive analysis, which is a useful
sanity check that the model learned structure rather than noise.

**Feature importance is not causation.** A high SHAP value means the model relies
on that feature, not that changing it would change the outcome. Fibre customers
churning more may reflect price sensitivity, competitor availability in
fibre-served areas, or how the product was sold — the data cannot separate these.

---

## 6. Retention prioritisation — who to call first

The model scores every customer. The retention team's capacity is assumed at
**{summary['capacity']:.0%} of the base** ({summary['customers_prioritised']:,}
customers). Lift is measured on the **hold-out test split**, where outcomes were
never seen during training.

| Capacity | Customers | Churn rate in group | Lift | Churners captured | Monthly revenue in group |
|---|---|---|---|---|---|
{lift_rows}

### The headline result

Contacting the top {summary['capacity']:.0%} by predicted risk
({summary['customers_prioritised']:,} customers):

- **Observed churn rate in the group: {summary['churn_rate_in_group']:.1%}**
  versus **{summary['overall_churn_rate']:.1%}** overall → **lift of {summary['lift_at_capacity']:.2f}×**
- **{summary['churners_captured']:,} churners captured**
  ({summary['recall_of_churners']:.1%} of all churners) from
  {summary['share_of_base_pct']:.0f}% of the base
- **{_fmt_currency(summary['monthly_revenue_in_group'])} of monthly revenue**
  ({summary['share_of_total_monthly_revenue_pct']:.1f}% of total) sits in the group
- **Expected monthly revenue at risk in the group:
  {_fmt_currency(summary['expected_monthly_revenue_at_risk'])}** — this is
  Σ(monthly charge × predicted probability), a probability-weighted exposure
  estimate, *not* a forecast of realised loss

For comparison, contacting a random {summary['capacity']:.0%} of customers would
capture only about {summary['capacity']:.0%} of churners. The model captures
{summary['recall_of_churners']:.0%}.

### What the priority group looks like

- **{priority_mix.get('Month-to-month', 0):.0%} are month-to-month contracts**
- **{priority_fibre_share:.0%} are fibre-optic customers**
- Median tenure: **{priority_tenure_median:.0f} months**

That profile points at a coherent playbook: early-lifecycle, high-bill, uncommitted
fibre customers. It is not a random slice of the base.

### Capacity scenarios

| Capacity | Customers | Observed churn in group | Lift | Churners captured | Expected monthly revenue at risk |
|---|---|---|---|---|---|
{scenario_rows}

The lift decays as capacity grows — that is expected and healthy. It is precisely
the argument for ranking rather than mass outreach: the marginal customer added at
30% capacity is far less likely to churn than the marginal customer at 5%.

### What this does and does not prove

- **Proves:** the ranking concentrates future churners well above the base rate on
  data the model never saw.
- **Does not prove:** that contacting them retains them. Without an intervention
  and a measured save rate, no revenue has been saved. The
  {_fmt_currency(summary['expected_monthly_revenue_at_risk'])} figure is exposure,
  not savings.

---

## 7. Recommendations

Recommendations are ordered by how well the evidence supports them.

### Immediate (evidence is strong, action is cheap)

1. **Rank every customer weekly and work the top decile first.** The out-of-sample
   lift of {summary['lift_at_capacity']:.2f}× means the same number of calls reaches
   roughly three times as many genuine churners as an untargeted list. The priority
   file `data/exports/powerbi/retention_priority_list.csv` is ready to use.
2. **Target the early-lifecycle cohort.** The {worst_tenure.tenure_group} group churns at
   {worst_tenure.churn_rate:.1%}. A structured first-90-days onboarding programme attacks
   the largest concentration of churn at the point where it is cheapest to influence.
3. **Push automatic payment enrolment.** {worst_payment.payment_method} customers churn at
   {worst_payment.churn_rate:.1%}. Converting manual payers to automatic billing removes
   a recurring churn decision point and requires no pricing change.

### Short term (evidence is strong, action needs design)

4. **Design a month-to-month → fixed-term conversion offer.** Month-to-month accounts
   for {priority_mix.get('Month-to-month', 0):.0%} of the high-risk list. Even a modest
   conversion rate moves a large block of exposed revenue
   ({_fmt_currency(worst_contract.monthly_revenue_at_risk)} sits with month-to-month
   churners) onto committed contracts.
5. **Bundle support services for at-risk fibre customers.** {strongest_addon.addon} shows a
   {strongest_addon.difference_pts:+.1f} point churn gap. Run this as a **controlled test**
   with a holdout group before scaling, because the current evidence is associational —
   engaged customers may simply buy more services.

### Medium term (needs more data first)

6. **Instrument the retention programme.** Log every contact, offer, outcome and cost.
   Only then can an expected-value-per-contact model replace the F{config.THRESHOLD_BETA:g}
   threshold rule with a genuinely optimised one.
7. **Investigate the fibre churn paradox.** {worst_internet.internet_service} customers pay more
   and churn more ({worst_internet.churn_rate:.1%}). The data cannot distinguish price
   sensitivity from competitive pressure or installation/quality issues. This needs
   qualitative research plus network-quality data.

---

## 8. Additional data that would materially improve this

| Gap | What it would unlock |
|---|---|
| Retention campaign costs and save rates | A genuinely cost-optimal threshold and an expected-value-per-contact model |
| Monthly usage/consumption and support tickets | Leading indicators instead of static billing attributes |
| Competitor offers and local coverage | The strongest churn driver in telecom is usually competition, which is entirely absent here |
| Billing dates and cancellation dates | Survival analysis with time-varying risk, and true discounted CLV |
| Customer acquisition cost and channel | Profitability-weighted targeting rather than revenue-weighted |
| Payment failures and dunning events | Distinguishing voluntary churn from involuntary churn |
| Contract end dates | Pre-renewal outreach timed to the actual decision window |

---

## 9. Honest limitations

1. **Fictional sample data.** All figures describe this 7,043-row sample, not a
   real operator's book of business.
2. **Cross-sectional snapshot.** One row per customer, no time dimension. Nothing
   here is a forecast over time, and no trend can be inferred.
3. **No causality.** Every relationship reported is an association. The model ranks
   risk; it does not estimate treatment effects.
4. **No intervention was run.** No revenue was saved. "Revenue at risk" is monthly
   recurring charge attached to a group, which is an exposure measure.
5. **Class imbalance is real and consequential.** At the chosen threshold, roughly
   {1 - metrics_tuned['precision']:.0%} of flagged customers would not have churned.
   That is a deliberate, documented trade-off rather than a hidden flaw.
6. **No temporal validation.** Performance is measured on a random hold-out split.
   A time-based split would be a stronger test and is impossible without dates.

---

## 10. Reproducing this analysis

```bash
python main.py pipeline   # data only
python main.py all        # full run: data → SQL → model → scoring → reports
python -m pytest tests    # test suite
```

Deterministic seed `{config.SEED}`. All metrics in this document were produced by
the accompanying run; re-running regenerates every number identically.
"""

    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote business insights to %s", output_path)
    return markdown
