# Customer Retention & Revenue Intelligence — Business Insights

**Generated:** 2026-11-19 15:25 UTC
**Dataset:** IBM Telco Customer Churn (fictional sample, 7,043 customers × 21 columns)
**Model:** `xgboost` · decision threshold 0.293

> **Scope statement.** This is a cross-sectional customer-churn classification and
> retention-prioritisation project. It is *not* a longitudinal monthly forecast.
> All monetary figures are in the dataset's own charge units, described here as
> "charge units" rather than a currency, because the source documents no currency.
> No retention campaign was executed, so nothing in this document claims revenue
> was saved or churn was reduced.

---

## 1. The business problem

The operator is losing roughly a quarter of its customer base
(26.5%, 1,869 of
7,043 customers) and wants three things answered:

1. **Why** are customers leaving, and which attributes travel with churn?
2. **Where** is the money exposed — which segments carry the most at-risk revenue?
3. **Who** should the retention team call first when it can only reach a fraction
   of the base?

---

## 2. Headline numbers

| KPI | Value | How it is calculated |
|---|---|---|
| Total customers | 7,043 | Row count of the cleaned dataset |
| Churned customers | 1,869 | Count where `churn = Yes` |
| Retained customers | 5,174 | Count where `churn = No` |
| Overall churn rate | 26.54% | Churned ÷ total |
| Average tenure | 32.4 months | Mean of `tenure` |
| Average monthly charges | 64.76 | Mean of `monthly_charges` |
| Total monthly revenue | 456,116.60 | Sum of `monthly_charges` — recurring, one month |
| Monthly revenue attached to churned customers | 139,130.85 | Sum of `monthly_charges` where churn = Yes |
| Share of monthly revenue lost with churners | 30.50% | Above ÷ total monthly revenue |
| Average monthly charge — churned | 74.44 | Mean charges among churners |
| Average monthly charge — retained | 61.27 | Mean charges among retained |
| Revenue realised to date (all customers) | 16,056,168.70 | Sum of `total_charges` |
| High-value customers (≥75th pct charges) | 1,771 | Monthly charges ≥ 89.85 |
| High-value churn rate | 32.75% | Churn among high-value customers |
| Revenue in top 10% of spenders | 16.7% | Share of monthly revenue |

**Read the revenue number carefully.** The
139,130.85 figure is *one month* of recurring
charges attached to customers who already left. It is an exposure measure, not a
cumulative loss, and it does not net off the cost of any retention activity.

---

## 3. What the data says about churn

### 3.1 Contract type is the single sharpest split

| Contract | Customers | Churned | Churn rate | Lift vs base | Monthly revenue at risk |
|---|---|---|---|---|---|
| Month-to-month | 3,875 | 1,655 | 42.7% | 1.61x | 120,847.10 |
| One year | 1,473 | 166 | 11.3% | 0.42x | 14,118.45 |
| Two year | 1,695 | 48 | 2.8% | 0.11x | 4,165.30 |

Month-to-month customers churn at 42.7%, which is
1.61× the base rate. Two year customers churn at
2.8% — a difference of
40 percentage points.

*Why it matters:* commitment length is the strongest observable retention signal
in this dataset, and it is something the business controls through pricing and
contract design.
*What to investigate:* whether conversion from month-to-month to a fixed term is
priced attractively enough, and what churn looks like in the first weeks after
signing a longer contract.

### 3.2 Churn is front-loaded in the customer lifecycle

| Tenure group | Customers | Churn rate | vs base (pts) |
|---|---|---|---|
| 0-6 months | 1,481 | 52.9% | +26.4 |
| 7-12 months | 705 | 35.9% | +9.3 |
| 13-24 months | 1,024 | 28.7% | +2.2 |
| 25-48 months | 1,594 | 20.4% | -6.2 |
| 49+ months | 2,239 | 9.5% | -17.0 |

The 0-6 months cohort churns at 52.9%. Churned
customers have a median tenure of 10
months versus 38 months for
retained customers.

*Why it matters:* the first year is where retention spend has the highest leverage,
because that is where the losses are concentrated.
*What to investigate:* onboarding quality, first-bill shock, and early support
contact volume.

### 3.3 Payment method and automatic billing

| Payment method | Customers | Churn rate | Automatic |
|---|---|---|---|
| Electronic check | 2,365 | 45.3% | No |
| Mailed check | 1,612 | 19.1% | No |
| Bank transfer (automatic) | 1,544 | 16.7% | Yes |
| Credit card (automatic) | 1,522 | 15.2% | Yes |

Electronic check customers churn at 45.3%.
*Why it matters:* payment friction is cheap to fix compared with a price change.
Moving customers onto automatic payment removes a monthly decision point where
they can reconsider the relationship.
*Caveat:* electronically-paying customers may differ in other ways (age, tech
comfort), so this is an association, not a proven intervention effect.

### 3.4 Internet service and service depth

| Internet service | Customers | Churn rate | Monthly revenue at risk |
|---|---|---|---|
| Fiber optic | 3,096 | 41.9% | 114,300.05 |
| DSL | 2,421 | 19.0% | 22,529.20 |
| No | 1,526 | 7.4% | 2,301.60 |

Fiber optic customers churn at 41.9%
despite paying more. Meanwhile add-on depth is protective:

| Add-on | Churn with | Churn without | Difference (pts) |
|---|---|---|---|
| Online Security | 14.6% | 41.8% | -27.2 |
| Tech Support | 15.2% | 41.6% | -26.5 |
| Online Backup | 21.5% | 39.9% | -18.4 |
| Device Protection | 22.5% | 39.1% | -16.6 |
| Streaming Movies | 29.9% | 33.7% | -3.7 |
| Streaming Tv | 30.1% | 33.5% | -3.5 |

The strongest single association is **Online Security**:
14.6% churn with it versus
41.8% without
(-27.2 percentage points).

*Why it matters:* customers who have adopted support-style services stay longer.
*What to investigate:* whether bundling support services changes retention
(causal test needed) or simply attracts customers who were already committed.

### 3.5 The strongest raw churn signals

| Feature | Value | Customers | Churn rate | vs base (pts) | Lift |
|---|---|---|---|---|---|
| tenure_group | 0-6 months | 1,481 | 52.9% | +26.4 | 1.99x |
| contract | Two year | 1,695 | 2.8% | -23.7 | 0.11x |
| online_security | No internet service | 1,526 | 7.4% | -19.1 | 0.28x |
| internet_service | No | 1,526 | 7.4% | -19.1 | 0.28x |
| tech_support | No internet service | 1,526 | 7.4% | -19.1 | 0.28x |
| payment_method | Electronic check | 2,365 | 45.3% | +18.8 | 1.71x |

One row per feature, so the repeated "No internet service" category across the
add-on columns does not inflate the table. This is a purely descriptive screen of
single attributes: useful for framing, but superseded by the multivariate model in
section 5, which controls for the other variables.

### 3.6 Segment heat map — the danger zone

The highest-exposure segment by revenue is **Month-to-month / Mid-High**:
1,244 customers, 53.2% churn,
52,866.22 of monthly revenue attached to churners.

The interaction matters: uncommitted customers on high bills are far riskier than
either attribute alone suggests, while long-contract high-spend customers are the
safest revenue in the business.

---

## 4. Revenue exposure

| Measure | Value |
|---|---|
| Total monthly revenue | 456,116.60 |
| Monthly revenue attached to churned customers | 139,130.85 |
| Share of monthly revenue | 30.50% |
| Revenue in top 10% of spenders | 16.7% |
| Revenue in top 20% of spenders | 31.9% |

Revenue is moderately concentrated: the top decile of spenders holds
16.7% of monthly revenue. Critically, churn is
*not* confined to low-value customers — churners pay
74.44 per month on average versus
61.27 for retained customers.
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
because churn is a 26.5% minority class and the business
decision is a ranked list, not a balanced classifier.

**Selected: `xgboost`** with `{"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9}`.

### 5.2 Test-set performance (never seen during training)

| Metric | At 0.50 cut-off | At tuned cut-off 0.29 |
|---|---|---|
| Accuracy | 0.751 | 0.666 |
| Precision | 0.520 | 0.439 |
| Recall | 0.791 | 0.922 |
| F1 | 0.628 | 0.595 |
| ROC-AUC | 0.844 | 0.844 |
| PR-AUC | 0.658 | 0.658 |

At the tuned cut-off the model catches **345 of 374 churners**
(92.2% recall) while raising
**441 false alarms** (43.9% precision). At the
default 0.5 cut-off it catches 296 churners with 273 false alarms.

**The trade-off, stated plainly.** Pushing recall from
79.1% to 92.2% costs
168 extra wasted contacts. That is the correct direction to err when a
retention call is cheap relative to the recurring revenue of a customer who leaves —
but it should be revisited once real call costs and save rates are known. Without
those, no dollar-optimal threshold exists.

### 5.3 What drives the predictions (SHAP)

| Rank | Feature (model input) | Mean absolute SHAP | Share of total |
|---|---|---|---|
| 1 | contract = Month-to-month | 0.6978 | 21.7% |
| 2 | tenure | 0.4504 | 14.0% |
| 3 | online security = No | 0.2458 | 7.6% |
| 4 | internet service = Fiber optic | 0.2157 | 6.7% |
| 5 | tech support = No | 0.1970 | 6.1% |
| 6 | payment method = Electronic check | 0.1681 | 5.2% |
| 7 | avg monthly charge | 0.1624 | 5.1% |
| 8 | total charges | 0.1385 | 4.3% |
| 9 | contract = Two year | 0.1252 | 3.9% |
| 10 | monthly charges | 0.1176 | 3.7% |

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
**10% of the base** (141
customers). Lift is measured on the **hold-out test split**, where outcomes were
never seen during training.

| Capacity | Customers | Churn rate in group | Lift | Churners captured | Monthly revenue in group |
|---|---|---|---|---|---|
| 5% | 70 | 81.4% | 3.07x | 15.2% | 5,892.05 |
| 10% | 141 | 77.3% | 2.91x | 29.1% | 11,329.70 |
| 20% | 282 | 67.0% | 2.52x | 50.5% | 21,789.10 |
| 30% | 423 | 58.9% | 2.22x | 66.6% | 32,299.90 |
| 50% | 704 | 46.3% | 1.75x | 87.2% | 51,977.15 |

### The headline result

Contacting the top 10% by predicted risk
(141 customers):

- **Observed churn rate in the group: 77.3%**
  versus **26.5%** overall → **lift of 2.91×**
- **109 churners captured**
  (29.1% of all churners) from
  10% of the base
- **11,329.70 of monthly revenue**
  (12.6% of total) sits in the group
- **Expected monthly revenue at risk in the group:
  10,168.95** — this is
  Σ(monthly charge × predicted probability), a probability-weighted exposure
  estimate, *not* a forecast of realised loss

For comparison, contacting a random 10% of customers would
capture only about 10% of churners. The model captures
29%.

### What the priority group looks like

- **100% are month-to-month contracts**
- **94% are fibre-optic customers**
- Median tenure: **3 months**

That profile points at a coherent playbook: early-lifecycle, high-bill, uncommitted
fibre customers. It is not a random slice of the base.

### Capacity scenarios

| Capacity | Customers | Observed churn in group | Lift | Churners captured | Expected monthly revenue at risk |
|---|---|---|---|---|---|
| 5% | 70 | 81.4% | 3.07x | 15.2% | 5,467.64 |
| 10% | 141 | 77.3% | 2.91x | 29.1% | 10,168.95 |
| 15% | 211 | 71.1% | 2.68x | 40.1% | 14,301.78 |
| 20% | 282 | 67.0% | 2.52x | 50.5% | 18,343.74 |
| 30% | 423 | 58.9% | 2.22x | 66.6% | 25,533.15 |

The lift decays as capacity grows — that is expected and healthy. It is precisely
the argument for ranking rather than mass outreach: the marginal customer added at
30% capacity is far less likely to churn than the marginal customer at 5%.

### What this does and does not prove

- **Proves:** the ranking concentrates future churners well above the base rate on
  data the model never saw.
- **Does not prove:** that contacting them retains them. Without an intervention
  and a measured save rate, no revenue has been saved. The
  10,168.95 figure is exposure,
  not savings.

---

## 7. Recommendations

Recommendations are ordered by how well the evidence supports them.

### Immediate (evidence is strong, action is cheap)

1. **Rank every customer weekly and work the top decile first.** The out-of-sample
   lift of 2.91× means the same number of calls reaches
   roughly three times as many genuine churners as an untargeted list. The priority
   file `data/exports/powerbi/retention_priority_list.csv` is ready to use.
2. **Target the early-lifecycle cohort.** The 0-6 months group churns at
   52.9%. A structured first-90-days onboarding programme attacks
   the largest concentration of churn at the point where it is cheapest to influence.
3. **Push automatic payment enrolment.** Electronic check customers churn at
   45.3%. Converting manual payers to automatic billing removes
   a recurring churn decision point and requires no pricing change.

### Short term (evidence is strong, action needs design)

4. **Design a month-to-month → fixed-term conversion offer.** Month-to-month accounts
   for 100% of the high-risk list. Even a modest
   conversion rate moves a large block of exposed revenue
   (120,847.10 sits with month-to-month
   churners) onto committed contracts.
5. **Bundle support services for at-risk fibre customers.** Online Security shows a
   -27.2 point churn gap. Run this as a **controlled test**
   with a holdout group before scaling, because the current evidence is associational —
   engaged customers may simply buy more services.

### Medium term (needs more data first)

6. **Instrument the retention programme.** Log every contact, offer, outcome and cost.
   Only then can an expected-value-per-contact model replace the F2
   threshold rule with a genuinely optimised one.
7. **Investigate the fibre churn paradox.** Fiber optic customers pay more
   and churn more (41.9%). The data cannot distinguish price
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
   56% of flagged customers would not have churned.
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

Deterministic seed `42`. All metrics in this document were produced by
the accompanying run; re-running regenerates every number identically.
