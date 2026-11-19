-- =====================================================================
-- Customer Retention & Revenue Intelligence Platform
-- Analytical SQL library (SQLite dialect)
--
-- Every block below is executable and every block is executed by
-- src/database/queries.py during the pipeline run. Results are printed to
-- the console, saved as CSV for Power BI, and summarised in the reports.
--
-- Block convention:
--   -- name:        unique identifier used by the runner
--   -- title:       human readable heading
--   -- description: what business question this answers
-- =====================================================================


-- name: q01_overall_churn_rate
-- title: Overall customer churn rate
-- description: Baseline KPI. Establishes the churn rate every other query is compared against.
SELECT
    COUNT(*)                                              AS total_customers,
    SUM(churn_label)                                      AS churned_customers,
    COUNT(*) - SUM(churn_label)                           AS retained_customers,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)         AS churn_rate_pct,
    ROUND(AVG(tenure), 2)                                 AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)                        AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2)                        AS total_monthly_revenue
FROM fact_billing;


-- name: q02_churn_by_contract
-- title: Churn by contract type
-- description: GROUP BY with a share-of-total window function. Tests whether commitment length is protective.
SELECT
    k.contract,
    COUNT(*)                                                          AS customers,
    SUM(b.churn_label)                                                AS churned,
    ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2)                   AS churn_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct_of_base,
    ROUND(SUM(b.monthly_charges), 2)                                  AS monthly_revenue,
    ROUND(SUM(CASE WHEN b.churn_label = 1 THEN b.monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM fact_billing b
JOIN dim_contract k ON k.customer_id = b.customer_id
GROUP BY k.contract
ORDER BY churn_rate_pct DESC;


-- name: q03_churn_by_tenure_group
-- title: Churn by tenure group
-- description: Uses CASE WHEN to order tenure bands chronologically instead of alphabetically.
SELECT
    b.tenure_group,
    COUNT(*)                                        AS customers,
    SUM(b.churn_label)                              AS churned,
    ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(b.tenure), 1)                         AS avg_tenure_months,
    ROUND(AVG(b.monthly_charges), 2)                AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN b.churn_label = 1 THEN b.monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM fact_billing b
GROUP BY b.tenure_group
ORDER BY CASE b.tenure_group
             WHEN '0-6 months'   THEN 1
             WHEN '7-12 months'  THEN 2
             WHEN '13-24 months' THEN 3
             WHEN '25-48 months' THEN 4
             WHEN '49+ months'   THEN 5
             ELSE 6
         END;


-- name: q04_churn_by_payment_method
-- title: Churn by payment method
-- description: Compares automatic payment uptake against churn; relevant because it is a cheap operational lever.
SELECT
    k.payment_method,
    k.auto_payment,
    COUNT(*)                                        AS customers,
    SUM(b.churn_label)                              AS churned,
    ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(b.monthly_charges), 2)                AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN b.churn_label = 1 THEN b.monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM fact_billing b
JOIN dim_contract k ON k.customer_id = b.customer_id
GROUP BY k.payment_method, k.auto_payment
ORDER BY churn_rate_pct DESC;


-- name: q05_churn_by_internet_service
-- title: Churn by internet service and add-on depth
-- description: Joins services to billing, buckets add-on depth with CASE WHEN, and filters tiny cells with HAVING.
SELECT
    s.internet_service,
    CASE
        WHEN s.num_addon_services = 0 THEN '0 add-ons'
        WHEN s.num_addon_services <= 2 THEN '1-2 add-ons'
        WHEN s.num_addon_services <= 4 THEN '3-4 add-ons'
        ELSE '5-6 add-ons'
    END                                             AS addon_bucket,
    COUNT(*)                                        AS customers,
    SUM(b.churn_label)                              AS churned,
    ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(b.monthly_charges), 2)                AS avg_monthly_charges
FROM fact_billing b
JOIN dim_service s ON s.customer_id = b.customer_id
GROUP BY
    s.internet_service,
    CASE
        WHEN s.num_addon_services = 0 THEN '0 add-ons'
        WHEN s.num_addon_services <= 2 THEN '1-2 add-ons'
        WHEN s.num_addon_services <= 4 THEN '3-4 add-ons'
        ELSE '5-6 add-ons'
    END
HAVING COUNT(*) >= 30
ORDER BY s.internet_service, churn_rate_pct DESC;


-- name: q06_monthly_revenue_lost_to_churn
-- title: Monthly revenue associated with churned customers
-- description: WHERE filter plus aggregate. This is recurring monthly revenue tied to customers who left, not a lifetime loss.
SELECT
    ROUND(SUM(CASE WHEN churn_label = 1 THEN monthly_charges ELSE 0 END), 2)       AS monthly_revenue_churned,
    ROUND(SUM(CASE WHEN churn_label = 0 THEN monthly_charges ELSE 0 END), 2)       AS monthly_revenue_retained,
    ROUND(SUM(monthly_charges), 2)                                                 AS monthly_revenue_total,
    ROUND(100.0 * SUM(CASE WHEN churn_label = 1 THEN monthly_charges ELSE 0 END)
          / SUM(monthly_charges), 2)                                               AS revenue_share_lost_pct,
    ROUND(AVG(CASE WHEN churn_label = 1 THEN monthly_charges END), 2)              AS avg_monthly_charge_churned,
    ROUND(AVG(CASE WHEN churn_label = 0 THEN monthly_charges END), 2)              AS avg_monthly_charge_retained
FROM fact_billing
WHERE tenure >= 0;


-- name: q07_revenue_contribution_by_segment
-- title: Revenue contribution by customer segment
-- description: Groups realised value by revenue band and reports each band's share of total monthly revenue.
SELECT
    b.revenue_band,
    COUNT(*)                                                AS customers,
    SUM(b.churn_label)                                      AS churned,
    ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2)         AS churn_rate_pct,
    ROUND(SUM(b.monthly_charges), 2)                        AS monthly_revenue,
    ROUND(100.0 * SUM(b.monthly_charges)
          / (SELECT SUM(monthly_charges) FROM fact_billing), 2) AS revenue_share_pct,
    ROUND(AVG(b.total_charges), 2)                          AS avg_realised_revenue_to_date
FROM fact_billing b
GROUP BY b.revenue_band
ORDER BY CASE b.revenue_band
             WHEN 'Low'      THEN 1
             WHEN 'Mid-Low'  THEN 2
             WHEN 'Mid-High' THEN 3
             WHEN 'High'     THEN 4
             ELSE 5
         END;


-- name: q08_customer_value_proxies
-- title: Customer value proxies and churn exposure
-- description: |
--   Two defensible value proxies, kept separate on purpose:
--     realised_value_to_date  = total_charges (billed to date, factual)
--     annualised_value        = monthly_charges * 12 (simple extrapolation, a proxy)
--   No discounting, no predicted lifetime, because the dataset has no dates or
--   survival horizon. NTILE ranks customers into value quintiles.
SELECT
    value_quintile,
    COUNT(*)                                                    AS customers,
    ROUND(MIN(monthly_charges), 2)                              AS min_monthly_charges,
    ROUND(MAX(monthly_charges), 2)                              AS max_monthly_charges,
    ROUND(AVG(annualised_value), 2)                             AS avg_annualised_value,
    ROUND(AVG(realised_value_to_date), 2)                       AS avg_realised_value_to_date,
    SUM(churn_label)                                            AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)               AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_label = 1 THEN annualised_value ELSE 0 END), 2) AS annualised_value_churned
FROM (
    SELECT
        monthly_charges,
        total_charges                                                  AS realised_value_to_date,
        monthly_charges * 12.0                                          AS annualised_value,
        churn_label,
        NTILE(5) OVER (ORDER BY monthly_charges DESC)                   AS value_quintile
    FROM fact_billing
) ranked
GROUP BY value_quintile
ORDER BY value_quintile;


-- name: q09_high_value_customer_segments
-- title: High-value customer segments
-- description: CTE builds high-value customers then breaks them down by contract and internet service. HAVING removes thin cells.
WITH high_value AS (
    SELECT
        c.customer_id,
        c.contract,
        c.payment_method,
        s.internet_service,
        s.num_addon_services,
        b.tenure,
        b.monthly_charges,
        b.churn_label
    FROM fact_billing  b
    JOIN dim_contract  c ON c.customer_id = b.customer_id
    JOIN dim_service   s ON s.customer_id = b.customer_id
    WHERE b.high_value_customer = 1
)
SELECT
    contract,
    internet_service,
    COUNT(*)                                        AS customers,
    ROUND(AVG(monthly_charges), 2)                  AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2)                  AS monthly_revenue,
    ROUND(AVG(tenure), 1)                           AS avg_tenure_months,
    SUM(churn_label)                                AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)   AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_label = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM high_value
GROUP BY contract, internet_service
HAVING COUNT(*) >= 20
ORDER BY monthly_revenue_at_risk DESC
LIMIT 15;


-- name: q10_churn_ranking_by_segment
-- title: Segment churn ranking within each contract type
-- description: Window function RANK() ranks payment-method/internet combinations inside each contract tier.
WITH segment_stats AS (
    SELECT
        c.contract,
        s.internet_service,
        c.payment_method,
        COUNT(*)                                        AS customers,
        SUM(b.churn_label)                              AS churned,
        ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct,
        ROUND(SUM(b.monthly_charges), 2)                AS monthly_revenue
    FROM fact_billing  b
    JOIN dim_contract  c ON c.customer_id = b.customer_id
    JOIN dim_service   s ON s.customer_id = b.customer_id
    GROUP BY c.contract, s.internet_service, c.payment_method
    HAVING COUNT(*) >= 25
)
SELECT
    contract,
    internet_service,
    payment_method,
    customers,
    churned,
    churn_rate_pct,
    monthly_revenue,
    RANK() OVER (PARTITION BY contract ORDER BY churn_rate_pct DESC) AS churn_rank_in_contract
FROM segment_stats
ORDER BY contract, churn_rank_in_contract;


-- name: q11_service_depth_vs_churn
-- title: Service depth versus churn
-- description: CTE + self-comparison subquery. Shows whether customers with more products stay longer.
WITH depth AS (
    SELECT
        s.service_count,
        b.tenure,
        b.monthly_charges,
        b.churn_label
    FROM fact_billing b
    JOIN dim_service  s ON s.customer_id = b.customer_id
)
SELECT
    service_count,
    COUNT(*)                                        AS customers,
    SUM(churn_label)                                AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)   AS churn_rate_pct,
    ROUND(AVG(tenure), 1)                           AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)                  AS avg_monthly_charges,
    ROUND(100.0 * SUM(churn_label) / COUNT(*))
        - (SELECT ROUND(100.0 * AVG(churn_label)) FROM fact_billing) AS churn_rate_vs_baseline_pts
FROM depth
GROUP BY service_count
ORDER BY service_count;


-- name: q12_contract_churn_lift
-- title: Contract churn lift versus baseline
-- description: Subquery computes the overall rate, then each contract's lift over it is derived.
SELECT
    contract,
    customers,
    churned,
    churn_rate_pct,
    ROUND(churn_rate_pct / (SELECT 100.0 * AVG(churn_label) FROM fact_billing), 2) AS lift_vs_baseline,
    ROUND(churn_rate_pct - (SELECT 100.0 * AVG(churn_label) FROM fact_billing), 2) AS pp_vs_baseline
FROM (
    SELECT
        c.contract,
        COUNT(*)                                        AS customers,
        SUM(b.churn_label)                              AS churned,
        ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct
    FROM fact_billing b
    JOIN dim_contract c ON c.customer_id = b.customer_id
    GROUP BY c.contract
) contract_stats
ORDER BY lift_vs_baseline DESC;


-- name: q13_tenure_revenue_cohorts
-- title: Tenure cohort revenue and cumulative revenue share
-- description: Window function running total over tenure cohorts, plus per-cohort churn.
WITH cohorts AS (
    SELECT
        tenure_group,
        COUNT(*)                                        AS customers,
        SUM(churn_label)                                AS churned,
        ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)   AS churn_rate_pct,
        ROUND(SUM(monthly_charges), 2)                  AS monthly_revenue,
        ROUND(SUM(total_charges), 2)                    AS realised_revenue
    FROM fact_billing
    GROUP BY tenure_group
)
SELECT
    tenure_group,
    customers,
    churned,
    churn_rate_pct,
    monthly_revenue,
    realised_revenue,
    ROUND(100.0 * monthly_revenue / SUM(monthly_revenue) OVER (), 2)      AS pct_of_monthly_revenue,
    ROUND(SUM(monthly_revenue) OVER (
        ORDER BY CASE tenure_group
                     WHEN '0-6 months'   THEN 1
                     WHEN '7-12 months'  THEN 2
                     WHEN '13-24 months' THEN 3
                     WHEN '25-48 months' THEN 4
                     ELSE 5
                 END
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2)             AS cumulative_monthly_revenue
FROM cohorts
ORDER BY CASE tenure_group
             WHEN '0-6 months'   THEN 1
             WHEN '7-12 months'  THEN 2
             WHEN '13-24 months' THEN 3
             WHEN '25-48 months' THEN 4
             ELSE 5
         END;


-- name: q14_customer_profile_churn
-- title: Churn by demographic profile
-- description: Multi-column GROUP BY with HAVING to suppress statistically thin cells.
SELECT
    c.senior_citizen,
    c.partner,
    c.dependents,
    COUNT(*)                                        AS customers,
    SUM(b.churn_label)                              AS churned,
    ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(b.monthly_charges), 2)                AS avg_monthly_charges,
    ROUND(AVG(b.tenure), 1)                         AS avg_tenure_months
FROM fact_billing b
JOIN dim_customer c ON c.customer_id = b.customer_id
GROUP BY c.senior_citizen, c.partner, c.dependents
HAVING COUNT(*) >= 25
ORDER BY churn_rate_pct DESC;


-- name: q15_addon_protection_effect
-- title: Protection add-on effect on churn
-- description: CASE WHEN turns each add-on into a has/has-not label, then aggregates across the whole base.
SELECT
    'Online security' AS addon,
    ROUND(100.0 * AVG(CASE WHEN online_security = 'Yes' THEN churn_label END), 2) AS churn_rate_with_pct,
    ROUND(100.0 * AVG(CASE WHEN online_security = 'No'  THEN churn_label END), 2) AS churn_rate_without_pct,
    SUM(CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END)                      AS customers_with
FROM fact_billing b
JOIN dim_service  s ON s.customer_id = b.customer_id
UNION ALL
SELECT 'Tech support',
    ROUND(100.0 * AVG(CASE WHEN tech_support = 'Yes' THEN churn_label END), 2),
    ROUND(100.0 * AVG(CASE WHEN tech_support = 'No'  THEN churn_label END), 2),
    SUM(CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END)
FROM fact_billing b JOIN dim_service s ON s.customer_id = b.customer_id
UNION ALL
SELECT 'Online backup',
    ROUND(100.0 * AVG(CASE WHEN online_backup = 'Yes' THEN churn_label END), 2),
    ROUND(100.0 * AVG(CASE WHEN online_backup = 'No'  THEN churn_label END), 2),
    SUM(CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END)
FROM fact_billing b JOIN dim_service s ON s.customer_id = b.customer_id
UNION ALL
SELECT 'Device protection',
    ROUND(100.0 * AVG(CASE WHEN device_protection = 'Yes' THEN churn_label END), 2),
    ROUND(100.0 * AVG(CASE WHEN device_protection = 'No'  THEN churn_label END), 2),
    SUM(CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END)
FROM fact_billing b JOIN dim_service s ON s.customer_id = b.customer_id
ORDER BY churn_rate_with_pct DESC;


-- name: q16_high_risk_revenue_concentration
-- title: Revenue concentration in the highest-risk decile
-- description: |
--   Ranks customers by an observable risk profile (short tenure, month-to-month,
--   fibre, electronic check, no tech support) to show where revenue concentrates.
--   This is a descriptive rule-based ranking, NOT the ML model's output; the
--   model-based version lives in data/exports/retention_summary.csv.
WITH risk_ranked AS (
    SELECT
        b.customer_id,
        b.monthly_charges,
        b.tenure,
        b.churn_label,
        NTILE(10) OVER (
            ORDER BY
                CASE WHEN c.contract = 'Month-to-month' THEN 3 ELSE 0 END
              + CASE WHEN s.internet_service = 'Fiber optic' THEN 2 ELSE 0 END
              + CASE WHEN k.payment_method = 'Electronic check' THEN 2 ELSE 0 END
              + CASE WHEN s.tech_support <> 'Yes' THEN 1 ELSE 0 END
              - CASE WHEN b.tenure >= 24 THEN 2 ELSE 0 END
                DESC,
                b.monthly_charges DESC
        ) AS risk_decile
    FROM fact_billing b
    JOIN dim_contract c ON c.customer_id = b.customer_id
    JOIN dim_service  s ON s.customer_id = b.customer_id
    JOIN dim_contract k ON k.customer_id = b.customer_id
)
SELECT
    risk_decile,
    COUNT(*)                                                    AS customers,
    SUM(churn_label)                                            AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)               AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2)                              AS monthly_revenue,
    ROUND(100.0 * SUM(monthly_charges) / SUM(SUM(monthly_charges)) OVER (), 2) AS pct_of_total_revenue
FROM risk_ranked
GROUP BY risk_decile
ORDER BY risk_decile;


-- name: q17_monthly_charge_bands
-- title: Churn across monthly charge bands
-- description: Banded revenue view with a churn-vs-baseline delta expressed in percentage points.
SELECT
    revenue_band,
    COUNT(*)                                                     AS customers,
    ROUND(AVG(monthly_charges), 2)                               AS avg_monthly_charges,
    SUM(churn_label)                                             AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2)                AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn_label = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk,
    ROUND(100.0 * SUM(churn_label) / COUNT(*)
          - (SELECT 100.0 * AVG(churn_label) FROM fact_billing), 2) AS pp_vs_baseline
FROM fact_billing
GROUP BY revenue_band
ORDER BY churn_rate_pct DESC;


-- name: q18_best_worst_segments
-- title: Best and worst retention segments in one table
-- description: CTE with two ranked views unioned, demonstrating UNION ALL plus window ranking.
WITH segment_summary AS (
    SELECT
        c.contract || ' | ' || s.internet_service || ' | ' || c.payment_method AS segment,
        COUNT(*)                                        AS customers,
        ROUND(100.0 * SUM(b.churn_label) / COUNT(*), 2) AS churn_rate_pct,
        ROUND(SUM(b.monthly_charges), 2)                AS monthly_revenue
    FROM fact_billing  b
    JOIN dim_contract  c ON c.customer_id = b.customer_id
    JOIN dim_service   s ON s.customer_id = b.customer_id
    GROUP BY c.contract, s.internet_service, c.payment_method
    HAVING COUNT(*) >= 25
),
ranked AS (
    SELECT
        segment, customers, churn_rate_pct, monthly_revenue,
        RANK() OVER (ORDER BY churn_rate_pct DESC) AS rnk_desc,
        RANK() OVER (ORDER BY churn_rate_pct ASC)  AS rnk_asc
    FROM segment_summary
)
SELECT 'Highest churn' AS bucket, segment, customers, churn_rate_pct, monthly_revenue
FROM ranked WHERE rnk_desc <= 5
UNION ALL
SELECT 'Lowest churn', segment, customers, churn_rate_pct, monthly_revenue
FROM ranked WHERE rnk_asc <= 5
ORDER BY bucket, churn_rate_pct DESC;
