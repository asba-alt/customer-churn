# SQL Analytics Results

Every table below is produced by executing `sql/analytics_queries.sql` against `data/processed/churn_analytics.sqlite`.

### Overall customer churn rate

_Query block: `q01_overall_churn_rate`_


| total_customers | churned_customers | retained_customers | churn_rate_pct | avg_tenure_months | avg_monthly_charges | total_monthly_revenue |
|---|---|---|---|---|---|---|
| 7043 | 1869 | 5174 | 26.54 | 32.37 | 64.76 | 456,116.60 |

### Churn by contract type

_Query block: `q02_churn_by_contract`_


| contract | customers | churned | churn_rate_pct | pct_of_base | monthly_revenue | monthly_revenue_at_risk |
|---|---|---|---|---|---|---|
| Month-to-month | 3875 | 1655 | 42.71 | 55.02 | 257,294.15 | 120,847.10 |
| One year | 1473 | 166 | 11.27 | 20.91 | 95,816.60 | 14,118.45 |
| Two year | 1695 | 48 | 2.83 | 24.07 | 103,005.85 | 4,165.30 |

### Churn by tenure group

_Query block: `q03_churn_by_tenure_group`_


| tenure_group | customers | churned | churn_rate_pct | avg_tenure_months | avg_monthly_charges | monthly_revenue_at_risk |
|---|---|---|---|---|---|---|
| 0-6 months | 1481 | 784 | 52.94 | 2.50 | 54.74 | 49,896.10 |
| 7-12 months | 705 | 253 | 35.89 | 9.40 | 58.95 | 19,058.15 |
| 13-24 months | 1024 | 294 | 28.71 | 18.40 | 61.36 | 23,081.65 |
| 25-48 months | 1594 | 325 | 20.39 | 36.20 | 65.93 | 27,462.50 |
| 49+ months | 2239 | 213 | 9.51 | 63.00 | 73.95 | 19,632.45 |

### Churn by payment method

_Query block: `q04_churn_by_payment_method`_


| payment_method | auto_payment | customers | churned | churn_rate_pct | avg_monthly_charges | monthly_revenue_at_risk |
|---|---|---|---|---|---|---|
| Electronic check | 0 | 2365 | 1071 | 45.29 | 76.26 | 84,288.75 |
| Mailed check | 0 | 1612 | 308 | 19.11 | 43.92 | 16,803.60 |
| Bank transfer (automatic) | 1 | 1544 | 258 | 16.71 | 67.19 | 20,091.90 |
| Credit card (automatic) | 1 | 1522 | 232 | 15.24 | 66.51 | 17,946.60 |

### Churn by internet service and add-on depth

_Query block: `q05_churn_by_internet_service`_


| internet_service | addon_bucket | customers | churned | churn_rate_pct | avg_monthly_charges |
|---|---|---|---|---|---|
| DSL | 0 add-ons | 294 | 121 | 41.16 | 40.28 |
| DSL | 1-2 add-ons | 869 | 226 | 26.01 | 49.42 |
| DSL | 3-4 add-ons | 834 | 95 | 11.39 | 62.99 |
| DSL | 5-6 add-ons | 424 | 17 | 4.01 | 78.65 |
| Fiber optic | 0 add-ons | 399 | 241 | 60.40 | 72.08 |
| Fiber optic | 1-2 add-ons | 1130 | 586 | 51.86 | 84.26 |
| Fiber optic | 3-4 add-ons | 1136 | 401 | 35.30 | 98.42 |
| Fiber optic | 5-6 add-ons | 431 | 69 | 16.01 | 110.22 |
| No | 0 add-ons | 1526 | 113 | 7.40 | 21.08 |

### Monthly revenue associated with churned customers

_Query block: `q06_monthly_revenue_lost_to_churn`_


| monthly_revenue_churned | monthly_revenue_retained | monthly_revenue_total | revenue_share_lost_pct | avg_monthly_charge_churned | avg_monthly_charge_retained |
|---|---|---|---|---|---|
| 139,130.85 | 316,985.75 | 456,116.60 | 30.50 | 74.44 | 61.27 |

### Revenue contribution by customer segment

_Query block: `q07_revenue_contribution_by_segment`_


| revenue_band | customers | churned | churn_rate_pct | monthly_revenue | revenue_share_pct | avg_realised_revenue_to_date |
|---|---|---|---|---|---|---|
| Low | 1731 | 188 | 10.86 | 38,077.15 | 8.35 | 637.06 |
| Mid-Low | 1721 | 407 | 23.65 | 94,177.15 | 20.65 | 1,505.45 |
| Mid-High | 1847 | 701 | 37.95 | 147,740.80 | 32.39 | 2,356.86 |
| High | 1744 | 573 | 32.86 | 176,121.50 | 38.61 | 4,592.56 |

### Customer value proxies and churn exposure

_Query block: `q08_customer_value_proxies`_


| value_quintile | customers | min_monthly_charges | max_monthly_charges | avg_annualised_value | avg_realised_value_to_date | churned | churn_rate_pct | annualised_value_churned |
|---|---|---|---|---|---|---|---|---|
| 1 | 1409 | 94.25 | 118.75 | 1,237.83 | 4,885.27 | 463 | 32.86 | 563,088.60 |
| 2 | 1409 | 79.10 | 94.25 | 1,031.82 | 2,896.92 | 509 | 36.12 | 524,118.00 |
| 3 | 1409 | 58.75 | 79.10 | 839.42 | 1,936.71 | 418 | 29.67 | 361,210.80 |
| 4 | 1408 | 25.05 | 58.75 | 527.99 | 1,085.39 | 351 | 24.93 | 189,174.60 |
| 5 | 1408 | 18.25 | 25.05 | 248.08 | 592.34 | 128 | 9.09 | 31,978.20 |

### High-value customer segments

_Query block: `q09_high_value_customer_segments`_


| contract | internet_service | customers | avg_monthly_charges | monthly_revenue | avg_tenure_months | churned | churn_rate_pct | monthly_revenue_at_risk |
|---|---|---|---|---|---|---|---|---|
| Month-to-month | Fiber optic | 877 | 98.30 | 86,207.75 | 29.70 | 456 | 52.00 | 44,787.80 |
| One year | Fiber optic | 446 | 101.94 | 45,464.45 | 53.40 | 93 | 20.85 | 9,637.55 |
| Two year | Fiber optic | 395 | 106.45 | 42,048.20 | 65.40 | 30 | 7.59 | 3,165.65 |
| Two year | DSL | 50 | 91.14 | 4,557.20 | 67.70 | 1 | 2.00 | 92.45 |

### Segment churn ranking within each contract type

_Query block: `q10_churn_ranking_by_segment`_


| contract | internet_service | payment_method | customers | churned | churn_rate_pct | monthly_revenue | churn_rank_in_contract |
|---|---|---|---|---|---|---|---|
| Month-to-month | Fiber optic | Electronic check | 1307 | 789 | 60.37 | 113,948.10 | 1 |
| Month-to-month | Fiber optic | Mailed check | 201 | 102 | 50.75 | 16,600.15 | 2 |
| Month-to-month | Fiber optic | Bank transfer (automatic) | 327 | 149 | 45.57 | 28,859.00 | 3 |
| Month-to-month | Fiber optic | Credit card (automatic) | 293 | 122 | 41.64 | 25,773.85 | 4 |
| Month-to-month | DSL | Electronic check | 474 | 192 | 40.51 | 23,342.45 | 5 |
| Month-to-month | DSL | Mailed check | 367 | 113 | 30.79 | 17,924.10 | 6 |
| Month-to-month | DSL | Credit card (automatic) | 185 | 50 | 27.03 | 9,650.85 | 7 |
| Month-to-month | No | Mailed check | 325 | 67 | 20.62 | 6,598.45 | 8 |
| Month-to-month | No | Bank transfer (automatic) | 65 | 13 | 20.00 | 1,331.25 | 9 |
| Month-to-month | DSL | Bank transfer (automatic) | 197 | 39 | 19.80 | 10,501.05 | 10 |
| Month-to-month | No | Electronic check | 69 | 13 | 18.84 | 1,440.00 | 11 |
| Month-to-month | No | Credit card (automatic) | 65 | 6 | 9.23 | 1,324.90 | 12 |

_Showing 12 of 34 rows._

### Service depth versus churn

_Query block: `q11_service_depth_vs_churn`_


| service_count | customers | churned | churn_rate_pct | avg_tenure_months | avg_monthly_charges | churn_rate_vs_baseline_pts |
|---|---|---|---|---|---|---|
| 1 | 1606 | 148 | 9.22 | 29.50 | 21.27 | -18.00 |
| 2 | 727 | 375 | 51.58 | 11.00 | 58.02 | 25.00 |
| 3 | 996 | 433 | 43.47 | 18.50 | 65.53 | 16.00 |
| 4 | 1041 | 361 | 34.68 | 27.40 | 73.16 | 8.00 |
| 5 | 1062 | 289 | 27.21 | 37.80 | 82.57 | 0.00 |
| 6 | 827 | 182 | 22.01 | 47.30 | 89.45 | -5.00 |
| 7 | 525 | 66 | 12.57 | 56.60 | 95.41 | -14.00 |
| 8 | 259 | 15 | 5.79 | 64.80 | 102.70 | -21.00 |

### Contract churn lift versus baseline

_Query block: `q12_contract_churn_lift`_


| contract | customers | churned | churn_rate_pct | lift_vs_baseline | pp_vs_baseline |
|---|---|---|---|---|---|
| Month-to-month | 3875 | 1655 | 42.71 | 1.61 | 16.17 |
| One year | 1473 | 166 | 11.27 | 0.42 | -15.27 |
| Two year | 1695 | 48 | 2.83 | 0.11 | -23.71 |

### Tenure cohort revenue and cumulative revenue share

_Query block: `q13_tenure_revenue_cohorts`_


| tenure_group | customers | churned | churn_rate_pct | monthly_revenue | realised_revenue | pct_of_monthly_revenue | cumulative_monthly_revenue |
|---|---|---|---|---|---|---|---|
| 0-6 months | 1481 | 784 | 52.94 | 81,067.95 | 211,146.90 | 17.77 | 81,067.95 |
| 7-12 months | 705 | 253 | 35.89 | 41,561.80 | 390,505.00 | 9.11 | 122,629.75 |
| 13-24 months | 1024 | 294 | 28.71 | 62,829.85 | 1,153,287.70 | 13.77 | 185,459.60 |
| 25-48 months | 1594 | 325 | 20.39 | 105,093.30 | 3,810,380.35 | 23.04 | 290,552.90 |
| 49+ months | 2239 | 213 | 9.51 | 165,563.70 | 10,490,848.75 | 36.30 | 456,116.60 |

### Churn by demographic profile

_Query block: `q14_customer_profile_churn`_


| senior_citizen | partner | dependents | customers | churned | churn_rate_pct | avg_monthly_charges | avg_tenure_months |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 561 | 276 | 49.20 | 76.86 | 25.10 |
| 1 | 1 | 0 | 490 | 178 | 36.33 | 82.75 | 40.50 |
| 0 | 0 | 0 | 2719 | 847 | 31.15 | 60.12 | 22.90 |
| 1 | 1 | 1 | 83 | 20 | 24.10 | 82.95 | 47.00 |
| 0 | 0 | 1 | 353 | 75 | 21.25 | 51.99 | 23.90 |
| 0 | 1 | 0 | 1163 | 242 | 20.81 | 71.70 | 43.60 |
| 0 | 1 | 1 | 1666 | 229 | 13.75 | 59.87 | 41.10 |

### Protection add-on effect on churn

_Query block: `q15_addon_protection_effect`_


| addon | churn_rate_with_pct | churn_rate_without_pct | customers_with |
|---|---|---|---|
| Device protection | 22.50 | 39.13 | 2422 |
| Online backup | 21.53 | 39.93 | 2429 |
| Tech support | 15.17 | 41.64 | 2044 |
| Online security | 14.61 | 41.77 | 2019 |

### Revenue concentration in the highest-risk decile

_Query block: `q16_high_risk_revenue_concentration`_


| risk_decile | customers | churned | churn_rate_pct | monthly_revenue | pct_of_total_revenue |
|---|---|---|---|---|---|
| 1 | 705 | 487 | 69.08 | 59,670.90 | 13.08 |
| 2 | 705 | 390 | 55.32 | 62,588.30 | 13.72 |
| 3 | 705 | 329 | 46.67 | 44,574.30 | 9.77 |
| 4 | 704 | 234 | 33.24 | 47,729.05 | 10.46 |
| 5 | 704 | 158 | 22.44 | 29,480.45 | 6.46 |
| 6 | 704 | 136 | 19.32 | 50,497.45 | 11.07 |
| 7 | 704 | 52 | 7.39 | 38,854.50 | 8.52 |
| 8 | 704 | 48 | 6.82 | 59,237.50 | 12.99 |
| 9 | 704 | 8 | 1.14 | 16,353.15 | 3.59 |
| 10 | 704 | 27 | 3.84 | 47,131.00 | 10.33 |

### Churn across monthly charge bands

_Query block: `q17_monthly_charge_bands`_


| revenue_band | customers | avg_monthly_charges | churned | churn_rate_pct | monthly_revenue_at_risk | pp_vs_baseline |
|---|---|---|---|---|---|---|
| Mid-High | 1847 | 79.99 | 701 | 37.95 | 55,926.90 | 11.42 |
| High | 1744 | 100.99 | 573 | 32.86 | 57,054.10 | 6.32 |
| Mid-Low | 1721 | 54.72 | 407 | 23.65 | 21,726.80 | -2.89 |
| Low | 1731 | 22.00 | 188 | 10.86 | 4,423.05 | -15.68 |

### Best and worst retention segments in one table

_Query block: `q18_best_worst_segments`_


| bucket | segment | customers | churn_rate_pct | monthly_revenue |
|---|---|---|---|---|
| Highest churn | Month-to-month | Fiber optic | Electronic check | 1307 | 60.37 | 113,948.10 |
| Highest churn | Month-to-month | Fiber optic | Mailed check | 201 | 50.75 | 16,600.15 |
| Highest churn | Month-to-month | Fiber optic | Bank transfer (automatic) | 327 | 45.57 | 28,859.00 |
| Highest churn | Month-to-month | Fiber optic | Credit card (automatic) | 293 | 41.64 | 25,773.85 |
| Highest churn | Month-to-month | DSL | Electronic check | 474 | 40.51 | 23,342.45 |
| Lowest churn | One year | No | Mailed check | 164 | 1.22 | 3,349.45 |
| Lowest churn | Two year | No | Credit card (automatic) | 190 | 1.05 | 4,176.00 |
| Lowest churn | Two year | No | Mailed check | 252 | 0.79 | 5,387.45 |
| Lowest churn | Two year | No | Bank transfer (automatic) | 178 | 0.56 | 3,934.30 |
| Lowest churn | Two year | DSL | Mailed check | 109 | 0.00 | 7,146.95 |