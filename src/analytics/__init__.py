"""Analytics package: descriptive and diagnostic business analysis."""

from .churn_analysis import (
    churn_by_addon,
    churn_by_contract,
    churn_by_internet_service,
    churn_by_payment_method,
    churn_by_tenure_group,
    churn_driver_summary,
)
from .customer_analysis import (
    customer_segments,
    demographic_profile,
    service_adoption_summary,
)
from .revenue_analysis import (
    customer_value_proxies,
    revenue_by_segment,
    revenue_exposure,
    revenue_summary,
)

__all__ = [
    "churn_by_addon",
    "churn_by_contract",
    "churn_by_internet_service",
    "churn_by_payment_method",
    "churn_by_tenure_group",
    "churn_driver_summary",
    "customer_segments",
    "demographic_profile",
    "service_adoption_summary",
    "customer_value_proxies",
    "revenue_by_segment",
    "revenue_exposure",
    "revenue_summary",
]
