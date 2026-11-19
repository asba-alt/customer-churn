"""Exploratory data analysis: figures plus a structured findings dictionary.

Each chart answers a specific management question. The returns of
:func:`run_full_eda` include the numbers behind every chart so the written
reports quote computed values rather than prose guesses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from . import style
from .churn_analysis import (
    churn_by_addon,
    churn_by_contract,
    churn_by_internet_service,
    churn_by_payment_method,
    churn_by_tenure_group,
)
from .customer_analysis import customer_segments
from .revenue_analysis import revenue_summary

logger = logging.getLogger(__name__)

TENURE_ORDER = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]


def plot_churn_distribution(df: pd.DataFrame) -> str:
    """How balanced is the target? Sets expectations for every later metric."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    counts = df["churn"].value_counts().reindex(["No", "Yes"])

    axes[0].bar(counts.index, counts.values,
                color=[style.CHURN_COLORS["No"], style.CHURN_COLORS["Yes"]], width=0.55)
    for i, (label, value) in enumerate(counts.items()):
        axes[0].text(i, value + 60, f"{value:,}\n({value / len(df):.1%})",
                     ha="center", fontsize=9, color="#333333")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Retained vs churned customers", loc="left")
    axes[0].set_ylim(0, counts.max() * 1.18)

    axes[1].pie(
        counts.values,
        labels=["Retained", "Churned"],
        colors=[style.CHURN_COLORS["No"], style.CHURN_COLORS["Yes"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Churn share of the base", loc="left")

    fig.suptitle("Churn distribution — 7,043 customers", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Roughly one in four customers churned. This imbalance is why PR-AUC, not accuracy, "
             "drives model selection.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return style.save(fig, "01_churn_distribution.png")


def plot_tenure_analysis(df: pd.DataFrame) -> str:
    """Tenure is the strongest continuous signal; show its shape two ways."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    rate = churn_by_tenure_group(df).set_index("tenure_group").reindex(TENURE_ORDER)
    axes[0].bar(rate.index, rate["churn_rate"] * 100, color=style.PALETTE["accent"], width=0.6)
    axes[0].axhline(df["churn_label"].mean() * 100, ls="--", color=style.PALETTE["grey"],
                    lw=1.2, label=f"Overall {df['churn_label'].mean():.1%}")
    for i, (count, churn) in enumerate(zip(rate["customers"], rate["churn_rate"])):
        axes[0].text(i, churn * 100 + 1.2, f"{churn:.1%}\nn={count:,}", ha="center", fontsize=8)
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_xlabel("Tenure group")
    axes[0].set_title("Churn rate by tenure group", loc="left")
    axes[0].set_ylim(0, rate["churn_rate"].max() * 130)
    axes[0].legend(loc="upper right")
    axes[0].tick_params(axis="x", rotation=15)

    for label, colour in [("No", style.CHURN_COLORS["No"]), ("Yes", style.CHURN_COLORS["Yes"])]:
        subset = df.loc[df["churn"] == label, "tenure"]
        axes[1].hist(subset, bins=24, alpha=0.62, color=colour,
                     label=f"{label} (median {subset.median():.0f} mo)", edgecolor="white")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Tenure distribution by outcome", loc="left")
    axes[1].legend()

    fig.suptitle("Tenure and churn", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Churn is heavily front-loaded: the earliest months carry the highest risk, and the "
             "risk falls steadily as tenure grows.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "02_tenure_analysis.png")


def plot_contract_and_payment(df: pd.DataFrame) -> str:
    """The two operational levers: contract commitment and payment mechanism."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    contract = churn_by_contract(df)
    axes[0].bar(contract["contract"], contract["churn_rate"] * 100,
                color=[style.PALETTE["danger"], style.PALETTE["amber"], style.PALETTE["teal"]],
                width=0.55)
    for i, row in contract.iterrows():
        axes[0].text(i, row["churn_rate"] * 100 + 1.0,
                     f"{row['churn_rate']:.1%}\nn={row['customers']:,}",
                     ha="center", fontsize=8)
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_title("Churn rate by contract type", loc="left")
    axes[0].set_ylim(0, contract["churn_rate"].max() * 125)

    payment = churn_by_payment_method(df).sort_values("churn_rate")
    colours = [style.PALETTE["teal"] if a else style.PALETTE["accent"]
               for a in payment["auto_payment"]]
    axes[1].barh(payment["payment_method"], payment["churn_rate"] * 100, color=colours, height=0.6)
    for i, row in enumerate(payment.itertuples()):
        axes[1].text(row.churn_rate * 100 + 0.8, i, f"{row.churn_rate:.1%}  (n={row.customers:,})",
                     va="center", fontsize=8)
    axes[1].set_xlabel("Churn rate (%)")
    axes[1].set_title("Churn rate by payment method", loc="left")
    axes[1].set_xlim(0, payment["churn_rate"].max() * 130)

    fig.suptitle("Contract and payment: the strongest controllable associations", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Teal bars are automatic payments. Month-to-month and electronic-cheque customers "
             "churn far more than committed, auto-paying customers.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "03_contract_payment.png")


def plot_service_analysis(df: pd.DataFrame) -> str:
    """Internet tier and add-on depth: what customers actually buy."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    internet = churn_by_internet_service(df).sort_values("churn_rate", ascending=False)
    axes[0].bar(internet["internet_service"], internet["churn_rate"] * 100,
                color=[style.PALETTE["danger"], style.PALETTE["amber"], style.PALETTE["teal"]],
                width=0.55)
    for i, row in internet.reset_index().iterrows():
        axes[0].text(i, row["churn_rate"] * 100 + 1.0,
                     f"{row['churn_rate']:.1%}\nn={row['customers']:,}", ha="center", fontsize=8)
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_title("Churn rate by internet service", loc="left")
    axes[0].set_ylim(0, internet["churn_rate"].max() * 125)

    depth = (
        df.groupby("num_addon_services", observed=True)
        .agg(customers=("customer_id", "count"), churn_rate=("churn_label", "mean"))
        .reset_index()
    )
    axes[1].bar(depth["num_addon_services"], depth["churn_rate"] * 100,
                color=style.PALETTE["primary"], width=0.65)
    for i, row in depth.iterrows():
        axes[1].text(row["num_addon_services"], row["churn_rate"] * 100 + 1.0,
                     f"{row['churn_rate']:.1%}\nn={row['customers']:,}", ha="center", fontsize=7.5)
    axes[1].set_xlabel("Number of add-on services")
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_title("Churn rate by add-on depth", loc="left")
    axes[1].set_ylim(0, depth["churn_rate"].max() * 130)

    fig.suptitle("Service mix and churn", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Fibre-optic customers churn most despite paying more; customers with more add-ons "
             "churn less. Both patterns are associations, not proven causes.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "04_service_analysis.png")


def plot_addon_effect(df: pd.DataFrame) -> str:
    """Difference in churn rate between having and not having each add-on."""
    addon = churn_by_addon(df).dropna(subset=["difference_pts"])
    fig, ax = style.new_figure(9.4, 4.8)
    colours = [style.PALETTE["danger"] if d > 0 else style.PALETTE["teal"]
               for d in addon["difference_pts"]]
    bars = ax.barh(addon["addon"], addon["difference_pts"], color=colours, height=0.6)
    ax.axvline(0, color="#333333", lw=0.9)
    for bar, value in zip(bars, addon["difference_pts"]):
        offset = 0.6 if value > 0 else -0.6
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f} pts", va="center",
                ha="left" if value > 0 else "right", fontsize=8.5)
    ax.set_xlabel("Churn rate difference: has add-on minus does not have it (percentage points)")
    ax.set_title("Protective service add-ons", loc="left")
    style.title_with_caption(
        ax, "Protective service add-ons",
        "Support-type add-ons (security, tech support) are associated with roughly 16-27 points "
        "lower churn. Streaming add-ons barely move the needle. Association only — customers who "
        "buy support may already be more committed.",
    )
    return style.save(fig, "05_addon_effect.png")


def plot_charges_analysis(df: pd.DataFrame) -> str:
    """Monthly and total charges, split by outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    data = [df.loc[df["churn"] == "No", "monthly_charges"], df.loc[df["churn"] == "Yes", "monthly_charges"]]
    parts = axes[0].violinplot(data, showmedians=True, widths=0.75)
    for pc, colour in zip(parts["bodies"], [style.CHURN_COLORS["No"], style.CHURN_COLORS["Yes"]]):
        pc.set_facecolor(colour)
        pc.set_alpha(0.65)
    axes[0].set_xticks([1, 2], ["Retained", "Churned"])
    axes[0].set_ylabel("Monthly charges")
    axes[0].set_title("Monthly charges by outcome", loc="left")
    for i, series in enumerate(data, start=1):
        axes[0].text(i, series.max() + 3, f"median {series.median():.0f}",
                     ha="center", fontsize=8, color="#333333")

    bands = df.groupby("revenue_band", observed=True).agg(
        customers=("customer_id", "count"), churn_rate=("churn_label", "mean")
    ).reindex(list(config.REVENUE_BANDS.keys()))
    axes[1].bar(bands.index, bands["churn_rate"] * 100, color=style.PALETTE["primary"], width=0.6)
    for i, row in enumerate(bands.itertuples()):
        axes[1].text(i, row.churn_rate * 100 + 1.0, f"{row.churn_rate:.1%}\nn={row.customers:,}",
                     ha="center", fontsize=8)
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_xlabel("Monthly charge band")
    axes[1].set_title("Churn rate by monthly charge band", loc="left")
    axes[1].set_ylim(0, bands["churn_rate"].max() * 130)

    fig.suptitle("Charges and churn", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Churners pay more per month: higher bills correlate with higher churn, concentrated "
             "in the Mid-High and High bands.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "06_charges_analysis.png")


def plot_revenue_exposure(df: pd.DataFrame) -> str:
    """Where monthly revenue sits versus where it is exposed to churn."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    contract = churn_by_contract(df)
    x = np.arange(len(contract))
    width = 0.36
    axes[0].bar(x - width / 2, contract["monthly_revenue"], width,
                label="Monthly revenue", color=style.PALETTE["primary"])
    axes[0].bar(x + width / 2, contract["monthly_revenue_at_risk"], width,
                label="Attached to churned customers", color=style.PALETTE["accent"])
    axes[0].set_xticks(x, contract["contract"], rotation=12)
    axes[0].set_ylabel("Monthly revenue")
    axes[0].set_title("Revenue and revenue exposure by contract", loc="left")
    axes[0].legend()

    segments = customer_segments(df).sort_values("revenue_at_risk", ascending=False).head(7)
    axes[1].barh(segments["segment"], segments["revenue_at_risk"],
                 color=style.PALETTE["danger"], height=0.62)
    for i, row in enumerate(segments.itertuples()):
        axes[1].text(row.revenue_at_risk + segments["revenue_at_risk"].max() * 0.015, i,
                     f"{row.revenue_at_risk:,.0f}  ({row.churn_rate:.0%} churn)",
                     va="center", fontsize=7.5)
    axes[1].set_xlabel("Monthly revenue attached to churned customers")
    axes[1].set_title("Highest-exposure contract/value segments", loc="left")
    axes[1].set_xlim(0, segments["revenue_at_risk"].max() * 1.42)

    fig.suptitle("Revenue exposure", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Month-to-month customers alone account for the large majority of exposed recurring "
             "revenue — the value at stake sits exactly where churn is highest.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "07_revenue_exposure.png")


def plot_segment_heatmap(df: pd.DataFrame) -> str:
    """Contract x revenue band heat map of churn rate."""
    pivot = df.pivot_table(
        index="contract", columns="revenue_band", values="churn_label",
        aggfunc="mean", observed=True,
    )
    counts = df.pivot_table(
        index="contract", columns="revenue_band", values="customer_id",
        aggfunc="count", observed=True,
    )
    order = ["Month-to-month", "One year", "Two year"]
    pivot = pivot.reindex(order)
    counts = counts.reindex(order)
    bands = [b for b in config.REVENUE_BANDS if b in pivot.columns]
    pivot = pivot[bands]
    counts = counts[bands]

    fig, ax = style.new_figure(8.4, 4.6)
    im = ax.imshow(pivot.values * 100, cmap=style.SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Monthly charge band")
    ax.set_ylabel("Contract type")
    ax.grid(False)  # gridlines would cut through the heat-map cells
    ax.tick_params(length=0)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            n = counts.values[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.1%}\nn={int(n):,}", ha="center", va="center", fontsize=8.5,
                    color="white" if value > 0.35 else "#1f3b73")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Churn rate (%)")
    ax.set_title("Churn rate by contract and monthly charge band", loc="left", pad=12)
    fig.text(
        0.01, -0.02,
        "Top row, right-hand cells are the danger zone: uncommitted customers on high bills. "
        "Bottom rows are the safest revenue in the business.",
        ha="left", va="top", fontsize=8, color="#5c6370",
    )
    return style.save(fig, "08_segment_heatmap.png")


def plot_correlation(df: pd.DataFrame) -> str:
    """Correlation among numeric features and the target, with a clear caveat."""
    numeric = df[
        ["tenure", "monthly_charges", "total_charges", "avg_monthly_charge",
         "num_addon_services", "service_count", "churn_label"]
    ].copy()
    corr = numeric.corr()

    fig, ax = style.new_figure(6.8, 5.4)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    im = ax.imshow(np.ma.masked_where(mask, corr.values), cmap=style.DIVERGING, vmin=-1, vmax=1)
    labels = [c.replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(i + 1):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(corr.values[i, j]) > 0.55 else "#1f3b73")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson correlation")
    style.title_with_caption(
        ax, "Numeric correlation with churn",
        "Correlation measures linear co-movement only. It does not imply causation, and it "
        "understates non-linear effects that the tree models capture.",
    )
    return style.save(fig, "09_correlation.png")


def plot_risk_distribution(scored: pd.DataFrame) -> str:
    """Predicted-probability distribution and tier sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    axes[0].hist(scored["churn_probability"], bins=40,
                 color=style.PALETTE["primary"], edgecolor="white")
    axes[0].axvline(scored["decision_threshold"].iloc[0], ls="--", color=style.PALETTE["danger"],
                    lw=1.5, label=f"Decision cut-off {scored['decision_threshold'].iloc[0]:.2f}")
    axes[0].set_xlabel("Predicted churn probability")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Distribution of predicted churn risk", loc="left")
    axes[0].legend()

    tiers = scored["risk_category"].value_counts().reindex(list(config.RISK_TIER_ORDER))
    colours = [style.PALETTE["danger"], style.PALETTE["accent"], style.PALETTE["amber"],
               style.PALETTE["teal"]]
    axes[1].bar(tiers.index, tiers.values, color=colours, width=0.6)
    for i, value in enumerate(tiers.values):
        axes[1].text(i, value + max(tiers.values) * 0.02, f"{value:,}\n({value / len(scored):.0%})",
                     ha="center", fontsize=8)
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Customers per risk tier", loc="left")
    axes[1].set_ylim(0, max(tiers.values) * 1.2)

    fig.suptitle("Predicted risk distribution", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "A wide spread is what makes prioritisation possible: the model separates a large "
             "low-risk majority from a concentrated high-risk group.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "10_risk_distribution.png")


def plot_lift_curve(gains: pd.DataFrame, capacity_point: dict) -> str:
    """Cumulative gains versus the random-targeting diagonal, out of sample."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    axes[0].plot(gains["population_share"] * 100, gains["cumulative_recall"] * 100,
                 color=style.PALETTE["accent"], lw=2.4, label="Model ranking")
    axes[0].plot([0, 100], [0, 100], "--", color=style.PALETTE["grey"], lw=1.2, label="Random targeting")
    axes[0].scatter([capacity_point["capacity"] * 100],
                    [capacity_point["recall_of_churners"] * 100],
                    s=110, color=style.PALETTE["primary"], zorder=5,
                    label=f"Top {capacity_point['capacity']:.0%} → {capacity_point['recall_of_churners']:.0%} caught")
    axes[0].set_xlabel("Share of customers contacted (%)")
    axes[0].set_ylabel("Share of churners captured (%)")
    axes[0].set_title("Cumulative gains — hold-out test", loc="left")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim(0, 100)

    lift_data = gains.iloc[1:]
    axes[1].plot(lift_data["population_share"] * 100, lift_data["cumulative_lift"],
                 color=style.PALETTE["primary"], lw=2.2)
    axes[1].axhline(1.0, ls="--", color=style.PALETTE["grey"], lw=1.2, label="No better than random")
    axes[1].set_xlabel("Share of customers contacted (%)")
    axes[1].set_ylabel("Lift (× base churn rate)")
    axes[1].set_title("Cumulative lift", loc="left")
    axes[1].legend()

    fig.suptitle("Retention prioritisation performance (out of sample)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=style.PALETTE["primary"])
    fig.text(0.01, 0.005,
             "Computed on the untouched test split. Contacting the top decile captures about three "
             "times as many churners as contacting a random decile.",
             fontsize=8, color="#5c6370")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return style.save(fig, "11_lift_curve.png")


def run_full_eda(
    df: pd.DataFrame,
    scored: pd.DataFrame | None = None,
    gains: pd.DataFrame | None = None,
    capacity_point: dict | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Generate every EDA figure and return the supporting numbers."""
    output_dir = Path(output_dir) if output_dir is not None else config.FIGURES
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "churn_distribution": plot_churn_distribution(df),
        "tenure": plot_tenure_analysis(df),
        "contract_payment": plot_contract_and_payment(df),
        "services": plot_service_analysis(df),
        "addons": plot_addon_effect(df),
        "charges": plot_charges_analysis(df),
        "revenue_exposure": plot_revenue_exposure(df),
        "segment_heatmap": plot_segment_heatmap(df),
        "correlation": plot_correlation(df),
    }
    if scored is not None:
        figures["risk_distribution"] = plot_risk_distribution(scored)
    if gains is not None and capacity_point is not None:
        figures["lift"] = plot_lift_curve(gains, capacity_point)

    findings = {
        "figures": figures,
        "summary": revenue_summary(df),
        "churn_by_contract": churn_by_contract(df).to_dict(orient="records"),
        "churn_by_tenure_group": churn_by_tenure_group(df).to_dict(orient="records"),
        "churn_by_payment_method": churn_by_payment_method(df).to_dict(orient="records"),
        "churn_by_internet_service": churn_by_internet_service(df).to_dict(orient="records"),
        "addon_effects": churn_by_addon(df).to_dict(orient="records"),
        "segments": customer_segments(df).to_dict(orient="records"),
    }
    logger.info("Generated %s EDA figures", len(figures))
    return findings
