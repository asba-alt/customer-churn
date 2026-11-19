"""Model explainability with SHAP.

Two layers of explanation are produced:

* **Global** — which features move churn risk across the whole customer base, and
  in which direction.
* **Local** — for one specific customer, which factors pushed their predicted
  risk up and which pulled it down, in probability space.

Everything is computed from the trained pipeline: the same preprocessor that the
model saw during training transforms the rows before SHAP runs. Explaining raw
untransformed rows would silently mismatch the model's feature space.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

import config
from src.analytics import style
from .preprocessing import coerce_feature_types, select_features

logger = logging.getLogger(__name__)


class ChurnExplainer:
    """Wraps a fitted pipeline so SHAP always sees the transformed feature space."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["clf"]
        self.feature_names = [self._pretty(n) for n in self.preprocessor.get_feature_names_out()]
        self._explainer = None
        # Gradient-boosted models explain in log-odds; scikit-learn forests
        # explain directly in probability space. Getting this wrong produces
        # silently meaningless "probability" conversions, so it is explicit.
        self.output_space = (
            "log_odds" if type(self.classifier).__name__.startswith("XGB") else "probability"
        )

    @staticmethod
    def _pretty(name: str) -> str:
        """Turn 'cat__contract_Month-to-month' into 'contract = Month-to-month'."""
        prefix, _, rest = name.partition("__")
        if prefix == "cat":
            # One-hot names are '<column>_<value>'; match against the known
            # column list so values that contain underscores stay intact.
            for col in sorted(config.CATEGORICAL_FEATURES, key=len, reverse=True):
                if rest.startswith(f"{col}_"):
                    return f"{col.replace('_', ' ')} = {rest[len(col) + 1:]}"
            return rest
        return rest.replace("_", " ")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = coerce_feature_types(select_features(df))
        matrix = self.preprocessor.transform(features)
        return pd.DataFrame(matrix, columns=self.feature_names, index=df.index)

    def explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """SHAP values in log-odds space for the churn class.

        Different SHAP explainers return different layouts for a binary target:
        ``(n, features)``, ``(n, features, 2)`` or a list of two arrays. All three
        are normalised to ``(n, features)`` for the positive (churn) class.
        """
        matrix = self.transform(df)
        values = self.explainer().shap_values(matrix)

        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            # (n, features, classes): take the last class (churn = 1).
            values = values[:, :, -1]
        return values

    def base_value(self) -> float:
        """Expected value in log-odds for the churn class."""
        base = self.explainer().expected_value
        if isinstance(base, (list, tuple, np.ndarray)):
            flat = np.ravel(np.asarray(base, dtype=float))
            base = flat[-1] if flat.size > 1 else flat[0]
        return float(base)


def global_importance(
    explainer: ChurnExplainer, df: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Mean absolute SHAP value per transformed feature."""
    values = explainer.shap_values(df)
    importance = np.abs(values).mean(axis=0)
    out = pd.DataFrame(
        {"feature": explainer.feature_names, "mean_abs_shap": importance}
    )
    out["mean_shap"] = values.mean(axis=0)
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["share_of_total_importance"] = (
        out["mean_abs_shap"] / out["mean_abs_shap"].sum() * 100
    ).round(2)
    out["mean_abs_shap"] = out["mean_abs_shap"].round(5)
    out["mean_shap"] = out["mean_shap"].round(5)
    return out.head(top_n)


def plot_global_importance(importance: pd.DataFrame, filename: str = "shap_global_importance.png") -> str:
    """Horizontal bar chart of the most influential features."""
    plot_data = importance.head(15).iloc[::-1]
    fig, ax = style.new_figure(9.2, 6.4)
    colours = [
        style.PALETTE["danger"] if v > 0 else style.PALETTE["teal"]
        for v in plot_data["mean_shap"]
    ]
    ax.barh(plot_data["feature"], plot_data["mean_abs_shap"], color=colours)
    ax.set_xlabel("Mean |SHAP value| (impact on log-odds of churn)")
    style.title_with_caption(
        ax, "Global feature importance from SHAP",
        "Red bars have a net positive average effect on churn risk, teal a net negative effect. "
        "Magnitude shows influence, not causality.",
    )
    return style.save(fig, filename)


def plot_shap_summary(explainer: ChurnExplainer, df: pd.DataFrame, filename: str = "shap_summary_beeswarm.png") -> str:
    """Beeswarm summary: direction and spread of each feature's effect."""
    matrix = explainer.transform(df)
    values = explainer.shap_values(df)
    style.apply_style()
    fig = plt.figure(figsize=(9.6, 6.6))
    shap.summary_plot(
        values,
        matrix,
        feature_names=explainer.feature_names,
        max_display=15,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (impact on log-odds of churn)")
    plt.title("SHAP summary — direction and spread of each effect", loc="left")
    return style.save(fig, filename)


def plot_shap_bar(explainer: ChurnExplainer, df: pd.DataFrame, filename: str = "shap_summary_bar.png") -> str:
    """Standard SHAP bar summary (mean absolute impact)."""
    matrix = explainer.transform(df)
    values = explainer.shap_values(df)
    style.apply_style()
    shap.summary_plot(
        values,
        matrix,
        feature_names=explainer.feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.set_size_inches(9.4, 6.2)
    fig.axes[0].set_xlabel("Mean |SHAP value| (average impact on log-odds of churn)")
    plt.title("SHAP mean absolute impact", loc="left")
    return style.save(fig, filename)


def dependence_plot(
    explainer: ChurnExplainer,
    df: pd.DataFrame,
    feature: str,
    filename: str = "shap_dependence_tenure.png",
) -> str:
    """SHAP dependence for one feature, showing the shape of the relationship."""
    matrix = explainer.transform(df)
    values = explainer.shap_values(df)
    if feature not in matrix.columns:
        candidates = [c for c in matrix.columns if feature in c]
        if not candidates:
            raise KeyError(f"Feature '{feature}' not found in transformed matrix")
        feature = candidates[0]

    style.apply_style()
    fig = plt.figure(figsize=(8.4, 5.4))
    shap.dependence_plot(
        feature,
        values,
        matrix,
        feature_names=explainer.feature_names,
        show=False,
        alpha=0.6,
    )
    fig = plt.gcf()
    plt.title(f"SHAP dependence — {feature}", loc="left")
    return style.save(fig, filename)


def explain_individual(
    explainer: ChurnExplainer,
    df: pd.DataFrame,
    customer_id: str,
    top_n: int = 10,
) -> dict:
    """Full local explanation for one customer.

    Contributions are reported in the space the explainer produces:
    ``log_odds`` for gradient-boosted models, ``probability`` for forests. The
    ``probability_delta_pts`` column is only populated when a valid conversion
    exists, so a probability-space model never gets a fabricated log-odds number.
    """
    row = df[df[config.ID_COL] == customer_id]
    if row.empty:
        raise KeyError(f"Customer {customer_id} not found")

    matrix = explainer.transform(row)
    values = explainer.shap_values(row)[0]
    base_value = explainer.base_value()
    final_probability = float(
        explainer.pipeline.predict_proba(coerce_feature_types(select_features(row)))[0, 1]
    )

    contributions = pd.DataFrame(
        {
            "feature": explainer.feature_names,
            "feature_value": matrix.iloc[0].values,
            "shap_value": values,
        }
    )
    contributions["impact_direction"] = np.where(
        contributions["shap_value"] > 0, "increases risk", "decreases risk"
    )
    contributions["shap_value"] = contributions["shap_value"].round(4)

    if explainer.output_space == "log_odds":
        base_probability = 1.0 / (1.0 + np.exp(-base_value))
        contributions["contribution_probability"] = (
            1.0 / (1.0 + np.exp(-(base_value + contributions["shap_value"])))
        ).round(4)
        contributions["probability_delta_pts"] = (
            (contributions["contribution_probability"] - base_probability) * 100
        ).round(2)
    else:
        base_probability = base_value
        contributions["contribution_probability"] = (
            base_value + contributions["shap_value"]
        ).clip(0, 1).round(4)
        contributions["probability_delta_pts"] = (
            (contributions["contribution_probability"] - base_probability) * 100
        ).round(2)

    ranked = contributions.reindex(
        contributions["shap_value"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return {
        "customer_id": customer_id,
        "output_space": explainer.output_space,
        "base_value": round(base_value, 4),
        "base_probability": round(float(base_probability), 4),
        "predicted_probability": round(final_probability, 4),
        "top_risk_increasing": ranked[ranked["shap_value"] > 0].head(top_n),
        "top_risk_decreasing": ranked[ranked["shap_value"] < 0].head(top_n),
        "all_contributions": ranked,
        "customer_context": {
            "contract": str(row["contract"].iloc[0]),
            "tenure": int(row["tenure"].iloc[0]),
            "monthly_charges": float(row["monthly_charges"].iloc[0]),
            "internet_service": str(row["internet_service"].iloc[0]),
            "payment_method": str(row["payment_method"].iloc[0]),
            "online_security": str(row["online_security"].iloc[0]),
            "tech_support": str(row["tech_support"].iloc[0]),
        },
    }


def plot_waterfall(
    explainer: ChurnExplainer,
    df: pd.DataFrame,
    customer_id: str,
    filename: str = "shap_waterfall_customer.png",
    max_display: int = 10,
) -> str:
    """SHAP waterfall for one customer: the classic 'why this customer?' chart."""
    row = df[df[config.ID_COL] == customer_id]
    if row.empty:
        raise KeyError(f"Customer {customer_id} not found")

    matrix = explainer.transform(row)
    explanation = shap.Explanation(
        values=explainer.shap_values(row)[0],
        base_values=explainer.base_value(),
        data=matrix.iloc[0].values,
        feature_names=explainer.feature_names,
    )
    style.apply_style()
    fig = plt.figure(figsize=(9.6, 6.0))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.title(f"Why is customer {customer_id} at risk?", loc="left")
    return style.save(fig, filename)


def save_explanations(
    explainer: ChurnExplainer,
    df: pd.DataFrame,
    scored: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """Generate every explainability artefact and return a manifest of paths.

    The sample passed to SHAP is the hold-out-relevant scoring population; a
    fixed seed keeps the sample stable across runs.
    """
    output_dir = Path(output_dir) if output_dir is not None else config.DATA_EXPORTS / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = min(1500, len(df))
    sample = df.sample(n=sample_size, random_state=config.SEED)

    importance = global_importance(explainer, sample, top_n=25)
    importance.to_csv(output_dir / "global_feature_importance.csv", index=False)

    figures = {
        "global_importance": plot_global_importance(importance),
        "summary_beeswarm": plot_shap_summary(explainer, sample),
        "summary_bar": plot_shap_bar(explainer, sample),
        "dependence_tenure": dependence_plot(explainer, sample, "tenure"),
    }
    if "monthly charges" in " ".join(explainer.feature_names):
        try:
            figures["dependence_monthly_charges"] = dependence_plot(
                explainer, sample, "monthly charges", "shap_dependence_monthly_charges.png"
            )
        except Exception as exc:  # pragma: no cover - optional chart
            logger.warning("Skipped monthly-charge dependence plot: %s", exc)

    # Highest-risk customer is the most useful individual example.
    top_customer = str(scored.iloc[0]["customer_id"])
    individual = explain_individual(explainer, df, top_customer)
    figures["waterfall"] = plot_waterfall(explainer, df, top_customer)

    individual_payload = {
        "customer_id": individual["customer_id"],
        "output_space": individual["output_space"],
        "base_probability": individual["base_probability"],
        "predicted_probability": individual["predicted_probability"],
        "customer_context": individual["customer_context"],
        "risk_increasing_factors": individual["top_risk_increasing"][
            ["feature", "feature_value", "shap_value", "probability_delta_pts"]
        ].to_dict(orient="records"),
        "risk_decreasing_factors": individual["top_risk_decreasing"][
            ["feature", "feature_value", "shap_value", "probability_delta_pts"]
        ].to_dict(orient="records"),
    }
    (output_dir / "individual_explanation.json").write_text(
        json.dumps(individual_payload, indent=2, default=str), encoding="utf-8"
    )

    # Explain a handful of customers so the report has more than one example.
    top_customers = scored.head(3)["customer_id"].astype(str).tolist()
    multi = {cid: explain_individual(explainer, df, cid) for cid in top_customers}
    multi_payload = {
        cid: {
            "predicted_probability": info["predicted_probability"],
            "context": info["customer_context"],
            "top_increasing": info["top_risk_increasing"][["feature", "probability_delta_pts"]].head(5).to_dict(orient="records"),
            "top_decreasing": info["top_risk_decreasing"][["feature", "probability_delta_pts"]].head(5).to_dict(orient="records"),
        }
        for cid, info in multi.items()
    }
    (output_dir / "customer_explanations_sample.json").write_text(
        json.dumps(multi_payload, indent=2, default=str), encoding="utf-8"
    )

    manifest = {
        "importance_csv": (output_dir / "global_feature_importance.csv").relative_to(config.ROOT).as_posix(),
        "figures": figures,
        "individual_json": (output_dir / "individual_explanation.json").relative_to(config.ROOT).as_posix(),
        "sample_json": (output_dir / "customer_explanations_sample.json").relative_to(config.ROOT).as_posix(),
        "explained_customer": top_customer,
        "sample_size": sample_size,
        "top_features": importance.head(10).to_dict(orient="records"),
    }
    (output_dir / "explainability_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote explainability artefacts to %s", output_dir)
    return manifest
