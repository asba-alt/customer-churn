"""End-to-end pipeline orchestration.

Stages are individually callable so a reviewer can run just the part they care
about, but ``run_all`` executes the full chain in dependency order.

The important methodological detail: retention lift is evaluated on the
**hold-out test split**, never on the rows the model was fitted on. Production
scoring covers all customers (because that is what a real deployment does), and
the tier cut-offs are learned from the validation split.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.analytics import eda
from src.analytics.churn_analysis import churn_by_contract
from src.analytics.customer_analysis import customer_segments, tenure_profile
from src.analytics.revenue_analysis import (
    customer_value_proxies,
    revenue_by_segment,
    revenue_concentration,
    revenue_exposure,
    revenue_summary,
)
from src.data.cleaning import clean_customers, load_clean, save_clean
from src.data.ingestion import (
    dataset_is_available,
    file_fingerprint,
    load_raw,
    missing_dataset_message,
)
from src.data.validation import build_data_quality_report
from src.database.db_setup import build_database
from src.database.queries import results_as_markdown, run_all_queries
from src.models.evaluate import (
    evaluate_on_test,
    plot_calibration,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall,
    plot_roc,
    write_model_evaluation_report,
)
from src.models.explainability import ChurnExplainer, save_explanations
from src.models.predict import (
    load_scores,
    risk_distribution,
    save_scores,
    score_customers,
    tier_thresholds,
)
from src.models.preprocessing import split_data
from src.models.train import (
    comparison_table,
    load_pipeline,
    refit_final,
    save_pipeline,
    select_best,
    train_candidates,
)
from src.retention.prioritization import (
    build_priority_list,
    cumulative_gains,
    lift_at_capacity,
    outreach_scenarios,
    prioritisation_summary,
    segment_priority_profile,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Values passed between stages so nothing is recomputed twice."""

    clean: pd.DataFrame | None = None
    splits: object | None = None
    pipeline: object | None = None
    threshold: float | None = None
    metadata: dict | None = None
    comparison: pd.DataFrame | None = None
    evaluation: dict | None = None
    scored: pd.DataFrame | None = None
    sql_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    explainability: dict | None = None
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Stage 1: data
# ----------------------------------------------------------------------
def stage_data(state: PipelineState) -> PipelineState:
    """Ingest, clean, validate, persist, build the SQLite database."""
    if not dataset_is_available():
        raise FileNotFoundError(missing_dataset_message())

    logger.info("=" * 72)
    logger.info("STAGE 1/6 — Data ingestion, cleaning and validation")
    logger.info("=" * 72)

    raw = load_raw()
    result = clean_customers(raw)
    clean = result.frame

    save_clean(clean)
    build_data_quality_report(raw, clean, result.log)
    build_database(clean)

    # Business-facing exports built straight from the cleaned data.
    exports = config.POWERBI_EXPORTS
    exports.mkdir(parents=True, exist_ok=True)
    clean.to_csv(exports / "customers.csv", index=False)
    churn_by_contract(clean).to_csv(exports / "churn_by_contract.csv", index=False)
    customer_segments(clean).to_csv(exports / "customer_segments.csv", index=False)
    revenue_exposure(clean).to_csv(exports / "revenue_exposure.csv", index=False)
    revenue_by_segment(clean).to_csv(exports / "revenue_by_segment.csv", index=False)
    customer_value_proxies(clean).to_csv(exports / "customer_value_proxies.csv", index=False)
    tenure_profile(clean).to_csv(exports / "tenure_profile.csv", index=False)

    kpis = revenue_summary(clean)
    kpis.update(revenue_concentration(clean))
    kpis["fingerprint"] = file_fingerprint()
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "business_kpis.json").write_text(
        json.dumps(kpis, indent=2, default=str), encoding="utf-8"
    )

    state.clean = clean
    state.extras["kpis"] = kpis
    state.extras["raw"] = raw
    state.extras["cleaning_log"] = result.log.as_dict()
    logger.info("Data stage complete: %s customers", len(clean))
    return state


# ----------------------------------------------------------------------
# Stage 2: SQL analytics
# ----------------------------------------------------------------------
def stage_sql(state: PipelineState) -> PipelineState:
    """Execute the SQL library against the SQLite database."""
    logger.info("=" * 72)
    logger.info("STAGE 2/6 — SQL analytics against SQLite")
    logger.info("=" * 72)
    with_results = run_all_queries()
    state.sql_results = with_results
    (config.REPORTS_DIR / "sql_results.md").write_text(
        "# SQL Analytics Results\n\n"
        "Every table below is produced by executing `sql/analytics_queries.sql` against "
        "`data/processed/churn_analytics.sqlite`.\n\n"
        + results_as_markdown(with_results),
        encoding="utf-8",
    )
    logger.info("SQL stage complete: %s query blocks", len(with_results))
    return state


# ----------------------------------------------------------------------
# Stage 3: modelling
# ----------------------------------------------------------------------
def stage_train(state: PipelineState) -> PipelineState:
    """Split, train candidates, select, refit, evaluate on the test set."""
    logger.info("=" * 72)
    logger.info("STAGE 3/6 — Model training and selection")
    logger.info("=" * 72)

    clean = state.clean if state.clean is not None else load_clean()
    splits = split_data(clean)
    results = train_candidates(splits)
    comparison = comparison_table(results)
    comparison.to_csv(config.MODEL_COMPARISON_FILE, index=False)

    best = select_best(results)
    pipeline, threshold = refit_final(best, splits)
    metadata = save_pipeline(pipeline, threshold, best, splits)
    metadata["tier_thresholds"] = tier_thresholds(
        pipeline.predict_proba(splits.X_val)[:, 1]
    )
    config.MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Selected %s (validation PR-AUC %.4f) at threshold %.3f",
                best.model_name, best.metrics["pr_auc"], threshold)

    state.splits = splits
    state.pipeline = pipeline
    state.threshold = threshold
    state.metadata = metadata
    state.comparison = comparison
    state.extras["best"] = best
    return state


def stage_evaluate(state: PipelineState) -> PipelineState:
    """Test-set evaluation, curves and the model evaluation report."""
    logger.info("=" * 72)
    logger.info("STAGE 4/6 — Test-set evaluation and explainability")
    logger.info("=" * 72)

    if state.pipeline is None:
        pipeline, threshold, metadata = load_pipeline()
        clean = state.clean if state.clean is not None else load_clean()
        state.splits = split_data(clean)
        state.pipeline, state.threshold, state.metadata = pipeline, threshold, metadata

    result = evaluate_on_test(state.pipeline, state.splits, state.threshold)
    figures = {
        "confusion": plot_confusion_matrix(result["confusion_tuned"], result["threshold"],
                                           "12_confusion_matrix.png"),
        "roc": plot_roc(result, "13_roc_curve.png"),
        "pr": plot_precision_recall(result, "14_precision_recall_curve.png"),
        "calibration": plot_calibration(result, "15_calibration_curve.png"),
    }
    if state.comparison is None:
        state.comparison = pd.read_csv(config.MODEL_COMPARISON_FILE)
    figures["comparison"] = plot_model_comparison(state.comparison, "16_model_comparison.png")

    write_model_evaluation_report(
        result, state.comparison, state.metadata, figures=figures
    )

    evaluation_payload = {
        "metrics_default_threshold": result["metrics_default"],
        "metrics_tuned_threshold": result["metrics_tuned"],
        "threshold": result["threshold"],
        "confusion_default": result["confusion_default"].tolist(),
        "confusion_tuned": result["confusion_tuned"].tolist(),
        "baseline_rate": result["baseline_rate"],
        "n_test": result["n_test"],
        "figures": figures,
    }
    (config.METRICS_DIR / "test_evaluation.json").write_text(
        json.dumps(evaluation_payload, indent=2), encoding="utf-8"
    )

    state.evaluation = result
    state.figures.update(figures)
    logger.info("Evaluation complete: PR-AUC %.4f, ROC-AUC %.4f (test)",
                result["metrics_default"]["pr_auc"], result["metrics_default"]["roc_auc"])
    return state


def stage_explain(state: PipelineState) -> PipelineState:
    """SHAP global and local explanations."""
    if state.pipeline is None:
        pipeline, threshold, metadata = load_pipeline()
        state.pipeline, state.threshold, state.metadata = pipeline, threshold, metadata
    if state.clean is None:
        state.clean = load_clean()
    if state.scored is None:
        state.scored = load_scores()

    explainer = ChurnExplainer(state.pipeline)
    manifest = save_explanations(explainer, state.clean, state.scored)
    state.explainability = manifest
    state.figures.update(manifest["figures"])
    return state


# ----------------------------------------------------------------------
# Stage 4: scoring and prioritisation
# ----------------------------------------------------------------------
def stage_score(state: PipelineState) -> PipelineState:
    """Score the full base and build the retention priority artefacts."""
    logger.info("=" * 72)
    logger.info("STAGE 5/6 — Scoring and retention prioritisation")
    logger.info("=" * 72)

    if state.pipeline is None:
        pipeline, threshold, metadata = load_pipeline()
        state.pipeline, state.threshold, state.metadata = pipeline, threshold, metadata
    if state.splits is None:
        if state.clean is None:
            state.clean = load_clean()
        state.splits = split_data(state.clean)

    # Tier cut-offs are learned from validation probabilities, which are
    # out-of-sample relative to the model that was just refit.
    reference_proba = state.pipeline.predict_proba(state.splits.X_val)[:, 1]
    scored_full = score_customers(
        state.clean, state.pipeline, state.threshold, reference_probabilities=reference_proba
    )

    # Out-of-sample evaluation set: predictions were produced by a model that
    # never saw these rows. This is the only place lift is legitimately measured.
    test_mask = state.clean[config.ID_COL].isin(
        state.clean.loc[state.splits.X_test.index, config.ID_COL]
    )
    test_rows = state.clean.loc[test_mask].copy()
    scored_test = score_customers(
        test_rows, state.pipeline, state.threshold, reference_probabilities=reference_proba
    )

    lift_table = lift_at_capacity(scored_test)
    primary_lift = lift_at_capacity(scored_test, capacities=(config.PRIMARY_CAPACITY,)).iloc[0].to_dict()
    gains = cumulative_gains(scored_test)
    summary = prioritisation_summary(scored_test)
    scenarios = outreach_scenarios(scored_test)
    priority = build_priority_list(scored_full)
    profile = segment_priority_profile(priority)

    exports = config.POWERBI_EXPORTS
    scored_full.to_csv(exports / "customer_risk_scores.csv", index=False)
    scored_test.to_csv(exports / "retention_test_evaluation.csv", index=False)
    priority.to_csv(exports / "retention_priority_list.csv", index=False)
    lift_table.to_csv(exports / "retention_lift.csv", index=False)
    gains.to_csv(exports / "retention_gains_curve.csv", index=False)
    scenarios.to_csv(exports / "retention_scenarios.csv", index=False)
    profile.to_csv(exports / "retention_segment_profile.csv", index=False)
    risk_distribution(scored_full).to_csv(exports / "risk_distribution.csv", index=False)

    # Preserve the base business KPIs. Running `score` standalone must not
    # overwrite business_kpis.json with only the retention subset.
    kpis = dict(state.extras.get("kpis", {}))
    if not kpis:
        kpis_path = config.METRICS_DIR / "business_kpis.json"
        if kpis_path.exists():
            kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
    if not kpis and state.clean is not None:
        kpis = revenue_summary(state.clean)
        kpis.update(revenue_concentration(state.clean))

    kpis.update(
        {
            "prioritised_customers": summary["customers_prioritised"],
            "prioritised_share_pct": summary["share_of_base_pct"],
            "priority_group_monthly_revenue": summary["monthly_revenue_in_group"],
            "priority_group_expected_monthly_at_risk": summary["expected_monthly_revenue_at_risk"],
            "lift_at_10pct_oos": summary["lift_at_capacity"],
            "oos_churn_rate_in_top_decile": summary["churn_rate_in_group"],
            "oos_overall_churn_rate": summary["overall_churn_rate"],
        }
    )
    state.extras["kpis"] = kpis
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "business_kpis.json").write_text(
        json.dumps(kpis, indent=2, default=str), encoding="utf-8"
    )
    (config.METRICS_DIR / "retention_summary.json").write_text(
        json.dumps(
            {
                "out_of_sample": summary,
                "lift_table": lift_table.to_dict(orient="records"),
                "scenarios": scenarios.to_dict(orient="records"),
                "tier_thresholds": state.metadata.get("tier_thresholds") if state.metadata else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    save_scores(scored_full)
    state.scored = scored_full
    state.extras.update(
        {
            "kpis": kpis,
            "lift_table": lift_table,
            "primary_lift": primary_lift,
            "gains": gains,
            "priority_summary": summary,
            "priority_list": priority,
            "scenarios": scenarios,
            "scored_test": scored_test,
        }
    )
    logger.info(
        "Out-of-sample lift@10%% = %.2fx (%.1f%% churn in group vs %.1f%% base)",
        summary["lift_at_capacity"], summary["churn_rate_in_group"] * 100,
        summary["overall_churn_rate"] * 100,
    )
    return state


# ----------------------------------------------------------------------
# Stage 5: reporting
# ----------------------------------------------------------------------
def stage_report(state: PipelineState) -> PipelineState:
    """EDA figures, Power BI inputs and the written business insights."""
    logger.info("=" * 72)
    logger.info("STAGE 6/6 — EDA, dashboard inputs and business reporting")
    logger.info("=" * 72)

    from src.reporting.business_report import write_business_insights

    if state.clean is None:
        state.clean = load_clean()
    if state.scored is None:
        state.scored = load_scores()

    # The report stage can run standalone, so reload anything a previous stage
    # would otherwise have left in memory.
    _hydrate_extras_from_disk(state)

    capacity_point = state.extras.get("primary_lift")
    gains = state.extras.get("gains")
    if capacity_point is None or gains is None:
        test_path = config.POWERBI_EXPORTS / "retention_test_evaluation.csv"
        if test_path.exists():
            scored_test = pd.read_csv(test_path)
            if gains is None:
                gains = cumulative_gains(scored_test)
            if capacity_point is None:
                capacity_point = lift_at_capacity(
                    scored_test, capacities=(config.PRIMARY_CAPACITY,)
                ).iloc[0].to_dict()

    findings = eda.run_full_eda(
        state.clean, state.scored, gains, capacity_point
    )
    state.figures.update(findings["figures"])
    (config.METRICS_DIR / "eda_findings.json").write_text(
        json.dumps(findings, indent=2, default=str), encoding="utf-8"
    )

    write_business_insights(state)
    _write_dashboard_data_dictionary(state)

    logger.info("Reporting stage complete")
    return state


def _hydrate_extras_from_disk(state: PipelineState) -> None:
    """Populate ``state.extras`` from artefacts when stages run individually.

    ``python main.py report`` is a valid standalone command, so the report writer
    cannot assume the earlier stages are still in memory.
    """
    kpis_path = config.METRICS_DIR / "business_kpis.json"
    if "kpis" not in state.extras and kpis_path.exists():
        state.extras["kpis"] = json.loads(kpis_path.read_text(encoding="utf-8"))
    elif "kpis" not in state.extras and state.clean is not None:
        merged = revenue_summary(state.clean)
        merged.update(revenue_concentration(state.clean))
        state.extras["kpis"] = merged

    test_path = config.POWERBI_EXPORTS / "retention_test_evaluation.csv"
    if state.extras.get("lift_table") is None and test_path.exists():
        scored_test = pd.read_csv(test_path)
        state.extras.setdefault("lift_table", lift_at_capacity(scored_test))
        state.extras.setdefault(
            "primary_lift",
            lift_at_capacity(scored_test, capacities=(config.PRIMARY_CAPACITY,)).iloc[0].to_dict(),
        )
        state.extras.setdefault("gains", cumulative_gains(scored_test))
        state.extras.setdefault("priority_summary", prioritisation_summary(scored_test))
        state.extras.setdefault("scenarios", outreach_scenarios(scored_test))
        state.extras["scored_test"] = scored_test

    priority_path = config.POWERBI_EXPORTS / "retention_priority_list.csv"
    if state.extras.get("priority_list") is None and priority_path.exists():
        state.extras["priority_list"] = pd.read_csv(priority_path)

    if state.evaluation is None:
        evaluation_path = config.METRICS_DIR / "test_evaluation.json"
        if evaluation_path.exists():
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            state.evaluation = {
                "metrics_default": payload["metrics_default_threshold"],
                "metrics_tuned": payload["metrics_tuned_threshold"],
                "threshold": payload["threshold"],
                "confusion_default": np.asarray(payload["confusion_default"]),
                "confusion_tuned": np.asarray(payload["confusion_tuned"]),
                "baseline_rate": payload["baseline_rate"],
                "n_test": payload["n_test"],
            }

    if state.metadata is None and config.MODEL_METADATA_FILE.exists():
        state.metadata = json.loads(config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))

    if state.explainability is None:
        manifest_path = config.DATA_EXPORTS / "explainability" / "explainability_manifest.json"
        if manifest_path.exists():
            state.explainability = json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_dashboard_data_dictionary(state: PipelineState) -> None:
    """Document every Power BI input file and its grain."""
    exports = config.POWERBI_EXPORTS
    rows = []
    for path in sorted(exports.glob("*.csv")):
        try:
            sample = pd.read_csv(path, nrows=5)
            total = sum(1 for _ in open(path, encoding="utf-8")) - 1
            rows.append(
                {
                    "file": path.name,
                    "rows": total,
                    "columns": len(sample.columns),
                    "column_list": ", ".join(sample.columns),
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not profile %s: %s", path, exc)

    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(exports / "data_dictionary.csv", index=False)
    state.extras["export_inventory"] = rows


# ----------------------------------------------------------------------
# Full run
# ----------------------------------------------------------------------
def run_all(include_explainability: bool = True) -> PipelineState:
    """Run every stage in order."""
    config.ensure_dirs()
    state = PipelineState()
    state = stage_data(state)
    state = stage_sql(state)
    state = stage_train(state)
    state = stage_evaluate(state)
    state = stage_score(state)
    if include_explainability:
        state = stage_explain(state)
    state = stage_report(state)
    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 72)
    return state
