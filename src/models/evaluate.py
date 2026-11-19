"""Model evaluation on the untouched test set, with plots and a written report.

Every number in the generated report comes from ``sklearn`` applied to real
predictions produced in the same run. Nothing is hardcoded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import config
from src.analytics import style
from .preprocessing import DataSplits
from .train import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_on_test(pipeline: Pipeline, splits: DataSplits, threshold: float) -> dict:
    """Compute the complete test-set evaluation bundle."""
    proba = pipeline.predict_proba(splits.X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= threshold).astype(int)

    metrics_default = compute_metrics(splits.y_test, proba, threshold=0.5)
    metrics_tuned = compute_metrics(splits.y_test, proba, threshold=threshold)

    cm_default = confusion_matrix(splits.y_test, pred_default)
    cm_tuned = confusion_matrix(splits.y_test, pred_tuned)

    fpr, tpr, _ = roc_curve(splits.y_test, proba)
    precision, recall, pr_thresholds = precision_recall_curve(splits.y_test, proba)
    frac_pos, mean_pred = calibration_curve(splits.y_test, proba, n_bins=10, strategy="quantile")

    baseline_rate = float(np.mean(splits.y_test))

    return {
        "proba": proba,
        "pred_default": pred_default,
        "pred_tuned": pred_tuned,
        "threshold": threshold,
        "metrics_default": metrics_default,
        "metrics_tuned": metrics_tuned,
        "confusion_default": cm_default,
        "confusion_tuned": cm_tuned,
        "classification_report_default": classification_report(
            splits.y_test, pred_default, target_names=["Retained", "Churned"], digits=3, zero_division=0
        ),
        "classification_report_tuned": classification_report(
            splits.y_test, pred_tuned, target_names=["Retained", "Churned"], digits=3, zero_division=0
        ),
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision, recall, pr_thresholds),
        "calibration": (mean_pred, frac_pos),
        "roc_auc_curve": float(auc(fpr, tpr)),
        "baseline_rate": baseline_rate,
        "n_test": int(len(splits.y_test)),
    }


def plot_confusion_matrix(cm: np.ndarray, threshold: float, filename: str) -> str:
    """Annotated confusion matrix with absolute counts and row percentages."""
    fig, ax = style.new_figure(6.2, 5.0)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    im = ax.imshow(cm, cmap=style.SEQUENTIAL)
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > cm.max() / 2 else "#1f3b73"
            ax.text(
                j, i,
                f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)",
                ha="center", va="center", color=colour, fontsize=11, fontweight="bold",
            )
    ax.set_title(f"Confusion Matrix — hold-out test (threshold {threshold:.2f})", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Customers")
    style.title_with_caption(
        ax,
        f"Confusion Matrix — hold-out test (threshold {threshold:.2f})",
        "Rows are actual outcomes. Percentages are within-row, so the Recall for churners "
        "is the bottom-right cell as a share of the bottom row.",
    )
    return style.save(fig, filename)


def plot_roc(result: dict, filename: str = "roc_curve.png") -> str:
    """ROC curve versus the random-classifier diagonal."""
    fpr, tpr = result["roc_curve"]
    fig, ax = style.new_figure(6.4, 5.2)
    ax.plot(fpr, tpr, color=style.PALETTE["primary"], lw=2.2,
            label=f"Model (ROC-AUC = {result['metrics_default']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=style.PALETTE["grey"], lw=1.2, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.08, color=style.PALETTE["primary"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("ROC Curve — hold-out test", loc="left")
    style.title_with_caption(
        ax, "ROC Curve — hold-out test",
        "ROC-AUC summarises ranking across all thresholds. It is threshold-free and "
        "therefore not the metric used to choose the outreach cut-off.",
    )
    return style.save(fig, filename)


def plot_precision_recall(result: dict, filename: str = "precision_recall_curve.png") -> str:
    """Precision-recall curve with the no-skill baseline and the chosen operating point."""
    precision, recall, thresholds = result["pr_curve"]
    fig, ax = style.new_figure(6.4, 5.2)
    ax.plot(recall, precision, color=style.PALETTE["accent"], lw=2.2,
            label=f"Model (PR-AUC = {result['metrics_default']['pr_auc']:.3f})")
    ax.axhline(result["baseline_rate"], ls="--", color=style.PALETTE["grey"], lw=1.2,
               label=f"No-skill baseline ({result['baseline_rate']:.3f})")

    # Operating point implied by the tuned threshold.
    idx = int(np.argmin(np.abs(thresholds - result["threshold"])))
    ax.scatter(recall[idx], precision[idx], s=90, color=style.PALETTE["primary"], zorder=5,
               label=f"Chosen cut-off {result['threshold']:.2f}")
    ax.annotate(
        f"P={precision[idx]:.2f}, R={recall[idx]:.2f}",
        (recall[idx], precision[idx]),
        textcoords="offset points", xytext=(8, 8), fontsize=8, color=style.PALETTE["primary"],
    )

    ax.set_xlabel("Recall (share of churners caught)")
    ax.set_ylabel("Precision (share of flagged customers who churn)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    style.title_with_caption(
        ax, "Precision-Recall Curve — hold-out test",
        "For an imbalanced problem PR-AUC is more informative than ROC-AUC: it shows the "
        "true cost of pushing recall higher on a small positive class.",
    )
    return style.save(fig, filename)


def plot_calibration(result: dict, filename: str = "calibration_curve.png") -> str:
    """Reliability curve: do predicted probabilities mean what they say?"""
    mean_pred, frac_pos = result["calibration"]
    fig, ax = style.new_figure(6.2, 5.0)
    ax.plot([0, 1], [0, 1], "--", color=style.PALETTE["grey"], lw=1.2, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=style.PALETTE["teal"], lw=2, label="Model")
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    style.title_with_caption(
        ax, "Calibration — hold-out test",
        "Points close to the diagonal mean a predicted 40% risk genuinely corresponds to about "
        "40% observed churn, which is what makes probability-weighted revenue estimates defensible.",
    )
    return style.save(fig, filename)


def plot_model_comparison(comparison: pd.DataFrame, filename: str = "model_comparison.png") -> str:
    """Grouped bar chart of validation metrics across candidates."""
    metrics = ["val_pr_auc", "val_roc_auc", "val_f1_tuned", "val_recall_tuned", "val_precision_tuned"]
    labels = ["PR-AUC", "ROC-AUC", "F1 (tuned)", "Recall (tuned)", "Precision (tuned)"]

    fig, ax = style.new_figure(10.5, 5.4)
    x = np.arange(len(comparison))
    width = 0.15
    colours = [style.PALETTE["primary"], style.PALETTE["teal"], style.PALETTE["accent"],
               style.PALETTE["amber"], style.PALETTE["grey"]]
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + i * width, comparison[metric], width, label=label, color=colours[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(
        [f"{m}\n#{i+1}" for i, m in enumerate(comparison["model"])], fontsize=8
    )
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style.title_with_caption(
        ax, "Candidate model comparison — validation split",
        "Bars are validation scores at the tuned threshold unless marked 'tuned'. "
        "Selection uses PR-AUC, not accuracy.",
    )
    return style.save(fig, filename)


def _metric_block(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| Metric | {name} |\n|---|---|\n"
        f"| Accuracy | {metrics['accuracy']:.4f} |\n"
        f"| Precision | {metrics['precision']:.4f} |\n"
        f"| Recall | {metrics['recall']:.4f} |\n"
        f"| F1-score | {metrics['f1']:.4f} |\n"
        f"| F2-score | {metrics['f2']:.4f} |\n"
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n"
        f"| PR-AUC (average precision) | {metrics['pr_auc']:.4f} |\n"
    )


def _cm_block(cm: np.ndarray) -> str:
    tn, fp, fn, tp = cm.ravel()
    return (
        f"|  | Predicted Retained | Predicted Churned |\n|---|---|---|\n"
        f"| **Actual Retained** | {tn:,} (TN) | {fp:,} (FP) |\n"
        f"| **Actual Churned** | {fn:,} (FN) | {tp:,} (TP) |\n"
    )


def write_model_evaluation_report(
    result: dict,
    comparison: pd.DataFrame,
    metadata: dict,
    figures: dict[str, str],
    output_path: Path | None = None,
) -> str:
    """Write reports/model_evaluation.md from the executed results."""
    output_path = output_path or config.REPORT_MODEL_EVALUATION
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md = result["metrics_default"]
    mt = result["metrics_tuned"]
    cm_default = result["confusion_default"]
    cm_tuned = result["confusion_tuned"]
    tn_d, fp_d, fn_d, tp_d = cm_default.ravel()
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()

    comparison_md = comparison.copy()
    for col in comparison_md.columns:
        if comparison_md[col].dtype.kind == "f":
            comparison_md[col] = comparison_md[col].round(4)
    table_lines = [
        "| " + " | ".join(str(c) for c in comparison_md.columns) + " |",
        "|" + "|".join(["---"] * len(comparison_md.columns)) + "|",
    ]
    for _, row in comparison_md.iterrows():
        table_lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    comparison_table_md = "\n".join(table_lines)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    split = metadata["split"]

    markdown = f"""# Model Evaluation Report

**Generated:** {generated}
**Selected model:** `{metadata['model_name']}` with `{json.dumps(metadata['params'])}`
**Decision threshold:** {metadata['decision_threshold']:.3f} (chosen by maximising F{config.THRESHOLD_BETA:g} on the validation split)
**Selection metric:** PR-AUC (average precision)
**Seed:** {config.SEED}

All figures below are computed on the **hold-out test set**, which was never used
for fitting, hyper-parameter selection or threshold selection.

```
train      : {split['train']:,} rows   churn {split['train_churn_rate']:.2%}
validation : {split['validation']:,} rows   churn {split['val_churn_rate']:.2%}
test       : {split['test']:,} rows   churn {split['test_churn_rate']:.2%}
```

---

## 1. Why PR-AUC drives the decision

Churn is a {result['baseline_rate']:.1%} minority class. A model that predicts
"nobody churns" reaches {1 - result['baseline_rate']:.1%} accuracy and is
completely useless. Accuracy therefore cannot be the selection criterion.

The business decision is a **ranked outreach list with finite capacity**. What
matters is whether the customers at the top of the list churn more often than the
base rate — that is exactly what PR-AUC and lift measure. ROC-AUC is reported for
completeness but rewards performance on a negative class the business already
handles fine.

---

## 2. Candidate comparison (validation split)

{comparison_table_md}

The winner is the row with the highest `val_pr_auc`. Note that the accuracy column
is nearly flat across candidates: accuracy hides almost all of the real difference
between these models, which is the point being made above.

---

## 3. Final model — test-set metrics

### At the default 0.5 cut-off

{_metric_block('Score', md)}

### At the tuned cut-off ({result['threshold']:.3f})

{_metric_block('Score', mt)}

### Confusion matrix at the default 0.5 cut-off

{_cm_block(cm_default)}

### Confusion matrix at the tuned cut-off

{_cm_block(cm_tuned)}

Interpretation of the trade-off, in business terms:

- At 0.5 the model catches **{tp_d:,} of {tp_d + fn_d:,}** churners
  (**{md['recall']:.1%} recall**) while flagging **{fp_d:,}** customers who stay
  (**{md['precision']:.1%} precision**).
- At {result['threshold']:.3f} it catches **{tp_t:,} of {tp_t + fn_t:,}**
  (**{mt['recall']:.1%} recall**), flagging **{fp_t:,}** false alarms
  (**{mt['precision']:.1%} precision**).
- Moving to the tuned cut-off changes the missed-churner count from
  **{fn_d:,} to {fn_t:,}** and the false-alarm count from **{fp_d:,} to {fp_t:,}**.
  Every extra churner caught costs additional wasted contacts; the F{config.THRESHOLD_BETA:g}
  threshold deliberately accepts that trade because missing a churner who pays recurring
  monthly charges is usually more expensive than one unnecessary retention call.

**The honest caveat:** without campaign cost data and a measured save rate, no
dollar-optimal threshold can be computed. The F{config.THRESHOLD_BETA:g} cut-off is a
defensible operating point, not a proven optimum.

---

## 4. Classification report — default threshold

```
{result['classification_report_default']}
```

## 5. Classification report — tuned threshold

```
{result['classification_report_tuned']}
```

---

## 6. Curves

### ROC curve
![ROC curve](../{figures['roc']})

### Precision-recall curve
![Precision-recall curve](../{figures['pr']})

### Calibration
![Calibration curve](../{figures['calibration']})

### Confusion matrix
![Confusion matrix](../{figures['confusion']})

### Candidate comparison
![Model comparison](../{figures['comparison']})

---

## 7. Methodology and leakage controls

1. **Identifier excluded.** `customer_id` is never a feature.
2. **Label-derived columns excluded.** `churn`, `churn_label` and any column whose
   name contains churn/target/outcome/label are blocked by an explicit assertion.
3. **No target encoding on the full dataset.** The one-hot vocabulary and the
   scaler are fitted inside the pipeline, on training rows only.
4. **Stratified splits.** Churn rate is held constant across train/validation/test.
5. **Overlap check.** The pipeline asserts that no customer appears in both train
   and test.
6. **Imbalance handled inside training.** Logistic Regression uses
   `class_weight='balanced'`, Random Forest `class_weight='balanced_subsample'`,
   XGBoost `scale_pos_weight` computed from the training split only.
7. **Threshold selected on validation**, then applied unchanged to test.
8. **Final refit on train + validation**, with the test set held out for the single
   evaluation reported above.

---

## 8. Limitations

- The dataset is a **fictional, single-snapshot sample**; these metrics describe the
  sample, not a production population.
- No temporal validation is possible: there are no dates, so performance cannot be
  assessed for drift over time.
- No campaign cost, save-rate or contact-capacity economics exist in the data, so
  the threshold is a business judgement rather than an optimisation result.
- Perfect separation is impossible here: some churn is driven by factors absent from
  the data (competitor offers, service outages, relocation).
"""

    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote model evaluation report to %s", output_path)
    return markdown
