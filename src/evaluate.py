"""Evaluation: the full metric suite, with the ordinal metrics that make this
problem different from a generic four-class classification.

Three principles, each a direct response to a v1 defect:

Every table leads with the majority-class baseline. An accuracy figure without
it is uninterpretable -- a constant predictor scores whatever the largest class
is worth, and v1 reported 74.2% without ever saying what beating nothing looked
like.

Macro and weighted averages are both reported and both labelled. v1 computed
AUC with `average='weighted'` and described it in the report as
"macro-averaged". They are different numbers and conflating them is how a model
acquires five different AUC values.

Per-class metrics always carry support. v1 reported Luxury precision/recall off
79 test rows without saying so, which made a number computed from a handful of
examples look like the others.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def majority_baseline(y_train: np.ndarray, y_test: np.ndarray, labels: list[str]) -> dict:
    """Score of always predicting the most frequent training class.

    This is the number every accuracy figure must be read against. Note it is
    computed from the TRAINING distribution and evaluated on test -- picking the
    majority class by looking at test labels would itself be a small leak.
    """
    counts = np.bincount(y_train, minlength=len(labels))
    majority = int(np.argmax(counts))
    predictions = np.full_like(y_test, majority)

    return {
        "strategy": "most_frequent",
        "predicted_class": labels[majority],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_test, predictions, weights="quadratic")
        ),
        "adjacent_accuracy": float(adjacent_accuracy(y_test, predictions)),
        "tier_mae": float(np.mean(np.abs(y_test - predictions))),
        "train_class_distribution": {
            label: int(counts[i]) for i, label in enumerate(labels)
        },
    }


# ---------------------------------------------------------------------------
# ordinal metrics -- the differentiator
# ---------------------------------------------------------------------------
def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of predictions within one tier of the truth.

    Budget mistaken for Mid-Range is a different kind of error from Budget
    mistaken for Luxury, and plain accuracy scores them identically. Requires
    the integer encoding to follow tier order, which `encode_target` guarantees.
    """
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def tier_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in tier units. 0.5 means the average prediction is
    half a tier away; 2.0 means it is routinely two tiers out."""
    return float(np.mean(np.abs(y_true - y_pred)))


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
        "linear_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="linear")
        ),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
        "tier_mae": tier_mae(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# per-class
# ---------------------------------------------------------------------------
def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> list[dict]:
    """Precision, recall, F1 and **support** for every class.

    Support is not optional. A recall of 0.203 computed from 79 examples and one
    computed from 5,600 are not comparable claims, and v1's report presented the
    former as though it were the latter.
    """
    indices = list(range(len(labels)))
    precision = precision_score(
        y_true, y_pred, labels=indices, average=None, zero_division=0
    )
    recall = recall_score(y_true, y_pred, labels=indices, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=indices, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=len(labels))

    return [
        {
            "class": label,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    ]


def error_profile(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """How far wrong the model is when it is wrong, and where.

    On an ordinal target the *size* of an error is as informative as its
    frequency: a model that is almost always one tier out is a different
    proposition from one that scatters. Accuracy cannot express that difference
    and neither can a confusion matrix at a glance.

    Recorded here rather than derived in the README generator so that every
    figure in the documentation traces back to this file -- scripts/
    verify_readme.py enforces exactly that.
    """
    n = len(y_true)
    distances = np.abs(y_true - y_pred)
    errors = int((distances > 0).sum())

    by_distance = []
    for d in range(1, len(labels)):
        count = int((distances == d).sum())
        by_distance.append(
            {
                "distance": d,
                "count": count,
                "share_of_errors": round(count / errors, 4) if errors else 0.0,
                "share_of_all": round(count / n, 4),
            }
        )

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cells = [
        {
            "true": labels[i],
            "predicted": labels[j],
            "count": int(matrix[i][j]),
            "share_of_errors": round(int(matrix[i][j]) / errors, 4) if errors else 0.0,
            "distance": abs(i - j),
        }
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j
    ]
    cells.sort(key=lambda c: c["count"], reverse=True)

    return {
        "n_evaluated": n,
        "n_correct": n - errors,
        "n_errors": errors,
        "error_rate": round(errors / n, 4),
        "by_distance": by_distance,
        "largest_confusions": cells[:5],
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Confusion matrix in raw counts and row-normalised form.

    Rows are true classes, columns predicted, both in tier order -- so the
    Luxury row reads left to right as Budget, Mid-Range, Premium, Luxury. v1's
    alphabetical encoding put Luxury second, which is why its published matrix
    was so hard to interpret.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.divide(
            matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0
        )
    return {
        "labels": labels,
        "counts": matrix.tolist(),
        "row_normalised": np.round(normalised, 4).tolist(),
    }


# ---------------------------------------------------------------------------
# probability-based
# ---------------------------------------------------------------------------
def probability_metrics(
    y_true: np.ndarray, y_proba: np.ndarray | None, n_classes: int
) -> dict:
    """AUC and average precision, one-vs-rest, in BOTH averagings.

    Reported separately and labelled explicitly. Returns nulls rather than
    zeros when probabilities are unavailable: v1 swallowed the failure in a bare
    except and wrote 0.0, which then appeared in a results table as though the
    model had genuinely scored zero.
    """
    if y_proba is None:
        return dict.fromkeys(
            [
                "auc_ovr_macro",
                "auc_ovr_weighted",
                "average_precision_macro",
                "average_precision_weighted",
            ]
        )

    binarised = label_binarize(y_true, classes=list(range(n_classes)))
    return {
        "auc_ovr_macro": float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        ),
        "auc_ovr_weighted": float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        ),
        "average_precision_macro": float(
            average_precision_score(binarised, y_proba, average="macro")
        ),
        "average_precision_weighted": float(
            average_precision_score(binarised, y_proba, average="weighted")
        ),
    }


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    cfg: dict,
) -> dict:
    """Percentile bootstrap confidence interval on the primary metric.

    v1 reported point estimates from a 1,214-row test set with no interval at
    all, which is how a Luxury recall computed from 79 examples came to sit in a
    table beside numbers computed from thousands.
    """
    rng = np.random.default_rng(cfg["seed"])
    n_iter = cfg["evaluation"]["bootstrap_iterations"]
    level = cfg["evaluation"]["bootstrap_ci"]
    scorer = _METRIC_FUNCTIONS[metric]

    n = len(y_true)
    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = scorer(y_true[idx], y_pred[idx])

    alpha = (1.0 - level) / 2.0
    return {
        "metric": metric,
        "point_estimate": float(scorer(y_true, y_pred)),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
        "level": level,
        "iterations": n_iter,
    }


_METRIC_FUNCTIONS = {
    "accuracy": lambda t, p: accuracy_score(t, p),
    "macro_f1": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
    "weighted_f1": lambda t, p: f1_score(t, p, average="weighted", zero_division=0),
    "quadratic_weighted_kappa": lambda t, p: cohen_kappa_score(t, p, weights="quadratic"),
    "adjacent_accuracy": adjacent_accuracy,
    "tier_mae": tier_mae,
}


def cross_validate_pipeline(
    pipe,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg: dict,
    weighted: bool = True,
    round_budget: int | None = None,
) -> dict:
    """Stratified k-fold CV with the ENTIRE pipeline inside each fold.

    v1 ran cross_val_score on features that had already been imputed and
    selected using every label in the dataset, so its CV score validated
    nothing -- each fold's "held-out" data had already influenced which features
    existed. Passing the unfitted pipeline means imputation, variance
    filtering, selection and scaling are all refitted per fold.
    """
    from sklearn.base import clone
    from sklearn.utils.class_weight import compute_sample_weight

    folds = cfg["evaluation"]["cv_folds"]
    metric = cfg["evaluation"]["primary_metric"]
    scoring = _CV_SCORING[metric]
    scorer = _METRIC_FUNCTIONS[metric]

    # Folds are iterated by hand rather than through cross_val_score, for two
    # reasons that both change the number.
    #
    # First, cross_val_score refits the clone with no eval_set, which XGBoost
    # rejects outright when early_stopping_rounds is set on the constructor. A
    # fold has no held-out slice to stop against, so CV necessarily measures the
    # model trained to full n_estimators.
    #
    # Second, and more subtly: fit_params passed to cross_val_score are handed
    # to every fold unsliced, so sample_weight cannot be routed correctly. The
    # earlier version simply omitted it, which meant CV silently measured an
    # UNWEIGHTED model while the reported test metric came from a weighted one
    # -- two different estimators presented side by side. Weights are now
    # recomputed per fold from that fold's training labels.
    template = clone(pipe)
    estimator = template.steps[-1][1]
    early_stopping_disabled = False
    params = estimator.get_params()
    for key in ("early_stopping_rounds", "estimator__early_stopping_rounds"):
        if params.get(key) is not None:
            estimator.set_params(**{key: None})
            early_stopping_disabled = True

    # With stopping disabled, a fold would run the full round budget. For the
    # tuned configuration that is 3,000 rounds at depth 8 -- both ruinously slow
    # and, worse, a different estimator from the ~450-round model whose test
    # score sits beside it. Cap each fold at where the real fit stopped.
    if early_stopping_disabled and round_budget:
        for key in ("n_estimators", "estimator__n_estimators"):
            if key in params:
                estimator.set_params(**{key: round_budget})

    # `weighted` must mirror what `src.pipeline.fit` actually does, which is
    # narrower than "balance everything": only XGBoost receives sample_weight,
    # because the other four carry class_weight='balanced' on the estimator
    # itself. Passing sample_weight to those as well applies the correction
    # twice -- an earlier version of this function did exactly that and drove
    # the decision tree's CV score 0.08 below its test score.
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X_train, y_train):
        fold_pipe = clone(template)
        X_fit, X_score = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_fit, y_score = y_train[train_idx], y_train[test_idx]

        if weighted:
            fold_pipe.fit(
                X_fit, y_fit,
                clf__sample_weight=compute_sample_weight("balanced", y_fit),
            )
        else:
            fold_pipe.fit(X_fit, y_fit)

        scores.append(float(scorer(y_score, fold_pipe.predict(X_score))))

    array = np.asarray(scores)
    return {
        "metric": metric,
        "scoring": scoring,
        "folds": folds,
        "scores": scores,
        "mean": float(array.mean()),
        "std": float(array.std()),
        "early_stopping_disabled_for_cv": early_stopping_disabled,
        "sample_weighted": bool(weighted),
        "round_budget_per_fold": round_budget,
    }


_CV_SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def evaluate_model(
    name: str,
    pipe,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    labels: list[str],
    cfg: dict,
    include_cv: bool = True,
    weighted: bool = True,
) -> dict[str, Any]:
    """Every metric for one fitted model.

    `weighted` must match how the model was actually fitted, so that the CV
    figure measures the same estimator as the test figure.
    """
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

    train_accuracy = float(accuracy_score(y_train, y_pred_train))
    test_accuracy = float(accuracy_score(y_test, y_pred_test))

    result: dict[str, Any] = {
        "model": name,
        "accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
        "overfitting_gap": train_accuracy - test_accuracy,
        "macro_precision": float(
            precision_score(y_test, y_pred_test, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred_test, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    result.update(probability_metrics(y_test, y_proba, len(labels)))
    result["ordinal"] = ordinal_metrics(y_test, y_pred_test)
    result["per_class"] = per_class_metrics(y_test, y_pred_test, labels)
    result["confusion"] = confusion(y_test, y_pred_test, labels)
    result["error_profile"] = error_profile(y_test, y_pred_test, labels)
    result["bootstrap"] = bootstrap_ci(
        y_test, y_pred_test, cfg["evaluation"]["primary_metric"], cfg
    )

    if include_cv:
        # Only models that receive sample_weight in src.pipeline.fit get it
        # here. Everything else is balanced via class_weight on the estimator.
        from src.pipeline import EARLY_STOPPING_MODELS

        stopped_at = getattr(pipe.steps[-1][1], "best_iteration", None)
        result["cross_validation"] = cross_validate_pipeline(
            pipe,
            X_train,
            y_train,
            cfg,
            weighted=weighted and name in EARLY_STOPPING_MODELS,
            round_budget=(int(stopped_at) + 1) if stopped_at is not None else None,
        )

    return result


def save_figures(
    artifacts: list[dict], results: list[dict], baseline: dict, labels: list[str], cfg: dict
) -> list[str]:
    """Write the figure set. Returns the paths written.

    Every comparison chart draws the majority baseline as an explicit reference
    line, for the same reason every table leads with it.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; CI has no display
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    out_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(fig, filename: str) -> None:
        path = out_dir / filename
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    n_classes = len(labels)

    # --- 1. confusion matrices, row-normalised ------------------------------
    cols = min(3, len(artifacts))
    rows = int(np.ceil(len(artifacts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    for ax, art in zip(axes.flat, artifacts, strict=False):
        matrix = confusion_matrix(art["y_true"], art["y_pred"], labels=range(n_classes))
        norm = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j, i, f"{norm[i, j]:.2f}\n({matrix[i, j]:,})",
                    ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
        ax.set_xticks(range(n_classes), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_classes), labels, fontsize=8)
        ax.set_title(art["name"], fontsize=10)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    for ax in axes.flat[len(artifacts):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="row-normalised")
    fig.suptitle("Confusion matrices (rows in tier order)", fontsize=12)
    _save(fig, "confusion_matrices.png")

    # --- 2. ROC, one-vs-rest macro ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for art in artifacts:
        if art["y_proba"] is None:
            continue
        y_bin = label_binarize(art["y_true"], classes=list(range(n_classes)))
        grid = np.linspace(0, 1, 500)
        mean_tpr = np.zeros_like(grid)
        for j in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, j], art["y_proba"][:, j])
            mean_tpr += np.interp(grid, fpr, tpr)
        mean_tpr /= n_classes
        ax.plot(grid, mean_tpr, lw=1.8, label=f"{art['name']} (AUC {auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, one-vs-rest macro-average")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "roc_curves.png")

    # --- 3. precision-recall -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for art in artifacts:
        if art["y_proba"] is None:
            continue
        y_bin = label_binarize(art["y_true"], classes=list(range(n_classes)))
        grid = np.linspace(0, 1, 500)
        mean_precision = np.zeros_like(grid)
        for j in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, j], art["y_proba"][:, j])
            mean_precision += np.interp(grid, rec[::-1], prec[::-1])
        mean_precision /= n_classes
        ap = average_precision_score(y_bin, art["y_proba"], average="macro")
        ax.plot(grid, mean_precision, lw=1.8, label=f"{art['name']} (AP {ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall, macro-average")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "precision_recall_curves.png")

    # --- 4. model comparison against the baseline ---------------------------
    primary = cfg["evaluation"]["primary_metric"]
    ordered = sorted(results, key=lambda r: r[primary], reverse=True)
    names = [r["model"] for r in ordered]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [
        ("macro F1", [r["macro_f1"] for r in ordered], baseline["macro_f1"]),
        ("accuracy", [r["accuracy"] for r in ordered], baseline["accuracy"]),
        (
            "quadratic weighted kappa",
            [r["ordinal"]["quadratic_weighted_kappa"] for r in ordered],
            baseline["quadratic_weighted_kappa"],
        ),
    ]
    for ax, (title, values, ref) in zip(axes, panels, strict=True):
        ax.barh(names[::-1], values[::-1], color="#4C72B0", alpha=0.85)
        ax.axvline(ref, color="crimson", ls="--", lw=1.5, label=f"baseline {ref:.3f}")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Model comparison, majority baseline marked", fontsize=12)
    _save(fig, "model_comparison.png")

    # --- 5. per-class recall, the Luxury story ------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(ordered), 1)
    positions = np.arange(n_classes)
    for i, r in enumerate(ordered):
        recalls = [c["recall"] for c in r["per_class"]]
        ax.bar(positions + i * width, recalls, width, label=r["model"], alpha=0.85)
    supports = [c["support"] for c in ordered[0]["per_class"]]
    ax.set_xticks(
        positions + width * (len(ordered) - 1) / 2,
        [f"{label}\n(n={n:,})" for label, n in zip(labels, supports, strict=True)],
    )
    ax.set_ylabel("recall")
    ax.set_title("Per-class recall with test support")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "per_class_recall.png")

    # --- 6. feature importance ----------------------------------------------
    with_importance = [a for a in artifacts if a.get("importances") is not None]
    if with_importance:
        art = max(
            with_importance,
            key=lambda a: next(r[primary] for r in results if r["model"] == a["name"]),
        )
        order = np.argsort(art["importances"])[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [art["feature_names"][i] for i in order][::-1],
            [art["importances"][i] for i in order][::-1],
            color="forestgreen", alpha=0.8,
        )
        ax.set_xlabel("importance")
        ax.set_title(f"Top 15 features -- {art['name']}")
        ax.grid(axis="x", alpha=0.3)
        _save(fig, "feature_importance.png")

    return written


def results_table(results: list[dict], baseline: dict, cfg: dict) -> str:
    """Console table with the baseline as row one, always."""
    primary = cfg["evaluation"]["primary_metric"]
    header = (
        f"{'model':<22}{'macro F1':>10}{'accuracy':>10}{'QWK':>8}"
        f"{'adj acc':>9}{'tier MAE':>10}{'AUC macro':>11}{'gap':>8}"
    )
    lines = [header, "-" * len(header)]

    lines.append(
        f"{'majority baseline':<22}{baseline['macro_f1']:>10.4f}"
        f"{baseline['accuracy']:>10.4f}{baseline['quadratic_weighted_kappa']:>8.4f}"
        f"{baseline['adjacent_accuracy']:>9.4f}{baseline['tier_mae']:>10.4f}"
        f"{'--':>11}{'--':>8}"
    )
    lines.append("-" * len(header))

    for r in sorted(results, key=lambda x: x[primary], reverse=True):
        auc = r.get("auc_ovr_macro")
        auc_text = f"{auc:>11.4f}" if auc is not None else f"{'--':>11}"
        lines.append(
            f"{r['model']:<22}{r['macro_f1']:>10.4f}{r['accuracy']:>10.4f}"
            f"{r['ordinal']['quadratic_weighted_kappa']:>8.4f}"
            f"{r['ordinal']['adjacent_accuracy']:>9.4f}"
            f"{r['ordinal']['tier_mae']:>10.4f}{auc_text}"
            f"{r['overfitting_gap']:>8.4f}"
        )

    lines.append("")
    lines.append(f"primary metric: {primary} (declared in conf/config.yaml)")
    lines.append(
        f"baseline predicts {baseline['predicted_class']!r} for every row"
    )
    return "\n".join(lines)
