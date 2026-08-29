"""Ablation: isolate what each fix was actually worth.

Six cumulative configurations, from v1 as-built to the corrected pipeline. Each
row changes exactly one thing relative to the row above it, so the delta is
attributable.

    python scripts/ablation.py                 # full data
    python scripts/ablation.py --sample 40000  # faster, clearly labelled

Two honest caveats, both recorded in the output:

The first row is a RECONSTRUCTION, not a reproduction. v1 loaded a 305,145-row
file whose price column was ~98% unusable; that file no longer exists and no
parse of the correct CSV recreates it (see results/v1_load_diagnosis.json). The
row is a stratified sample at v1's exact reported class proportions, which
reproduces its *shape* but cannot reproduce its *rows*.

Accuracy is expected to fall when leakage is removed. That is the correct
outcome and nothing here is tuned to recover it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluate import adjacent_accuracy, ordinal_metrics, per_class_metrics  # noqa: E402
from src.pipeline import (  # noqa: E402
    build_estimator,
    build_pipeline,
    encode_target,
    fit,
    stratified_split,
)
from src.train import load_gold  # noqa: E402


@dataclass(frozen=True)
class Configuration:
    """One row of the ablation table."""

    name: str
    rows: str          # "v1" or "full"
    leaky: bool        # fit preprocessing on all data, using all labels
    scale_all: bool    # scale every model, or only LogisticRegression as v1 did
    balanced: bool     # balanced sample weights on XGBoost
    early_stopping: bool
    changed: str       # what this row changes relative to the one above
    # Swap in a different estimator for the boosted slot. Everything above is a
    # correctness fix; this is the one row that is a performance change, and it
    # is kept visibly separate for that reason.
    xgb_variant: str = "xgboost"


CONFIGURATIONS = [
    Configuration(
        "v1 as-built", "v1", True, False, False, False,
        "baseline: 6,067-row reconstruction, leaky, unscaled, unweighted",
    ),
    Configuration(
        "+ correct data load", "full", True, False, False, False,
        "426,880 rows instead of a 6,067-row subset",
    ),
    Configuration(
        "+ leakage removed", "full", False, False, False, False,
        "impute/variance/select fitted inside the pipeline, train only",
    ),
    Configuration(
        "+ scaling for all models", "full", False, True, False, False,
        "RobustScaler applied to every model, not just LogisticRegression",
    ),
    Configuration(
        "+ balanced sample weights", "full", False, True, True, False,
        "compute_sample_weight('balanced') on the XGBoost fit",
    ),
    Configuration(
        "+ early stopping on val", "full", False, True, True, True,
        "validation set drives early stopping instead of being discarded",
    ),
    # Everything above restores correctness. The two below are improvements,
    # separated so that "what the fixes were worth" cannot be confused with
    # "what tuning was worth".
    Configuration(
        "+ tuned hyperparameters", "full", False, True, True, True,
        "depth 8 and a real round budget, from scripts/tune.py; v1's depth-4 "
        "settings were chosen for a 6,067-row experiment",
        xgb_variant="xgboost_tuned",
    ),
    Configuration(
        "+ ordinal decomposition", "full", False, True, True, True,
        "tier ordering moved into the objective via cumulative binary models, "
        "at the tuned hyperparameters -- rows are cumulative, so pointing this "
        "at the untuned variant would charge the decomposition for losing the "
        "tuning as well",
        xgb_variant="xgboost_ordinal_tuned",
    ),
    # NOT a cumulative step. This removes balanced weighting from the tuned
    # configuration, to answer a question the cumulative table cannot: is the
    # weighting still earning its place once the model has enough capacity?
    #
    # At v1's max_depth=4 it plainly did -- unweighted Luxury recall was 0.41.
    # At depth 8 the unweighted model already reaches 0.66 recall at much higher
    # precision, so the same precision-for-recall trade now costs more than it
    # returns on the declared primary metric. Published because the brief lists
    # balanced weighting as a defect fix, and at final hyperparameters that
    # framing does not survive measurement.
    Configuration(
        "(tuned, weighting removed)", "full", False, True, False, True,
        "comparison row, not a cumulative step: balanced weighting removed from "
        "the tuned configuration",
        xgb_variant="xgboost_tuned",
    ),
]

# The models the table tracks. XGBoost is the one v1 selected and reported;
# SVM is included because the scaling row is invisible on a tree model and
# dramatic on a kernel one.
TRACKED = ["xgboost", "svm"]


def reconstruct_v1_sample(
    X: pd.DataFrame, y: pd.Series, cfg: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified sample at v1's reported class counts.

    Not a reproduction. See the module docstring.
    """
    wanted = cfg["v1_reconstruction"]["class_counts"]
    seed = cfg["seed"]
    parts = []
    for label, n in wanted.items():
        pool = y[y == label]
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} rows of {label!r}, need {n}")
        parts.append(pool.sample(n=n, random_state=seed).index)
    idx = np.concatenate(parts)
    return X.loc[idx], y.loc[idx]


def disable_early_stopping(estimator) -> None:
    """Turn off early stopping wherever the parameter lives.

    On a bare XGBClassifier it is a top-level parameter; on the ordinal wrapper
    it belongs to the nested base estimator.
    """
    params = estimator.get_params()
    for key in ("early_stopping_rounds", "estimator__early_stopping_rounds"):
        if key in params:
            estimator.set_params(**{key: None})


def set_class_weight(estimator, value: str | None) -> None:
    """Turn estimator-level class balancing on or off, wherever it lives.

    Without this every ablation row carried class_weight='balanced' from
    conf/config.yaml, including the row that is meant to represent v1 as-built.
    That made the balancing row a no-op for every model except XGBoost and
    quietly overstated the first row.
    """
    params = estimator.get_params()
    for key in ("class_weight", "estimator__class_weight"):
        if key in params:
            estimator.set_params(**{key: value})


def leaky_transform(
    X: pd.DataFrame, y: np.ndarray, cfg: dict
) -> pd.DataFrame:
    """v1's ordering: fit every transform on the full dataset, before splitting.

    SelectKBest sees every label, which is the strongest of the three leaks.
    """
    p = cfg["pipeline"]
    imputed = SimpleImputer(strategy=p["imputer_strategy"]).fit_transform(X)
    variance = VarianceThreshold(threshold=p["variance_threshold"])
    reduced = variance.fit_transform(imputed)
    selector = SelectKBest(score_func=f_classif, k=p["select_k_best"])
    selected = selector.fit_transform(reduced, y)
    names = np.asarray(list(X.columns))[variance.get_support()][selector.get_support()]
    return pd.DataFrame(selected, columns=names, index=X.index)


def evaluate_configuration(
    conf: Configuration, X_all: pd.DataFrame, y_all: pd.Series, cfg: dict
) -> dict:
    """Fit and score the tracked models under one configuration."""
    if conf.rows == "v1":
        X, y_labels = reconstruct_v1_sample(X_all, y_all, cfg)
    else:
        X, y_labels = X_all, y_all

    y, labels = encode_target(y_labels, cfg)

    # When leaky, transform first and split second -- exactly v1's ordering.
    X_used = leaky_transform(X, y, cfg) if conf.leaky else X

    splits = stratified_split(X_used, y, cfg)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    row: dict = {
        "configuration": conf.name,
        "changed": conf.changed,
        "rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "flags": {k: v for k, v in asdict(conf).items() if isinstance(v, bool)},
        "models": {},
    }

    for tracked_name in TRACKED:
        # The boosted slot can hold a variant; the SVM slot never does.
        name = conf.xgb_variant if tracked_name == "xgboost" else tracked_name
        if conf.leaky:
            # The preprocessing already happened outside; give the estimator the
            # bare features, scaling only where v1 scaled.
            estimator = build_estimator(
                name, cfg, n_classes=len(labels), for_v1=(conf.rows == "v1")
            )
            set_class_weight(estimator, "balanced" if conf.balanced else None)
            if conf.scale_all or name == "logistic_regression":
                scaler = RobustScaler().fit(X_train)
                Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
            else:
                Xtr, Xte = X_train.to_numpy(), X_test.to_numpy()

            weight = (
                compute_sample_weight("balanced", y_train)
                if conf.balanced and name.startswith("xgboost")
                else None
            )
            if name.startswith("xgboost"):
                disable_early_stopping(estimator)
                estimator.fit(Xtr, y_train, sample_weight=weight)
            else:
                estimator.fit(Xtr, y_train)
            y_pred = estimator.predict(Xte)
        else:
            pipe = build_pipeline(
                name, cfg, n_classes=len(labels), for_v1=(conf.rows == "v1")
            )
            set_class_weight(pipe.named_steps["clf"], "balanced" if conf.balanced else None)
            if not conf.scale_all and name != "logistic_regression":
                pipe.set_params(scale="passthrough")
            if not conf.early_stopping and name.startswith("xgboost"):
                disable_early_stopping(pipe.named_steps["clf"])
            fitted = fit(
                pipe,
                name,
                X_train,
                y_train,
                X_val if conf.early_stopping else None,
                y_val if conf.early_stopping else None,
                balanced=conf.balanced,
            )
            y_pred = fitted.predict(X_test)

        from sklearn.metrics import accuracy_score, f1_score

        per_class = per_class_metrics(y_test, y_pred, labels)
        luxury = next(c for c in per_class if c["class"] == "Luxury")
        # Keyed by the tracked slot, not the variant, so the table can be read
        # down a column. `variant` records which estimator actually ran.
        row["models"][tracked_name] = {
            "variant": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "luxury_recall": luxury["recall"],
            "luxury_precision": luxury["precision"],
            "luxury_support": luxury["support"],
            "adjacent_accuracy": adjacent_accuracy(y_test, y_pred),
            **ordinal_metrics(y_test, y_pred),
        }
        print(
            f"      {name:<16} macro F1 {row['models'][tracked_name]['macro_f1']:.4f}"
            f"  acc {row['models'][tracked_name]['accuracy']:.4f}"
            f"  luxury recall {luxury['recall']:.4f}"
        )

    return row


def baseline_row(y_all: pd.Series, cfg: dict) -> dict:
    """Majority-class baseline on the full data, for the 'vs baseline' column."""
    y, labels = encode_target(y_all, cfg)
    counts = np.bincount(y, minlength=len(labels))
    majority = int(np.argmax(counts))
    preds = np.full_like(y, majority)
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "predicted_class": labels[majority],
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
    }


def render_table(rows: list[dict], baseline: dict, model: str) -> str:
    header = (
        f"{'configuration':<28}{'rows':>10}{'macro F1':>10}{'accuracy':>10}"
        f"{'vs base':>9}{'lux recall':>12}{'lux n':>8}{'QWK':>8}"
    )
    lines = [header, "-" * len(header)]
    lines.append(
        f"{'majority baseline':<28}{'--':>10}{baseline['macro_f1']:>10.4f}"
        f"{baseline['accuracy']:>10.4f}{'--':>9}{'--':>12}{'--':>8}{0.0:>8.4f}"
    )
    lines.append("-" * len(header))
    for r in rows:
        m = r["models"][model]
        delta = m["macro_f1"] - baseline["macro_f1"]
        lines.append(
            f"{r['configuration']:<28}{r['rows']:>10,}{m['macro_f1']:>10.4f}"
            f"{m['accuracy']:>10.4f}{delta:>+9.4f}{m['luxury_recall']:>12.4f}"
            f"{m['luxury_support']:>8,}{m['quadratic_weighted_kappa']:>8.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="conf/config.yaml")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--out", default="results/ablation.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])  # noqa: NPY002

    X, y = load_gold(cfg)
    if args.sample and args.sample < len(X):
        from src.train import subsample

        X, y = subsample(X, y, args.sample, cfg["seed"])
        print(f"SUBSAMPLE: {len(X):,} rows -- indicative only\n")

    baseline = baseline_row(y, cfg)

    rows = []
    for conf in CONFIGURATIONS:
        print(f"  [{conf.name}]")
        rows.append(evaluate_configuration(conf, X, y, cfg))

    payload = {
        "note": (
            "Row 1 is a reconstruction at v1's reported class proportions, not a "
            "reproduction: v1's source file no longer exists and no parse of the "
            "correct CSV recreates it. See results/v1_load_diagnosis.json."
        ),
        "subsampled_to": args.sample,
        "baseline": baseline,
        "tracked_models": TRACKED,
        "configurations": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for model in TRACKED:
        print(f"\n{'=' * 95}\nABLATION -- {model}\n{'=' * 95}")
        print(render_table(rows, baseline, model))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
