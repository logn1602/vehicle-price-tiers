"""Training orchestration with MLflow tracking.

Loads the gold feature store, splits 60/20/20, fits every model inside its own
leakage-free Pipeline, evaluates, and writes `results/metrics.json`.

`metrics.json` is the single source of truth. Nothing downstream -- README,
model card, ablation table -- may state a number that is not read from it.
Volatile fields (timestamps, wall-clock durations) are confined to a single
`run_metadata` block so the rest of the file can be compared byte-for-byte
between runs, which is what the reproducibility check asserts.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_hash, file_hash
from src.evaluate import evaluate_model, majority_baseline, results_table, save_figures
from src.pipeline import (
    MODEL_NAMES,
    build_pipeline,
    encode_target,
    fit,
    selected_features,
    stratified_split,
)

# MLflow 3 serialises sklearn models with skops, which refuses to persist types
# it does not recognise. These are the ones our pipeline legitimately contains:
# the scoring callable held by SelectKBest, and numpy dtype objects carried by
# the fitted transformers. Listing them explicitly is the sanctioned escape
# hatch and keeps the trust decision visible rather than blanket-disabled.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.feature_selection._univariate_selection.f_classif",
]


def load_gold(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Read the model-ready feature store produced by the gold stage."""
    path = Path(cfg["paths"]["gold"]) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/run_all.py` first."
        )
    df = pd.read_parquet(path)
    target = cfg["target"]["name"]
    feature_cols = [c for c in df.columns if c not in {target, "id"}]
    return df[feature_cols], df[target]


def subsample(
    X: pd.DataFrame, y: pd.Series, n: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample, for validating the code path before a full run."""
    if n >= len(X):
        return X, y
    frac = n / len(X)
    idx = (
        y.groupby(y, observed=True)
        .apply(lambda s: s.sample(frac=frac, random_state=seed))
        .index.get_level_values(-1)
    )
    return X.loc[idx], y.loc[idx]


def _mlflow_run(cfg: dict, name: str, params: dict, metrics: dict, pipe) -> str | None:
    """Log one model to MLflow. Returns the run id, or None if unavailable.

    Import is local so that a broken or missing MLflow install degrades to
    "tracking skipped" rather than taking the whole training run down with it.
    """
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.exceptions import MlflowException
    except ImportError:
        print("    mlflow not installed -- tracking skipped")
        return None

    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    except MlflowException as exc:
        # Tracking is observability, not the deliverable. A broken backend must
        # not destroy an hour of training -- but it must be loud, not silent.
        print(f"\n    WARNING: MLflow unavailable, continuing untracked: {exc}")
        return None

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        flat = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        mlflow.log_metrics(flat)
        for entry in metrics["per_class"]:
            cls = entry["class"].lower().replace("-", "_")
            mlflow.log_metric(f"{cls}_precision", entry["precision"])
            mlflow.log_metric(f"{cls}_recall", entry["recall"])
            mlflow.log_metric(f"{cls}_f1", entry["f1"])
            mlflow.log_metric(f"{cls}_support", entry["support"])
        for key, value in metrics["ordinal"].items():
            mlflow.log_metric(key, value)
        try:
            mlflow.sklearn.log_model(
                pipe, name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES
            )
        except TypeError:
            # MLflow 2.x used artifact_path and had no skops backend.
            mlflow.sklearn.log_model(pipe, artifact_path="model")
        except MlflowException as exc:
            print(f"\n    WARNING: model artifact not logged: {exc}")
        return run.info.run_id


def train_all(
    cfg: dict,
    models: list[str] | None = None,
    sample_size: int | None = None,
    include_cv: bool = True,
    track: bool = True,
) -> dict[str, Any]:
    """Fit and evaluate every model. Returns the full metrics payload."""
    started = time.perf_counter()
    models = models or MODEL_NAMES

    X, y_labels = load_gold(cfg)
    if sample_size:
        X, y_labels = subsample(X, y_labels, sample_size, cfg["seed"])

    y, labels = encode_target(y_labels, cfg)
    splits = stratified_split(X, y, cfg)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    baseline = majority_baseline(y_train, y_test, labels)

    print(f"\n  {len(X):,} rows x {X.shape[1]} features")
    print(f"  train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,}")
    print(
        f"  majority baseline: {baseline['predicted_class']!r} -> "
        f"accuracy {baseline['accuracy']:.4f}, macro F1 {baseline['macro_f1']:.4f}\n"
    )

    results: list[dict] = []
    artifacts: list[dict] = []
    fitted_models: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for name in models:
        print(f"  training {name} ...", end="", flush=True)
        t0 = time.perf_counter()

        pipe = build_pipeline(name, cfg, n_classes=len(labels))
        fitted = fit(pipe, name, X_train, y_train, X_val, y_val)
        elapsed = time.perf_counter() - t0

        metrics = evaluate_model(
            name, fitted, X_train, y_train, X_test, y_test, labels, cfg,
            include_cv=include_cv,
        )
        metrics["selected_features"] = selected_features(fitted, list(X.columns))
        metrics["n_selected_features"] = len(metrics["selected_features"])

        if name == "xgboost":
            booster = fitted.named_steps["clf"]
            metrics["best_iteration"] = int(booster.best_iteration)
            metrics["early_stopped"] = bool(
                booster.best_iteration < cfg["model"]["xgboost"]["n_estimators"] - 1
            )

        if track:
            metrics["mlflow_run_id"] = _mlflow_run(
                cfg, name, _params_for(name, cfg), metrics, fitted
            )

        results.append(metrics)
        fitted_models[name] = fitted
        # Timings are volatile, so they live in run_metadata rather than beside
        # the metrics; otherwise two identical runs would never compare equal.
        timings[name] = round(elapsed, 2)

        # Kept in memory for plotting only; probabilities are far too large to
        # serialise into metrics.json.
        estimator = fitted.named_steps["clf"]
        artifacts.append(
            {
                "name": name,
                "y_true": y_test,
                "y_pred": fitted.predict(X_test),
                "y_proba": (
                    fitted.predict_proba(X_test)
                    if hasattr(fitted, "predict_proba")
                    else None
                ),
                "importances": getattr(estimator, "feature_importances_", None),
                "feature_names": metrics["selected_features"],
            }
        )

        print(
            f" macro F1 {metrics['macro_f1']:.4f}  acc {metrics['accuracy']:.4f}"
            f"  ({elapsed:.1f}s)"
        )

    primary = cfg["evaluation"]["primary_metric"]
    tie_breaker = cfg["evaluation"]["tie_breaker"]
    best = max(
        results,
        key=lambda r: (r[primary], r["ordinal"][tie_breaker]),
    )

    figures = save_figures(artifacts, results, baseline, labels, cfg)
    print(f"\n  wrote {len(figures)} figures to {cfg['paths']['figures']}")

    model_path = save_model(
        fitted_models[best["model"]], best["model"], labels, list(X.columns), cfg
    )
    print(f"  saved {best['model']} to {model_path}")

    return {
        "run_metadata": {
            # Everything volatile lives here so the rest of the file is stable
            # and two runs can be compared byte-for-byte.
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - started, 1),
            "fit_seconds": timings,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "figures": [Path(f).name for f in figures],
        },
        "provenance": {
            "config_hash": config_hash(dict(cfg)),
            "data_hash": file_hash(Path(cfg["paths"]["gold"]) / "features.parquet"),
            "seed": cfg["seed"],
            "n_rows": int(len(X)),
            "n_features_in": int(X.shape[1]),
            "subsampled_to": sample_size,
        },
        "selection": {
            "primary_metric": primary,
            "tie_breaker": tie_breaker,
            "best_model": best["model"],
            # v1 ranked models by 0.4*acc + 0.3*F1 + 0.3*AUC, arbitrary weights
            # with no justification. Explicitly recorded as not used.
            "composite_score_used": cfg["evaluation"]["composite_score"],
        },
        "class_labels": labels,
        "split": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "baseline": baseline,
        "models": results,
    }


def _params_for(name: str, cfg: dict) -> dict:
    """Hyperparameters for one model, flattened for MLflow."""
    key = {"svm": "svm"}.get(name, name)
    params = dict(cfg["model"].get(key, {}))
    params.update(
        {
            "seed": cfg["seed"],
            "select_k_best": cfg["pipeline"]["select_k_best"],
            "variance_threshold": cfg["pipeline"]["variance_threshold"],
            "imputer_strategy": cfg["pipeline"]["imputer_strategy"],
            "scaler": cfg["pipeline"]["scaler"],
        }
    )
    return {k: str(v) for k, v in params.items()}


def save_model(
    pipe, name: str, labels: list[str], feature_names: list[str], cfg: dict
) -> Path:
    """Persist the selected pipeline plus everything the API needs to use it.

    The sidecar metadata matters as much as the weights: without the exact
    feature order and class labels, a served model will happily accept
    misaligned columns and return confident nonsense.
    """
    import joblib

    out_dir = Path(cfg["paths"]["artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    (out_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "model": name,
                "class_labels": labels,
                "feature_names": feature_names,
                "n_features_in": len(feature_names),
                "config_hash": config_hash(dict(cfg)),
                "current_year": cfg["features"]["current_year"],
                "target_bins": [str(b) for b in cfg["target"]["bins"]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path


def write_metrics(payload: dict, cfg: dict) -> Path:
    path = Path(cfg["paths"]["metrics"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def print_report(payload: dict, cfg: dict) -> None:
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(results_table(payload["models"], payload["baseline"], cfg))
