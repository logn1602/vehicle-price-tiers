"""Does the approach itself hold up? Three checks that could invalidate it.

The rest of this project verifies that the pipeline is *correct*. This script
asks the prior question: is a four-class classifier over binned asking prices
the right thing to be building at all?

    1. Are errors just boundary artifacts?
       If most mistakes sit next to a tier cut, the accuracy figure is measuring
       the arbitrariness of the cuts rather than the model. It would be
       convenient to believe this, so it is worth measuring rather than
       assuming.

    2. Is classification the right framing?
       Binning discards information before the model sees it. A regressor on the
       continuous target keeps it and can be binned afterwards. If that wins,
       the project is the wrong shape.

    3. Is balanced weighting still earning its place?
       The brief treats the missing class weights as a defect. At v1's
       hyperparameters that is clearly right. At tuned hyperparameters it is
       worth re-testing, because weighting trades precision for recall rather
       than adding skill.

Every claim these produce is written to `results/approach_validation.json` so
that the README can cite them and `scripts/verify_readme.py` can trace them.
Findings that appear in prose but not in an artifact are exactly what this
rebuild exists to eliminate.

    python scripts/validate_approach.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import cohen_kappa_score, f1_score  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluate import per_class_metrics  # noqa: E402
from src.pipeline import build_pipeline, encode_target, fit, stratified_split  # noqa: E402
from src.train import load_gold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y_true).mean()),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, pred, weights="quadratic")
        ),
        "tier_mae": float(np.mean(np.abs(y_true - pred))),
        "adjacent_accuracy": float(np.mean(np.abs(y_true - pred) <= 1)),
    }


def boundary_analysis(
    prices: np.ndarray, wrong: np.ndarray, edges: np.ndarray, bands: tuple[float, ...]
) -> dict:
    """Error rate as a function of distance to the nearest tier boundary."""
    nearest = np.argmin(np.abs(np.subtract.outer(prices, edges)), axis=1)
    relative = np.abs(prices - edges[nearest]) / edges[nearest]

    out = {
        "median_relative_distance_correct": float(np.median(relative[~wrong])),
        "median_relative_distance_errors": float(np.median(relative[wrong])),
        "bands": [],
    }
    for band in bands:
        near = relative <= band
        if not near.any() or near.all():
            continue
        out["bands"].append(
            {
                "within_pct_of_boundary": band * 100,
                "listings": int(near.sum()),
                "share_of_test": round(float(near.mean()), 4),
                "error_rate_near": round(float(wrong[near].mean()), 4),
                "error_rate_elsewhere": round(float(wrong[~near].mean()), 4),
            }
        )
    out["share_of_errors_within_10pct"] = round(float((relative[wrong] <= 0.10).mean()), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/approach_validation.json")
    args = p.parse_args(argv)

    cfg = load_config()
    bins = [float(b) for b in cfg["target"]["bins"]]
    edges = np.array(bins[1:-1])

    X, y_labels = load_gold(cfg)
    y, labels = encode_target(y_labels, cfg)
    splits = stratified_split(X, y, cfg)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    gold_ids = pd.read_parquet(
        REPO / cfg["paths"]["gold"] / "features.parquet", columns=["id"]
    )["id"]
    silver = pq.read_table(REPO / cfg["paths"]["silver"], columns=["id", "price"]).to_pandas()
    price = silver.set_index("id")["price"].reindex(gold_ids.to_numpy()).to_numpy()
    positions = X.index.get_indexer(X_test.index)
    price_test = price[positions]
    price_train = price[X.index.get_indexer(X_train.index)]
    price_val = price[X.index.get_indexer(X_val.index)]

    report: dict = {"note": __doc__.strip().splitlines()[0]}

    # --- 1 and 3: classification variants ----------------------------------
    print("fitting classifier, balanced ...")
    weighted = fit(
        build_pipeline("xgboost_tuned", cfg, n_classes=len(labels)),
        "xgboost_tuned", X_train, y_train, X_val, y_val, balanced=True,
    )
    pred_weighted = weighted.predict(X_test)

    print("fitting classifier, unweighted ...")
    unweighted = fit(
        build_pipeline("xgboost_tuned", cfg, n_classes=len(labels)),
        "xgboost_tuned", X_train, y_train, X_val, y_val, balanced=False,
    )
    pred_unweighted = unweighted.predict(X_test)

    report["boundary_effect"] = boundary_analysis(
        price_test, pred_weighted != y_test, edges, (0.05, 0.10, 0.20)
    )

    report["class_weighting"] = {
        "question": (
            "Does balanced weighting still help once the model has the capacity "
            "to learn the minority tier unaided?"
        ),
        "balanced": {
            **score(y_test, pred_weighted),
            "per_class": per_class_metrics(y_test, pred_weighted, labels),
        },
        "unweighted": {
            **score(y_test, pred_unweighted),
            "per_class": per_class_metrics(y_test, pred_unweighted, labels),
        },
    }

    # --- 2: framing ---------------------------------------------------------
    print("fitting regressors ...")
    prefix = clone(build_pipeline("xgboost_tuned", cfg, n_classes=len(labels))[:-1])
    Xt_train = prefix.fit_transform(X_train, y_train)
    Xt_val = prefix.transform(X_val)
    Xt_test = prefix.transform(X_test)
    params = {**cfg["model"]["xgboost_tuned"], "eval_metric": "rmse"}

    def to_tier(values: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(values, edges, right=True), 0, len(labels) - 1)

    framings = {"classification": score(y_test, pred_weighted)}
    for tag, transform, inverse in (
        ("regression_raw_price", lambda v: v, lambda v: v),
        ("regression_log_price", np.log1p, np.expm1),
    ):
        model = XGBRegressor(
            objective="reg:squarederror", random_state=cfg["seed"], n_jobs=-1, **params
        )
        model.fit(
            Xt_train, transform(price_train),
            eval_set=[(Xt_val, transform(price_val))], verbose=False,
        )
        framings[tag] = score(y_test, to_tier(inverse(model.predict(Xt_test))))

    best_framing = max(framings, key=lambda k: framings[k]["macro_f1"])
    report["framing"] = {
        "question": "Does regressing the continuous price and binning afterwards beat "
                    "classifying the bins directly?",
        "results": framings,
        "best": best_framing,
        "classification_wins": best_framing == "classification",
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- console summary ----------------------------------------------------
    print("\n" + "=" * 76)
    print("APPROACH VALIDATION")
    print("=" * 76)

    print("\n[1] Are errors just boundary artifacts?")
    for band in report["boundary_effect"]["bands"]:
        print(
            f"    within {band['within_pct_of_boundary']:>4.0f}% of a boundary: "
            f"error {band['error_rate_near'] * 100:5.2f}%  vs "
            f"{band['error_rate_elsewhere'] * 100:5.2f}% elsewhere"
        )
    print(
        f"    share of all errors within 10% of a boundary: "
        f"{report['boundary_effect']['share_of_errors_within_10pct'] * 100:.2f}%"
    )

    print("\n[2] Is classification the right framing?")
    for tag, values in sorted(
        framings.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True
    ):
        print(f"    {tag:<24} macro F1 {values['macro_f1']:.4f}"
              f"  acc {values['accuracy']:.4f}  tier MAE {values['tier_mae']:.4f}")
    print(f"    -> classification wins: {report['framing']['classification_wins']}")

    print("\n[3] Is balanced weighting still earning its place?")
    for tag in ("balanced", "unweighted"):
        block = report["class_weighting"][tag]
        lux = next(c for c in block["per_class"] if c["class"] == "Luxury")
        print(
            f"    {tag:<12} macro F1 {block['macro_f1']:.4f}"
            f"  Luxury P {lux['precision']:.4f} R {lux['recall']:.4f} F1 {lux['f1']:.4f}"
        )

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
