"""Hyperparameter search for XGBoost, scored on the validation split.

Why this exists: every model in this project inherits v1's hyperparameters --
`max_depth=4`, `min_child_weight=7` -- which were chosen for a 6,067-row
experiment and are now applied to 233,544 training rows. The symptoms are
unambiguous. The train/test gap is 0.007, and early stopping never fires: the
booster is still improving when it exhausts its 500-round budget. That is
underfitting, and it is an artifact of the inherited configuration rather than a
property of the data.

The search deliberately does NOT replace v1's parameters in the ablation. Those
stay fixed so the ablation remains apples-to-apples. The tuned configuration is
reported as a separate, labelled result.

Scoring uses the validation split, not cross-validation. The validation set
exists precisely for model selection, using it here is what it is for, and it
keeps the test set untouched by any tuning decision.

    python scripts/tune.py --trials 15 --sample 120000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.base import clone  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import load_config  # noqa: E402
from src.pipeline import build_pipeline, encode_target, stratified_split  # noqa: E402
from src.train import load_gold, subsample  # noqa: E402

# Ranges centred on where the diagnosis points: more capacity than depth 4, and
# enough rounds that early stopping can actually choose a stopping point.
SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10, 12],
    "min_child_weight": [1, 3, 5, 7, 10],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0.0, 0.1, 0.5],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
}

MAX_ROUNDS = 3000
EARLY_STOPPING = 50


def sample_params(rng: np.random.Generator) -> dict:
    return {key: rng.choice(values).item() for key, values in SEARCH_SPACE.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument(
        "--sample",
        type=int,
        default=120_000,
        help="tune on a stratified subsample; the winner is verified on full data",
    )
    p.add_argument("--out", default="results/tuning.json")
    args = p.parse_args(argv)

    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])

    X, y_labels = load_gold(cfg)
    if args.sample and args.sample < len(X):
        X, y_labels = subsample(X, y_labels, args.sample, cfg["seed"])
        print(f"tuning on a {len(X):,}-row stratified subsample\n")

    y, labels = encode_target(y_labels, cfg)
    splits = stratified_split(X, y, cfg)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    # Fit the preprocessing prefix once. It does not depend on the classifier's
    # hyperparameters, and refitting it per trial would waste most of the budget
    # -- but it is still fitted on TRAIN ONLY, so no tuning decision sees
    # validation or test rows.
    template = build_pipeline("xgboost", cfg, n_classes=len(labels))
    prefix = clone(template[:-1])
    Xt_train = prefix.fit_transform(X_train, y_train)
    Xt_val = prefix.transform(X_val)
    weight = compute_sample_weight("balanced", y_train)

    # Strip the fields this search controls itself: the round budget and the
    # stopping rule are set below for every trial, including the reference, so
    # each configuration is judged on where early stopping puts it rather than
    # on an inherited cap.
    baseline_params = dict(cfg["model"]["xgboost"])
    for controlled in ("n_estimators", "early_stopping_rounds", "eval_metric"):
        baseline_params.pop(controlled, None)

    trials: list[dict] = []
    print(f"{'trial':>6}  {'macro F1':>9}  {'rounds':>7}  params")
    print("-" * 100)

    def evaluate(params: dict, label: str) -> dict:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(labels),
            n_estimators=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING,
            eval_metric="mlogloss",
            random_state=cfg["seed"],
            n_jobs=-1,
            **params,
        )
        started = time.perf_counter()
        model.fit(
            Xt_train, y_train,
            sample_weight=weight,
            eval_set=[(Xt_val, y_val)],
            verbose=False,
        )
        score = float(
            f1_score(y_val, model.predict(Xt_val), average="macro", zero_division=0)
        )
        record = {
            "label": label,
            "params": params,
            "val_macro_f1": score,
            "best_iteration": int(model.best_iteration),
            "seconds": round(time.perf_counter() - started, 1),
        }
        compact = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"{label:>6}  {score:>9.4f}  {model.best_iteration:>7}  {compact}")
        return record

    # Trial 0 is v1's configuration, so the table has a reference point.
    trials.append(evaluate(baseline_params, "v1"))
    for i in range(args.trials):
        trials.append(evaluate(sample_params(rng), str(i + 1)))

    best = max(trials, key=lambda t: t["val_macro_f1"])
    reference = trials[0]
    lift = best["val_macro_f1"] - reference["val_macro_f1"]

    payload = {
        "note": (
            "Scored on the validation split. The test set was not consulted. "
            "v1's hyperparameters remain the ablation's baseline; this is a "
            "separate, labelled configuration."
        ),
        "tuned_on_rows": int(len(X)),
        "trials": trials,
        "reference": reference,
        "best": best,
        "lift_vs_v1_params": lift,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"v1 params      val macro F1 {reference['val_macro_f1']:.4f} "
          f"(stopped at round {reference['best_iteration']})")
    print(f"best found     val macro F1 {best['val_macro_f1']:.4f} "
          f"(stopped at round {best['best_iteration']})")
    print(f"lift           {lift:+.4f}")
    print("\nAdd to conf/config.yaml under model.xgboost_tuned:\n")
    for key, value in best["params"].items():
        print(f"    {key}: {value}")
    print(f"    n_estimators: {best['best_iteration'] + 1}")
    print("    early_stopping_rounds: 50")
    print("    eval_metric: mlogloss")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
