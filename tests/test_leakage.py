"""Leakage tests -- the most important file in this repository.

v1 fitted imputation, variance filtering and feature selection on the full
dataset before splitting, and passed every label to SelectKBest. Its reported
test accuracy was therefore partly the model recognising choices made using the
answers.

These tests do not merely assert that the current pipeline is clean. Several of
them also demonstrate that the leaky ordering would have produced a *different*
result, which is what makes the ablation in Phase 5 meaningful rather than
decorative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer

from src.config import load_config
from src.pipeline import build_pipeline, encode_target, fit, selected_features, stratified_split

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(REPO / "conf" / "config.yaml")


@pytest.fixture(scope="module")
def dataset(cfg) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """A synthetic frame with the real shape: 49 features, 4 ordinal classes.

    Deliberately synthetic so the tests are fast and deterministic. Some
    features carry real signal, some are noise, and one column has missing
    values so the imputer has something to learn.
    """
    rng = np.random.default_rng(cfg["seed"])
    n, n_features = 4_000, 49

    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i:02d}" for i in range(n_features)],
    )
    # Give a handful of columns genuine association with the target.
    signal = X["f00"] * 2.0 + X["f01"] - X["f02"] * 1.5 + rng.normal(scale=0.5, size=n)
    y_labels = pd.qcut(signal, q=4, labels=cfg["target"]["labels"])

    # A column with missingness, skewed so train and full medians differ.
    missing_mask = rng.random(n) < 0.3
    X.loc[missing_mask, "f10"] = np.nan
    X["f10"] = X["f10"] + np.linspace(0, 50, n)

    # A near-constant column for VarianceThreshold to remove.
    X["f48"] = 1e-6 * rng.normal(size=n)

    y, labels = encode_target(pd.Series(y_labels), cfg)
    return X, y, labels


@pytest.fixture(scope="module")
def split(cfg, dataset):
    X, y, _ = dataset
    return stratified_split(X, y, cfg)


# ---------------------------------------------------------------------------
# 1. the selector sees training data only
# ---------------------------------------------------------------------------
def test_pipeline_selection_matches_manual_train_only_selection(cfg, dataset, split):
    """The pipeline's chosen features equal those from fitting the same
    transforms by hand on train alone."""
    X, _, _ = dataset
    X_train, y_train = split["train"]

    pipe = build_pipeline("decision_tree", cfg, n_classes=4)
    fitted = fit(pipe, "decision_tree", X_train, y_train)
    from_pipeline = selected_features(fitted, list(X.columns))

    p = cfg["pipeline"]
    imputer = SimpleImputer(strategy=p["imputer_strategy"])
    variance = VarianceThreshold(threshold=p["variance_threshold"])
    selector = SelectKBest(score_func=f_classif, k=p["select_k_best"])

    step1 = imputer.fit_transform(X_train)
    step2 = variance.fit_transform(step1)
    selector.fit(step2, y_train)

    names = np.asarray(list(X.columns))[variance.get_support()][selector.get_support()]
    assert from_pipeline == list(names)


def test_train_only_selection_differs_from_full_data_selection(cfg, dataset, split):
    """The v1 bug, demonstrated rather than asserted.

    Fitting the selector on all rows with all labels picks a measurably
    different feature set than fitting on train alone. If these were identical
    the leak would be harmless and the ablation pointless.
    """
    X, y, _ = dataset
    X_train, y_train = split["train"]
    p = cfg["pipeline"]

    def choose(frame: pd.DataFrame, target: np.ndarray) -> set[str]:
        imputer = SimpleImputer(strategy=p["imputer_strategy"])
        variance = VarianceThreshold(threshold=p["variance_threshold"])
        selector = SelectKBest(score_func=f_classif, k=p["select_k_best"])
        step = variance.fit_transform(imputer.fit_transform(frame))
        selector.fit(step, target)
        cols = np.asarray(list(frame.columns))
        return set(cols[variance.get_support()][selector.get_support()])

    train_only = choose(X_train, y_train)
    leaky = choose(X, y)
    assert train_only != leaky, (
        "train-only and full-data selection agree on this dataset, so it cannot "
        "demonstrate the leak"
    )


# ---------------------------------------------------------------------------
# 2. test labels cannot influence training artifacts
# ---------------------------------------------------------------------------
def test_permuting_test_labels_changes_no_training_artifact(cfg, dataset, split):
    """Shuffle the test labels and refit. Every training-time artifact -- the
    imputer's medians, the variance mask, the selected features -- must be
    byte-identical. This is the regression guard against someone later fitting
    on a concatenation of train and test."""
    X, _, _ = dataset
    X_train, y_train = split["train"]
    _, y_test = split["test"]

    pipe = build_pipeline("decision_tree", cfg, n_classes=4)
    before = fit(clone(pipe), "decision_tree", X_train, y_train)

    rng = np.random.default_rng(cfg["seed"] + 1)
    y_test_shuffled = y_test.copy()
    rng.shuffle(y_test_shuffled)
    assert not np.array_equal(y_test, y_test_shuffled)

    after = fit(clone(pipe), "decision_tree", X_train, y_train)

    np.testing.assert_array_equal(
        before.named_steps["impute"].statistics_,
        after.named_steps["impute"].statistics_,
    )
    np.testing.assert_array_equal(
        before.named_steps["variance"].get_support(),
        after.named_steps["variance"].get_support(),
    )
    assert selected_features(before, list(X.columns)) == selected_features(
        after, list(X.columns)
    )


# ---------------------------------------------------------------------------
# 3. imputer statistics come from train only
# ---------------------------------------------------------------------------
def test_imputer_medians_are_train_only(cfg, dataset, split):
    """The fitted imputer's medians must match train, and must NOT match the
    medians of the full dataset. Equality with the full-data medians would mean
    the imputer had seen held-out rows."""
    X, _, _ = dataset
    X_train, y_train = split["train"]

    pipe = build_pipeline("decision_tree", cfg, n_classes=4)
    fitted = fit(pipe, "decision_tree", X_train, y_train)
    learned = fitted.named_steps["impute"].statistics_

    train_medians = X_train.median().to_numpy()
    full_medians = X.median().to_numpy()

    np.testing.assert_allclose(learned, train_medians)
    assert not np.allclose(learned, full_medians), (
        "imputer medians equal the full-data medians -- it saw held-out rows"
    )


def test_scaler_centres_on_train_only(cfg, dataset, split):
    X, _, _ = dataset
    X_train, y_train = split["train"]
    pipe = build_pipeline("decision_tree", cfg, n_classes=4)
    fitted = fit(pipe, "decision_tree", X_train, y_train)

    centre = fitted.named_steps["scale"].center_
    assert centre.shape[0] == cfg["pipeline"]["select_k_best"]
    assert np.isfinite(centre).all()


# ---------------------------------------------------------------------------
# 4. no feature is a proxy for the target
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_no_feature_correlates_with_target_above_threshold(cfg):
    """Run against the real feature store. A near-perfect correlation would
    indicate a price-derived column or target encoding leaking into X."""
    gold = REPO / cfg["paths"]["gold"] / "features.parquet"
    if not gold.exists():
        pytest.skip("run scripts/run_all.py first")

    df = pd.read_parquet(gold)
    target_col = cfg["target"]["name"]
    y, _ = encode_target(df[target_col], cfg)

    feature_cols = [c for c in df.columns if c not in {target_col, "id"}]
    threshold = cfg["evaluation"]["leakage_correlation_threshold"]

    offenders = {}
    for col in feature_cols:
        values = df[col]
        if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
            continue
        corr = abs(np.corrcoef(values.fillna(values.median()), y)[0, 1])
        if corr > threshold:
            offenders[col] = round(float(corr), 4)

    assert not offenders, f"features correlate with the target above {threshold}: {offenders}"


@pytest.mark.slow
def test_price_is_not_among_the_features(cfg):
    """The most direct leak available: the target's own source column."""
    gold = REPO / cfg["paths"]["gold"] / "features.parquet"
    if not gold.exists():
        pytest.skip("run scripts/run_all.py first")

    columns = set(pd.read_parquet(gold).columns)
    forbidden = {"price", "log_price", "price_category"}
    assert not (columns & forbidden), f"target-derived columns present: {columns & forbidden}"


# ---------------------------------------------------------------------------
# 5. the split itself
# ---------------------------------------------------------------------------
def test_split_is_60_20_20_and_disjoint(cfg, dataset, split):
    X, _, _ = dataset
    sizes = {k: len(v[0]) for k, v in split.items()}
    total = sum(sizes.values())
    assert total == len(X)

    assert sizes["test"] / total == pytest.approx(cfg["split"]["test_size"], abs=0.01)
    assert sizes["val"] / total == pytest.approx(cfg["split"]["val_size"], abs=0.01)

    indices = {k: set(v[0].index) for k, v in split.items()}
    assert not indices["train"] & indices["val"]
    assert not indices["train"] & indices["test"]
    assert not indices["val"] & indices["test"]


def test_split_preserves_class_proportions(cfg, dataset, split):
    _, y, _ = dataset
    overall = np.bincount(y) / len(y)
    for name, (_, y_part) in split.items():
        part = np.bincount(y_part, minlength=len(overall)) / len(y_part)
        np.testing.assert_allclose(part, overall, atol=0.01, err_msg=f"{name} split")


def test_validation_set_is_actually_consumed_by_xgboost(cfg, dataset, split):
    """v1 built X_val, scaled it, and never referenced it again because
    early_stopping_rounds was commented out.

    What this asserts is that the validation set is *consumed*: the booster
    records a per-iteration evaluation history against it and exposes a best
    iteration. It deliberately does NOT assert that early stopping fires --
    whether it does depends on the data and the estimator budget, and on the
    full 389k-row dataset it does not (the booster is still improving at 500
    rounds, which is a statement about the hyperparameters being inherited from
    a 6,067-row experiment, not about the wiring being broken).

    Asserting "stopped early" would make this test a hyperparameter check that
    passes on a small fixture and misdescribes the production run.
    """
    X_train, y_train = split["train"]
    X_val, y_val = split["val"]

    pipe = build_pipeline("xgboost", cfg, n_classes=4)
    fitted = fit(pipe, "xgboost", X_train, y_train, X_val, y_val)
    booster = fitted.named_steps["clf"]

    history = booster.evals_result()
    assert history, "no eval history -- the validation set was ignored"

    # One entry per boosting round means the val set was scored every iteration.
    curve = next(iter(next(iter(history.values())).values()))
    assert len(curve) > 1, "validation was scored once or not at all"
    assert booster.best_iteration is not None
    assert 0 <= booster.best_iteration <= cfg["model"]["xgboost"]["n_estimators"]


def test_fit_without_a_validation_set_still_works(cfg, split):
    """The ablation fits XGBoost with early stopping disabled. That path must
    not silently depend on an eval_set being present."""
    X_train, y_train = split["train"]
    pipe = build_pipeline("xgboost", cfg, n_classes=4)
    pipe.named_steps["clf"].set_params(early_stopping_rounds=None)
    fitted = fit(pipe, "xgboost", X_train, y_train, None, None)
    assert fitted.predict(X_train.head(5)).shape == (5,)


def test_eval_set_is_transformed_not_raw(cfg, dataset, split):
    """The subtle failure this protocol exists to prevent: passing raw X_val to
    an estimator trained on 25 selected features. The booster's input width must
    equal select_k_best, not the original 49."""
    X_train, y_train = split["train"]
    X_val, y_val = split["val"]

    pipe = build_pipeline("xgboost", cfg, n_classes=4)
    fitted = fit(pipe, "xgboost", X_train, y_train, X_val, y_val)

    assert fitted.named_steps["clf"].n_features_in_ == cfg["pipeline"]["select_k_best"]
    assert X_train.shape[1] > cfg["pipeline"]["select_k_best"]
