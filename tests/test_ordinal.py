"""Tests for the ordinal binary decomposition.

The arithmetic is tested against a stub estimator with hand-chosen cumulative
probabilities, so the telescoping and the clipping are verified exactly rather
than inferred from a model's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.ordinal import OrdinalClassifier


class StubBinary(ClassifierMixin, BaseEstimator):
    """Returns a fixed P(positive), whatever it is asked.

    Each clone must return a different value, so the constructor argument is
    supplied by a factory that hands out successive probabilities.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def fit(self, X, y, **kwargs):
        self.fitted_ = True
        self.n_seen_ = len(y)
        self.positives_ = int(np.sum(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class SequencedStub(StubBinary):
    """A stub whose clones step through a queue of probabilities."""

    queue: list[float] = []

    def fit(self, X, y, **kwargs):
        self.p = SequencedStub.queue.pop(0)
        return super().fit(X, y, **kwargs)


@pytest.fixture
def X():
    return np.zeros((10, 3))


@pytest.fixture
def y():
    # Two of each tier, plus two extra Budget, in tier order 0..3.
    return np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 1])


def fit_with(cumulative: list[float], X, y) -> OrdinalClassifier:
    SequencedStub.queue = list(cumulative)
    model = OrdinalClassifier(estimator=SequencedStub(), n_classes=4)
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# the decomposition itself
# ---------------------------------------------------------------------------
def test_fits_k_minus_one_binary_models(X, y):
    model = fit_with([0.8, 0.5, 0.2], X, y)
    assert len(model.estimators_) == 3


def test_each_submodel_sees_every_row(X, y):
    """Unlike one-vs-rest, every sub-problem uses the whole training set."""
    model = fit_with([0.8, 0.5, 0.2], X, y)
    for sub in model.estimators_:
        assert sub.n_seen_ == len(y)


def test_submodel_targets_are_cumulative(X, y):
    """Model k must be trained on the question 'is y > k?'."""
    model = fit_with([0.8, 0.5, 0.2], X, y)
    for k, sub in enumerate(model.estimators_):
        assert sub.positives_ == int(np.sum(y > k)), f"model {k}"


def test_telescoping_is_exact(X, y):
    """P(y=0)=1-P(y>0); P(y=k)=P(y>k-1)-P(y>k); P(y=K-1)=P(y>K-2)."""
    model = fit_with([0.8, 0.5, 0.2], X, y)
    proba = model.predict_proba(X)
    expected = np.array([1 - 0.8, 0.8 - 0.5, 0.5 - 0.2, 0.2])
    np.testing.assert_allclose(proba[0], expected, atol=1e-12)


def test_probabilities_sum_to_one(X, y):
    model = fit_with([0.9, 0.4, 0.1], X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))


def test_probabilities_are_non_negative(X, y):
    model = fit_with([0.9, 0.4, 0.1], X, y)
    assert (model.predict_proba(X) >= 0).all()


def test_predict_is_argmax_of_predict_proba(X, y):
    model = fit_with([0.9, 0.8, 0.7], X, y)
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(model.predict(X), np.argmax(proba, axis=1))


# ---------------------------------------------------------------------------
# monotonicity violations
# ---------------------------------------------------------------------------
def test_monotonic_cumulative_records_no_violation(X, y):
    model = fit_with([0.8, 0.5, 0.2], X, y)  # properly decreasing
    model.predict_proba(X)
    assert model.monotonicity_violation_rate_ == 0.0


def test_inverted_cumulative_is_clipped_and_reported(X, y):
    """Independent fits can produce P(y>1) > P(y>0), which telescopes to a
    negative probability. It must be clipped, renormalised, and counted."""
    model = fit_with([0.3, 0.9, 0.1], X, y)  # 0.3 -> 0.9 is an inversion
    proba = model.predict_proba(X)

    assert model.monotonicity_violation_rate_ == 1.0
    assert (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))


def test_all_zero_row_falls_back_to_uniform(X, y):
    """A pathological case where clipping removes all mass must not divide by
    zero -- it returns a uniform distribution instead."""
    model = fit_with([1.0, 1.0, 0.0], X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
    assert np.isfinite(proba).all()


# ---------------------------------------------------------------------------
# integration with the real pipeline
# ---------------------------------------------------------------------------
def test_ordinal_pipeline_produces_valid_probabilities():
    """End to end with a real booster on synthetic ordered data."""
    import pandas as pd

    from src.config import load_config
    from src.pipeline import build_pipeline, encode_target, fit, stratified_split

    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    n, n_features = 1_500, 12

    frame = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i:02d}" for i in range(n_features)],
    )
    signal = frame["f00"] * 2 - frame["f01"] + rng.normal(scale=0.4, size=n)
    labels = pd.qcut(signal, q=4, labels=cfg["target"]["labels"])
    y, class_labels = encode_target(pd.Series(labels), cfg)

    splits = stratified_split(frame, y, cfg)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    pipe = build_pipeline("xgboost_ordinal", cfg, n_classes=len(class_labels))
    fitted = fit(pipe, "xgboost_ordinal", X_train, y_train, X_val, y_val)

    proba = fitted.predict_proba(X_test)
    assert proba.shape == (len(X_test), 4)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-6)
    assert (proba >= 0).all()

    predictions = fitted.predict(X_test)
    assert set(np.unique(predictions)).issubset(set(range(4)))
    # Ordered signal, so it must beat chance comfortably.
    assert (predictions == y_test).mean() > 0.4


def test_ordinal_exposes_averaged_feature_importances(X, y):
    class WithImportances(SequencedStub):
        def fit(self, X, y, **kwargs):
            super().fit(X, y, **kwargs)
            self.feature_importances_ = np.array([self.p, 1 - self.p, 0.0])
            return self

    SequencedStub.queue = [0.8, 0.6, 0.4]
    model = OrdinalClassifier(estimator=WithImportances(), n_classes=4).fit(X, y)
    importances = model.feature_importances_
    np.testing.assert_allclose(importances, [0.6, 0.4, 0.0], atol=1e-12)
