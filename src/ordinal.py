"""Ordinal classification by binary decomposition (Frank & Hall, 2001).

The tiers are ordered: Budget < Mid-Range < Premium < Luxury. Standard
multiclass softmax throws that away -- it treats predicting Luxury for a Budget
vehicle as exactly as wrong as predicting Mid-Range, when the first is three
tiers out and the second is one.

Everything else in this project *measures* the ordering (quadratic weighted
kappa, adjacent accuracy, tier MAE) but does not *model* it. This does.

The decomposition
-----------------
For K ordered classes, train K-1 binary classifiers, each answering a
cumulative question:

    model 0:  is this above Budget?          P(y > 0)
    model 1:  is this above Mid-Range?       P(y > 1)
    model 2:  is this above Premium?         P(y > 2)

Class probabilities telescope out of the cumulative ones:

    P(y = 0) = 1 - P(y > 0)
    P(y = k) = P(y > k-1) - P(y > k)
    P(y = K-1) = P(y > K-2)

Every sub-problem is monotone in price, which is exactly the structure the flat
softmax has to rediscover from scratch. It also means each binary model sees
every training row, rather than the one-vs-rest split where the Luxury model
sees 7% positives.

Why the clipping matters
------------------------
The K-1 models are fitted independently, so nothing forces
P(y>0) >= P(y>1) >= P(y>2). Where they invert, a naive difference produces a
negative "probability". Differences are clipped at zero and renormalised;
`monotonicity_violation_rate_` records how often that was necessary, because a
high rate would mean the decomposition is not fitting this data well and should
be reported rather than hidden.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Wrap any binary classifier into an ordinal multiclass one.

    Parameters
    ----------
    estimator
        A binary classifier exposing `predict_proba`. Cloned K-1 times.
    n_classes
        Number of ordered classes. Labels must be integers 0..n_classes-1 in
        tier order -- `src.pipeline.encode_target` guarantees this.
    """

    def __init__(self, estimator=None, n_classes: int = 4) -> None:
        self.estimator = estimator
        self.n_classes = n_classes

    def fit(self, X, y, sample_weight=None, eval_set=None, **fit_params):
        if self.estimator is None:
            raise ValueError("OrdinalClassifier requires a base estimator")

        y = np.asarray(y)
        self.classes_ = np.arange(self.n_classes)
        self.estimators_ = []

        for k in range(self.n_classes - 1):
            # "Is this vehicle above tier k?" Every row contributes to every
            # sub-problem, unlike one-vs-rest.
            target = (y > k).astype(int)
            model = clone(self.estimator)

            kwargs = dict(fit_params)
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            if eval_set is not None:
                kwargs["eval_set"] = [
                    (Xv, (np.asarray(yv) > k).astype(int)) for Xv, yv in eval_set
                ]
                kwargs.setdefault("verbose", False)

            model.fit(X, target, **kwargs)
            self.estimators_.append(model)

        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "estimators_")

        # cumulative[k] = P(y > k)
        cumulative = np.column_stack(
            [m.predict_proba(X)[:, 1] for m in self.estimators_]
        )

        n = cumulative.shape[0]
        proba = np.empty((n, self.n_classes), dtype=float)
        proba[:, 0] = 1.0 - cumulative[:, 0]
        for k in range(1, self.n_classes - 1):
            proba[:, k] = cumulative[:, k - 1] - cumulative[:, k]
        proba[:, -1] = cumulative[:, -1]

        # Independent fits can invert the cumulative ordering; where they do,
        # the telescoped difference goes negative.
        self.monotonicity_violation_rate_ = float(np.mean(proba.min(axis=1) < 0))

        np.clip(proba, 0.0, None, out=proba)
        totals = proba.sum(axis=1, keepdims=True)
        # A row where every clipped value is zero cannot be normalised; fall
        # back to uniform rather than dividing by zero.
        degenerate = totals[:, 0] == 0
        proba[degenerate] = 1.0 / self.n_classes
        totals[degenerate] = 1.0
        return proba / totals

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        return tags

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean importance across the cumulative models.

        Averaging is a simplification: a feature can matter a great deal for
        "above Premium?" and not at all for "above Budget?". Per-model
        importances remain available on `estimators_`.
        """
        check_is_fitted(self, "estimators_")
        per_model = [
            m.feature_importances_
            for m in self.estimators_
            if hasattr(m, "feature_importances_")
        ]
        if not per_model:
            raise AttributeError("base estimator exposes no feature_importances_")
        return np.mean(per_model, axis=0)

    @property
    def best_iteration(self) -> int | None:
        """Largest best_iteration across the cumulative models, if any stopped early."""
        check_is_fitted(self, "estimators_")
        iterations = [
            getattr(m, "best_iteration", None) for m in self.estimators_
        ]
        present = [i for i in iterations if i is not None]
        return max(present) if present else None

    def evals_result(self) -> dict:
        check_is_fitted(self, "estimators_")
        return {
            f"above_tier_{k}": m.evals_result()
            for k, m in enumerate(self.estimators_)
            if hasattr(m, "evals_result")
        }
