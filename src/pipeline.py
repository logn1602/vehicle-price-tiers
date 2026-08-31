"""Pipeline factory and the leakage-free fitting protocol.

Every transform that learns anything from data lives inside an sklearn
Pipeline: imputation, variance filtering, feature selection, scaling. The
Pipeline is fitted on the training split only, so none of those quantities can
see validation or test rows.

v1 fitted the first three of those on the full dataset before splitting, and
passed every label to SelectKBest. Only the scaler was train-only.

The step order is exactly as specified in the brief, flat rather than nested,
so `pipe[:-1]` slices off the preprocessing prefix -- which is what makes
XGBoost early stopping expressible without breaking the leakage guarantee.
See `fit` below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.ordinal import OrdinalClassifier

MODEL_NAMES = [
    "decision_tree",
    "random_forest",
    "xgboost",
    "logistic_regression",
    "svm",
    # Same booster, same hyperparameters, but decomposed into cumulative binary
    # problems so the tier ordering is in the objective rather than only in the
    # evaluation. See src/ordinal.py.
    "xgboost_ordinal",
    # Same architecture as `xgboost`, hyperparameters from scripts/tune.py.
    # Separate from `xgboost` so v1's parameters remain the ablation reference.
    "xgboost_tuned",
    # The ordinal decomposition at tuned hyperparameters. Without this pair,
    # "ordinal does not help" would only be established for two underfit
    # models, which is a far weaker claim than it sounds.
    "xgboost_ordinal_tuned",
]

# Models fitted with an explicit early-stopping protocol rather than a plain
# Pipeline.fit.
EARLY_STOPPING_MODELS = {
    "xgboost",
    "xgboost_ordinal",
    "xgboost_tuned",
    "xgboost_ordinal_tuned",
}

# Which config block supplies each model's hyperparameters. The ordinal
# variants share a block with their flat counterpart so the only difference
# between the pair is the decomposition.
HYPERPARAMETER_BLOCK = {
    "xgboost": "xgboost",
    "xgboost_ordinal": "xgboost",
    "xgboost_tuned": "xgboost_tuned",
    "xgboost_ordinal_tuned": "xgboost_tuned",
}


# ---------------------------------------------------------------------------
# target encoding
# ---------------------------------------------------------------------------
def encode_target(y: pd.Series, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Map tier labels to integers **in tier order**, not alphabetically.

    v1 used LabelEncoder, which sorts: [Budget, Luxury, Mid-Range, Premium].
    That is why its confusion matrix was unreadable -- the Luxury row
    `[1, 16, 4, 58]` only makes sense once you notice that 58 sits in the
    *Premium* column, three positions from where a reader would expect it.

    Encoding in tier order is also a precondition for the ordinal metrics:
    quadratic weighted kappa and tier MAE are meaningless if the integer
    distance between classes is not the distance between price tiers.
    """
    labels = list(cfg["target"]["labels"])
    lookup = {label: i for i, label in enumerate(labels)}
    encoded = y.astype(str).map(lookup)
    if encoded.isna().any():
        unknown = sorted(set(y.astype(str)) - set(labels))
        raise ValueError(f"Target contains labels outside {labels}: {unknown}")
    return encoded.to_numpy(dtype=np.int64), labels


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def stratified_split(
    X: pd.DataFrame, y: np.ndarray, cfg: dict
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Stratified 60/20/20 train/validation/test.

    The validation set exists to drive early stopping. In v1 it was built,
    scaled, and then never referenced again because `early_stopping_rounds`
    was commented out -- 20% of the data spent on nothing.
    """
    seed = cfg["seed"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    stratify = cfg["split"]["stratify"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None
    )
    # val_size is a fraction of the ORIGINAL, so rescale against what remains.
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_temp if stratify else None,
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def build_estimator(name: str, cfg: dict, n_classes: int, for_v1: bool = False) -> Any:
    """Construct one classifier from config. No hyperparameter is literal here."""
    seed = cfg["seed"]
    m = cfg["model"]

    if name == "decision_tree":
        p = m["decision_tree"]
        return DecisionTreeClassifier(random_state=seed, **p)

    if name == "random_forest":
        p = m["random_forest"]
        return RandomForestClassifier(random_state=seed, **p)

    if name in {"xgboost", "xgboost_tuned"}:
        p = dict(m[HYPERPARAMETER_BLOCK[name]])
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            random_state=seed,
            n_jobs=m.get("n_jobs", 4),
            # xgboost >= 2.0 takes early_stopping_rounds on the constructor,
            # not on fit(). v1's commented-out line was already stale syntax.
            **p,
        )

    if name in {"xgboost_ordinal", "xgboost_ordinal_tuned"}:
        p = dict(m[HYPERPARAMETER_BLOCK[name]])
        # Identical hyperparameters to the flat model, so any difference in the
        # results is attributable to the decomposition and not to tuning. The
        # one necessary change: each sub-model solves a BINARY problem, so the
        # multiclass log-loss inherited from config would be evaluated against
        # two-class labels and xgboost rejects it.
        p["eval_metric"] = "logloss"
        base = XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            n_jobs=m.get("n_jobs", 4),
            **p,
        )
        return OrdinalClassifier(estimator=base, n_classes=n_classes)

    if name == "logistic_regression":
        p = m["logistic_regression"]
        return LogisticRegression(random_state=seed, **p)

    if name == "svm":
        p = m["svm"]
        if for_v1:
            # n=6,067 makes the exact RBF kernel cheap, so the v1 reconstruction
            # uses the same estimator v1 used.
            #
            # `probability=True` is deprecated in scikit-learn 1.9 and removed
            # in 1.11, and the replacement -- CalibratedClassifierCV(SVC(),
            # ensemble=False) -- is not numerically identical to SVC's internal
            # Platt scaling. Fidelity wins here: the entire purpose of this
            # branch is to run what v1 ran. When the pin has to move past 1.11
            # this row's numbers will shift slightly and that should be recorded
            # rather than absorbed silently.
            return SVC(
                C=p["C"],
                kernel="rbf",
                probability=True,
                class_weight=p["class_weight"],
                random_state=seed,
            )
        # SVC(kernel='rbf') is between quadratic and cubic in sample count and
        # does not terminate in reasonable time on ~233k training rows.
        # LinearSVC is the substitute; it is stated in the results table rather
        # than quietly swapped. Calibration supplies the predict_proba that AUC
        # and average precision require.
        base = LinearSVC(
            C=p["C"],
            max_iter=p["max_iter"],
            class_weight=p["class_weight"],
            dual=False,          # n_samples >> n_features
            random_state=seed,
        )
        return CalibratedClassifierCV(base, cv=p["calibration_cv"], method="sigmoid")

    raise ValueError(f"Unknown model {name!r}. Expected one of {MODEL_NAMES}.")


def build_pipeline(
    name: str, cfg: dict, n_classes: int, for_v1: bool = False
) -> Pipeline:
    """The full pipeline, in the order the brief specifies.

    Scaling is applied to every model, not just LogisticRegression. In v1 the
    RBF-kernel SVM trained on unscaled features and scored 0.554 -- the worst of
    the five -- and the report attributed that to the algorithm rather than to
    the missing scaler.
    """
    p = cfg["pipeline"]
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy=p["imputer_strategy"])),
            ("variance", VarianceThreshold(threshold=p["variance_threshold"])),
            ("select", SelectKBest(score_func=f_classif, k=p["select_k_best"])),
            ("scale", RobustScaler()),
            ("clf", build_estimator(name, cfg, n_classes, for_v1=for_v1)),
        ]
    )


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fit(
    pipe: Pipeline,
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    balanced: bool = True,
) -> Pipeline:
    """Fit the pipeline on training data only, using the validation set where
    the estimator can exploit it.

    Why this is not simply `pipe.fit(X_train, y_train, clf__eval_set=[(X_val,
    y_val)])`: that hands XGBoost the *raw* validation frame while it is
    training on transformed features. The preprocessing prefix reduces 49
    columns to 25, so the eval set would have the wrong width -- a shape error
    at best, and a silent mismatch at worst.

    Instead the prefix is fitted on train, used to transform both train and
    val, and the classifier is fitted on the transformed pair. The prefix never
    sees a validation label, so the leakage guarantee is unchanged; the
    validation set influences only when boosting stops.
    """
    sample_weight = (
        compute_sample_weight("balanced", y_train) if balanced else None
    )

    if name not in EARLY_STOPPING_MODELS or X_val is None:
        if sample_weight is not None and name in EARLY_STOPPING_MODELS:
            return pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
        return pipe.fit(X_train, y_train)

    prefix = clone(pipe[:-1])
    Xt_train = prefix.fit_transform(X_train, y_train)
    Xt_val = prefix.transform(X_val)

    clf = clone(pipe[-1])
    clf.fit(
        Xt_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(Xt_val, y_val)],
        verbose=False,
    )

    # Reassemble from the already-fitted parts. sklearn's Pipeline does not
    # re-fit on predict, so this is a usable fitted estimator.
    return Pipeline([*prefix.steps, ("clf", clf)])


def selected_features(pipe: Pipeline, feature_names: list[str]) -> list[str]:
    """Names of the features the fitted pipeline actually kept.

    Walks the prefix in order because VarianceThreshold drops columns before
    SelectKBest sees them, so the selector's support mask indexes into the
    post-variance subset, not the original frame.
    """
    names = np.asarray(feature_names)
    if "variance" in pipe.named_steps:
        names = names[pipe.named_steps["variance"].get_support()]
    if "select" in pipe.named_steps:
        names = names[pipe.named_steps["select"].get_support()]
    return list(names)
