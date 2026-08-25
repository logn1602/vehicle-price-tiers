"""Feature-layer tests: purity, determinism, and the specific v1 defects.

The purity tests matter more than they look. Feature engineering that is truly
row-independent cannot leak information between train and test no matter how
the data is split, which is what lets the leakage tests in test_leakage.py
focus solely on the Pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import GROUP_ORDER, build_features, feature_matrix, make_target

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(REPO / "conf" / "config.yaml")


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small silver-shaped frame covering the awkward cases deliberately:
    a post-scrape year, a missing condition, an unknown fuel, a salvage title."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6"],
            "price": [4500.0, 12000.0, 28000.0, 65000.0, 8000.0, 20000.0],
            "year": [2005.0, 2015.0, 2019.0, 2022.0, 1995.0, 2010.0],
            "odometer": [185000.0, 78000.0, 22000.0, 1200.0, 240000.0, 95000.0],
            "manufacturer": ["ford", "toyota", "bmw", "tesla", None, "HONDA "],
            "type": ["sedan", "suv", "coupe", None, "truck", "hatchback"],
            "condition": ["fair", "good", "excellent", None, "salvage", None],
            "fuel": ["gas", "gas", "diesel", "electric", None, "hybrid"],
            "transmission": ["automatic", "automatic", "manual", None, "other", None],
            "title_status": ["clean", "clean", "clean", "clean", "salvage", "lien"],
            "state": ["ca", "tx", "ny", "ca", "fl", "wa"],
        }
    )


# ---------------------------------------------------------------------------
# purity
# ---------------------------------------------------------------------------
def test_does_not_mutate_input(cfg, frame):
    before = frame.copy(deep=True)
    build_features(frame, cfg)
    pd.testing.assert_frame_equal(frame, before)


def test_deterministic_across_repeated_calls(cfg, frame):
    a, _ = build_features(frame, cfg)
    b, _ = build_features(frame, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_row_order_does_not_change_any_feature(cfg, frame):
    """Property: features are row-independent.

    Computing on a shuffled frame and re-sorting must reproduce the original
    exactly. Any feature that secretly depended on a column statistic -- a mean,
    a rank, a group aggregate -- would fail this.
    """
    full, manifest = build_features(frame, cfg)
    shuffled = frame.sample(frac=1.0, random_state=0)
    shuffled_out, _ = build_features(shuffled, cfg)

    cols = manifest["feature_names"]
    pd.testing.assert_frame_equal(
        full[cols], shuffled_out[cols].reindex(full.index), check_like=False
    )


def test_single_row_matches_batch(cfg, frame):
    """Stronger form of the same property: one row at a time equals all at once.

    This is the invariant the FastAPI service depends on -- a single-listing
    request must produce the same features as a batch job.
    """
    batch, manifest = build_features(frame, cfg)
    cols = manifest["feature_names"]

    for idx in frame.index:
        one, _ = build_features(frame.loc[[idx]], cfg)
        pd.testing.assert_frame_equal(
            one[cols].reset_index(drop=True),
            batch.loc[[idx], cols].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# the v1 defects
# ---------------------------------------------------------------------------
def test_vehicle_age_is_never_negative(cfg, frame):
    """133 real listings post-date the 2021 scrape. In v1 they produced age -1."""
    out, _ = build_features(frame, cfg)
    assert (out["vehicle_age"] >= 0).all()
    # Row 4 is the year-2022 listing.
    assert out.loc[3, "vehicle_age"] == 0


def test_mileage_per_year_has_no_division_by_zero(cfg, frame):
    """v1 computed odometer / (vehicle_age + 1) with age -1, producing inf that
    was silently coerced to 0 much later."""
    out, _ = build_features(frame, cfg)
    assert np.isfinite(out["mileage_per_year"]).all()
    assert not (out["mileage_per_year"] == np.inf).any()


def test_unknown_condition_is_missing_not_fair(cfg, frame):
    """v1 mapped unknown condition to 2 -- identical to 'fair'. Rows 4 and 6
    report no condition and must stay NaN for the imputer to resolve."""
    out, _ = build_features(frame, cfg)
    assert out.loc[3, "condition_numeric"] != 2
    assert pd.isna(out.loc[3, "condition_numeric"])
    assert pd.isna(out.loc[5, "condition_numeric"])
    # A genuine "fair" reading is still 2.
    assert out.loc[0, "condition_numeric"] == 2


def test_categorical_cleaning_is_case_and_whitespace_insensitive(cfg, frame):
    """Row 6 is 'HONDA ' with trailing space; it must register as reliable."""
    out, _ = build_features(frame, cfg)
    assert out.loc[5, "is_reliable"] == 1


def test_missing_categoricals_yield_false_not_error(cfg, frame):
    """A listing with no fuel reported is not electric. Row 5 has fuel None."""
    out, _ = build_features(frame, cfg)
    for col in ("is_electric", "is_hybrid", "is_diesel", "is_gas"):
        assert out.loc[4, col] == 0


# ---------------------------------------------------------------------------
# counts are computed, never declared
# ---------------------------------------------------------------------------
def test_manifest_counts_match_actual_columns(cfg, frame):
    """v1 printed 'Created 10 temporal features' while creating nine."""
    out, manifest = build_features(frame, cfg)
    for group in GROUP_ORDER:
        produced = manifest["groups"][group]
        assert manifest["counts"][group] == len(produced)
        for col in produced:
            assert col in out.columns
    assert manifest["n_features"] == sum(manifest["counts"].values())
    assert manifest["n_features"] == len(set(manifest["feature_names"]))


def test_no_feature_name_collisions(cfg, frame):
    _, manifest = build_features(frame, cfg)
    names = manifest["feature_names"]
    assert len(names) == len(set(names)), "a later group overwrote an earlier feature"


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------
def test_target_is_ordered(cfg, frame):
    y = make_target(frame, cfg)
    assert y.cat.ordered
    assert list(y.cat.categories) == cfg["target"]["labels"]


def test_target_bin_boundaries_are_upper_inclusive(cfg, frame):
    """price 8000 -> Budget, 20000 -> Mid-Range. The v1 report described these
    as '<= $8,000' and '$8,001-$20,000', which only agrees with the code if
    every price is an integer."""
    y = make_target(frame, cfg)
    assert y.iloc[4] == "Budget"      # exactly 8000
    assert y.iloc[5] == "Mid-Range"   # exactly 20000
    assert y.iloc[3] == "Luxury"      # 65000


def test_feature_matrix_is_all_numeric(cfg, frame):
    X, y, manifest = feature_matrix(frame, cfg)
    assert X.shape[1] == manifest["n_features"]
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    assert len(X) == len(y) == len(frame)


def test_only_condition_derived_features_may_be_nan(cfg, frame):
    """Missingness should be confined to where it is meaningful. Anything else
    would silently hand NaN to the imputer and hide a bug in a transform."""
    X, _, _ = feature_matrix(frame, cfg)
    nan_cols = {c for c in X.columns if X[c].isna().any()}
    assert nan_cols <= {"condition_numeric", "condition_age_interaction"}, nan_cols
