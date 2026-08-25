"""Feature engineering: pure, deterministic, stateless.

Every function here is a pure transform. Nothing is fitted, nothing reads a
statistic from the data it is given, and nothing depends on module-level
mutable state. That is what makes it safe to apply the identical code to train,
validation and test without leaking anything between them -- all learned
quantities (medians, variances, feature selection, scaling) live inside the
sklearn Pipeline instead.

Ported from the regression notebook's `comprehensive_feature_engineering_final`,
which the v1 classification pipeline never finished porting -- it stopped at a
`# [Previous feature engineering code continues here...]` placeholder having
built only the first three groups.

Four behaviours were changed in the port, each for a stated reason:

1. Group counts are computed, never asserted. The source printed
   "Created 10 temporal features" while creating nine.
2. Negative vehicle ages are clipped at 0. 133 listings have a year after the
   2021 scrape date; in the source these produced age -1, which made
   `odometer / (vehicle_age + 1)` a division by zero, yielding inf that was
   silently replaced with 0 two hundred lines later.
3. Unknown `condition` maps to NaN, not to 2. The source used 2 -- identical to
   "fair" -- which asserted a condition for the ~40% of listings that do not
   report one. Missingness is now resolved by the Pipeline's imputer, fitted on
   train only.
4. `odometer` is not coerced with `.fillna(0)`. Silver guarantees it non-null;
   treating a missing reading as zero miles would be a fabricated value, not a
   neutral one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Domain taxonomies.
#
# These are vocabulary, not tuning knobs, so they live in code where they can be
# read alongside the logic that uses them. Numeric bucket edges live in
# conf/config.yaml because those genuinely are tunable.
# ---------------------------------------------------------------------------
LUXURY_BRANDS = frozenset(
    {"bmw", "mercedes-benz", "audi", "lexus", "acura", "infiniti",
     "cadillac", "lincoln", "volvo", "jaguar", "porsche", "tesla"}
)
RELIABLE_BRANDS = frozenset(
    {"toyota", "honda", "nissan", "mazda", "subaru", "hyundai", "kia"}
)
AMERICAN_BRANDS = frozenset(
    {"ford", "chevrolet", "gmc", "dodge", "jeep", "chrysler", "buick", "pontiac"}
)
EUROPEAN_BRANDS = frozenset(
    {"bmw", "mercedes-benz", "audi", "volkswagen", "volvo", "jaguar",
     "land rover", "mini", "porsche", "ferrari", "alfa-romeo"}
)
JAPANESE_BRANDS = frozenset(
    {"toyota", "honda", "nissan", "mazda", "subaru", "lexus", "acura", "infiniti"}
)

LUXURY_TYPES = frozenset({"convertible", "coupe", "wagon"})
UTILITY_TYPES = frozenset({"truck", "suv", "pickup"})
ECONOMY_TYPES = frozenset({"sedan", "hatchback", "mini-van"})
SPORTS_TYPES = frozenset({"coupe", "convertible"})

EXCELLENT_CONDITIONS = frozenset({"new", "like new", "excellent"})
POOR_CONDITIONS = frozenset({"poor", "salvage"})
NEEDS_WORK_CONDITIONS = frozenset({"fair", "poor", "salvage"})

ALTERNATIVE_FUELS = frozenset({"electric", "hybrid"})
PREMIUM_FUELS = frozenset({"electric", "hybrid", "diesel"})

TITLE_ISSUES = frozenset({"lien", "salvage", "rebuilt"})

# Order matters: interactions depend on features from earlier groups.
GROUP_ORDER = [
    "temporal",
    "odometer",
    "manufacturer",
    "vehicle_type",
    "condition",
    "fuel",
    "transmission",
    "title_status",
    "interactions",
]


def _clean(series: pd.Series) -> pd.Series:
    """Lowercase and strip a categorical column, preserving missingness.

    Missing stays missing. `.isin()` against a NaN yields False, which is the
    correct answer for a binary indicator: an unreported fuel type is not
    electric. Where missingness must not be silently answered -- condition --
    the value stays NaN and the imputer decides.

    Object dtype, not pandas StringDtype, is deliberate. StringDtype propagates
    pd.NA through `.isin()` and `.map()` into a masked boolean, which then
    refuses to cast to int8. Object dtype gives the plain NaN-to-False
    behaviour these indicators are written against.
    """
    return series.astype("object").str.lower().str.strip()


def _bucketise(values: pd.Series, edges: list[float]) -> pd.Series:
    """Map values to ordinal 1..len(edges)+1 by upper-inclusive edges.

    NaN in, NaN out -- unlike the source, which returned 0 for missing and so
    placed "unknown" below "low mileage" on the same ordinal scale.
    """
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    assigned = pd.Series(False, index=values.index)
    for level, edge in enumerate(edges, start=1):
        mask = (~assigned) & values.notna() & (values <= edge)
        out[mask] = level
        assigned |= mask
    out[(~assigned) & values.notna()] = len(edges) + 1
    return out


# ---------------------------------------------------------------------------
# feature groups
# ---------------------------------------------------------------------------
def temporal_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Age and era effects.

    Depreciation is steep and non-linear in the first years and flattens later,
    so a linear age term alone underfits; the squared term and the ordinal
    bucket give tree models explicit split points. The vintage flag exists
    because the curve reverses at the collectible end -- a 30-year-old vehicle
    is not simply a more depreciated 15-year-old one.

    `current_year` is 2021 because that is when the data was scraped. Using the
    calendar year would add a constant offset to every age and drift the model
    every January.
    """
    t = cfg["features"]["thresholds"]
    year = df["year"]

    age = cfg["features"]["current_year"] - year
    if cfg["features"]["clip_negative_age"]:
        # 133 listings post-date the scrape. Clipping rather than dropping keeps
        # them: they are otherwise valid, and they skew new -- i.e. toward the
        # Luxury tier, the class already hardest to predict.
        age = age.clip(lower=0)

    return {
        "vehicle_age": age,
        "age_squared": age**2,
        "is_new": (age <= t["age_new"]).astype("int8"),
        "is_old": (age >= t["age_old"]).astype("int8"),
        "is_vintage": (age >= t["age_vintage"]).astype("int8"),
        "age_category_numeric": _bucketise(age, t["age_buckets"]),
        "decade": (year // t["decade_divisor"]) * t["decade_divisor"],
        "is_2010s": year.between(2010, 2019).astype("int8"),
        "is_2000s": year.between(2000, 2009).astype("int8"),
    }


def odometer_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Mileage, absolute and relative to age.

    Raw mileage is heavily right-skewed, so log and sqrt transforms give linear
    models a usable scale. The threshold flags exist because buyer behaviour is
    discontinuous at round numbers -- 100k miles costs more value than the
    thousand miles either side of it would suggest.

    `mileage_per_year` separates a high-mileage vehicle that is simply old from
    one that has been driven hard, which raw odometer cannot distinguish.
    """
    t = cfg["features"]["thresholds"]
    odo = df["odometer"]
    age = cfg["features"]["current_year"] - df["year"]
    if cfg["features"]["clip_negative_age"]:
        age = age.clip(lower=0)

    # age is clipped at 0, so the denominator is >= 1 and this cannot divide by
    # zero. In the source it could, and did, for 133 rows.
    per_year = odo / (age + 1)

    return {
        "log_odometer": np.log1p(odo),
        "sqrt_odometer": np.sqrt(odo),
        "mileage_category_numeric": _bucketise(odo, t["mileage_buckets"]),
        "low_mileage": (odo <= t["mileage_low"]).astype("int8"),
        "high_mileage": (odo >= t["mileage_high"]).astype("int8"),
        "very_high_mileage": (odo >= t["mileage_very_high"]).astype("int8"),
        "mileage_per_year": per_year,
        "low_mileage_for_age": (per_year < t["mileage_per_year_low"]).astype("int8"),
        "high_mileage_for_age": (per_year > t["mileage_per_year_high"]).astype("int8"),
        "mileage_age_interaction": odo * age,
    }


def manufacturer_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:  # noqa: ARG001
    """Brand as market positioning rather than as 40-odd dummy columns.

    One-hot encoding the manufacturer would add high-cardinality sparse columns
    that tree models split on poorly. Collapsing to prestige, reliability and
    origin keeps the pricing signal and stays interpretable.

    Note the groups deliberately overlap: BMW is both luxury and European.
    """
    brand = _clean(df["manufacturer"])
    tier = pd.Series(1, index=df.index, dtype="int8")
    tier[brand.isin(RELIABLE_BRANDS)] = 2
    tier[brand.isin(LUXURY_BRANDS)] = 3

    return {
        "is_luxury": brand.isin(LUXURY_BRANDS).astype("int8"),
        "is_reliable": brand.isin(RELIABLE_BRANDS).astype("int8"),
        "is_american": brand.isin(AMERICAN_BRANDS).astype("int8"),
        "is_european": brand.isin(EUROPEAN_BRANDS).astype("int8"),
        "is_japanese": brand.isin(JAPANESE_BRANDS).astype("int8"),
        "brand_tier_numeric": tier,
    }


def vehicle_type_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:  # noqa: ARG001
    """Body style as a proxy for use case and buyer segment."""
    body = _clean(df["type"])
    return {
        "is_luxury_type": body.isin(LUXURY_TYPES).astype("int8"),
        "is_utility_type": body.isin(UTILITY_TYPES).astype("int8"),
        "is_economy_type": body.isin(ECONOMY_TYPES).astype("int8"),
        "is_sports_type": body.isin(SPORTS_TYPES).astype("int8"),
    }


def condition_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Seller-reported condition, with missingness preserved.

    ~40% of listings report no condition. v1 encoded those as 2, the same value
    as "fair", which is an imputation decision disguised as a feature. Here the
    ordinal stays NaN and the Pipeline's median imputer resolves it, fitted on
    training data only.

    The binary flags do answer False for missing, which is deliberate and
    different: "is not reported to be excellent" is a true statement about a
    listing with no condition field, whereas "is fair" is not.
    """
    scale = cfg["features"]["condition_scale"]
    cond = _clean(df["condition"])
    return {
        "condition_numeric": cond.map(scale).astype("float64"),
        "is_excellent_condition": cond.isin(EXCELLENT_CONDITIONS).astype("int8"),
        "is_poor_condition": cond.isin(POOR_CONDITIONS).astype("int8"),
        "needs_work": cond.isin(NEEDS_WORK_CONDITIONS).astype("int8"),
    }


def fuel_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:  # noqa: ARG001
    """Fuel type. Gas dominates, so the value is in the rare categories --
    electric and hybrid carry a price premium out of proportion to their share."""
    fuel = _clean(df["fuel"])
    return {
        "is_electric": (fuel == "electric").astype("int8"),
        "is_hybrid": (fuel == "hybrid").astype("int8"),
        "is_diesel": (fuel == "diesel").astype("int8"),
        "is_gas": (fuel == "gas").astype("int8"),
        "is_alternative_fuel": fuel.isin(ALTERNATIVE_FUELS).astype("int8"),
        "is_premium_fuel": fuel.isin(PREMIUM_FUELS).astype("int8"),
    }


def transmission_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:  # noqa: ARG001
    """Transmission. The three flags are mutually exclusive and exhaustive, so
    one is redundant by construction -- VarianceThreshold and SelectKBest in the
    Pipeline decide whether to keep it rather than this module guessing."""
    trans = _clean(df["transmission"])
    return {
        "is_manual": (trans == "manual").astype("int8"),
        "is_automatic": (trans == "automatic").astype("int8"),
        "is_other_transmission": (~trans.isin({"manual", "automatic"})).astype("int8"),
    }


def title_status_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:  # noqa: ARG001
    """Title status. A salvage or lien title is a large, legible price discount
    and one of the few fields here that is close to objective."""
    title = _clean(df["title_status"])
    return {
        "has_clean_title": (title == "clean").astype("int8"),
        "has_lien": (title == "lien").astype("int8"),
        "is_salvage": (title == "salvage").astype("int8"),
        "title_issues": title.isin(TITLE_ISSUES).astype("int8"),
    }


def interaction_features(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:  # noqa: ARG001
    """Cross-terms the tree models would otherwise need depth to discover.

    Luxury vehicles depreciate on a different curve from economy ones, so age
    and mileage mean different things depending on brand tier. Condition
    likewise matters more on an old vehicle than a new one.

    These read from columns produced by earlier groups, which is why
    `build_features` applies the groups in order.
    """
    return {
        "luxury_age_interaction": df["is_luxury"] * df["vehicle_age"],
        "luxury_mileage_interaction": df["is_luxury"] * df["mileage_per_year"],
        "condition_age_interaction": df["condition_numeric"] * df["vehicle_age"],
    }


GROUP_FUNCTIONS = {
    "temporal": temporal_features,
    "odometer": odometer_features,
    "manufacturer": manufacturer_features,
    "vehicle_type": vehicle_type_features,
    "condition": condition_features,
    "fuel": fuel_features,
    "transmission": transmission_features,
    "title_status": title_status_features,
    "interactions": interaction_features,
}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply every group in order and return (features, manifest).

    The manifest records which features each group actually produced, counted
    from the output rather than declared in a print statement.
    """
    out = df.copy()
    manifest: dict[str, list[str]] = {}

    for name in GROUP_ORDER:
        produced = GROUP_FUNCTIONS[name](out, cfg)
        for column, values in produced.items():
            out[column] = values
        manifest[name] = list(produced)

    feature_names = [c for group in GROUP_ORDER for c in manifest[group]]
    return out, {
        "groups": manifest,
        "counts": {g: len(manifest[g]) for g in GROUP_ORDER},
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }


def make_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Bin price into ordered tiers.

    Boundaries are conventional financing/segment thresholds, not learned from
    the distribution -- stated as a limitation rather than presented as a
    finding. `ordered=True` is what makes the downstream ordinal metrics (QWK,
    adjacent accuracy, tier MAE) meaningful.
    """
    t = cfg["target"]
    tier = pd.cut(
        df[t["source_column"]],
        bins=[float(b) for b in t["bins"]],
        labels=t["labels"],
        ordered=True,
    )
    return tier.rename(t["name"])


def feature_matrix(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Convenience wrapper: silver frame in, (X, y, manifest) out."""
    enriched, manifest = build_features(df, cfg)
    y = make_target(enriched, cfg)
    X = enriched[manifest["feature_names"]].astype("float64")
    return X, y, manifest
