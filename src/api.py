"""FastAPI inference service for the price-tier classifier.

Two endpoints: `/health` for liveness and model provenance, `/predict` for
classification. A prediction returns the tier *and* the full probability
distribution across all four tiers, because a listing the model places at
0.41 Premium / 0.39 Luxury is a materially different answer from one it places
at 0.95 Premium, and returning only the argmax discards that.

The service reuses `src.features` unchanged. That is the point of those
functions being pure: the transform applied to a single incoming listing is
provably the same code that produced the training data, and a test asserts
single-row output equals batch output.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.config import load_config
from src.features import build_features

app = FastAPI(
    title="Vehicle Price Tier Classifier",
    description=(
        "Classifies used-vehicle listings into four ordinal price tiers. "
        "Trained on Craigslist asking prices, which are not transaction prices."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
class Listing(BaseModel):
    """One vehicle listing.

    Only `year` and `odometer` are required: the data layer drops rows missing
    either, so the model has never seen such a listing and should refuse rather
    than guess. Everything else is genuinely optional -- ~40% of training rows
    report no condition, and the pipeline's imputer is fitted to handle that.
    """

    year: int = Field(..., ge=1900, le=2022, examples=[2015])
    odometer: float = Field(..., ge=0, le=500_000, examples=[78_000])
    manufacturer: str | None = Field(None, examples=["toyota"])
    condition: str | None = Field(None, examples=["good"])
    fuel: str | None = Field(None, examples=["gas"])
    transmission: str | None = Field(None, examples=["automatic"])
    title_status: str | None = Field(None, examples=["clean"])
    type: str | None = Field(None, examples=["sedan"])

    @field_validator("manufacturer", "condition", "fuel", "transmission",
                     "title_status", "type")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        """An empty string is missing data, not a category called ''."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class TierProbability(BaseModel):
    tier: str
    probability: float = Field(..., ge=0.0, le=1.0)


class Prediction(BaseModel):
    tier: str = Field(..., description="Most likely price tier")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[TierProbability]
    model_name: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    n_features: int | None = None
    class_labels: list[str] | None = None
    config_hash: str | None = None


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
class ModelBundle:
    """The fitted pipeline plus the metadata needed to use it correctly."""

    def __init__(self, pipe: Any, meta: dict, cfg: dict) -> None:
        self.pipe = pipe
        self.meta = meta
        self.cfg = cfg

    @property
    def labels(self) -> list[str]:
        return self.meta["class_labels"]

    @property
    def feature_names(self) -> list[str]:
        return self.meta["feature_names"]


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle | None:
    """Load the persisted model once per process.

    Returns None rather than raising when no model exists, so `/health` can
    report the situation honestly instead of the whole service failing to boot.
    """
    import joblib

    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    model_path = artifacts / "model.joblib"
    meta_path = artifacts / "model_meta.json"

    if not (model_path.exists() and meta_path.exists()):
        return None

    return ModelBundle(
        joblib.load(model_path),
        json.loads(meta_path.read_text(encoding="utf-8")),
        cfg,
    )


def require_bundle() -> ModelBundle:
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model available. Run `python scripts/run_all.py` "
                "to produce artifacts/model.joblib."
            ),
        )
    return bundle


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=Health)
def health() -> Health:
    bundle = get_bundle()
    if bundle is None:
        return Health(status="degraded", model_loaded=False)
    return Health(
        status="ok",
        model_loaded=True,
        model_name=bundle.meta["model"],
        n_features=bundle.meta["n_features_in"],
        class_labels=bundle.labels,
        config_hash=bundle.meta["config_hash"],
    )


@app.post("/predict", response_model=Prediction)
def predict(
    listing: Listing,
    bundle: Annotated[ModelBundle, Depends(require_bundle)],
) -> Prediction:
    frame = pd.DataFrame([listing.model_dump()])
    enriched, manifest = build_features(frame, bundle.cfg)

    # Reindex to the exact training feature order. Without this a schema change
    # would silently misalign columns and the model would return confident
    # nonsense rather than an error.
    X = enriched.reindex(columns=bundle.feature_names).astype("float64")
    missing = [c for c in bundle.feature_names if c not in enriched.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature pipeline did not produce required columns: {missing}",
        )

    proba = bundle.pipe.predict_proba(X)[0]
    labels = bundle.labels
    _ = manifest  # feature manifest unused at serve time; kept for parity

    # Probabilities are returned in TIER order, so a caller can read the
    # distribution as an ordinal curve rather than having to re-sort it.
    distribution = [
        TierProbability(tier=label, probability=float(p))
        for label, p in zip(labels, proba, strict=True)
    ]
    best = max(distribution, key=lambda t: t.probability)

    return Prediction(
        tier=best.tier,
        confidence=best.probability,
        probabilities=distribution,
        model_name=bundle.meta["model"],
    )
