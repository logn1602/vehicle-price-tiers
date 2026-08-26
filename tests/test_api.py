"""API tests.

Self-contained: a small model is trained in a fixture and injected, so these
run in CI without the 1.35 GB dataset or a prior training run.

The substantive test is `test_single_request_matches_batch_prediction`. A
serving path that quietly disagrees with the training path is the classic way a
model that looked good offline behaves differently in production, and the pure
feature functions are what let us assert it does not happen here.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import ModelBundle, app, get_bundle
from src.config import load_config
from src.features import feature_matrix
from src.pipeline import build_pipeline, encode_target, fit

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(REPO / "conf" / "config.yaml")


@pytest.fixture(scope="module")
def training_frame(cfg) -> pd.DataFrame:
    """A synthetic silver-shaped frame large enough to fit a real pipeline."""
    rng = np.random.default_rng(cfg["seed"])
    n = 600
    year = rng.integers(1995, 2022, n).astype(float)
    odometer = rng.integers(1_000, 300_000, n).astype(float)
    # Price correlated with age and mileage so the model learns something.
    price = np.clip(
        60_000 - (2021 - year) * 1_800 - odometer * 0.06 + rng.normal(0, 4_000, n),
        600,
        190_000,
    )
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "price": price,
            "year": year,
            "odometer": odometer,
            "manufacturer": rng.choice(["ford", "toyota", "bmw", "tesla"], n),
            "type": rng.choice(["sedan", "suv", "coupe", "truck"], n),
            "condition": rng.choice(["good", "excellent", "fair", None], n),
            "fuel": rng.choice(["gas", "diesel", "electric", "hybrid"], n),
            "transmission": rng.choice(["automatic", "manual"], n),
            "title_status": rng.choice(["clean", "salvage", "lien"], n),
            "state": rng.choice(["ca", "tx"], n),
        }
    )


@pytest.fixture(scope="module")
def bundle(cfg, training_frame, tmp_path_factory) -> ModelBundle:
    X, y_labels, manifest = feature_matrix(training_frame, cfg)
    y, labels = encode_target(y_labels, cfg)

    pipe = build_pipeline("decision_tree", cfg, n_classes=len(labels))
    fitted = fit(pipe, "decision_tree", X, y)

    out = tmp_path_factory.mktemp("artifacts")
    joblib.dump(fitted, out / "model.joblib")
    meta = {
        "model": "decision_tree",
        "class_labels": labels,
        "feature_names": manifest["feature_names"],
        "n_features_in": manifest["n_features"],
        "config_hash": "test",
        "current_year": cfg["features"]["current_year"],
        "target_bins": [str(b) for b in cfg["target"]["bins"]],
    }
    (out / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ModelBundle(fitted, meta, cfg)


@pytest.fixture
def client(bundle) -> TestClient:
    app.dependency_overrides = {}
    get_bundle.cache_clear()
    app.dependency_overrides[__import__("src.api", fromlist=["require_bundle"]).require_bundle] = (
        lambda: bundle
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


VALID = {
    "year": 2015,
    "odometer": 78000,
    "manufacturer": "toyota",
    "condition": "good",
    "fuel": "gas",
    "transmission": "automatic",
    "title_status": "clean",
    "type": "sedan",
}


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------
def test_predict_returns_200_with_valid_payload(client, cfg):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] in cfg["target"]["labels"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_probabilities_sum_to_one(client, cfg):
    body = client.post("/predict", json=VALID).json()
    total = sum(p["probability"] for p in body["probabilities"])
    assert total == pytest.approx(1.0, abs=1e-6)
    assert len(body["probabilities"]) == len(cfg["target"]["labels"])


def test_reported_tier_is_the_argmax(client):
    body = client.post("/predict", json=VALID).json()
    best = max(body["probabilities"], key=lambda p: p["probability"])
    assert body["tier"] == best["tier"]
    assert body["confidence"] == pytest.approx(best["probability"])


def test_probability_tiers_are_in_tier_order(client, cfg):
    """Not alphabetical. v1's LabelEncoder ordering is what made its confusion
    matrix unreadable; the API must not reintroduce it."""
    body = client.post("/predict", json=VALID).json()
    assert [p["tier"] for p in body["probabilities"]] == cfg["target"]["labels"]


def test_optional_fields_may_be_omitted(client):
    """~40% of training rows report no condition; the service must accept the
    same shape of listing it was trained on."""
    minimal = {"year": 2015, "odometer": 78000}
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({}, "no fields at all"),
        ({"odometer": 78000}, "year missing"),
        ({"year": 2015}, "odometer missing"),
        ({"year": 1850, "odometer": 78000}, "year below the quality gate"),
        ({"year": 2050, "odometer": 78000}, "year above the quality gate"),
        ({"year": 2015, "odometer": -1}, "negative odometer"),
        ({"year": 2015, "odometer": 900000}, "odometer above the quality gate"),
        ({"year": "not-a-year", "odometer": 78000}, "wrong type"),
    ],
)
def test_malformed_payload_returns_422(client, payload, why):
    assert client.post("/predict", json=payload).status_code == 422, why


def test_blank_strings_are_treated_as_missing(client):
    payload = {**VALID, "condition": "   ", "manufacturer": ""}
    assert client.post("/predict", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
def test_health_reports_model_details(bundle):
    import src.api

    get_bundle.cache_clear()
    src.api.get_bundle = lambda: bundle  # type: ignore[assignment]
    try:
        with TestClient(app) as c:
            body = c.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "decision_tree"
        assert body["class_labels"] == bundle.labels
    finally:
        src.api.get_bundle = get_bundle  # type: ignore[assignment]


def test_health_is_honest_when_no_model_exists(monkeypatch):
    """Reports degraded rather than failing to boot -- an orchestrator needs a
    truthful answer, not a crashed container."""
    import src.api

    monkeypatch.setattr(src.api, "get_bundle", lambda: None)
    with TestClient(app) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_predict_returns_503_when_no_model(monkeypatch):
    import src.api

    monkeypatch.setattr(src.api, "get_bundle", lambda: None)
    app.dependency_overrides = {}
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID)
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# the serving path must agree with the training path
# ---------------------------------------------------------------------------
def test_single_request_matches_batch_prediction(client, bundle, cfg, training_frame):
    """Serve three listings one at a time and score the same three as a batch.

    Any divergence would mean the API's feature construction differs from the
    training pipeline's -- the failure mode that makes an offline metric a lie.
    """
    sample = training_frame.head(3)
    X_batch, _, _ = feature_matrix(sample, cfg)
    batch_pred = bundle.pipe.predict(X_batch.reindex(columns=bundle.feature_names))

    for i, (_, row) in enumerate(sample.iterrows()):
        payload = {
            "year": int(row["year"]),
            "odometer": float(row["odometer"]),
            "manufacturer": row["manufacturer"],
            "condition": row["condition"] if pd.notna(row["condition"]) else None,
            "fuel": row["fuel"],
            "transmission": row["transmission"],
            "title_status": row["title_status"],
            "type": row["type"],
        }
        body = client.post("/predict", json=payload).json()
        assert body["tier"] == bundle.labels[batch_pred[i]], f"row {i} diverged"
