"""Data-layer tests: schema contracts, lineage integrity, and row-count chains.

The schema tests are the important ones. They assert that deliberately corrupted
rows are *rejected*, which is the property that would have caught v1's unusable
input file at load time instead of 300 lines later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.schema import ContractViolation, raw_schema, silver_schema
from src.schema import validate as validate_schema
from src.validate import ContractFailure, LineageTracker, LineageViolation, check_contract

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(REPO / "conf" / "config.yaml")


@pytest.fixture
def good_silver_row() -> pd.DataFrame:
    """One plausible listing that satisfies every silver gate."""
    return pd.DataFrame(
        {
            "id": ["7316814884"],
            "price": [15995.0],
            "year": [2015.0],
            "odometer": [78000.0],
            "state": ["ca"],
        }
    )


# ---------------------------------------------------------------------------
# silver schema rejects corrupted rows
# ---------------------------------------------------------------------------
def test_silver_accepts_a_valid_row(cfg, good_silver_row):
    out = validate_schema(good_silver_row, silver_schema(cfg), stage="test")
    assert len(out) == 1


@pytest.mark.parametrize(
    ("column", "bad_value", "why"),
    [
        ("price", 0.0, "zero-price listings are 30,759 rows of the source"),
        ("price", -1.0, "negative price"),
        ("price", 3_736_928_711.0, "the $3.7bn outlier that survives in raw"),
        ("price", 200_001.0, "just past the configured upper bound"),
        ("year", 1899.0, "before the configured lower bound"),
        ("year", 2023.0, "after the scrape year plus tolerance"),
        ("odometer", -5.0, "negative mileage"),
        ("odometer", 500_001.0, "just past the configured upper bound"),
    ],
)
def test_silver_rejects_corrupted_value(cfg, good_silver_row, column, bad_value, why):
    corrupted = good_silver_row.copy()
    corrupted.loc[0, column] = bad_value

    with pytest.raises(ContractViolation) as exc:
        validate_schema(corrupted, silver_schema(cfg), stage="test")

    # The failure message must name the offending column and show the value,
    # not just report a count.
    assert column in str(exc.value), why
    assert str(int(bad_value)) in str(exc.value).replace(",", "") or "nullable" in str(exc.value)


def test_silver_rejects_null_required_field(cfg, good_silver_row):
    corrupted = good_silver_row.copy()
    corrupted.loc[0, "price"] = None
    with pytest.raises(ContractViolation):
        validate_schema(corrupted, silver_schema(cfg), stage="test")


def test_silver_rejects_duplicate_ids(cfg, good_silver_row):
    duped = pd.concat([good_silver_row, good_silver_row], ignore_index=True)
    with pytest.raises(ContractViolation) as exc:
        validate_schema(duped, silver_schema(cfg), stage="test")
    assert "id" in str(exc.value)


# ---------------------------------------------------------------------------
# raw schema is deliberately permissive where silver is strict
# ---------------------------------------------------------------------------
def test_raw_schema_tolerates_what_silver_rejects():
    """The $3.7bn price and a post-scrape year are real rows in the source.

    Raw must accept them -- they are the rows the pipeline exists to remove, so
    failing on them at ingest would make the file unloadable.
    """
    from src.schema import RAW_COLUMNS

    row = dict.fromkeys(RAW_COLUMNS, "x")
    row.update(
        {
            "id": "7316814884",
            "price": 3_736_928_711.0,
            "year": 2022.0,
            "odometer": 900_000.0,
            "lat": 34.0,
            "long": -118.0,
        }
    )
    df = pd.DataFrame([row])
    for col in ("price", "year", "odometer", "lat", "long"):
        df[col] = df[col].astype("float64")

    out = validate_schema(df, raw_schema(), stage="test")
    assert len(out) == 1


def test_raw_schema_rejects_column_shift():
    """A column-shifted export -- the corruption class that produced v1's file --
    puts a non-numeric value where `year` belongs and fails immediately."""
    from src.schema import RAW_COLUMNS

    row = dict.fromkeys(RAW_COLUMNS, "x")
    row.update({"id": "1", "price": 1000.0, "year": 3.0, "odometer": 1.0,
                "lat": 0.0, "long": 0.0})
    df = pd.DataFrame([row])
    for col in ("price", "year", "odometer", "lat", "long"):
        df[col] = df[col].astype("float64")

    with pytest.raises(ContractViolation) as exc:
        validate_schema(df, raw_schema(), stage="test")
    assert "year" in str(exc.value)


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------
def test_lineage_blocks_a_silent_mass_drop(cfg):
    """The v1 defect, reproduced against the guard that now prevents it."""
    tracker = LineageTracker(cfg)
    with (
        pytest.raises(LineageViolation) as exc,
        tracker.stage("v1_style_price_filter", 305_145, "price > 0") as st,
    ):
        st.rows_out = 6_067
    assert "98" in f"{(1 - 6067 / 305145) * 100:.0f}"
    assert "exceeding" in str(exc.value)


def test_lineage_allows_whitelisted_stage(cfg):
    tracker = LineageTracker(cfg)
    with tracker.stage("drop_invalid_price", 1000, "whitelisted in config") as st:
        st.rows_out = 100          # 90% drop, would normally fail
    assert tracker.records[-1].whitelisted is True


def test_lineage_requires_rows_out(cfg):
    tracker = LineageTracker(cfg)
    with (
        pytest.raises(ValueError, match="did not set rows_out"),
        tracker.stage("forgetful", 100, "reason"),
    ):
        pass


def test_contract_tolerance_is_applied_only_to_percentages():
    observed = {"a_pct": 71.77, "b_count": 101}
    with pytest.raises(ContractFailure):
        check_contract(observed, {"b_count": 100}, 0.1, "counts are exact")
    ok = check_contract(observed, {"a_pct": 71.7}, 0.1, "pcts are rounded")
    assert all(r["pass"] for r in ok)


# ---------------------------------------------------------------------------
# integration: only run when the pipeline has produced artifacts
# ---------------------------------------------------------------------------
def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.mark.slow
def test_lineage_row_counts_form_an_unbroken_chain(cfg):
    lineage = _load(REPO / cfg["paths"]["lineage"])
    if lineage is None:
        pytest.skip("run scripts/run_all.py first")

    for earlier, later in zip(lineage, lineage[1:], strict=False):
        assert earlier["rows_out"] == later["rows_in"], (
            f"gap between {earlier['stage']!r} and {later['stage']!r}: "
            f"{earlier['rows_out']:,} != {later['rows_in']:,}"
        )


@pytest.mark.slow
def test_ingest_asserted_the_raw_contract(cfg):
    manifest = _load(REPO / cfg["paths"]["bronze"] / "_manifest.json")
    if manifest is None:
        pytest.skip("run scripts/run_all.py first")

    assert manifest["stats"]["rows"] == cfg["data_contract"]["raw"]["rows"]
    failures = [c for c in manifest["contract"] if not c["pass"]]
    assert not failures, f"raw contract violations: {failures}"


@pytest.mark.slow
def test_first_lineage_stage_starts_from_the_full_file(cfg):
    lineage = _load(REPO / cfg["paths"]["lineage"])
    if lineage is None:
        pytest.skip("run scripts/run_all.py first")
    assert lineage[0]["rows_in"] == cfg["data_contract"]["raw"]["rows"]


# ---------------------------------------------------------------------------
# end-to-end ingest on a synthetic file
#
# Exercises bronze -> silver -> gold without the 1.35 GB source, so the layer
# machinery (chunking, Parquet round-trip, caching, partitioning, lineage) is
# covered by the test suite rather than only by running the real pipeline.
# ---------------------------------------------------------------------------
def _synthetic_csv(path: Path, n: int = 120) -> None:
    import numpy as np

    from src.schema import RAW_COLUMNS

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({col: ["x"] * n for col in RAW_COLUMNS})
    frame["id"] = [str(1_000_000 + i) for i in range(n)]
    frame["price"] = rng.integers(500, 90_000, n).astype(float)
    frame["year"] = rng.integers(1995, 2022, n).astype(float)
    frame["odometer"] = rng.integers(1_000, 300_000, n).astype(float)
    frame["lat"] = rng.uniform(25, 48, n)
    frame["long"] = rng.uniform(-124, -70, n)
    frame["county"] = None
    frame["state"] = rng.choice(["ca", "tx", "ny"], n)
    frame["manufacturer"] = rng.choice(["ford", "bmw", "toyota"], n)
    frame["condition"] = rng.choice(["good", "excellent", None], n)
    frame["fuel"] = rng.choice(["gas", "diesel", "electric"], n)
    frame["transmission"] = rng.choice(["automatic", "manual"], n)
    frame["title_status"] = rng.choice(["clean", "salvage"], n)
    frame["type"] = rng.choice(["sedan", "suv", "coupe"], n)

    # Rows the quality gates must remove: a zero price and a null odometer.
    frame.loc[0, "price"] = 0.0
    frame.loc[1, "odometer"] = None

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def synthetic_cfg(cfg, tmp_path) -> dict:
    """Config pointed at a throwaway tree, with the data contract relaxed to
    the synthetic file's shape. The contract machinery is tested separately in
    test_contract_tolerance_is_applied_only_to_percentages."""
    import copy

    local = copy.deepcopy(dict(cfg))
    local["paths"] = dict(local["paths"])
    local["paths"]["raw"] = str(tmp_path / "raw" / "vehicles.csv")
    for layer in ("bronze", "silver", "gold"):
        local["paths"][layer] = str(tmp_path / layer)
    local["paths"]["results"] = str(tmp_path / "results")
    local["ingest"] = dict(local["ingest"])
    local["ingest"]["chunk_size"] = 50
    local["data_contract"] = copy.deepcopy(local["data_contract"])
    local["data_contract"]["raw"] = {"columns": 26}
    _synthetic_csv(Path(local["paths"]["raw"]))
    return local


def test_end_to_end_ingest_produces_all_three_layers(synthetic_cfg):
    from src.config import Config
    from src.ingest import build_gold, build_silver, ingest_bronze

    local = Config(synthetic_cfg)
    tracker = LineageTracker(local)

    bronze = ingest_bronze(local, tracker)
    assert bronze["stats"]["rows"] == 120
    assert bronze["parquet_bytes"] > 0

    silver = build_silver(local, tracker)
    # The zero-price row and the null-odometer row must both be gone.
    assert silver["rows_out"] == 118
    assert silver["n_partitions"] >= 1

    gold = build_gold(local, tracker)
    assert gold["rows_out"] == 118
    assert gold["n_features"] == sum(gold["feature_groups"].values())

    stages = [r.stage for r in tracker.records]
    assert stages[0] == "ingest_bronze"
    assert "build_gold" in stages


def test_ingest_is_idempotent_and_replays_lineage(synthetic_cfg):
    """A cached run must produce the same lineage as the run that built it,
    otherwise results/lineage.json depends on cache state rather than on data."""
    from src.config import Config
    from src.ingest import build_gold, build_silver, ingest_bronze

    local = Config(synthetic_cfg)

    first = LineageTracker(local)
    ingest_bronze(local, first)
    build_silver(local, first)
    build_gold(local, first)

    second = LineageTracker(local)
    ingest_bronze(local, second)
    build_silver(local, second)
    build_gold(local, second)

    assert first.as_list() == second.as_list()
