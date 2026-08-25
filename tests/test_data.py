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
