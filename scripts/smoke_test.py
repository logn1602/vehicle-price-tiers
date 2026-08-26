"""End-to-end smoke test on a synthetic 5,000-row dataset.

CI has no access to the 1.35 GB Kaggle file, so this generates a file with the
same schema and runs every stage against it: chunked ingest, schema validation,
quality gates, lineage, feature engineering, split, and a full model fit.

It is a wiring check, not a quality check. The metrics it produces are
meaningless -- the data is random -- and nothing here is ever published. What it
catches is the class of breakage that only shows up when the stages run
together: a renamed config key, a column dropped in the wrong layer, a
transform that no longer composes.

Must finish well inside two minutes.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.ingest import build_gold, build_silver, ingest_bronze  # noqa: E402
from src.schema import RAW_COLUMNS  # noqa: E402
from src.train import train_all  # noqa: E402
from src.validate import LineageTracker  # noqa: E402

BUDGET_SECONDS = 120


def synthetic_csv(path: Path, n: int, seed: int) -> None:
    """A file with the real schema and plausible ranges, but random content."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({col: ["x"] * n for col in RAW_COLUMNS})

    year = rng.integers(1995, 2022, n).astype(float)
    odometer = rng.integers(1_000, 320_000, n).astype(float)
    # Correlate price with age and mileage so the models have signal to find and
    # a degenerate single-class target cannot occur.
    price = np.clip(
        70_000 - (2021 - year) * 1_900 - odometer * 0.07 + rng.normal(0, 5_000, n),
        400,
        195_000,
    )

    frame["id"] = [str(9_000_000 + i) for i in range(n)]
    frame["price"] = price
    frame["year"] = year
    frame["odometer"] = odometer
    frame["lat"] = rng.uniform(25, 48, n)
    frame["long"] = rng.uniform(-124, -70, n)
    frame["county"] = None
    frame["state"] = rng.choice(["ca", "tx", "ny", "fl"], n)
    frame["manufacturer"] = rng.choice(["ford", "toyota", "bmw", "tesla", "honda"], n)
    frame["type"] = rng.choice(["sedan", "suv", "coupe", "truck", None], n)
    frame["condition"] = rng.choice(["good", "excellent", "fair", None], n)
    frame["fuel"] = rng.choice(["gas", "diesel", "electric", "hybrid"], n)
    frame["transmission"] = rng.choice(["automatic", "manual", None], n)
    frame["title_status"] = rng.choice(["clean", "salvage", "lien"], n)

    # Rows the quality gates must remove.
    frame.loc[0, "price"] = 0.0
    frame.loc[1, "odometer"] = None
    frame.loc[2, "year"] = None

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def smoke_config(base: dict, root: Path, raw: Path) -> Config:
    """Point every path at a throwaway tree and relax the row-count contract.

    Only the row count is relaxed. Column names, dtypes, ranges, uniqueness and
    the lineage drop threshold are all still enforced -- those are the parts
    worth exercising.
    """
    cfg = copy.deepcopy(dict(base))
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["raw"] = str(raw)
    for layer in ("bronze", "silver", "gold"):
        cfg["paths"][layer] = str(root / layer)
    cfg["paths"]["results"] = str(root / "results")
    cfg["paths"]["figures"] = str(root / "results" / "figures")
    cfg["paths"]["artifacts"] = str(root / "artifacts")
    cfg["paths"]["metrics"] = str(root / "results" / "metrics.json")
    cfg["paths"]["lineage"] = str(root / "results" / "lineage.json")

    cfg["data_contract"] = copy.deepcopy(cfg["data_contract"])
    cfg["data_contract"]["raw"] = {"columns": 26}

    cfg["ingest"] = dict(cfg["ingest"])
    cfg["ingest"]["chunk_size"] = 1_000
    return Config(cfg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5_000)
    p.add_argument("--budget", type=float, default=BUDGET_SECONDS)
    args = p.parse_args(argv)

    started = time.perf_counter()
    base = load_config()

    with tempfile.TemporaryDirectory(prefix="vpt-smoke-") as tmp:
        root = Path(tmp)
        raw = root / "raw" / "vehicles.csv"

        print(f"generating {args.rows:,} synthetic rows ...")
        synthetic_csv(raw, args.rows, base["seed"])
        cfg = smoke_config(base, root, raw)

        tracker = LineageTracker(cfg)
        print("ingest ...")
        bronze = ingest_bronze(cfg, tracker)
        print("silver ...")
        silver = build_silver(cfg, tracker)
        print("gold ...")
        gold = build_gold(cfg, tracker)

        print("train ...")
        payload = train_all(cfg, include_cv=False, track=False)

        elapsed = time.perf_counter() - started

        print("\n" + tracker.table())
        print(
            f"\nbronze {bronze['stats']['rows']:,} rows"
            f" | silver {silver['rows_out']:,}"
            f" | gold {gold['rows_out']:,} x {gold['n_features']} features"
        )
        print(f"models trained: {len(payload['models'])}")
        print(f"best: {payload['selection']['best_model']}")

        # Assertions -- the point of the exercise.
        assert bronze["stats"]["rows"] == args.rows, "bronze lost rows"
        assert silver["rows_out"] < bronze["stats"]["rows"], "quality gates removed nothing"
        assert gold["n_features"] == sum(gold["feature_groups"].values())
        assert len(payload["models"]) == 5, "not every model trained"
        assert payload["baseline"]["accuracy"] > 0
        for model in payload["models"]:
            assert 0.0 <= model["macro_f1"] <= 1.0
            assert model["n_test"] > 0
            assert all(c["support"] >= 0 for c in model["per_class"])

        print(f"\nsmoke test passed in {elapsed:.1f}s (budget {args.budget:.0f}s)")
        if elapsed > args.budget:
            print(f"FAIL: exceeded the {args.budget:.0f}s budget")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
