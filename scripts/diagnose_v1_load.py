"""Phase 1 diagnostic: verify the raw CSV and root-cause the v1 row loss.

This script answers three questions before any pipeline code is written:

1. Is `data/raw/vehicles.csv` the real file?  (426,880 x 26)
2. Do the known data facts from the brief hold, exactly?
3. What parse configuration could turn 426,880 rows into the 305,145 that
   v1 reported loading?

Memory note
-----------
The raw file is ~1.35 GB and the `description` column is long free text. Loading
all 26 columns into a single frame exhausts memory on an 8 GB machine -- the
first version of this script did exactly that and died with an ArrayMemoryError.
Everything below is therefore computed chunk-wise: we hold accumulators, never
the full frame. This is the same constraint that makes the bronze Parquet layer
necessary rather than decorative.

Nothing here is hand-typed downstream: every result is written to
`results/v1_load_diagnosis.json` and read from there by later stages.

Run:
    python -u scripts/diagnose_v1_load.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/vehicles.csv")
OUT = Path("results/v1_load_diagnosis.json")

# Rows small enough that one chunk stays well inside available memory.
CHUNK = 50_000

# Row count v1 reported at load time, from the notebook output cell.
V1_REPORTED_ROWS = 305_145
# Row count v1 reported after its price filter.
V1_REPORTED_MODEL_ROWS = 6_067

# Columns that make `df.duplicated()` meaningless: unique by construction.
IDENTITY_COLS = ["id", "url", "image_url"]

# Explicit dtypes for the columns we reason about numerically. Everything else
# is read as str so pandas never infers a type from a sample.
NUMERIC_COLS = {"price": "float64", "year": "float64", "odometer": "float64"}


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}s"


def count_physical_lines(path: Path) -> int:
    """Count b'\\n' occurrences without decoding. Cheap record-count check."""
    total = 0
    with path.open("rb") as fh:
        while chunk := fh.read(8 << 20):
            total += chunk.count(b"\n")
    return total


def header_of(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


class Accumulator:
    """Streaming statistics over the raw file. Holds O(rows) hashes, not rows."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.payload_cols = [c for c in columns if c not in IDENTITY_COLS]
        self.rows = 0
        self.nulls = dict.fromkeys(columns, 0)
        self.price_eq_0 = 0
        self.price_gt_100k = 0
        self.price_max = float("-inf")
        self.price_in_v1_filter = 0
        self.year_gt_2021 = 0
        self.odometer_gt_500k = 0
        self.year_and_odometer_present = 0
        self.row_hashes: list[np.ndarray] = []
        self.id_hashes: list[np.ndarray] = []

    def update(self, chunk: pd.DataFrame) -> None:
        self.rows += len(chunk)

        for col in self.columns:
            self.nulls[col] += int(chunk[col].isna().sum())

        price = chunk["price"]
        self.price_eq_0 += int((price == 0).sum())
        self.price_gt_100k += int((price > 100_000).sum())
        chunk_max = float(price.max())
        if chunk_max > self.price_max:
            self.price_max = chunk_max
        self.price_in_v1_filter += int(((price > 0) & (price <= 200_000)).sum())

        self.year_gt_2021 += int((chunk["year"] > 2021).sum())
        self.odometer_gt_500k += int((chunk["odometer"] > 500_000).sum())
        self.year_and_odometer_present += int(
            (chunk["year"].notna() & chunk["odometer"].notna()).sum()
        )

        # Content hashes let us find true duplicates across chunk boundaries
        # without ever materialising the whole frame.
        self.row_hashes.append(
            pd.util.hash_pandas_object(chunk[self.payload_cols], index=False).to_numpy()
        )
        self.id_hashes.append(
            pd.util.hash_pandas_object(chunk[["id"]], index=False).to_numpy()
        )

    def duplicates(self) -> dict:
        rows = pd.Series(np.concatenate(self.row_hashes))
        ids = pd.Series(np.concatenate(self.id_hashes))
        return {
            "duplicates_excluding_identity": int(rows.duplicated().sum()),
            "identity_columns_excluded": IDENTITY_COLS,
            "duplicate_ids": int(ids.duplicated().sum()),
            "note": (
                "v1 ran df.duplicated() across all 26 columns including id/url/"
                "image_url, which returns 0 by construction because id is unique."
            ),
        }

    def facts(self) -> list[dict]:
        """Every known fact from the brief, section 3.

        Percentages carry a 0.1pp tolerance because the brief quotes them
        rounded. Counts are compared exactly.
        """
        n = self.rows

        def null_pct(col: str) -> float:
            return round(self.nulls[col] / n * 100, 2)

        checks: list[tuple[str, float, float, float]] = [
            ("rows", n, 426_880, 0),
            ("columns", len(self.columns), 26, 0),
            ("county_null_pct", null_pct("county"), 100.0, 0.1),
            ("size_null_pct", null_pct("size"), 71.6, 0.1),
            ("cylinders_null_pct", null_pct("cylinders"), 41.5, 0.1),
            ("condition_null_pct", null_pct("condition"), 40.5, 0.1),
            ("VIN_null_pct", null_pct("VIN"), 37.8, 0.1),
            ("paint_color_null_pct", null_pct("paint_color"), 30.4, 0.1),
            ("year_null", self.nulls["year"], 1_205, 0),
            ("odometer_null", self.nulls["odometer"], 4_400, 0),
            ("rows_after_year_odometer_dropna", self.year_and_odometer_present, 421_344, 0),
            ("price_eq_0", self.price_eq_0, 30_759, 0),
            ("price_gt_100k", self.price_gt_100k, 647, 0),
            ("price_max", self.price_max, 3_736_928_711.0, 0),
            ("year_gt_2021", self.year_gt_2021, 133, 0),
            ("odometer_gt_500k", self.odometer_gt_500k, 1_385, 0),
        ]

        return [
            {
                "fact": name,
                "actual": actual,
                "expected": expected,
                "pass": bool(abs(actual - expected) <= tol),
            }
            for name, actual, expected, tol in checks
        ]

    def v1_filter(self) -> dict:
        survivors = self.price_in_v1_filter
        return {
            "filter": "(price > 0) & (price <= 200000)",
            "rows_in": self.rows,
            "rows_out": survivors,
            "pct_dropped": round((1 - survivors / self.rows) * 100, 2),
            "v1_reported_rows_out": V1_REPORTED_MODEL_ROWS,
            "ratio_vs_v1": round(survivors / V1_REPORTED_MODEL_ROWS, 1),
        }


def scan_reference(path: Path, columns: list[str]) -> Accumulator:
    dtypes: dict[str, str] = {c: "str" for c in columns}
    dtypes.update(NUMERIC_COLS)

    acc = Accumulator(columns)
    reader = pd.read_csv(path, dtype=dtypes, chunksize=CHUNK)
    for i, chunk in enumerate(reader, 1):
        acc.update(chunk)
        print(f"      chunk {i:>3}  rows so far: {acc.rows:>9,}")
        del chunk
    return acc


def count_rows_with(path: Path, **kwargs) -> int:
    """Row count under an alternative parse config, chunk-wise."""
    total = 0
    reader = pd.read_csv(path, chunksize=CHUNK, **kwargs)
    for chunk in reader:
        total += len(chunk)
        del chunk
    return total


def misparse_hypotheses(path: Path) -> list[dict]:
    """Parse configurations that could plausibly yield 305,145 rows.

    Each is a hypothesis about how v1's file came to be, not a random sweep.
    Note the physical-line count already rules out embedded newlines as a
    cause, so these test field-level rather than record-level breakage.
    """
    hypotheses = [
        {
            "name": "quote_none",
            "why": "quoting ignored, commas inside description explode the field count",
            "kwargs": {"quoting": csv.QUOTE_NONE, "on_bad_lines": "skip", "usecols": [0]},
        },
        {
            "name": "skip_bad_lines",
            "why": "old pandas error_bad_lines=False silently dropped malformed rows",
            "kwargs": {"on_bad_lines": "skip", "usecols": [0]},
        },
        {
            "name": "python_engine_skip",
            "why": "engine='python' parses quoting differently from the C engine",
            "kwargs": {"engine": "python", "on_bad_lines": "skip", "usecols": [0]},
        },
        {
            "name": "escapechar_backslash",
            "why": "backslash-escaped quotes in description mis-terminate fields",
            "kwargs": {"escapechar": "\\", "on_bad_lines": "skip", "usecols": [0]},
        },
    ]

    out = []
    for h in hypotheses:
        started = time.perf_counter()
        record = {"name": h["name"], "why": h["why"], "kwargs": str(h["kwargs"])}
        try:
            rows = count_rows_with(path, **h["kwargs"])
            record["rows"] = rows
            record["matches_v1"] = bool(rows == V1_REPORTED_ROWS)
            record["delta_vs_v1"] = rows - V1_REPORTED_ROWS
        except (ValueError, MemoryError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["matches_v1"] = False
        record["duration_s"] = round(time.perf_counter() - started, 1)
        print(
            f"      {record['name']:<22} "
            f"rows={record.get('rows', 'ERROR')!s:<12} "
            f"match={record['matches_v1']}  ({_fmt(record['duration_s'])})"
        )
        out.append(record)
    return out


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. See data/README.md for Kaggle download steps."
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"source": str(RAW), "size_bytes": RAW.stat().st_size, "chunk_size": CHUNK}

    print("=" * 72)
    print("PHASE 1 DIAGNOSTIC")
    print("=" * 72)

    print("\n[1/5] Counting physical lines...")
    t0 = time.perf_counter()
    lines = count_physical_lines(RAW)
    report["physical_newlines"] = lines
    print(f"      {lines:,} newline bytes  ({_fmt(time.perf_counter() - t0)})")

    columns = header_of(RAW)
    report["columns"] = columns

    print("\n[2/5] Streaming reference scan (explicit dtypes, RFC-4180 quoting)...")
    t0 = time.perf_counter()
    acc = scan_reference(RAW, columns)
    print(f"      rows={acc.rows:,}  cols={len(columns)}  ({_fmt(time.perf_counter() - t0)})")
    report["reference_shape"] = [acc.rows, len(columns)]
    report["embedded_newlines_present"] = bool(lines - 1 != acc.rows)

    print("\n[3/5] Known-fact assertions (brief section 3)...")
    facts = acc.facts()
    report["facts"] = facts
    for f in facts:
        flag = "PASS" if f["pass"] else "FAIL"
        print(f"      [{flag}] {f['fact']:<34} actual={f['actual']!s:<16} expected={f['expected']}")
    report["facts_all_pass"] = all(f["pass"] for f in facts)

    print("\n[4/5] Duplicates and v1 price filter...")
    report["duplicates"] = acc.duplicates()
    for k, v in report["duplicates"].items():
        if k != "note":
            print(f"      {k}: {v}")
    report["v1_filter"] = acc.v1_filter()
    print(
        f"      v1 filter on the correct file -> {report['v1_filter']['rows_out']:,} rows "
        f"({report['v1_filter']['ratio_vs_v1']}x v1's {V1_REPORTED_MODEL_ROWS:,})"
    )

    del acc

    # Checkpoint before the slow passes: steps 1-4 are the load-bearing results
    # and should survive even if the hypothesis sweep is interrupted.
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"      checkpointed -> {OUT}")

    print("\n[5/5] Misparse hypotheses (target: 305,145 rows)...")
    report["misparse_hypotheses"] = misparse_hypotheses(RAW)
    report["misparse_root_cause_found"] = any(
        h.get("matches_v1") for h in report["misparse_hypotheses"]
    )

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
