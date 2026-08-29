"""Phase 1 follow-up: explain the two open questions from diagnose_v1_load.py.

Question 1 -- why do seven of the brief's section-3 facts miss?
    Every miss is in the same direction (our counts are higher). Hypothesis:
    the brief's statistics were computed on the 421,344-row frame that remains
    after dropping null `year`/`odometer`, not on the raw 426,880-row file.

Question 2 -- what actually reduced v1 to 6,067 rows?
    v1's price filter is not the cause: on the correct file it retains 393,861
    rows (92.3%). Hypothesis: a listwise dropna() across columns that are
    40-72% null collapses the frame to roughly that order of magnitude.

Both are tested chunk-wise; nothing holds the full frame.

Run:
    python -u scripts/diagnose_v1_rowloss.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/vehicles.csv")
OUT = Path("results/v1_rowloss_diagnosis.json")
CHUNK = 50_000

V1_REPORTED_MODEL_ROWS = 6_067

NUMERIC_COLS = {"price": "float64", "year": "float64", "odometer": "float64"}

# The brief's section-3 figures, as stated. Percentages are quoted rounded to
# one decimal; counts are exact.
BRIEF_FACTS = {
    "size_null_pct": 71.6,
    "cylinders_null_pct": 41.5,
    "condition_null_pct": 40.5,
    "VIN_null_pct": 37.8,
    "paint_color_null_pct": 30.4,
    "price_eq_0": 30_759,
    "price_gt_100k": 647,
    "odometer_gt_500k": 1_385,
}


def reader(path: Path, columns: list[str]):
    dtypes: dict[str, str] = {c: "str" for c in columns}
    dtypes.update(NUMERIC_COLS)
    return pd.read_csv(path, dtype=dtypes, chunksize=CHUNK)


def header_of(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def main() -> None:
    columns = header_of(RAW)
    null_cols = ["size", "cylinders", "condition", "VIN", "paint_color"]

    # --- accumulators -------------------------------------------------------
    post_rows = 0
    post_nulls = dict.fromkeys(null_cols, 0)
    post_price_eq_0 = 0
    post_price_gt_100k = 0
    post_odo_gt_500k = 0

    # v1 collapse hypotheses
    dropna_all = 0
    dropna_no_county = 0
    dropna_no_county_then_price = 0
    price_then_dropna_no_county = 0

    non_county = [c for c in columns if c != "county"]

    print("=" * 72)
    print("PHASE 1 FOLLOW-UP: row-loss root cause")
    print("=" * 72)
    print("\nScanning...")

    for i, chunk in enumerate(reader(RAW, columns), 1):
        # --- Q1: statistics on the post-dropna frame ------------------------
        post = chunk.dropna(subset=["year", "odometer"])
        post_rows += len(post)
        for col in null_cols:
            post_nulls[col] += int(post[col].isna().sum())
        post_price_eq_0 += int((post["price"] == 0).sum())
        post_price_gt_100k += int((post["price"] > 100_000).sum())
        post_odo_gt_500k += int((post["odometer"] > 500_000).sum())

        # --- Q2: listwise dropna variants -----------------------------------
        dropna_all += int(len(chunk.dropna()))
        no_county = chunk[non_county]
        kept = no_county.dropna()
        dropna_no_county += len(kept)
        dropna_no_county_then_price += int(
            ((kept["price"] > 0) & (kept["price"] <= 200_000)).sum()
        )
        priced = no_county[(no_county["price"] > 0) & (no_county["price"] <= 200_000)]
        price_then_dropna_no_county += int(len(priced.dropna()))

        print(f"      chunk {i:>3}  scanned: {i * CHUNK:>9,}")
        del chunk, post, no_county, kept, priced

    # --- Q1 report ----------------------------------------------------------
    def pct(col: str) -> float:
        return round(post_nulls[col] / post_rows * 100, 2)

    q1 = {
        "denominator_rows": post_rows,
        "checks": [],
    }
    for name, expected in BRIEF_FACTS.items():
        if name.endswith("_null_pct"):
            actual = pct(name.replace("_null_pct", ""))
            tol = 0.05
        elif name == "price_eq_0":
            actual, tol = post_price_eq_0, 0
        elif name == "price_gt_100k":
            actual, tol = post_price_gt_100k, 0
        else:
            actual, tol = post_odo_gt_500k, 0
        q1["checks"].append(
            {
                "fact": name,
                "actual_post_dropna": actual,
                "brief_expected": expected,
                "pass": bool(abs(actual - expected) <= tol),
            }
        )
    q1["all_pass"] = all(c["pass"] for c in q1["checks"])

    print("\n[Q1] Do the brief's figures match the POST-dropna frame?")
    print(f"      denominator: {post_rows:,} rows")
    for c in q1["checks"]:
        flag = "PASS" if c["pass"] else "FAIL"
        print(
            f"      [{flag}] {c['fact']:<24} "
            f"post-dropna={c['actual_post_dropna']!s:<12} brief={c['brief_expected']}"
        )

    # --- Q2 report ----------------------------------------------------------
    q2 = {
        "v1_reported_model_rows": V1_REPORTED_MODEL_ROWS,
        "dropna_all_26_columns": dropna_all,
        "dropna_excluding_county": dropna_no_county,
        "dropna_excluding_county_then_price_filter": dropna_no_county_then_price,
        "price_filter_then_dropna_excluding_county": price_then_dropna_no_county,
    }

    print("\n[Q2] What collapses the frame to v1's order of magnitude?")
    for k, v in q2.items():
        if k == "v1_reported_model_rows":
            continue
        ratio = v / V1_REPORTED_MODEL_ROWS if V1_REPORTED_MODEL_ROWS else 0
        print(f"      {k:<46} {v:>9,}  ({ratio:>6.1f}x v1)")

    report = {"question_1_stats_denominator": q1, "question_2_row_collapse": q2}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
