"""Pipeline entry point.

    python scripts/run_all.py                       # every stage
    python scripts/run_all.py --stages ingest,silver
    python scripts/run_all.py --force               # ignore the stage cache
    python scripts/run_all.py --set ingest.chunk_size=25000

Stages are idempotent. Re-running with unchanged inputs and config is a no-op
that logs a cache hit rather than redoing the work.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/run_all.py` from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.ingest import build_silver, ingest_bronze  # noqa: E402
from src.validate import LineageTracker  # noqa: E402

ALL_STAGES = ["ingest", "silver"]


def set_seeds(seed: int) -> None:
    """Seed every source of randomness we control.

    Recorded here rather than scattered through the modules so that the
    reproducibility requirement has one place to audit.
    """
    random.seed(seed)
    # Legacy global seeding is deliberate here, not an oversight. scikit-learn
    # and xgboost fall back to numpy's global RandomState wherever an explicit
    # random_state is not threaded through, so seeding the modern Generator API
    # alone would leave those paths unseeded and break the byte-identical-runs
    # requirement.
    np.random.seed(seed)  # noqa: NPY002


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="conf/config.yaml")
    p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"comma-separated subset of: {', '.join(ALL_STAGES)}",
    )
    p.add_argument("--force", action="store_true", help="ignore stage caches")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override a config value; repeatable",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    set_seeds(cfg["seed"])

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(sorted(unknown))}")

    tracker = LineageTracker(cfg)
    bronze_manifest: dict | None = None

    print("=" * 72)
    print("PIPELINE")
    print("=" * 72)

    if "ingest" in stages:
        print("\n[ingest] raw CSV -> bronze Parquet")
        bronze_manifest = ingest_bronze(cfg, tracker, force=args.force)
        m = bronze_manifest
        print(
            f"  CSV {m['csv_bytes'] / 1e6:,.1f} MB -> Parquet "
            f"{m['parquet_bytes'] / 1e6:,.1f} MB "
            f"({m['compression_ratio']}x, {m['size_reduction_pct']}% smaller)"
        )

    if "silver" in stages:
        print("\n[silver] bronze -> cleaned, partitioned silver")
        sm = build_silver(cfg, tracker, force=args.force)
        print(
            f"  {sm['rows_out']:,} rows across {sm['n_partitions']} partitions "
            f"({sm['silver_bytes'] / 1e6:,.1f} MB)"
        )

    lineage_path = tracker.write(cfg["paths"]["lineage"])

    print("\n" + "=" * 72)
    print("LINEAGE")
    print("=" * 72)
    print(tracker.table())
    print(f"\nWrote {lineage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
