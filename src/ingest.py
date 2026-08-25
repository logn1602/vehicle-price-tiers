"""Ingestion: raw CSV -> bronze Parquet -> silver Parquet.

Bronze is a faithful, typed copy of the source: every row, every column,
nothing filtered. Keeping `description` makes it large, but it also makes the
CSV-to-Parquet size comparison honest -- dropping the biggest column before
measuring compression would flatter the number.

Silver is bronze cleaned: rows with unusable `year`/`odometer`/`price` removed,
free-text and URL columns dropped, partitioned by state.

Everything streams. The raw file is ~1.35 GB and a full-frame load exhausts
memory on an 8 GB machine, so each stage reads in chunks and holds only
accumulators. Stages are idempotent: each writes a manifest recording the hash
of its inputs and config, and re-running with unchanged inputs is a no-op.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import config_hash, file_hash
from src.features import feature_matrix
from src.schema import RAW_COLUMNS, RAW_DTYPES, raw_schema, silver_schema
from src.schema import validate as validate_schema
from src.validate import LineageTracker, check_contract

MANIFEST = "_manifest.json"
CACHE_VERSION = 1

_ARROW_TYPES = {"str": pa.string(), "float64": pa.float64()}


def arrow_schema() -> pa.Schema:
    """Explicit Arrow schema so every chunk writes identical types.

    Without this, a chunk in which some column happens to be entirely null can
    be inferred as a different type from its neighbours, and the resulting
    Parquet file has an inconsistent schema across row groups.
    """
    return pa.schema(
        [pa.field(name, _ARROW_TYPES[dtype]) for name, dtype in RAW_DTYPES.items()]
    )


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def _manifest_path(layer_dir: Path) -> Path:
    return layer_dir / MANIFEST


def read_manifest(layer_dir: Path) -> dict | None:
    path = _manifest_path(layer_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A truncated manifest means an interrupted write; treat as a cache miss
        # rather than trusting it.
        return None


def write_manifest(layer_dir: Path, payload: dict) -> None:
    _manifest_path(layer_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_fresh(layer_dir: Path, cache_key: dict) -> bool:
    manifest = read_manifest(layer_dir)
    if manifest is None:
        return False
    return all(manifest.get(k) == v for k, v in cache_key.items())


# ---------------------------------------------------------------------------
# bronze
# ---------------------------------------------------------------------------
class _RawStats:
    """Streaming statistics for the raw data contract."""

    def __init__(self) -> None:
        self.rows = 0
        self.nulls: dict[str, int] = dict.fromkeys(RAW_COLUMNS, 0)
        self.year_gt_2021 = 0
        self.price_max = float("-inf")
        self._row_hashes: list[np.ndarray] = []
        self._id_hashes: list[np.ndarray] = []

    def update(self, chunk: pd.DataFrame, identity_cols: list[str]) -> None:
        self.rows += len(chunk)
        for col in RAW_COLUMNS:
            self.nulls[col] += int(chunk[col].isna().sum())
        self.year_gt_2021 += int((chunk["year"] > 2021).sum())
        chunk_max = float(chunk["price"].max())
        if chunk_max > self.price_max:
            self.price_max = chunk_max

        payload = [c for c in chunk.columns if c not in identity_cols]
        self._row_hashes.append(
            pd.util.hash_pandas_object(chunk[payload], index=False).to_numpy()
        )
        self._id_hashes.append(
            pd.util.hash_pandas_object(chunk[["id"]], index=False).to_numpy()
        )

    def observed(self) -> dict[str, Any]:
        rows = pd.Series(np.concatenate(self._row_hashes))
        ids = pd.Series(np.concatenate(self._id_hashes))
        return {
            "rows": self.rows,
            "columns": len(RAW_COLUMNS),
            "county_null_pct": round(self.nulls["county"] / self.rows * 100, 2),
            "year_null": self.nulls["year"],
            "odometer_null": self.nulls["odometer"],
            "year_gt_2021": self.year_gt_2021,
            "price_max": self.price_max,
            "duplicates_excluding_identity": int(rows.duplicated().sum()),
            "duplicate_ids": int(ids.duplicated().sum()),
        }


def ingest_bronze(
    cfg: dict,
    tracker: LineageTracker,
    force: bool = False,
) -> dict[str, Any]:
    """Stream the raw CSV into typed Parquet and assert the raw data contract."""
    raw_path = Path(cfg["paths"]["raw"])
    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)
    dst = bronze_dir / "vehicles.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. See data/README.md for download instructions."
        )

    print(f"  hashing {raw_path.name} ...")
    src_hash = file_hash(raw_path)
    cache_key = {
        "version": CACHE_VERSION,
        "input_hash": src_hash,
        "config_hash": config_hash(
            {"ingest": cfg["ingest"], "data_contract": cfg["data_contract"]}
        ),
    }

    if dst.exists() and is_fresh(bronze_dir, cache_key) and not force:
        manifest = read_manifest(bronze_dir)
        print(f"  cache hit -- reusing {dst}")
        tracker.replay(manifest["lineage"])
        return manifest

    identity_cols = cfg["data_contract"]["identity_columns"]
    stats = _RawStats()
    schema = arrow_schema()
    writer: pq.ParquetWriter | None = None
    chunk_size = cfg["ingest"]["chunk_size"]

    try:
        reader = pd.read_csv(raw_path, dtype=RAW_DTYPES, chunksize=chunk_size)
        for i, chunk in enumerate(reader, 1):
            # Structural check on the first chunk only. Column names, dtypes and
            # coarse ranges are properties of the file, not of individual rows,
            # so one chunk is enough to catch a shifted or re-exported CSV -- the
            # corruption class that produced v1's unusable input. Running it on
            # every chunk would triple ingest time for no additional signal;
            # global uniqueness is covered by `duplicate_ids` in the contract.
            if i == 1:
                validate_schema(chunk, raw_schema(), stage="bronze:first_chunk")

            stats.update(chunk, identity_cols)
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    dst, schema, compression=cfg["ingest"]["compression"]
                )
            writer.write_table(table)
            print(f"    chunk {i:>3}  rows: {stats.rows:>9,}")
            del chunk, table
    finally:
        if writer is not None:
            writer.close()

    observed = stats.observed()

    # Contract assertion. A wrong or partial file fails here, loudly, before any
    # modelling code sees it.
    contract = check_contract(
        observed,
        cfg["data_contract"]["raw"],
        cfg["data_contract"]["pct_tolerance"],
        label="Raw data contract",
    )

    csv_bytes = raw_path.stat().st_size
    parquet_bytes = dst.stat().st_size
    compression_ratio = round(csv_bytes / parquet_bytes, 2)

    first_record = len(tracker.records)
    with tracker.stage(
        "ingest_bronze", observed["rows"], "typed copy of source; no rows filtered"
    ) as st:
        st.rows_out = observed["rows"]
        st.extra = {
            "csv_mb": round(csv_bytes / 1e6, 1),
            "parquet_mb": round(parquet_bytes / 1e6, 1),
            "compression_ratio": compression_ratio,
        }

    manifest = {
        **cache_key,
        "source": str(raw_path),
        "destination": str(dst),
        "stats": observed,
        "contract": contract,
        "csv_bytes": csv_bytes,
        "parquet_bytes": parquet_bytes,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": round((1 - parquet_bytes / csv_bytes) * 100, 2),
        # Stored so a cached run can replay identical lineage.
        "lineage": tracker.as_list()[first_record:],
    }
    write_manifest(bronze_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# silver
# ---------------------------------------------------------------------------
def build_silver(
    cfg: dict,
    tracker: LineageTracker,
    force: bool = False,
) -> dict[str, Any]:
    """Clean bronze into silver: drop unusable rows, drop text columns, partition."""
    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    src = bronze_dir / "vehicles.parquet"

    if not src.exists():
        raise FileNotFoundError(f"{src} not found -- run ingest_bronze first.")

    bronze_manifest = read_manifest(bronze_dir) or {}
    cache_key = {
        "version": CACHE_VERSION,
        "input_hash": bronze_manifest.get("input_hash", ""),
        "config_hash": config_hash(
            {
                "ingest": cfg["ingest"],
                "quality": cfg["quality"],
                "data_contract": cfg["data_contract"],
            }
        ),
    }

    if silver_dir.exists() and is_fresh(silver_dir, cache_key) and not force:
        manifest = read_manifest(silver_dir)
        print(f"  cache hit -- reusing {silver_dir}")
        tracker.replay(manifest["lineage"])
        return manifest

    if silver_dir.exists():
        shutil.rmtree(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)

    first_record = len(tracker.records)
    drop_cols = set(cfg["ingest"]["drop_columns"]) | set(cfg["ingest"]["text_columns"])
    keep = [c for c in RAW_COLUMNS if c not in drop_cols]

    print(f"  reading bronze ({len(keep)} of {len(RAW_COLUMNS)} columns)...")
    df = pq.read_table(src, columns=keep).to_pandas()
    rows_bronze = len(df)

    # --- stage: drop rows with no usable year or odometer -------------------
    with tracker.stage(
        "drop_null_year_odometer",
        rows_bronze,
        "year and odometer are the two strongest predictors; a listing missing "
        "either cannot be aged or mileage-scored, and imputing them would "
        "fabricate the signal the model is meant to learn",
    ) as st:
        df = df.dropna(subset=["year", "odometer"])
        st.rows_out = len(df)

    # --- stage: drop implausible prices -------------------------------------
    q = cfg["quality"]["price"]
    with tracker.stage(
        "drop_invalid_price",
        len(df),
        f"price must be in ({q['min']}, {q['max']}]; the source contains 30,759 "
        f"zero-price listings and a $3.7bn outlier",
    ) as st:
        df = df[(df["price"] > q["min"]) & (df["price"] <= q["max"])]
        st.rows_out = len(df)

    # --- stage: enforce remaining quality gates -----------------------------
    yq, oq = cfg["quality"]["year"], cfg["quality"]["odometer"]
    with tracker.stage(
        "apply_quality_gates",
        len(df),
        f"year in [{yq['min']}, {yq['max']}], odometer in [{oq['min']}, {oq['max']}]",
    ) as st:
        df = df[
            df["year"].between(yq["min"], yq["max"])
            & df["odometer"].between(oq["min"], oq["max"])
            & df["state"].notna()
        ]
        st.rows_out = len(df)

    # Schema contract on the cleaned frame. Anything that slipped past the
    # filters above fails here with the offending rows attached.
    df = validate_schema(df, silver_schema(cfg), stage="silver")

    partition_on = cfg["ingest"]["partition_on"]
    print(f"  writing silver partitioned by {partition_on!r}...")
    pq.write_to_dataset(
        pa.Table.from_pandas(df, preserve_index=False),
        root_path=str(silver_dir),
        partition_cols=[partition_on],
        compression=cfg["ingest"]["compression"],
        existing_data_behavior="overwrite_or_ignore",
    )

    n_partitions = sum(1 for p in silver_dir.iterdir() if p.is_dir())
    silver_bytes = sum(f.stat().st_size for f in silver_dir.rglob("*.parquet"))

    manifest = {
        **cache_key,
        "rows_in": rows_bronze,
        "rows_out": len(df),
        "columns": list(df.columns),
        "partition_on": partition_on,
        "n_partitions": n_partitions,
        "silver_bytes": silver_bytes,
        # Stored so a cached run can replay identical lineage.
        "lineage": tracker.as_list()[first_record:],
    }
    write_manifest(silver_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------
def build_gold(
    cfg: dict,
    tracker: LineageTracker,
    force: bool = False,
) -> dict[str, Any]:
    """Apply the feature transforms to silver and materialise the model input.

    No fitting happens here. Every learned quantity -- medians, variances,
    feature selection, scaling -- belongs to the sklearn Pipeline and is fitted
    on the training split only. Gold is a deterministic function of silver, so
    it can safely be computed once for the whole dataset before the split.
    """
    silver_dir = Path(cfg["paths"]["silver"])
    gold_dir = Path(cfg["paths"]["gold"])
    if not silver_dir.exists():
        raise FileNotFoundError(f"{silver_dir} not found -- run build_silver first.")

    silver_manifest = read_manifest(silver_dir) or {}
    cache_key = {
        "version": CACHE_VERSION,
        "input_hash": silver_manifest.get("input_hash", ""),
        "config_hash": config_hash({"features": cfg["features"], "target": cfg["target"]}),
    }

    if gold_dir.exists() and is_fresh(gold_dir, cache_key) and not force:
        manifest = read_manifest(gold_dir)
        print(f"  cache hit -- reusing {gold_dir}")
        tracker.replay(manifest["lineage"])
        return manifest

    if gold_dir.exists():
        shutil.rmtree(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    print("  reading silver...")
    df = pq.read_table(silver_dir).to_pandas()
    rows_in = len(df)

    first_record = len(tracker.records)
    with tracker.stage(
        "build_gold",
        rows_in,
        "deterministic feature transforms; rows are only dropped if price falls "
        "outside every tier bin, which the silver price gate already prevents",
    ) as st:
        X, y, feat_manifest = feature_matrix(df, cfg)
        keep = y.notna()
        X, y = X[keep], y[keep]
        st.rows_out = len(X)
        st.extra = {"n_features": feat_manifest["n_features"]}

    out = X.copy()
    out[cfg["target"]["name"]] = y.astype(str)
    out["id"] = df.loc[keep, "id"].to_numpy()

    dst = gold_dir / "features.parquet"
    pq.write_table(
        pa.Table.from_pandas(out, preserve_index=False),
        dst,
        compression=cfg["ingest"]["compression"],
    )

    class_counts = y.value_counts().reindex(cfg["target"]["labels"]).to_dict()
    manifest = {
        **cache_key,
        "rows_in": rows_in,
        "rows_out": int(len(X)),
        "destination": str(dst),
        "gold_bytes": dst.stat().st_size,
        "feature_groups": feat_manifest["counts"],
        "feature_names": feat_manifest["feature_names"],
        "n_features": feat_manifest["n_features"],
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "lineage": tracker.as_list()[first_record:],
    }
    write_manifest(gold_dir, manifest)

    # A standalone copy under results/ so the README generator and the feature
    # tests can read the manifest without touching the gitignored data tree.
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "groups": feat_manifest["counts"],
                "feature_names": feat_manifest["feature_names"],
                "n_features": feat_manifest["n_features"],
                "class_counts": manifest["class_counts"],
                "rows": manifest["rows_out"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest
