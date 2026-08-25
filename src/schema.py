"""Schema contracts for each data layer.

Two schemas, deliberately different in strictness:

`raw_schema` documents the file as it actually is -- including a $3.7bn price
and 133 vehicles built after the scrape date. It asserts structure (column
names, types, what may be null) but not plausibility, because asserting
plausibility on raw data would fail on the very rows the pipeline exists to
remove.

`silver_schema` asserts plausibility. It encodes the quality gates from
`conf/config.yaml` and is applied after cleaning, so a regression that lets a
zero-price or 3-million-mile listing through fails loudly.

Validation failures raise with the offending rows attached rather than a bare
count, so a failing run tells you which records broke the contract.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# Dtypes for pd.read_csv. Explicit so pandas never infers a type from a sample
# -- inference on a chunked read can assign different dtypes to different
# chunks of the same column.
RAW_DTYPES: dict[str, str] = {
    "id": "str",
    "url": "str",
    "region": "str",
    "region_url": "str",
    "price": "float64",
    "year": "float64",
    "manufacturer": "str",
    "model": "str",
    "condition": "str",
    "cylinders": "str",
    "fuel": "str",
    "odometer": "float64",
    "title_status": "str",
    "transmission": "str",
    "VIN": "str",
    "drive": "str",
    "size": "str",
    "type": "str",
    "paint_color": "str",
    "image_url": "str",
    "description": "str",
    "county": "str",
    "state": "str",
    "lat": "float64",
    "long": "float64",
    "posting_date": "str",
}

RAW_COLUMNS: list[str] = list(RAW_DTYPES)

# Columns that are entirely null in the source, or free text the model does not
# consume. Kept here so `county` being 100% null is an asserted fact rather than
# a surprise discovered at feature time.
ALL_NULL_COLUMNS = ["county"]


class ContractViolation(Exception):
    """Raised when a layer fails its schema contract."""


def _nullable_str(**kwargs) -> pa.Column:
    return pa.Column(str, nullable=True, required=True, **kwargs)


def raw_schema() -> pa.DataFrameSchema:
    """Structural contract for the raw CSV.

    Type and nullability only. The one range check is on `year`, which has a
    hard lower bound that a column-shifted export would violate immediately --
    this is the cheapest possible detector for the class of corruption that
    produced v1's unusable file.
    """
    columns: dict[str, pa.Column] = {name: _nullable_str() for name in RAW_COLUMNS}

    columns["price"] = pa.Column(float, nullable=True, checks=pa.Check.ge(0))
    columns["year"] = pa.Column(
        float,
        nullable=True,
        checks=pa.Check.in_range(1900, 2023, include_min=True, include_max=True),
    )
    columns["odometer"] = pa.Column(float, nullable=True, checks=pa.Check.ge(0))
    columns["lat"] = pa.Column(
        float, nullable=True, checks=pa.Check.in_range(-90, 90)
    )
    columns["long"] = pa.Column(
        float, nullable=True, checks=pa.Check.in_range(-180, 180)
    )
    columns["id"] = pa.Column(str, nullable=False, unique=True)

    return pa.DataFrameSchema(
        columns,
        strict=True,       # an unexpected column is a failure, not a warning
        ordered=False,
        coerce=False,
        name="raw_vehicles",
    )


def silver_schema(cfg: dict) -> pa.DataFrameSchema:
    """Plausibility contract, built from the quality gates in config.

    Every bound here comes from `conf/config.yaml`; none is hard-coded.
    """
    q = cfg["quality"]
    price, year, odo = q["price"], q["year"], q["odometer"]

    return pa.DataFrameSchema(
        {
            "id": pa.Column(str, nullable=False, unique=True),
            "price": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(
                    price["min"],
                    price["max"],
                    include_min=price.get("min_inclusive", True),
                    include_max=True,
                ),
            ),
            "year": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(year["min"], year["max"]),
            ),
            "odometer": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(odo["min"], odo["max"]),
            ),
            "state": pa.Column(str, nullable=False),
        },
        strict=False,      # feature columns are added downstream
        coerce=False,
        name="silver_vehicles",
    )


def validate(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    stage: str,
    max_examples: int = 10,
) -> pd.DataFrame:
    """Validate `df`, raising ContractViolation with the offending rows.

    Uses lazy validation so a failing frame reports every broken check at once
    rather than stopping at the first. The message includes concrete failure
    cases because "3 rows failed" is not actionable.
    """
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        failures = exc.failure_cases
        summary = (
            failures.groupby(["column", "check"], dropna=False)
            .size()
            .reset_index(name="n_failed")
            .to_string(index=False)
        )
        examples = failures.head(max_examples).to_string(index=False)
        raise ContractViolation(
            f"[{stage}] schema '{schema.name}' failed "
            f"({len(failures)} failing cases)\n\n"
            f"Summary:\n{summary}\n\n"
            f"First {min(max_examples, len(failures))} cases:\n{examples}"
        ) from exc
    except SchemaError as exc:
        raise ContractViolation(f"[{stage}] schema '{schema.name}' failed: {exc}") from exc
