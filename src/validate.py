"""Quality gates and lineage tracking.

Lineage is the defect detector this project is built around. Every stage that
changes the row count records how many rows went in, how many came out, and
why. A stage that drops more than `lineage.max_drop_pct` fails the run unless
it is whitelisted in config with a stated reason.

That single rule is what would have caught the v1 defect in four seconds: a
price filter that silently discarded 98% of the data produced no warning at
all, and the resulting 6,067-row training set was reported as though it were
the full 426,880.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class LineageViolation(Exception):
    """A stage dropped more rows than the configured threshold allows."""


class ContractFailure(Exception):
    """Observed data does not match the asserted data contract."""


@dataclass
class StageRecord:
    stage: str
    rows_in: int
    rows_out: int
    rows_dropped: int
    pct_dropped: float
    reason: str
    duration_s: float
    timestamp: str
    whitelisted: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class _StageHandle:
    """Mutable handle yielded by `LineageTracker.stage`; the caller sets
    `rows_out` once the stage has actually run."""

    __slots__ = ("rows_in", "rows_out", "extra")

    def __init__(self, rows_in: int) -> None:
        self.rows_in = rows_in
        self.rows_out: int | None = None
        self.extra: dict[str, Any] = {}


class LineageTracker:
    """Accumulates stage records and enforces the drop threshold."""

    def __init__(self, cfg: dict) -> None:
        lineage_cfg = cfg.get("lineage", {})
        self.max_drop_pct: float = float(lineage_cfg.get("max_drop_pct", 25.0))
        self._whitelist: dict[str, str] = {
            entry["stage"]: entry.get("reason", "")
            for entry in lineage_cfg.get("whitelist", [])
        }
        self.records: list[StageRecord] = []

    @contextmanager
    def stage(self, name: str, rows_in: int, reason: str) -> Iterator[_StageHandle]:
        """Time a stage, record its row delta, and enforce the threshold.

        The handle's `rows_out` must be set inside the block; leaving it unset
        is a programming error and raises, rather than silently recording a
        stage that appears to have dropped everything.
        """
        handle = _StageHandle(rows_in)
        started = time.perf_counter()
        yield handle
        duration = time.perf_counter() - started

        if handle.rows_out is None:
            raise ValueError(
                f"Stage {name!r} did not set rows_out. Every tracked stage must "
                f"report its output row count."
            )

        dropped = handle.rows_in - handle.rows_out
        pct = (dropped / handle.rows_in * 100) if handle.rows_in else 0.0
        whitelisted = name in self._whitelist

        record = StageRecord(
            stage=name,
            rows_in=handle.rows_in,
            rows_out=handle.rows_out,
            rows_dropped=dropped,
            pct_dropped=round(pct, 4),
            reason=reason,
            duration_s=round(duration, 3),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            whitelisted=whitelisted,
            extra=handle.extra,
        )
        self.records.append(record)

        if pct > self.max_drop_pct and not whitelisted:
            raise LineageViolation(
                f"Stage {name!r} dropped {dropped:,} of {handle.rows_in:,} rows "
                f"({pct:.2f}%), exceeding the {self.max_drop_pct}% threshold.\n"
                f"Reason given: {reason}\n\n"
                f"If this is intended, whitelist it in conf/config.yaml under "
                f"lineage.whitelist with an explicit reason."
            )

    def replay(self, records: list[dict]) -> None:
        """Re-append stage records recovered from a cache manifest.

        A cached stage did not re-run, but it did still happen -- the data it
        produced is on disk. Replaying its records verbatim (original durations
        and timestamps included) keeps `lineage.json` byte-identical between a
        fresh run and a cached one. Collapsing cached stages into a single
        summary entry instead would make the artifact depend on cache state
        rather than on the data, which is exactly the kind of drift the
        reproducibility check exists to catch.
        """
        for record in records:
            self.records.append(StageRecord(**record))

    def as_list(self) -> list[dict]:
        return [asdict(r) for r in self.records]

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_list(), indent=2), encoding="utf-8")
        return path

    def table(self) -> str:
        """Human-readable lineage table for the console and the README."""
        if not self.records:
            return "(no stages recorded)"
        header = f"{'stage':<28}{'rows in':>12}{'rows out':>12}{'dropped':>12}{'%':>9}"
        lines = [header, "-" * len(header)]
        for r in self.records:
            flag = " *" if r.whitelisted else ""
            lines.append(
                f"{r.stage:<28}{r.rows_in:>12,}{r.rows_out:>12,}"
                f"{r.rows_dropped:>12,}{r.pct_dropped:>8.2f}%{flag}"
            )
        if any(r.whitelisted for r in self.records):
            lines.append("")
            lines.append("* whitelisted in conf/config.yaml with a stated reason")
        return "\n".join(lines)


def check_contract(
    observed: dict[str, Any],
    expected: dict[str, Any],
    pct_tolerance: float,
    label: str,
    raise_on_fail: bool = True,
) -> list[dict]:
    """Compare observed statistics against an asserted contract.

    Keys ending in `_pct` are compared with `pct_tolerance` because the source
    figures are quoted rounded; everything else must match exactly.

    Note the two contracts in config have DIFFERENT denominators -- `raw` is
    measured on all 426,880 rows, `post_dropna` on the 421,344 that remain once
    null year/odometer rows are removed. Conflating them produces seven
    spurious failures; this is verified in results/v1_rowloss_diagnosis.json.
    """
    results: list[dict] = []
    for key, want in expected.items():
        if key not in observed:
            results.append(
                {"fact": key, "observed": None, "expected": want, "pass": False,
                 "note": "not measured"}
            )
            continue
        got = observed[key]
        tol = pct_tolerance if key.endswith("_pct") else 0
        ok = abs(got - want) <= tol if isinstance(got, int | float) else got == want
        results.append(
            {"fact": key, "observed": got, "expected": want, "pass": bool(ok)}
        )

    failed = [r for r in results if not r["pass"]]
    if failed and raise_on_fail:
        detail = "\n".join(
            f"  {r['fact']:<32} observed={r['observed']!s:<18} expected={r['expected']}"
            for r in failed
        )
        raise ContractFailure(
            f"{label}: {len(failed)} of {len(results)} facts do not hold.\n{detail}\n\n"
            f"This usually means the input file is a different revision than the "
            f"one this project was built against. See data/README.md."
        )
    return results
