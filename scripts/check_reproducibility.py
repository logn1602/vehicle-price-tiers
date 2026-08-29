"""Assert that two full training runs produce an identical metrics.json.

Section 7.4 of the brief: run the pipeline twice from a clean state and confirm
the results are identical apart from timestamps. If they are not, something is
unseeded and every published figure is a sample from an unknown distribution
rather than a measurement.

Volatile fields live entirely inside `run_metadata` -- timestamp, wall-clock
duration, per-model fit seconds, platform -- so the comparison is a plain
equality check on everything else. Keeping the volatile fields quarantined in
one block, rather than scattered beside the metrics, is what makes that
possible.

    python scripts/check_reproducibility.py            # runs training twice
    python scripts/check_reproducibility.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]

# The only block permitted to differ between runs.
VOLATILE_TOP_LEVEL = {"run_metadata"}


def strip_volatile(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in VOLATILE_TOP_LEVEL}


def canonical(payload: dict) -> str:
    return json.dumps(strip_volatile(payload), sort_keys=True, separators=(",", ":"))


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Every divergence, not just the first.

    Reporting only the first is actively misleading: a trivial difference early
    in the document hides whatever comes after it. The original version of this
    function stopped at an MLflow run id -- a UUID that differs by design -- and
    said nothing about whether any actual metric had moved.
    """
    if type(a) is not type(b):
        return [f"{path or '<root>'}: type {type(a).__name__} vs {type(b).__name__}"]

    if isinstance(a, dict):
        found: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a:
                found.append(f"{path}.{key}: missing from run 1")
            elif key not in b:
                found.append(f"{path}.{key}: missing from run 2")
            else:
                found.extend(differences(a[key], b[key], f"{path}.{key}"))
        return found

    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        found = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found.extend(differences(x, y, f"{path}[{i}]"))
        return found

    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def run_training(config: str) -> dict:
    subprocess.run(
        [sys.executable, "scripts/run_all.py", "--stages", "train", "--config", config],
        cwd=REPO,
        check=True,
    )
    metrics = REPO / "results" / "metrics.json"
    return json.loads(metrics.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="conf/config.yaml")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="compare two existing metrics files instead of running training",
    )
    args = p.parse_args(argv)

    if args.compare:
        first = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        second = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        labels = tuple(args.compare)
    else:
        print("=" * 72)
        print("RUN 1 of 2")
        print("=" * 72)
        first = run_training(args.config)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(first, fh)
        print("\n" + "=" * 72)
        print("RUN 2 of 2")
        print("=" * 72)
        second = run_training(args.config)
        labels = ("run 1", "run 2")

    print("\n" + "=" * 72)
    print("REPRODUCIBILITY")
    print("=" * 72)

    same = canonical(first) == canonical(second)
    print(f"comparing:   {labels[0]}  vs  {labels[1]}")
    print(f"excluded:    {', '.join(sorted(VOLATILE_TOP_LEVEL))}")
    print(f"config hash: {first['provenance']['config_hash'][:16]}")
    print(f"data hash:   {first['provenance']['data_hash'][:16]}")
    print(f"seed:        {first['provenance']['seed']}")

    if same:
        print("\nIDENTICAL -- every metric reproduced exactly.")
        return 0

    found = differences(strip_volatile(first), strip_volatile(second))
    print(f"\nDIFFER in {len(found)} field(s):\n")
    for line in found[:40]:
        print(f"  {line}")
    if len(found) > 40:
        print(f"  ... and {len(found) - 40} more")
    print(
        "\nSomething in the pipeline is unseeded. Every published figure is "
        "therefore a draw from an unknown distribution, not a measurement."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
