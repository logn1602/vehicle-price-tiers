"""Assert that every numeric claim in README.md exists in a generated artifact.

This is the mechanism that makes the no-hand-typed-numbers rule enforceable
instead of aspirational. Every error in the v1 report came from one process:
a number was read off a screen, retyped into prose, the code was re-run, and
the prose was never updated. The report ended up carrying five different AUC
values for one model and a headline figure matching none of them.

So: if a number appears in the README and cannot be traced to
`results/metrics.json`, `results/lineage.json`, `results/feature_manifest.json`,
`results/ablation.json` or `conf/config.yaml`, this script fails and CI fails
with it.

    python scripts/verify_readme.py
    python scripts/verify_readme.py --readme MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

ARTIFACTS = [
    "results/metrics.json",
    "results/lineage.json",
    "results/feature_manifest.json",
    "results/ablation.json",
    "conf/config.yaml",
]

# Numbers that are structural rather than empirical: they describe the document
# or the world, not a result. Each needs a reason to be here.
ALLOWED = {
    2021: "dataset scrape year, stated in data/README.md",
    2022: "upper bound of the year quality gate",
    1900: "lower bound of the year quality gate",
    3: "Python minor-version references and small counts in prose",
    11: "Python 3.11",
    12: "Python 3.12",
    0: "zero",
    1: "one",
    2: "two",
    4: "the four price tiers",
    5: "the five models compared",
    100: "percentage denominators",
}

# Markdown constructs that are not claims: ordered-list markers, heading
# levels, table alignment rows, code fences, and URLs.
_STRIP_PATTERNS = [
    re.compile(r"^\s{0,3}\d+\.\s", re.MULTILINE),   # "1. " list markers
    re.compile(r"^\|[\s:|-]+\|$", re.MULTILINE),    # table separator rows
    re.compile(r"```.*?```", re.DOTALL),            # fenced code blocks
    re.compile(r"`[^`]*`"),                         # inline code
    re.compile(r"https?://\S+"),                    # URLs
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),            # images
    re.compile(r"\[[^\]]*\]\([^)]*\)"),             # links
]

_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(%?)")


def strip_non_claims(text: str) -> str:
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def extract_claims(text: str) -> list[tuple[str, float, bool]]:
    """Return (raw token, numeric value, is_percentage) for each claim."""
    claims = []
    for match in _NUMBER.finditer(strip_non_claims(text)):
        raw, suffix = match.group(1), match.group(2)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        claims.append((raw + suffix, value, suffix == "%"))
    return claims


def walk(node: Any, sink: set[float]) -> None:
    """Collect every numeric leaf from a nested structure."""
    if isinstance(node, bool):
        return
    if isinstance(node, int | float):
        sink.add(float(node))
        return
    if isinstance(node, dict):
        for key, value in node.items():
            # Dict keys can be numeric too (e.g. class-count maps).
            with contextlib.suppress(TypeError, ValueError):
                sink.add(float(key))
            walk(value, sink)
        return
    if isinstance(node, list):
        for item in node:
            walk(item, sink)


def known_values(paths: list[str]) -> tuple[set[float], list[str]]:
    """Every number appearing anywhere in the generated artifacts."""
    values: set[float] = set()
    found: list[str] = []
    for rel in paths:
        path = REPO / rel
        if not path.exists():
            continue
        found.append(rel)
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
        walk(data, values)
    return values, found


def matches(value: float, is_pct: bool, known: set[float]) -> bool:
    """Does this claim correspond to a known value?

    A README rounds: 0.6104 may appear as 0.61, and a proportion of 0.3327 may
    be written as 33.3%. Both are accepted, at the precision the README used.
    """
    candidates = [value, value / 100.0] if is_pct else [value]
    for candidate in candidates:
        decimals = len(str(candidate).split(".")[1]) if "." in str(candidate) else 0
        for k in known:
            if round(k, decimals) == round(candidate, decimals):
                return True
            # A proportion written as a percentage, or the reverse.
            if round(k * 100, decimals) == round(candidate, decimals):
                return True
    return False


def verify(readme: Path, artifact_paths: list[str]) -> tuple[list[dict], list[str]]:
    known, found = known_values(artifact_paths)
    if not known:
        raise SystemExit(
            "No generated artifacts found. Run `python scripts/run_all.py` first."
        )

    unverified = []
    for raw, value, is_pct in extract_claims(readme.read_text(encoding="utf-8")):
        if value in ALLOWED:
            continue
        if not matches(value, is_pct, known):
            unverified.append({"claim": raw, "value": value})
    return unverified, found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readme", default="README.md")
    p.add_argument("--artifact", action="append", default=None)
    args = p.parse_args(argv)

    readme = REPO / args.readme
    if not readme.exists():
        raise SystemExit(f"{readme} not found")

    unverified, found = verify(readme, args.artifact or ARTIFACTS)

    print(f"Checking {args.readme} against: {', '.join(found)}")
    if unverified:
        print(f"\n{len(unverified)} unverified numeric claim(s):\n")
        for item in unverified:
            print(f"  {item['claim']:<16} (parsed as {item['value']})")
        print(
            "\nEvery number in the README must come from a generated artifact.\n"
            "Either regenerate the README from results/metrics.json, or add the "
            "value to ALLOWED in this script with a reason if it is structural."
        )
        return 1

    print("\nAll numeric claims verified against generated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
