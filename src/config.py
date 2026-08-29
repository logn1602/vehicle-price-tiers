"""Configuration loading and content hashing.

Every tunable value lives in `conf/config.yaml`. This module loads it, allows
dotted-path overrides from the CLI, and produces a stable hash of the resolved
configuration so that a training run can be tied to the exact settings that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("conf/config.yaml")


class Config(dict):
    """A plain dict with dotted-path assignment, so a CLI override can be
    written the way it reads: `--set model.xgboost.max_depth=8`.

    Reads stay ordinary subscripting. A matching `get_path` existed here and was
    never called once -- config access throughout the codebase is
    `cfg["model"]["xgboost"]`, which is clearer at the point of use.
    """

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML rules, so `4`, `0.7`, `true` and
    `[a, b]` all arrive as the right type."""
    return yaml.safe_load(text)


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    overrides: list[str] | None = None,
) -> Config:
    """Load YAML config, applying `key.path=value` overrides in order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = Config(yaml.safe_load(path.read_text(encoding="utf-8")))

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Override must be key.path=value, got: {override!r}"
            )
        key, _, raw = override.partition("=")
        cfg.set_path(key.strip(), _coerce(raw.strip()))

    return cfg


def config_hash(cfg: dict) -> str:
    """Stable SHA-256 over the resolved config.

    Sorted keys and a fixed separator make this reproducible across runs and
    platforms -- it is recorded in metrics.json so a metric can always be traced
    back to the settings that produced it.
    """
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_hash(path: Path | str, chunk_size: int = 8 << 20) -> str:
    """SHA-256 of a file, streamed. Used to record the exact input data."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
