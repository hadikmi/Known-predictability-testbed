"""Shared experiment utilities: configuration, seeding, provenance, logging."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger("knownpred")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def derive_seed(master: int, *keys) -> int:
    """Deterministic, independent sub-seed for a named component.

    Hashing (master, keys) gives each component its own random stream, so
    adding a model or a lead does not change the data other components see.
    """
    h = hashlib.sha256(json.dumps([master, *keys], default=str).encode()).digest()
    return int.from_bytes(h[:4], "little")


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return None


def provenance(cfg: dict) -> dict:
    """Metadata written next to every set of results."""
    versions = {}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "torch", "xarray"):
        try:
            versions[mod] = importlib.import_module(mod).__version__
        except Exception:
            versions[mod] = None
    from knownpred import __version__

    return {
        "knownpred_version": __version__,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
    }


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
