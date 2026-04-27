"""Helpers for discovering input recording files."""

from __future__ import annotations

from pathlib import Path


def find_braidz(root: str | Path, *, recursive: bool = False) -> list[str]:
    """Return sorted `.braidz` paths under `root`."""
    root_path = Path(root)
    pattern = "**/*.braidz" if recursive else "*.braidz"
    return [str(path) for path in sorted(root_path.glob(pattern))]
