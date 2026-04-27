"""Loading `.braidz` archives."""

from __future__ import annotations

import gzip
import hashlib
import zipfile
from pathlib import Path
from typing import Optional

import polars as pl


def _braidz_cache_path(file_path: str | Path, cache_dir: str | Path) -> Path:
    source = Path(file_path)
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"{source.stem}-{digest}"


def load_braidz(
    file_path: str,
    cache_dir: Optional[str] = None,
) -> tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Load kalman estimates and stimulus data from a `.braidz` file.

    A `.braidz` archive is a zip containing `kalman_estimates.csv.gz` and
    (optionally) `stim.csv` or `visual_stimuli.csv`.

    If `cache_dir` is set, parquet files are checked first; if not found,
    the `.braidz` is loaded and cached for future runs.

    Args:
        file_path: Path to the `.braidz` file.
        cache_dir: Optional directory to cache parquet files. Auto-creates
            a subdirectory per `.braidz` file. If not set, always loads from
            the archive (no caching).

    Returns:
        Tuple of `(df_kalman, df_stim)`. `df_stim` is `None` if the archive
        contains no stimulus file.
    """
    cache_path = _braidz_cache_path(file_path, cache_dir) if cache_dir else None

    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        kalman_cache = cache_path / "kalman_estimates.parquet"
        stim_cache = cache_path / "stim.parquet"

        if kalman_cache.exists():
            df_kalman = pl.read_parquet(kalman_cache)
            df_stim = pl.read_parquet(stim_cache) if stim_cache.exists() else None
            return df_kalman, df_stim

    with zipfile.ZipFile(file_path, "r") as z:
        with z.open("kalman_estimates.csv.gz") as f, gzip.open(f) as gz:
            df_kalman = pl.read_csv(gz)

        df_stim: Optional[pl.DataFrame] = None
        if "stim.csv" in z.namelist():
            with z.open("stim.csv") as f:
                df_stim = pl.read_csv(f)
        elif "visual_stimuli.csv" in z.namelist():
            with z.open("visual_stimuli.csv") as f:
                df_stim = pl.read_csv(f)

    if cache_path:
        df_kalman.write_parquet(cache_path / "kalman_estimates.parquet")
        if df_stim is not None:
            df_stim.write_parquet(cache_path / "stim.parquet")

    return df_kalman, df_stim
