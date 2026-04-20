"""Loading `.braidz` archives."""

from __future__ import annotations

import gzip
import zipfile
from typing import Optional

import polars as pl


def load_braidz(file_path: str) -> tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Load kalman estimates and stimulus data from a `.braidz` file.

    A `.braidz` archive is a zip containing `kalman_estimates.csv.gz` and
    (optionally) `stim.csv` or `visual_stimuli.csv`.

    Args:
        file_path: Path to the `.braidz` file.

    Returns:
        Tuple of `(df_kalman, df_stim)`. `df_stim` is `None` if the archive
        contains no stimulus file.
    """
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

    return df_kalman, df_stim
