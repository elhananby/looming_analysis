"""Convert responses to tidy DataFrames (scalar or long format)."""

from __future__ import annotations

import numpy as np
import polars as pl

from ._types import Response


def responses_to_dataframe(
    responses: list[Response],
    kind: str = "scalar",
    backend: str = "polars",
) -> pl.DataFrame:
    """Convert a list of response dicts to a tidy DataFrame.

    Args:
        responses: List of response dicts from extract/classify pipeline.
        kind: "scalar" (one row per trial) or "long" (one row per trial × timepoint).
        backend: "polars" (default) or "pandas". Return type matches backend.

    Returns:
        Polars or pandas DataFrame, depending on `backend`.

    Raises:
        ValueError: If `kind` or `backend` is invalid.
    """
    if kind not in ("scalar", "long"):
        raise ValueError(f"kind must be 'scalar' or 'long', got {kind!r}")
    if backend not in ("polars", "pandas"):
        raise ValueError(f"backend must be 'polars' or 'pandas', got {backend!r}")

    if kind == "scalar":
        rows = _build_scalar_rows(responses)
    else:
        rows = _build_long_rows(responses)

    if backend == "polars":
        return pl.DataFrame(rows)
    else:
        import pandas as pd

        return pd.DataFrame(rows)


def _build_scalar_rows(responses: list[Response]) -> list[dict]:
    """Extract scalar columns (one row per trial)."""
    rows = []
    for r in responses:
        row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        rows.append(row)
    return rows


def _build_long_rows(responses: list[Response]) -> list[dict]:
    """Expand to long format (one row per trial × timepoint)."""
    rows = []
    for trial_id, r in enumerate(responses):
        scalars = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        time = r["time"]
        ang_vel_deg_s = np.rad2deg(r["ang_vel"])
        for t, av in zip(time, ang_vel_deg_s):
            rows.append(
                {
                    "trial_id": trial_id,
                    "time": float(t),
                    "ang_vel_deg_s": float(av),
                    **scalars,
                }
            )
    return rows
