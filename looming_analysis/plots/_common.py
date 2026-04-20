"""Shared plotting helpers: faceting, hue colors, baseline subtraction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

Response = dict


def unique_values(responses: list[Response], key: Optional[str]) -> list[Any]:
    """Return sorted unique values for `key` across responses (skips None/missing)."""
    if key is None:
        return []
    return sorted({r.get(key) for r in responses if r.get(key) is not None})


def iter_facets(
    responses: list[Response],
    row_by: Optional[str],
    col_by: Optional[str],
) -> Iterator[tuple[Any, Any, list[Response], tuple[int, int, int, int]]]:
    """Yield one tuple per facet cell.

    Each yield is `(row_val, col_val, subset, (row_idx, col_idx, n_rows, n_cols))`.
    When a dimension is `None`, its value is `None` and the grid collapses to 1.

    Args:
        responses: Response list.
        row_by: Column name for subplot rows (or None).
        col_by: Column name for subplot columns (or None).
    """
    row_vals = unique_values(responses, row_by)
    col_vals = unique_values(responses, col_by)
    n_rows = len(row_vals) if row_vals else 1
    n_cols = len(col_vals) if col_vals else 1
    effective_rows = row_vals if row_vals else [None]
    effective_cols = col_vals if col_vals else [None]

    for ri, rv in enumerate(effective_rows):
        for ci, cv in enumerate(effective_cols):
            subset = [
                r
                for r in responses
                if (row_by is None or r.get(row_by) == rv)
                and (col_by is None or r.get(col_by) == cv)
            ]
            yield rv, cv, subset, (ri, ci, n_rows, n_cols)


def build_hue_colormap(
    responses: list[Response],
    hue_by: Optional[str],
) -> dict[Any, Any]:
    """Build a {hue_value: rgba-color} mapping. Single-entry when hue_by is None."""
    if hue_by is None:
        return {None: "steelblue"}
    hue_vals = unique_values(responses, hue_by)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(hue_vals), 1)))
    return dict(zip(hue_vals, colors))


def prepare_ang_vel(
    response_list: list[Response],
    time_axis: np.ndarray,
    baseline_subtract: bool,
) -> np.ndarray:
    """Stack trials into `|angular velocity|` array (deg/s), optionally baseline-subtracted.

    When `baseline_subtract` is True, each trial's mean `|ω|` during the
    pre-stimulus window (`time < 0`) is subtracted from that trial's trace.
    """
    data_abs_deg = np.abs(np.rad2deg(np.stack([r["ang_vel"] for r in response_list])))
    if baseline_subtract:
        pre_mask = time_axis < 0
        if pre_mask.any():
            with np.errstate(all="ignore"):
                baseline = np.nanmean(data_abs_deg[:, pre_mask], axis=1, keepdims=True)
            data_abs_deg = data_abs_deg - baseline
    return data_abs_deg


def annotate_facet(
    ax,
    row_val: Any,
    col_val: Any,
    row_by: Optional[str],
    col_by: Optional[str],
    position: tuple[int, int, int, int],
    base_ylabel: str,
    xlabel: str = "Time relative to stimulus start (s)",
) -> None:
    """Apply standard facet-grid axis titles/labels.

    Top row gets column titles, left column gets row-prefixed y-labels,
    bottom row gets x-labels.
    """
    row_idx, col_idx, n_rows, n_cols = position
    is_top = row_idx == 0
    is_left = col_idx == 0
    is_bottom = row_idx == n_rows - 1

    if is_top and col_by is not None:
        ax.set_title(f"{col_by} = {col_val}")
    if is_left:
        if row_by is not None:
            ax.set_ylabel(f"{row_by} = {row_val}\n{base_ylabel}")
        else:
            ax.set_ylabel(base_ylabel)
    if is_bottom:
        ax.set_xlabel(xlabel)
