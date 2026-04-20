"""Angular velocity time-series plots (faceted)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._common import (
    annotate_facet,
    build_hue_colormap,
    iter_facets,
    prepare_ang_vel,
    unique_values,
)

Response = dict


def _draw_traces(
    ax,
    subset: list[Response],
    hue_by: Optional[str],
    color_map: dict,
    baseline_subtract: bool,
) -> None:
    """Draw mean ± SEM angular velocity traces on `ax`, split by hue_by."""
    if hue_by is None:
        hue_vals: list = [None]
    else:
        hue_vals = unique_values(subset, hue_by)

    for hv in hue_vals:
        if hue_by is None:
            hue_subset = subset
        else:
            hue_subset = [r for r in subset if r.get(hue_by) == hv]
        if not hue_subset:
            continue

        time_axis = hue_subset[0]["time"]
        data = prepare_ang_vel(hue_subset, time_axis, baseline_subtract)
        with np.errstate(all="ignore"):
            mean_resp = np.nanmean(data, axis=0)
            sem_resp = np.nanstd(data, axis=0)

        color = color_map.get(hv, "steelblue")
        label = (
            f"{hv} (n={len(hue_subset)})" if hv is not None else f"n={len(hue_subset)}"
        )
        ax.plot(time_axis, mean_resp, color=color, label=label)
        ax.fill_between(
            time_axis,
            mean_resp - sem_resp,
            mean_resp + sem_resp,
            alpha=0.2,
            color=color,
        )

    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def plot_responses(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    baseline_subtract: bool = True,
    sharey: bool = True,
    ax_size: tuple[float, float] = (5, 4),
) -> Figure:
    """Faceted mean ± SEM `|angular velocity|` over time.

    Args:
        responses: Response list.
        row_by: Column name mapped to subplot rows (None → single row).
        col_by: Column name mapped to subplot columns (None → single col).
        hue_by: Column name mapped to line colors within each subplot.
        baseline_subtract: Subtract per-trial pre-stim baseline from |ω|.
        sharey: Share y-axis across all subplots.
        ax_size: (width, height) per subplot in inches.

    Returns:
        The matplotlib Figure. The caller is responsible for display/save.
    """
    row_vals = unique_values(responses, row_by)
    col_vals = unique_values(responses, col_by)
    n_rows = len(row_vals) if row_vals else 1
    n_cols = len(col_vals) if col_vals else 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(ax_size[0] * n_cols, ax_size[1] * n_rows),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    color_map = build_hue_colormap(responses, hue_by)
    ylabel = (
        "Δ Angular velocity (deg/s)"
        if baseline_subtract
        else "Angular velocity (deg/s)"
    )

    for row_val, col_val, subset, position in iter_facets(responses, row_by, col_by):
        ax = axes[position[0], position[1]]
        if subset:
            _draw_traces(ax, subset, hue_by, color_map, baseline_subtract)
        annotate_facet(ax, row_val, col_val, row_by, col_by, position, ylabel)

    _suptitle(fig, row_by, col_by, hue_by, prefix="Fly Response to Looming")
    fig.tight_layout()
    return fig


def plot_responses_by_responsiveness(
    responses: list[Response],
    *,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    baseline_subtract: bool = True,
    sharey: bool = True,
    ax_size: tuple[float, float] = (5, 4),
) -> Figure:
    """2-row grid (responsive / non-responsive) × `col_by` columns.

    Each response must already have `is_responsive` set (call
    `classify_responsiveness` first).

    Args:
        responses: Response list with `is_responsive` set.
        col_by: Column name mapped to subplot columns.
        hue_by: Column name mapped to line colors within each subplot.
        baseline_subtract: Subtract per-trial pre-stim baseline from |ω|.
        sharey: Share y within each row (responsive vs non-responsive).
        ax_size: (width, height) per subplot.

    Returns:
        The matplotlib Figure.
    """
    col_vals = unique_values(responses, col_by)
    n_cols = len(col_vals) if col_vals else 1
    n_rows = 2  # responsive, non-responsive

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(ax_size[0] * n_cols, ax_size[1] * n_rows),
        sharex=True,
        sharey="row" if sharey else False,
        squeeze=False,
    )

    color_map = build_hue_colormap(responses, hue_by)
    ylabel = (
        "Δ Angular velocity (deg/s)"
        if baseline_subtract
        else "Angular velocity (deg/s)"
    )
    row_labels = ["Responsive", "Non-responsive"]

    effective_cols = col_vals if col_vals else [None]

    for row_idx, is_resp in enumerate([True, False]):
        for col_idx, col_val in enumerate(effective_cols):
            ax = axes[row_idx, col_idx]
            subset = [
                r
                for r in responses
                if r.get("is_responsive") == is_resp
                and (col_by is None or r.get(col_by) == col_val)
            ]
            if subset:
                _draw_traces(ax, subset, hue_by, color_map, baseline_subtract)

            if row_idx == 0 and col_by is not None:
                ax.set_title(f"{col_by} = {col_val}")
            if col_idx == 0:
                ax.set_ylabel(f"{row_labels[row_idx]}\n{ylabel}")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time relative to stimulus start (s)")

    _suptitle(
        fig,
        row_by=None,
        col_by=col_by,
        hue_by=hue_by,
        prefix="Angular velocity: Responsive vs Non-responsive",
    )
    fig.tight_layout()
    return fig


def _suptitle(fig: Figure, row_by, col_by, hue_by, *, prefix: str) -> None:
    parts = [prefix]
    dims = []
    if row_by:
        dims.append(f"rows={row_by}")
    if col_by:
        dims.append(f"cols={col_by}")
    if hue_by:
        dims.append(f"hue={hue_by}")
    if dims:
        parts.append("  |  " + ", ".join(dims))
    fig.suptitle("".join(parts), y=1.02)
