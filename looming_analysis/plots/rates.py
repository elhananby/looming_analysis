"""Responsiveness rate bar plots, faceted."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._types import Response
from ._common import build_hue_colormap, effective_axis, filter_real, grouped_offsets, require_responsiveness, unique_values


def plot_responsiveness_rates(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    exclude_sham: bool = True,
    ax_size: tuple[float, float] = (7, 5),
) -> Figure:
    """Bar chart of the percentage of responsive trials.

    Dimension roles (same as `plot_heading_changes`):
        - `row_by`:  subplot rows (None → single row)
        - `col_by`:  x-axis tick categories within each subplot
        - `hue_by`:  grouped bars within each x tick

    Each response must have `is_responsive` set (call
    `classify_responsiveness` first).

    Args:
        responses: Response list with `is_responsive` set.
        row_by: Column for subplot rows.
        col_by: Column for x-axis ticks. If None, a single aggregate bar is shown.
        hue_by: Column for grouped side-by-side bars.
        ax_size: (width, height) per subplot.

    Returns:
        The matplotlib Figure.
    """
    require_responsiveness(responses)
    if exclude_sham:
        responses = filter_real(responses)

    effective_rows, n_rows = effective_axis(responses, row_by)
    effective_x, _ = effective_axis(responses, col_by)
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]

    width = max(ax_size[0], 1.1 * max(len(effective_x), 1) * max(len(hue_vals), 1))

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(width, ax_size[1] * n_rows), squeeze=False
    )
    color_map = build_hue_colormap(responses, hue_by)

    for row_idx, row_val in enumerate(effective_rows):
        ax = axes[row_idx, 0]
        _draw_grouped_bars(
            ax,
            responses=responses,
            row_by=row_by,
            row_val=row_val,
            col_by=col_by,
            x_vals=effective_x,
            hue_by=hue_by,
            hue_vals=hue_vals,
            color_map=color_map,
        )

        if row_idx == 0 and hue_by is not None:
            ax.legend(title=hue_by, bbox_to_anchor=(1.02, 1), loc="upper left")
        if row_by is not None:
            ax.set_ylabel(f"{row_by} = {row_val}\nResponsive trials (%)")
        else:
            ax.set_ylabel("Responsive trials (%)")
        if row_idx == n_rows - 1 and col_by is not None:
            ax.set_xlabel(col_by)
        ax.set_ylim(0, 115)
        ax.grid(True, alpha=0.3, axis="y")

    title = "Response Rate"
    dims = [
        f"{k}={v}" for k, v in [("rows", row_by), ("x", col_by), ("hue", hue_by)] if v
    ]
    if dims:
        title += "  |  " + ", ".join(dims)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def _draw_grouped_bars(
    ax,
    *,
    responses,
    row_by,
    row_val,
    col_by,
    x_vals,
    hue_by,
    hue_vals,
    color_map,
):
    all_offsets, bar_width = grouped_offsets(max(len(hue_vals), 1), width=0.8)
    x_positions = np.arange(len(x_vals))

    for i, hv in enumerate(hue_vals):
        rates: list[float] = []
        ns: list[int] = []
        for xv in x_vals:
            subset = [
                r
                for r in responses
                if (row_by is None or r.get(row_by) == row_val)
                and (col_by is None or r.get(col_by) == xv)
                and (hue_by is None or r.get(hue_by) == hv)
            ]
            n_total = len(subset)
            n_resp = sum(1 for r in subset if r.get("is_responsive"))
            rates.append(100 * n_resp / n_total if n_total > 0 else 0.0)
            ns.append(n_total)

        offset = all_offsets[i] if hue_by else 0
        color = color_map[hv]
        label = str(hv) if hv is not None else None
        bars = ax.bar(
            x_positions + offset,
            rates,
            width=bar_width,
            color=color,
            label=label,
            alpha=0.85,
            edgecolor="white",
        )
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(xv) if xv is not None else "all" for xv in x_vals])
