"""Heading direction time-series plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._types import Response
from ._common import annotate_facet, build_hue_colormap, iter_facets, unique_values


def plot_heading_traces(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    sharey: bool = True,
    ax_size: tuple[float, float] = (5, 4),
) -> Figure:
    """Faceted mean heading direction traces over time."""
    if responses and "heading_deg" not in responses[0]:
        raise ValueError(
            "Responses do not include 'heading_deg'. "
            "Re-run extraction with the current package version."
        )

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

    for row_val, col_val, subset, position in iter_facets(responses, row_by, col_by):
        ax = axes[position[0], position[1]]
        hue_vals = unique_values(subset, hue_by) if hue_by else [None]
        for hv in hue_vals:
            hue_subset = (
                subset if hue_by is None else [r for r in subset if r.get(hue_by) == hv]
            )
            if not hue_subset:
                continue
            time_axis = hue_subset[0]["time"]
            data = np.stack([r["heading_deg"] for r in hue_subset])
            with np.errstate(all="ignore"):
                mean_heading = np.nanmean(data, axis=0)
                sem_heading = np.nanstd(data, axis=0)
            color = color_map.get(hv, "steelblue")
            label = (
                f"{hv} (n={len(hue_subset)})"
                if hv is not None
                else f"n={len(hue_subset)}"
            )
            ax.plot(time_axis, mean_heading, color=color, label=label)
            ax.fill_between(
                time_axis,
                mean_heading - sem_heading,
                mean_heading + sem_heading,
                color=color,
                alpha=0.2,
            )
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        annotate_facet(
            ax,
            row_val,
            col_val,
            row_by,
            col_by,
            position,
            "Heading direction (deg)",
        )

    fig.suptitle("Mean Heading Direction", y=1.02)
    fig.tight_layout()
    return fig
