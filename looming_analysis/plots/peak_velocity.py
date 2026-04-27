"""Peak angular velocity violin plots, faceted."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from .._types import Response
from ._common import plot_violin_facets


def plot_peak_velocity(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    responsive_only: bool = False,
    ax_size: tuple[float, float] = (7, 5),
) -> Figure:
    """Violin plot of peak |angular velocity| from saccade detection.

    Values come from ``peak_ang_vel_deg_s`` set by ``classify_responsiveness``
    (the first qualifying ``find_peaks`` hit in the detection window). Requires
    ``classify_responsiveness`` to have been called first.

    Dimension roles:
        - ``row_by``:  subplot rows
        - ``col_by``:  x-axis tick categories within each subplot
        - ``hue_by``:  side-by-side violins within each x tick

    Args:
        responses: Response list.
        row_by: Column for subplot rows.
        col_by: Column for x-axis ticks.
        hue_by: Column for side-by-side violins.
        ax_size: (width, height) per subplot in inches.

    Returns:
        The matplotlib Figure.
    """
    return plot_violin_facets(
        responses,
        value_fn=lambda r: (
            r["peak_ang_vel_deg_s"]
            if not np.isnan(r.get("peak_ang_vel_deg_s", float("nan")))
            else r.get("mean_ang_vel_window_deg_s")
        ),
        row_by=row_by,
        col_by=col_by,
        hue_by=hue_by,
        responsive_only=responsive_only,
        ylabel="Peak |angular velocity| (deg/s)",
        title="Peak Angular Velocity",
        show_zero_line=False,
        ax_size=ax_size,
    )
