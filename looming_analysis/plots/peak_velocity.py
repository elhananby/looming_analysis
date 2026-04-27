"""Peak angular velocity violin plots, faceted."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from .._types import Response
from ._common import plot_violin_facets


def _extract_peak(
    r: Response,
    start_offset_s: float,
    end_offset_s: float,
) -> float:
    """Return peak |ang_vel| (deg/s) within [0 + start_offset_s, end_expansion_time + end_offset_s]."""
    time = r["time"]
    ang_vel_abs = np.abs(np.rad2deg(r["ang_vel"]))
    end_t = r["end_expansion_time"]
    mask = (time >= start_offset_s) & (time <= end_t + end_offset_s)
    vals = ang_vel_abs[mask]
    if vals.size == 0 or np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmax(vals))


def plot_peak_velocity(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    start_offset_s: float = 0.0,
    end_offset_s: float = 0.0,
    responsive_only: bool = False,
    ax_size: tuple[float, float] = (7, 5),
) -> Figure:
    """Violin plot of peak |angular velocity| within the stimulus window.

    The extraction window per trial is
    ``[0 + start_offset_s,  end_expansion_time + end_offset_s]``.
    Defaults correspond to the full expansion period (onset → end of expansion).

    Dimension roles:
        - ``row_by``:  subplot rows
        - ``col_by``:  x-axis tick categories within each subplot
        - ``hue_by``:  side-by-side violins within each x tick

    Args:
        responses: Response list (each must have ``time``, ``ang_vel``,
            ``end_expansion_time``).
        row_by: Column for subplot rows.
        col_by: Column for x-axis ticks.
        hue_by: Column for side-by-side violins.
        start_offset_s: Seconds added to stimulus onset (t=0) to define window
            start.  Use a negative value to start before onset.
        end_offset_s: Seconds added to ``end_expansion_time`` to define window
            end.  Use a positive value to extend past the end of expansion.
        ax_size: (width, height) per subplot in inches.

    Returns:
        The matplotlib Figure.
    """
    return plot_violin_facets(
        responses,
        value_fn=lambda r: _extract_peak(r, start_offset_s, end_offset_s),
        row_by=row_by,
        col_by=col_by,
        hue_by=hue_by,
        responsive_only=responsive_only,
        ylabel="Peak |angular velocity| (deg/s)",
        title="Peak Angular Velocity",
        show_zero_line=False,
        ax_size=ax_size,
    )
