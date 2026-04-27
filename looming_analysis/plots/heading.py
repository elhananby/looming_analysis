"""Heading change distribution (violin) plots, faceted."""

from __future__ import annotations

from typing import Optional

from matplotlib.figure import Figure

from .._types import Response
from ._common import plot_violin_facets


def plot_heading_changes(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    absolute: bool = False,
    responsive_only: bool = False,
    ax_size: tuple[float, float] = (7, 5),
) -> Figure:
    """Violin plot of heading changes.

    Dimension roles for this plot:
        - `row_by`:  subplot rows (None → single row)
        - `col_by`:  x-axis tick categories within each subplot
        - `hue_by`:  side-by-side violins within each x tick

    Args:
        responses: Response list (each must have `heading_change`).
        row_by: Column for subplot rows.
        col_by: Column for x-axis ticks. If None, all responses form a single tick.
        hue_by: Column for side-by-side violins at each tick.
        absolute: If True, plot |heading_change| instead of the signed value.
            The zero-reference line is hidden in this mode.
        ax_size: (width, height) per subplot.

    Returns:
        The matplotlib Figure.
    """
    value_fn = (
        (lambda r: abs(r["heading_change"]))
        if absolute
        else (lambda r: r["heading_change"])
    )
    return plot_violin_facets(
        responses,
        value_fn=value_fn,
        row_by=row_by,
        col_by=col_by,
        hue_by=hue_by,
        responsive_only=responsive_only,
        ylabel="|Heading change| (deg)" if absolute else "Heading change (deg)",
        title="Heading Change Distribution",
        show_zero_line=not absolute,
        ax_size=ax_size,
    )
