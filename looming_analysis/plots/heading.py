"""Heading change distribution (violin) plots, faceted."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ._common import build_hue_colormap, unique_values

Response = dict


def plot_heading_changes(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    absolute: bool = False,
    ax_size: tuple[float, float] = (7, 5),
) -> Figure:
    """Violin plot of heading changes.

    Dimension roles for this plot:
        - `row_by`:  subplot rows (None → single row)
        - `col_by`:  x-axis tick categories within each subplot
        - `hue_by`:  side-by-side violins within each x tick

    This mirrors how you'd typically describe violins: "x on `col_by`,
    hue on `hue_by`, facet rows on `row_by`."

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
    row_vals = unique_values(responses, row_by)
    x_vals = unique_values(responses, col_by)
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]

    n_rows = len(row_vals) if row_vals else 1
    effective_rows = row_vals if row_vals else [None]
    effective_x = x_vals if x_vals else [None]

    width = max(ax_size[0], 1.0 * max(len(effective_x), 1) * max(len(hue_vals), 1))

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(width, ax_size[1] * n_rows), squeeze=False
    )
    color_map = build_hue_colormap(responses, hue_by)

    for row_idx, row_val in enumerate(effective_rows):
        ax = axes[row_idx, 0]
        _draw_violins(
            ax,
            responses=responses,
            row_by=row_by,
            row_val=row_val,
            col_by=col_by,
            x_vals=effective_x,
            hue_by=hue_by,
            hue_vals=hue_vals,
            color_map=color_map,
            absolute=absolute,
        )

        if row_idx == 0 and hue_by is not None:
            ax.legend(
                handles=[
                    Patch(facecolor=color_map[hv], alpha=0.7, label=str(hv))
                    for hv in hue_vals
                ],
                title=hue_by,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
        ylabel_base = "|Heading change| (deg)" if absolute else "Heading change (deg)"
        if row_by is not None:
            ax.set_ylabel(f"{row_by} = {row_val}\n{ylabel_base}")
        else:
            ax.set_ylabel(ylabel_base)
        if row_idx == n_rows - 1 and col_by is not None:
            ax.set_xlabel(col_by)
        if not absolute:
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")

    title = "Heading Change Distribution"
    dims = [
        f"{k}={v}" for k, v in [("rows", row_by), ("x", col_by), ("hue", hue_by)] if v
    ]
    if dims:
        title += "  |  " + ", ".join(dims)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def _draw_violins(
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
    absolute,
):
    violin_width = 0.7
    gap = 1.0
    x = 0.0
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for xv in x_vals:
        group_start = x
        for hv in hue_vals:
            raw = [
                r["heading_change"]
                for r in responses
                if (row_by is None or r.get(row_by) == row_val)
                and (col_by is None or r.get(col_by) == xv)
                and (hue_by is None or r.get(hue_by) == hv)
            ]
            vals = [abs(v) for v in raw] if absolute else raw
            if vals:
                vp = ax.violinplot(
                    [vals], positions=[x], widths=violin_width, showmeans=True
                )
                for pc in vp["bodies"]:
                    pc.set_facecolor(color_map[hv])
                    pc.set_alpha(0.7)
                for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
                    if partname in vp:
                        vp[partname].set_color(color_map[hv])
                top = max(vals)
                ax.text(
                    x,
                    top,
                    f"n={len(vals)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            x += 1
        tick_positions.append((group_start + x - 1) / 2)
        tick_labels.append(str(xv) if xv is not None else "all")
        x += gap

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
