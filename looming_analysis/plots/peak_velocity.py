"""Peak angular velocity violin plots, faceted."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ._common import build_hue_colormap, unique_values

Response = dict


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
            start_offset_s=start_offset_s,
            end_offset_s=end_offset_s,
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
        ylabel = "Peak |angular velocity| (deg/s)"
        if row_by is not None:
            ax.set_ylabel(f"{row_by} = {row_val}\n{ylabel}")
        else:
            ax.set_ylabel(ylabel)
        if row_idx == n_rows - 1 and col_by is not None:
            ax.set_xlabel(col_by)
        ax.grid(True, alpha=0.3, axis="y")

    title = "Peak Angular Velocity"
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
    start_offset_s,
    end_offset_s,
):
    violin_width = 0.7
    gap = 1.0
    x = 0.0
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for xv in x_vals:
        group_start = x
        for hv in hue_vals:
            subset = [
                r
                for r in responses
                if (row_by is None or r.get(row_by) == row_val)
                and (col_by is None or r.get(col_by) == xv)
                and (hue_by is None or r.get(hue_by) == hv)
            ]
            vals = [
                v
                for v in (
                    _extract_peak(r, start_offset_s, end_offset_s) for r in subset
                )
                if not np.isnan(v)
            ]
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
                ax.text(
                    x,
                    max(vals),
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
